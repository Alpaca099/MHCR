"""Controlled serving ablations for MHCR-FastScan.

Arms share the same index, FP32 base, cost model and refinement budget;
they differ only in the per-cell score and the prefix policy:

  full             learned p_k(q) from the frozen router, adaptive prefix (lambda)
  fixed_probe      same learned p_k, fixed prefix length m
  cosine_adaptive  softmax(qhat . ahat_k / T) with a fitted scalar T, adaptive prefix

The full arm delegates routing to the reference engine. Fixed-probe and cosine
controls retain stable tie ordering and the same refinement implementation.
"""

import numpy as np, torch, torch.nn.functional as F, math, faiss

EPS32 = float(np.finfo(np.float32).eps)


def _prefix(p, density, penalty, cost64, eps32):
    k = density.shape[1]
    width = k
    if penalty > 0:
        guard = 2 * eps32 * torch.clamp(p.double().sum(1), min=1)
        possible = p.double() - penalty * cost64 >= -guard[:, None]
        floor = torch.where(possible, density, torch.inf).amin(1)
        width = min(
            k,
            max(64, ((int((density >= floor[:, None]).sum(1).max()) + 63) // 64) * 64),
        )
    if width < k:
        values, cells = torch.topk(density, width, dim=1, sorted=True)
        if torch.any(values[:, 1:] == values[:, :-1]) or torch.any(
            (density == values[:, -1:]).sum(1) > 1
        ):
            cells = torch.argsort(density, dim=1, descending=True, stable=True)
    else:
        cells = torch.argsort(density, dim=1, descending=True, stable=True)
    coverage = torch.cumsum(p.gather(1, cells), 1).double()
    total = torch.cumsum(cost64[cells], 1)
    obj = torch.cat(
        [
            torch.zeros((len(density), 1), dtype=torch.float64),
            coverage - penalty * total,
        ],
        1,
    )
    budget = obj.argmax(1).clamp(min=1)
    return cells[0, : int(budget[0])]


def _topm(density, m):
    k = len(density)
    idx = np.argpartition(-density, m - 1)[:m]
    vals = density[idx]
    low = vals.min()
    ties = int(np.count_nonzero(density == low))
    held = int(np.count_nonzero(vals == low))
    if ties > held or np.any(vals[1:] == vals[:-1]) or ties > 1:
        return np.argsort(-density, kind="stable")[:m]
    return np.take(idx, np.lexsort((idx, -vals)))


class AblationEngine:
    MODES = ("full", "fixed_probe", "cosine_adaptive", "cosine_fixed", "ivf")

    def __init__(
        self, reference, router, index, centers, costs, mode, param=None, T=None
    ):
        assert mode in self.MODES, mode
        assert router is not None
        self.mode = mode
        self.param = param
        self.T = T
        self.index = index
        self.base = reference.base
        self.normalize_routing = getattr(reference, "normalize_routing", True)
        self.sizes = reference.sizes
        self.cr = router
        self.top100 = reference.top100 if hasattr(reference, "top100") else None
        self.costs = np.asarray(costs, np.float64)
        self.cost64_np = self.costs
        self.k = int(costs.shape[0])
        cent = np.asarray(centers, np.float64)
        nrm = np.linalg.norm(cent, axis=1)
        zero = np.flatnonzero(nrm == 0.0)
        assert zero.size == 0, "zero-norm centroids: %s" % zero[:8].tolist()
        self.cnorm = (cent / nrm[:, None]).astype(np.float32)
        self.ct = torch.from_numpy(self.cnorm)
        self.n_guard = 0
        self.n_extend = 0
        self.last_cells = None
        self.last_entries = 0
        self.T = None if T is None else float(T)
        if mode in ("cosine_adaptive", "cosine_fixed"):
            assert self.T is not None and self.T > 0, "cosine_adaptive needs a fitted T"
        if mode in ("fixed_probe", "cosine_fixed", "ivf"):
            assert (
                isinstance(param, int) and 0 < param <= self.k
            ), "Probe count must be in [1, K]"

    def _head(self, feat):
        """Return (p, density) as 1xK torch tensors for the learned arms."""
        logits = (
            self.cr.scorer(F.normalize(self.cr.encoder(feat - self.cr.mean), dim=-1))
            / self.cr.tau
        )
        p = torch.softmax(logits, 1)
        density = logits - self.cr.logcost
        return p, density

    def _geo(self, raw):
        qn = (raw[0] / np.linalg.norm(raw[0])).astype(np.float32)
        s = torch.mv(self.ct, torch.from_numpy(qn)).reshape(1, -1) / self.T
        p = torch.softmax(s, 1)
        density = s - self.cr.logcost
        return p, density

    def cells(self, feature, raw, point):
        penalty = float(point.get("penalty", 0.0))
        if self.mode == "ivf":
            _, ids = self.index.quantizer.search(raw, int(self.param))
            return np.ascontiguousarray(ids[0], np.int64)
        if self.mode == "full":
            return np.ascontiguousarray(
                self.cr.route(torch.from_numpy(feature), penalty), np.int64
            )
        with torch.inference_mode():
            if self.mode == "cosine_fixed":
                p, density = self._geo(raw)
                cells = _topm(density[0].numpy(), int(self.param))
                if int(self.sizes[cells].sum()) < 100:
                    self.n_guard += 1
                    full = np.argsort(-density[0].numpy(), kind="stable")
                    width = int(np.searchsorted(np.cumsum(self.sizes[full]), 100)) + 1
                    cells = full[: max(width, len(cells))]
                    self.n_extend += 1
                return cells
            if self.mode == "fixed_probe":
                p, density = self._head(torch.from_numpy(feature))
                cells = _topm(density[0].numpy(), int(self.param))
                if int(self.sizes[cells].sum()) < 100:
                    self.n_guard += 1
                    full = np.argsort(-density[0].numpy(), kind="stable")
                    width = int(np.searchsorted(np.cumsum(self.sizes[full]), 100)) + 1
                    cells = full[: max(width, len(cells))]
                    self.n_extend += 1
                return cells
            p, density = self._geo(raw)
            cost64 = torch.from_numpy(self.cost64_np)
            cells = _prefix(p, density, penalty, cost64, EPS32).numpy().astype(np.int64)
            if int(self.sizes[cells].sum()) < 100:
                self.n_guard += 1
                order = np.argsort(-density[0].numpy(), kind="stable")
                width = int(np.searchsorted(np.cumsum(self.sizes[order]), 100)) + 1
                cells = order[: max(width, len(cells))]
                self.n_extend += 1
            return cells

    def scored_prefix(self, feature, raw, point):
        """(cells, p_selected_sum, observed_unavailable) for probability calibration."""
        if feature is None:
            feature = np.ascontiguousarray(raw.reshape(1, -1), np.float32)
            if self.normalize_routing:
                faiss.normalize_L2(feature)
        cells = self.cells(feature, raw, point)
        if self.mode == "ivf":
            return cells, float("nan")
        with torch.inference_mode():
            if self.mode in ("cosine_adaptive", "cosine_fixed"):
                p, _ = self._geo(raw)
            else:
                p, _ = self._head(torch.from_numpy(feature))
            mass = float(p[0, torch.from_numpy(cells)].sum())
        return cells, mass

    def search(self, query, point):
        import time

        t0 = time.perf_counter_ns()
        raw = np.array(query.reshape(1, -1), dtype=np.float32, order="C", copy=True)
        feature = raw.copy()
        if self.normalize_routing:
            faiss.normalize_L2(feature)
        cells = self.cells(feature, raw, point)
        if int(self.sizes[cells].sum()) < 100:
            self.n_guard += 1
            order = self.cr.route(torch.from_numpy(feature), 0.0)
            width = int(np.searchsorted(np.cumsum(self.sizes[order]), 100)) + 1
            cells = order[: max(width, len(cells))]
            self.n_extend += 1
        chosen = np.ascontiguousarray(cells[None], np.int64)
        coarse = np.zeros(chosen.shape, np.float32)
        self.index.nprobe = chosen.shape[1]
        assert int(point["rerank"]) >= 100
        _, short = self.index.search_preassigned(
            raw, int(point["rerank"]), chosen, coarse
        )
        ids = np.ascontiguousarray(short[0, short[0] >= 0], np.int64)
        from mhcr.serving.fastscan import top100, refine_scores

        scores = refine_scores(self.base, raw, ids)
        order = (self.top100 or top100)(ids, scores)
        entries = int(self.sizes[cells].sum())
        ms = (time.perf_counter_ns() - t0) / 1e6
        return ids[order], scores[order], cells, entries, ms
