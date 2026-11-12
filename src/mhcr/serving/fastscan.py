"""FastScan search with explicit routing conventions and FP32 refinement.

Prefix coverage uses torch.cumsum to preserve the accumulation convention of
published checkpoints. Equal density values use stable cell order. The encoder
is scripted when supported. Only non-residual inner-product indices are valid.
"""

import math, time
import numpy as np
import torch
import faiss
from torch.nn import functional as F

JIT_STATUS = None


class CachedRouter:
    def __init__(self, router, variant="stable"):
        if variant not in ("stable", "datacomp"):
            raise ValueError("Unknown routing variant")
        self.variant = variant
        self.encoder = router.encoder
        self.scorer = router.scorer
        self.mean = router.mean
        self.tau = router.tau
        self.cost = router.cost.clone()
        self.cost64 = self.cost.double()
        self.logcost = self.cost.log()
        assert torch.isfinite(self.cost).all() and torch.all(self.cost > 0)
        self.cost64_np = np.ascontiguousarray(self.cost64.numpy())
        self.eps32 = 2 * np.finfo(np.float32).eps
        global JIT_STATUS
        if variant == "datacomp":
            self.encoder_jit = None
            return
        try:
            self.encoder_jit = torch.jit.script(router.encoder)
            JIT_STATUS = "scripted"
        except Exception as error:
            self.encoder_jit = None
            JIT_STATUS = repr(error)

    @torch.inference_mode()
    def route(self, features, penalty):
        assert len(features) == 1 and math.isfinite(penalty) and penalty >= 0
        encoder = self.encoder_jit if self.encoder_jit is not None else self.encoder
        logits = (
            self.scorer(F.normalize(encoder(features - self.mean), dim=-1)) / self.tau
        )
        assert torch.isfinite(logits).all()
        p = torch.softmax(logits, 1)
        density = logits - self.logcost
        k = logits.shape[1]
        width = k
        if penalty > 0:
            guard = self.eps32 * torch.clamp(p.double().sum(1), min=1)
            possible = p.double() - penalty * self.cost64 >= -guard[:, None]
            floor = torch.where(possible, density, torch.inf).amin(1)
            width = min(
                k,
                max(
                    64,
                    ((int((density >= floor[:, None]).sum(1).max()) + 63) // 64) * 64,
                ),
            )
        if self.variant == "datacomp":
            # Preserve torch.topk ordering and accumulation of the 128M run.
            _, cells = torch.topk(density, width, dim=1, sorted=True)
            objective = torch.cat(
                [
                    torch.zeros((1, 1), dtype=torch.float64),
                    torch.cumsum(p.gather(1, cells), 1).double()
                    - penalty * torch.cumsum(self.cost64[cells], 1),
                ],
                1,
            )
            budget = int(objective.argmax(1).clamp(min=1)[0])
            return np.ascontiguousarray(cells.numpy()[0, :budget], np.int64)
        if penalty == 0:
            cells = np.argsort(-density[0].numpy(), kind="stable")
            return np.ascontiguousarray(cells, np.int64)
        cells, budget = self.prefix(p[0], density[0].numpy(), width, penalty)
        return np.ascontiguousarray(cells[:budget], np.int64)

    def prefix(self, probability, row, width, penalty):
        """Frozen topk+fallback semantics, reproduced with stable numpy orders."""
        index = np.argpartition(-row, width - 1)[:width]
        values = row[index]
        low = values.min()
        ties = int(np.count_nonzero(row == low))
        held = int(np.count_nonzero(values == low))
        if ties > held or np.any(values[1:] == values[:-1]) or ties > 1:
            cells = np.argsort(-row, kind="stable")[:width]
        else:
            cells = np.take(index, np.lexsort((index, -values)))
        picked = torch.from_numpy(np.ascontiguousarray(cells))
        coverage = (
            torch.cumsum(torch.index_select(probability, 0, picked), 0).double().numpy()
        )
        total = np.cumsum(self.cost64_np[cells])
        objective = np.concatenate(
            (np.zeros(1, dtype=np.float64), coverage - penalty * total)
        )
        return cells, max(int(np.argmax(objective)), 1)


def top100(ids, scores, stable_ties=True):
    """Partial top-100 selection with deterministic score/ID boundary ties."""
    assert len(ids) == len(scores) and len(ids) >= 100 and np.isfinite(scores).all()
    if len(ids) == 100:
        return np.lexsort((ids, -scores))
    keep = np.argpartition(-scores, 99)[:100]
    if not stable_ties:
        return keep[np.lexsort((ids[keep], -scores[keep]))]
    threshold = scores[keep].min()
    ties = np.flatnonzero(scores == threshold)
    greater = keep[scores[keep] > threshold]
    needed = 100 - len(greater)
    if len(ties) > needed:
        tied = ties[np.argpartition(ids[ties], needed - 1)[:needed]]
        keep = np.concatenate([greater, tied])
    return keep[np.lexsort((ids[keep], -scores[keep]))]


def refine_scores(base, query, ids):
    """Use the gallery's canonical kernel without materializing its full corpus."""
    if hasattr(base, "scores"):
        return base.scores(query, ids)
    scores = np.empty((1, len(ids)), np.float32)
    faiss.fvec_inner_products_by_idx(
        faiss.swig_ptr(scores),
        faiss.swig_ptr(query),
        faiss.swig_ptr(base),
        faiss.swig_ptr(ids),
        query.shape[1],
        1,
        len(ids),
    )
    return scores[0]


class Engine:
    def __init__(self, reference):
        self.index = reference.index
        self.base = reference.base
        self.base_format = getattr(reference, "base_format", "fp32")
        self.sizes = reference.sizes
        assert isinstance(self.index, faiss.IndexIVFPQFastScan)
        assert (
            not self.index.by_residual
            and self.index.metric_type == faiss.METRIC_INNER_PRODUCT
        )
        self.routing_variant = getattr(reference, "routing_variant", "stable")
        self.normalize_routing = self.routing_variant != "datacomp"
        self.router = CachedRouter(reference.router, self.routing_variant)

    def top100(self, ids, scores):
        return top100(ids, scores, stable_ties=self.routing_variant != "datacomp")

    def search(self, query, point, profile=False):
        start = time.perf_counter_ns() if profile else 0
        raw = np.array(query.reshape(1, -1), dtype=np.float32, order="C", copy=True)
        assert raw.shape == (1, self.index.d) and np.isfinite(raw).all()
        feature = raw.copy()
        if self.normalize_routing:
            faiss.normalize_L2(feature)
        if point["method"] == "IVF":
            coarse, cells = self.index.quantizer.search(raw, int(point["nprobe"]))
            chosen = cells[0]
        else:
            assert point["method"] == "MHCR"
            chosen = self.router.route(torch.from_numpy(feature), point["penalty"])
            coarse = None
        if int(self.sizes[chosen].sum()) < 100:
            if point["method"] == "IVF":
                coarse, full = self.index.quantizer.search(raw, self.index.nlist)
                order = full[0]
            else:
                order = self.router.route(torch.from_numpy(feature), 0.0)
            width = int(np.searchsorted(np.cumsum(self.sizes[order]), 100)) + 1
            chosen = order[: max(width, len(chosen))]
        chosen = np.ascontiguousarray(chosen[None], np.int64)
        coarse = (
            np.zeros(chosen.shape, np.float32)
            if coarse is None
            else np.ascontiguousarray(coarse[:, : chosen.shape[1]], np.float32)
        )
        self.index.nprobe = chosen.shape[1]
        assert point["rerank"] >= 100
        routed = time.perf_counter_ns() if profile else 0
        _, short = self.index.search_preassigned(
            raw, int(point["rerank"]), chosen, coarse
        )
        scanned = time.perf_counter_ns() if profile else 0
        ids = np.ascontiguousarray(short[0, short[0] >= 0], np.int64)
        assert len(ids) >= 100 and len(np.unique(ids)) == len(ids)
        scores = refine_scores(self.base, raw, ids)
        order = self.top100(ids, scores)
        found = ids[order]
        values = scores[order]
        if not profile:
            return found, values
        end = time.perf_counter_ns()
        return (
            found,
            values,
            dict(
                route_prepare_ms=(routed - start) / 1e6,
                scan_ms=(scanned - routed) / 1e6,
                refine_validate_topk_ms=(end - scanned) / 1e6,
                total_ms=(end - start) / 1e6,
                selected_cells=int(chosen.shape[1]),
                scanned_entries=int(self.sizes[chosen[0]].sum()),
                refined_candidates=int(len(ids)),
            ),
        )
