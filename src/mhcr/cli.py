"""Portable preparation, training and Recall@100 evaluation entrypoints."""

import argparse, os
from pathlib import Path
import json
import time
import numpy as np
import torch
import faiss
import yaml
from .data import load_array, validate_truth, open_gallery, gallery_identity
from .storage import read, save, sha


def config(path):
    c = yaml.safe_load(Path(path).read_text())
    if c.get("top_k") != 100:
        raise ValueError("Configuration must explicitly set top_k=100")
    return c


def recall100(found, truth):
    validate_truth(truth)
    if found.shape != truth.shape:
        raise ValueError("Returned IDs must have shape Q x 100")
    validate_truth(found)
    return np.array([len(np.intersect1d(a, b)) / 100 for a, b in zip(found, truth)])


def build_main(argv=None):
    from .index.build import build

    p = argparse.ArgumentParser(
        description="Build a full non-residual PQ4 FastScan index"
    )
    p.add_argument("--config", required=True)
    p.add_argument(
        "--base",
        required=True,
        help="Normalized FP32 array or configured FP16 shard manifest",
    )
    p.add_argument("--out", required=True)
    p.add_argument("--threads", type=int, default=1)
    a = p.parse_args(argv)
    c = config(a.config)
    required = [
        "seed",
        "partition_sample",
        "partition_iterations",
        "pq_sample",
        "pq_iterations",
        "build_chunk",
    ]
    missing = [key for key in required if c.get(key) is None]
    if missing:
        p.error("Supply explicit index-training recipe fields: " + ", ".join(missing))
    faiss.omp_set_num_threads(a.threads)
    base = open_gallery(a.base, c.get("base_format", "fp32"))
    if base.shape != (c["n"], c["d"]) or base.dtype != np.float32:
        raise ValueError("Gallery shape/dtype differs from config")
    # Bounded chunks; never materialize the full gallery.
    for lo in range(0, len(base), 32768):
        x = base[lo : lo + 32768]
        if not np.isfinite(x).all() or not np.allclose(
            np.linalg.norm(x, axis=1), 1, atol=1e-4
        ):
            raise ValueError("Gallery must be pre-normalized")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    identity = dict(
        config=c, gallery=gallery_identity(a.base, c.get("base_format", "fp32"))
    )
    pp = out / "protocol.json"
    if pp.exists() and read(pp) != identity:
        raise ValueError("Build identity changed")
    if not pp.exists():
        save(pp, identity)
    build(base, out, c, sha(pp))


def train_main(argv=None):
    from .training import train_epochs
    from .teacher import train_teacher
    from .projection import project_teacher
    from .router import QueryCellRouter
    from .serving.engine import load_model

    p = argparse.ArgumentParser(
        description="Train teacher or student using verified GT100 cell labels"
    )
    for key in ["config", "features", "cells", "out", "identity"]:
        p.add_argument("--" + key, required=True)
    p.add_argument("--stage", choices=["teacher", "student"], required=True)
    p.add_argument("--teacher")
    p.add_argument("--centroids")
    p.add_argument("--cost")
    p.add_argument("--sizes")
    p.add_argument("--threads", type=int, default=1)
    a = p.parse_args(argv)
    c = config(a.config)
    required = [
        "training_rows",
        "hidden",
        "epochs",
        "batch",
        "lr",
        "weight_decay",
        "seed",
    ]
    required += (
        ["rank_gamma"]
        if a.stage == "teacher"
        else ["dimension", "budget_unit", "budget_range", "family"]
    )
    tc = c.get(a.stage, {})
    missing = [key for key in required if tc.get(key) is None]
    if missing:
        p.error("Supply explicit " + a.stage + " recipe fields: " + ", ".join(missing))
    identity = read(a.identity)
    torch.set_num_threads(a.threads)
    torch.manual_seed(tc["seed"])
    x = np.asarray(load_array(a.features), np.float32)
    cells = np.asarray(load_array(a.cells), np.int64)
    if x.shape != (tc["training_rows"], c["d"]) or cells.shape != (len(x), 100):
        raise ValueError("Training dimensions/count do not match the explicit recipe")
    if not np.allclose(np.linalg.norm(x, axis=1), 1, atol=1e-4):
        raise ValueError("Expected normalized training queries")
    if np.any(cells < 0) or np.any(cells >= c["k"]):
        raise ValueError("Invalid target cell")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.stage == "teacher":
        if not a.centroids:
            p.error("--centroids required for teacher")
        centers = np.asarray(load_array(a.centroids), np.float32)
        if centers.shape != (c["k"], c["d"]):
            raise ValueError("Invalid centroids")
        model = QueryCellRouter(
            c["d"],
            c["d"],
            c["k"],
            x.mean(0),
            arch="res_mlp",
            hidden=tc["hidden"],
            tau=tc.get("tau", 0.07),
            proto_init=centers,
        )
        result = train_teacher(model, x, cells, out, identity, tc)
    else:
        if not all([a.teacher, a.cost, a.sizes]):
            p.error("--teacher --cost --sizes required for student")
        teacher = load_model(a.teacher)
        model = project_teacher(
            teacher, tc["dimension"], tc["hidden"], out / "projection.npy", identity
        )
        result = train_epochs(
            model,
            x,
            cells,
            load_array(a.cost),
            load_array(a.sizes),
            out,
            identity,
            tc,
            family=tc.get("family", "budget_tail"),
            teacher=teacher,
        )
    from .storage import atomic

    payload = dict(
        model=model.state_dict(),
        model_config=dict(
            d_clip=model.d_clip,
            d_r=model.d_r,
            K=model.K,
            arch=model.arch,
            hidden=tc["hidden"],
            tau=model.tau,
        ),
    )
    if not (out / "model.pt").exists():
        atomic(out / "model.pt", lambda f: torch.save(payload, f))
    else:
        from .training import model_hash

        if model_hash(load_model(out / "model.pt")) != model_hash(model):
            raise ValueError("Existing model export differs from completed training")
    print(json.dumps(result, indent=2))


def evaluate_main(argv=None):
    from .serving.engine import Engine

    p = argparse.ArgumentParser(
        description="Evaluate end-to-end Recall@100; optional calibration on separate queries"
    )
    p.add_argument("--engine", required=True)
    p.add_argument("--queries", required=True)
    p.add_argument(
        "--point", required=True, help="JSON method, penalty/nprobe and rerank"
    )
    p.add_argument("--out", required=True)
    p.add_argument("--calibration", help="Separate Q x D q, Q x 100 gt, query_ids NPZ")
    p.add_argument(
        "--dataset-config", help="Dataset YAML: assert gallery and index identity"
    )
    modes = p.add_mutually_exclusive_group()
    modes.add_argument(
        "--breakdown",
        action="store_true",
        help="Record routing, scan and refinement phases",
    )
    modes.add_argument(
        "--ablation",
        choices=["full", "fixed_probe", "cosine_adaptive", "cosine_fixed", "ivf"],
    )
    p.add_argument("--penalties", help="Comma separated finite nonnegative penalties")
    p.add_argument("--target", type=float)
    a = p.parse_args(argv)
    torch.set_num_threads(1)
    faiss.omp_set_num_threads(1)
    if Path(a.out).exists():
        raise FileExistsError("Evaluation output must be a new directory")
    engine = Engine.from_manifest(a.engine)
    point = read(a.point)
    if a.dataset_config:
        c = config(a.dataset_config)
        if (engine.index.ntotal, engine.index.d, engine.index.nlist) != (
            c["n"],
            c["d"],
            c["k"],
        ):
            raise ValueError("Dataset configuration differs from loaded index")
        for key in ("base_format", "routing_variant"):
            if key in c and getattr(engine, key) != c[key]:
                raise ValueError(f"Dataset/engine {key} mismatch")
    if a.ablation:
        from .benchmark.ablation import AblationEngine

        reference_engine = engine
        variant = AblationEngine(
            engine,
            engine.router,
            engine.index,
            engine.index.quantizer.reconstruct_n(0, engine.index.nlist),
            engine.router.cost.numpy(),
            a.ablation,
            param=point.get("probes"),
            T=point.get("temperature"),
        )

        class Adapter:
            index = reference_engine.index

            def search(self, query, operating_point):
                return variant.search(query, operating_point)[:2]

        engine = Adapter()
        if a.calibration and a.ablation in ("fixed_probe", "cosine_fixed", "ivf"):
            p.error(
                "Fixed-probe arms require externally calibrated probes in point.json"
            )
    with np.load(a.queries) as z:
        q, gt, ids = z["q"], z["gt"], z["query_ids"]
    validate_truth(gt, engine.index.ntotal)
    if (
        q.ndim != 2
        or len(q) == 0
        or q.shape[1] != engine.index.d
        or not np.isfinite(q).all()
        or len(q) != len(gt)
        or len(ids) != len(q)
        or len(np.unique(ids)) != len(ids)
    ):
        raise ValueError("Invalid evaluation cohort")
    if a.calibration:
        from .calibrate import calibrate_end_to_end

        if a.penalties is None or a.target is None:
            p.error("Calibration needs --penalties and --target")
        with np.load(a.calibration) as z:
            cq, cgt, cids = z["q"], z["gt"], z["query_ids"]
        if len(cids) != len(cq) or len(np.unique(cids)) != len(cids):
            raise ValueError("Invalid calibration IDs")
        if np.intersect1d(cids, ids).size:
            raise ValueError("Calibration/test query IDs overlap")
        if set(map(bytes, np.ascontiguousarray(cq))).intersection(
            map(bytes, np.ascontiguousarray(q))
        ):
            raise ValueError("Calibration/test query content overlaps")
        selected = calibrate_end_to_end(
            engine, cq, cgt, point, [float(v) for v in a.penalties.split(",")], a.target
        )
        point = dict(point, penalty=selected["penalty"])
    else:
        selected = None
    found = []
    scores = []
    latencies = []
    major = []
    minor = []
    phases = []
    import resource

    for query in q:
        before = resource.getrusage(resource.RUSAGE_SELF)
        start = time.perf_counter_ns()
        if a.breakdown:
            ii, ss, parts = engine.search(query, point, profile=True)
            phases.append(parts)
        else:
            ii, ss = engine.search(query, point)
        latencies.append((time.perf_counter_ns() - start) / 1e6)
        after = resource.getrusage(resource.RUSAGE_SELF)
        major.append(after.ru_majflt - before.ru_majflt)
        minor.append(after.ru_minflt - before.ru_minflt)
        found.append(ii)
        scores.append(ss)
    found = np.array(found)
    recall = recall100(found, gt)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    np.savez(
        out / "raw.npz",
        ids=found,
        scores=scores,
        query_ids=ids,
        recall100=recall,
        latency_ms=latencies,
        major_faults=major,
        minor_faults=minor,
        **(
            {key: np.asarray([row[key] for row in phases]) for key in phases[0]}
            if phases
            else {}
        ),
    )
    save(
        out / "summary.json",
        dict(
            point=point,
            calibration=selected,
            measurement=(
                "breakdown" if a.breakdown else "ablation" if a.ablation else "latency"
            ),
            ablation=a.ablation,
            phase_means=(
                {key: float(np.mean([row[key] for row in phases])) for key in phases[0]}
                if phases
                else None
            ),
            recall100=float(recall.mean()),
            mean_ms=float(np.mean(latencies)),
            p99_ms=float(np.percentile(latencies, 99)),
            protocol="Single-thread sequential portable evaluation; no warmup; not historical timing replay",
        ),
    )
