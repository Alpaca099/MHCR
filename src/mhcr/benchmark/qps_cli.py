"""Standalone audited process-QPS command; never tunes operating points."""

import argparse
import functools
import os
from pathlib import Path
import platform
import json
import numpy as np
from .processes import trial_processes
from .qps import audit_trial, TrialFailed
from ..storage import save, sha
from ..data import validate_truth


def load_engine(path):
    import torch
    import faiss
    from ..serving.engine import Engine

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    faiss.omp_set_num_threads(1)
    return Engine.from_manifest(path)


def persist(directory, result):
    directory.mkdir(parents=True, exist_ok=True)
    meta = {k: v for k, v in result.items() if k != "records"}
    meta["records"] = []
    for record in result.get("records", []):
        arrays = {k: v for k, v in record.items() if isinstance(v, np.ndarray)}
        np.savez_compressed(directory / f"worker-{record['worker']}.npz", **arrays)
        meta["records"].append({k: v for k, v in record.items() if k not in arrays})
    save(directory / "trial.json", meta)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("engine", "queries", "point", "cores", "out"):
        p.add_argument("--" + key, required=True)
    p.add_argument("--workers", default="1,2,4,8")
    p.add_argument(
        "--dataset-config", help="Assert dataset N, D and K against the engine assets"
    )
    p.add_argument("--cycles", type=int, default=3)
    p.add_argument("--warmup", type=int, default=24)
    p.add_argument("--start-method", choices=["spawn", "fork"], default="spawn")
    p.add_argument("--ready-timeout", type=float, default=600)
    p.add_argument("--run-timeout", type=float, default=900)
    p.add_argument("--min-recall", type=float, default=0)
    a = p.parse_args()
    cores = [int(x) for x in a.cores.split(",")]
    wanted = [int(x) for x in a.workers.split(",")]
    if (
        not wanted
        or min(wanted) < 1
        or len(set(wanted)) != len(wanted)
        or max(wanted) > len(cores)
    ):
        p.error(
            "Worker counts must be positive, unique and no greater than the core count"
        )
    if len(set(cores)) != len(cores) or not set(cores) <= os.sched_getaffinity(0):
        p.error("Cores must be distinct and within current CPU affinity")
    if (
        a.cycles < 1
        or a.warmup < 0
        or min(a.ready_timeout, a.run_timeout) <= 0
        or not 0 <= a.min_recall <= 1
    ):
        p.error("Invalid cycles, warmup, timeout or recall threshold")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[name] = "1"
    with np.load(a.queries, allow_pickle=False) as z:
        q, gt, qids = z["q"], z["gt"], z["query_ids"]
    validate_truth(gt)
    if (
        q.ndim != 2
        or len(q) == 0
        or len(q) != len(gt)
        or len(q) != len(qids)
        or len(np.unique(qids)) != len(qids)
        or not np.isfinite(q).all()
    ):
        raise ValueError("Invalid evaluation cohort")
    point = json.loads(Path(a.point).read_text())
    manifest = str(Path(a.engine).resolve())
    factory = functools.partial(load_engine, manifest)
    engine = factory() if a.start_method == "fork" else None
    import faiss

    config = json.loads(Path(manifest).read_text())
    # Avoid loading an extra index in the parent: canonical gallery header gives N.
    from ..serving.engine import load_gallery

    base = load_gallery(config, Path(manifest).parent)
    n, dimension = base.shape
    if hasattr(base, "close"):
        base.close()
    if a.dataset_config:
        from ..cli import config as read_config

        dc = read_config(a.dataset_config)
        if (n, dimension) != (dc["n"], dc["d"]):
            raise ValueError("Dataset/gallery mismatch")
        for key, default in (("base_format", "fp32"), ("routing_variant", "stable")):
            if key in dc and config.get(key, default) != dc[key]:
                raise ValueError(f"Dataset/engine {key} mismatch")
        # Dimension and partition checks occur in each Engine loader; a declared
        # partition count is passed through the manifest check below.
        if config.get("expected_nlist") != dc["k"]:
            raise ValueError("Set engine manifest expected_nlist to dataset K")
    if q.shape[1] != dimension:
        raise ValueError("Query dimension differs from gallery")
    validate_truth(gt, n)
    protocol = dict(
        options=vars(a),
        point=point,
        engine_manifest_sha256=sha(manifest),
        queries_sha256=sha(a.queries),
        platform=platform.platform(),
        python=platform.python_version(),
        faiss=faiss.__version__,
        timing="End-to-end request loop; output capture and fault sampling included",
        memory="Worker peak RSS includes setup; shared pages counted in each RSS; do not sum as unique RAM",
    )
    save(out / "protocol.json", protocol)
    rows = []

    def run(name, workers, cycles, reference=None):
        directory = out / name
        try:
            result = trial_processes(
                engine,
                q,
                qids,
                point,
                cycles,
                cores[:workers],
                engine_factory=factory if engine is None else None,
                start_method=a.start_method,
                warmup=a.warmup,
                ready_timeout=a.ready_timeout,
                run_timeout=a.run_timeout,
            )
        except TrialFailed as exc:
            persist(directory, exc.evidence)
            save(out / "status.json", dict(complete=False, failed_stage=name))
            raise
        persist(directory, dict(result, valid=False, audit_pending=True))
        if reference is None:
            reference = [None] * len(q)
            for rec in result["records"]:
                for i, j in enumerate(rec["order"]):
                    reference[int(j)] = (rec["ids"][i].copy(), rec["scores"][i].copy())
        try:
            audit = audit_trial(result, qids, gt, n, reference)
            if audit["recall100"] < a.min_recall:
                raise ValueError("Frozen point misses minimum recall")
        except BaseException as exc:
            save(directory / "audit.json", dict(passed=False, error=repr(exc)))
            save(out / "status.json", dict(complete=False, failed_stage=name))
            raise
        save(directory / "audit.json", audit)
        persist(directory, dict(result, valid=True, audit_pending=False))
        return result, audit, reference

    _, _, reference = run("reference", 1, 1)
    for workers in wanted:
        result, audit, _ = run(f"workers-{workers}", workers, a.cycles, reference)
        lat = np.concatenate(
            [(r["end_ns"] - r["start_ns"]) / 1e6 for r in result["records"]]
        )
        rows.append(
            dict(
                workers=result["workers"],
                qps=result["qps"],
                recall100=audit["recall100"],
                mean_ms=float(lat.mean()),
                p99_ms=float(np.percentile(lat, 99)),
                wall_seconds=result["wall_seconds"],
            )
        )
        baseline = next((r["qps"] for r in rows if r["workers"] == 1), None)
        for r in rows:
            r["speedup_vs_one_worker"] = r["qps"] / baseline if baseline else None
        save(out / "qps.json", dict(rows=rows, complete=False))
        print(json.dumps(rows[-1]), flush=True)
    save(out / "qps.json", dict(rows=rows, complete=True))
    save(out / "status.json", dict(complete=True))


if __name__ == "__main__":
    main()
