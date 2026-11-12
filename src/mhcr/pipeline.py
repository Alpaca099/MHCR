"""Build, train, calibrate and evaluate from vectors in a fresh output directory."""

import argparse
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import numpy as np
import torch
import faiss
import yaml
from .storage import read, save, sha, atomic, memory
from .data import (
    open_gallery,
    gallery_identity,
    file_identity,
    label_rows,
    validate_truth,
)
from .preparation import prepare_queries, half_manifest, exact_truth, array_file

STAGES = (
    "prepare",
    "build",
    "labels",
    "cost",
    "teacher",
    "student",
    "calibrate",
    "evaluate",
)


def bind(path, value):
    if path.exists() and read(path) != value:
        raise ValueError(f"Immutable protocol changed: {path}")
    if not path.exists():
        save(path, value)


def prepare(c, inputs, out, threads):
    folder = out / "prepared"
    prepare_queries(inputs, c, folder)
    if "base_shards" in inputs:
        if c.get("base_format") != "fp16_shards":
            raise ValueError("base_shards requires base_format: fp16_shards")
        base = half_manifest(inputs["base_shards"], folder, inputs.get("inverse_norm"))
    else:
        base = Path(inputs["base"])
    gallery = open_gallery(base, c.get("base_format", "fp32"))
    if gallery.shape != (c["n"], c["d"]):
        raise ValueError("Full gallery shape differs from dataset configuration")
    save(
        folder / "gallery.json",
        dict(
            path=str(base.resolve()),
            base_format=c.get("base_format", "fp32"),
            identity=gallery_identity(base, c.get("base_format", "fp32")),
        ),
    )
    return (
        list(folder.glob("*.json"))
        + list(folder.glob("*.npz"))
        + list(folder.glob("*.npy"))
        + list(folder.glob("*.txt"))
    )


def build(c, inputs, out, threads):
    from .cli import build_main

    build_main(
        [
            "--config",
            str(out / "config.yaml"),
            "--base",
            read(out / "prepared/gallery.json")["path"],
            "--out",
            str(out / "built"),
            "--threads",
            str(threads),
        ]
    )
    return [
        out / "built/build_complete.json",
        out / "built/index/fastscan.faiss",
        *[out / "built" / n for n in ("centroids.npy", "assignment.npy", "sizes.npy")],
    ]


def labels(c, inputs, out, threads):
    g = read(out / "prepared/gallery.json")
    base = open_gallery(g["path"], g["base_format"])
    assign = np.load(out / "built/assignment.npy", mmap_mode="r")
    folder = out / "supervision"
    folder.mkdir(exist_ok=True)
    protocol = sha(out / "protocol.json")
    files = []
    origins = {}
    for name in ("training", "calibration", "test"):
        with np.load(out / "prepared" / (name + ".npz")) as z:
            q = z["q"]
            ids = z["query_ids"]
            gt = z["gt"] if "gt" in z else None
        if gt is None:
            gt = exact_truth(
                base,
                q,
                folder / (name + "_gt"),
                protocol,
                c.get("preparation", {}).get("query_chunk", 32),
                c.get("preparation", {}).get("base_chunk", 32768),
            )
            origins[name] = "Exact full-gallery blocked FP32 inner products"
        else:
            origins[name] = (
                "Supplied vector GT100; shape/IDs checked, source identity recorded"
            )
        validate_truth(gt, c["n"])
        if name == "training":
            array_file(folder / "features.npy", q)
            array_file(folder / "neighbors.npy", gt)
            label_rows(assign, gt, folder / "cells.npy", protocol, c["n"])
            count = c["student"]["training_rows"]
            if not 0 < count <= len(q):
                raise ValueError("Student training count exceeds clean teacher pool")
            rows = np.sort(
                np.random.default_rng(c["student"]["seed"]).choice(len(q), count, False)
            )
            array_file(folder / "student_rows.npy", rows)
            array_file(folder / "student_features.npy", q[rows])
            array_file(
                folder / "student_cells.npy",
                np.load(folder / "cells.npy", mmap_mode="r")[rows],
            )
        else:
            path = folder / (name + ".npz")
            if not path.exists():
                atomic(path, lambda f: np.savez(f, q=q, gt=gt, query_ids=ids))
    save(folder / "truth_sources.json", origins)
    return (
        list(folder.glob("*.npy"))
        + list(folder.glob("*.json"))
        + list(folder.glob("*.npz"))
    )


def cost(c, inputs, out, threads):
    from .cost import measure

    index = faiss.read_index(str(out / "built/index/fastscan.faiss"))
    q = np.load(out / "supervision/features.npy", mmap_mode="r")
    count = min(c.get("cost", {}).get("queries", 64), len(q))
    rows = np.sort(np.random.default_rng(c["seed"] + 3).choice(len(q), count, False))
    sizes = np.load(out / "built/sizes.npy")
    opts = c.get("cost", {})
    folder = out / "cost"
    folder.mkdir(exist_ok=True)
    costs, model, observations = measure(
        index,
        q[rows],
        sizes,
        opts.get("probes", [1, 4, 16, 64, 256, 1024]),
        opts.get("rerank", 5000),
        opts.get("repeats", 3),
        folder / "observations.json",
    )
    save(folder / "observations.json", observations)
    save(folder / "fit.json", dict(model, training_rows=rows.tolist()))
    array_file(folder / "cost.npy", costs)
    return list(folder.glob("*"))


def teacher(c, inputs, out, threads):
    from .cli import train_main

    identity = read(out / "prepared/identity.json")
    identity.update(
        pipeline_protocol=sha(out / "protocol.json"),
        labels_sha256=sha(out / "supervision/cells.npy"),
    )
    bind(out / "teacher_identity.json", identity)
    train_main(
        [
            "--config",
            str(out / "config.yaml"),
            "--features",
            str(out / "supervision/features.npy"),
            "--cells",
            str(out / "supervision/cells.npy"),
            "--out",
            str(out / "teacher"),
            "--identity",
            str(out / "teacher_identity.json"),
            "--stage",
            "teacher",
            "--centroids",
            str(out / "built/centroids.npy"),
            "--threads",
            str(threads),
        ]
    )
    return [
        out / "teacher/model.pt",
        out / "teacher/complete.json",
        out / "teacher_identity.json",
    ]


def student(c, inputs, out, threads):
    from .cli import train_main
    from .serving.engine import load_model
    from .projection import project_teacher
    from .algorithm import Router
    from .cost import derive_budget

    identity = read(out / "teacher_identity.json")
    identity["teacher_training_protocol"] = sha(out / "teacher/protocol.json")
    bind(out / "student_identity.json", identity)
    folder = out / "student"
    folder.mkdir(exist_ok=True)
    resolved = json.loads(json.dumps(c))
    if c["student"].get("budget_range") is None:
        t = load_model(out / "teacher/model.pt")
        m = project_teacher(
            t,
            c["student"]["dimension"],
            c["student"]["hidden"],
            folder / "projection.npy",
            identity,
        )
        router = Router(m, np.load(out / "cost/cost.npy"), True)
        q = np.load(out / "supervision/features.npy", mmap_mode="r")
        cells = np.load(out / "supervision/cells.npy", mmap_mode="r")
        rows = np.random.default_rng(c["seed"] + 4).choice(
            len(q), min(512, len(q)), False
        )
        b = derive_budget(
            router,
            q[rows],
            cells[rows],
            np.load(out / "built/sizes.npy"),
            c["student"]["budget_unit"],
        )
        bind(folder / "budget.json", dict(b, training_rows=rows.tolist()))
        resolved["student"]["budget_range"] = b["range"]
        del t, m, router
    path = out / "resolved_config.yaml"
    text = yaml.safe_dump(resolved, sort_keys=False)
    if path.exists() and path.read_text() != text:
        raise ValueError("Resolved training recipe changed")
    path.write_text(text)
    train_main(
        [
            "--config",
            str(path),
            "--features",
            str(out / "supervision/student_features.npy"),
            "--cells",
            str(out / "supervision/student_cells.npy"),
            "--out",
            str(folder),
            "--identity",
            str(out / "student_identity.json"),
            "--stage",
            "student",
            "--teacher",
            str(out / "teacher/model.pt"),
            "--cost",
            str(out / "cost/cost.npy"),
            "--sizes",
            str(out / "built/sizes.npy"),
            "--threads",
            str(threads),
        ]
    )
    gallery = read(out / "prepared/gallery.json")
    engine = dict(
        index=str(out / "built/index/fastscan.faiss"),
        base=gallery["path"],
        base_format=gallery["base_format"],
        model=str(folder / "model.pt"),
        cost=str(out / "cost/cost.npy"),
        expected_nlist=c["k"],
        top_k=100,
        quantized=True,
        routing_variant=c.get("routing_variant", "stable"),
    )
    if gallery["base_format"] == "fp16_shards":
        from .serving.fp16 import build_library

        library = out / "libmhcr_fp16.so"
        if not library.exists():
            build_library(library)
        engine["native_library"] = str(library)
    bind(out / "engine.json", engine)
    return [
        folder / "model.pt",
        folder / "complete.json",
        out / "engine.json",
        path,
        out / "student_identity.json",
    ]


def calibrate(c, inputs, out, threads):
    from .serving.engine import Engine
    from .cli import recall100

    engine = Engine.from_manifest(out / "engine.json")
    with np.load(out / "supervision/calibration.npz") as z:
        q = z["q"]
        gt = z["gt"]
    folder = out / "calibration"
    folder.mkdir(exist_ok=True)
    cfg = c.get("evaluation", {})
    costs = engine.router.cost.numpy()
    penalties = cfg.get(
        "penalties", [0.0] + np.geomspace(1e-8, 1 / float(costs.min()), 12).tolist()
    )
    probes = cfg.get(
        "probes",
        sorted(set([1, c["k"]] + [min(c["k"], 2**i) for i in range(2, 17, 2)])),
    )
    reranks = cfg.get("reranks", [1000, 5000, 20000, 40000])
    rows = []

    def measure_point(point):
        tag = (
            __import__("hashlib")
            .sha256(json.dumps(point, sort_keys=True).encode())
            .hexdigest()[:16]
        )
        path = folder / (tag + ".json")
        if path.exists():
            row = read(path)
            if (
                row["point"] != point
                or sha(folder / (tag + ".npz")) != row["raw_sha256"]
            ):
                raise ValueError("Calibration checkpoint changed")
        else:
            found = []
            scores = []
            times = []
            for x in q:
                t = time.perf_counter_ns()
                ii, ss = engine.search(x, point)
                times.append((time.perf_counter_ns() - t) / 1e6)
                found.append(ii)
                scores.append(ss)
            raw = folder / (tag + ".npz")
            if raw.exists():
                raise ValueError(
                    "Unpublished calibration raw checkpoint; inspect before resuming"
                )
            atomic(
                raw,
                lambda f: np.savez(f, ids=found, scores=scores, latency_ms=times),
            )
            row = dict(
                point=point,
                recall100=float(recall100(np.array(found), gt).mean()),
                mean_ms=float(np.mean(times)),
                raw_sha256=sha(raw),
            )
            save(path, row)
        rows.append(row)

    for method, values in [("MHCR", penalties), ("IVF", probes)]:
        for rerank in sorted(set(min(c["n"], int(x)) for x in reranks)):
            for value in values:
                measure_point(
                    dict(
                        method=method,
                        rerank=rerank,
                        **(
                            {"penalty": float(value)}
                            if method == "MHCR"
                            else {"nprobe": int(value)}
                        ),
                    )
                )
            from .calibrate import refinement_candidate

            for target in cfg.get("targets", [0.90, 0.95, 0.98, 0.99]):
                for _ in range(cfg.get("refinement_steps", 8)):
                    group = [
                        r
                        for r in rows
                        if r["point"]["method"] == method
                        and r["point"]["rerank"] == rerank
                    ]
                    point = refinement_candidate(
                        group, method, target, cfg.get("recall_tolerance", 0.002)
                    )
                    if point is None:
                        break
                    measure_point(point)
    selected = []
    for method in ("MHCR", "IVF"):
        for target in cfg.get("targets", [0.90, 0.95, 0.98, 0.99]):
            eligible = [
                r
                for r in rows
                if r["point"]["method"] == method and r["recall100"] >= target
            ]
            selected.append(
                dict(
                    method=method,
                    target=target,
                    attained=bool(eligible),
                    selected=(
                        min(eligible, key=lambda r: r["mean_ms"]) if eligible else None
                    ),
                )
            )
    save(
        folder / "selection.json",
        dict(
            rows=selected,
            selection_on="calibration only; test never used to select parameters",
        ),
    )
    return list(folder.glob("*.json")) + list(folder.glob("*.npz"))


def evaluate(c, inputs, out, threads):
    from .cli import evaluate_main

    selection = read(out / "calibration/selection.json")
    folder = out / "evaluation"
    folder.mkdir(exist_ok=True)
    for r in selection["rows"]:
        if not r["attained"]:
            continue
        name = f"{r['method'].lower()}_{r['target']:.3f}"
        point = folder / (name + ".json")
        bind(point, r["selected"]["point"])
        for mode in ("latency", "breakdown"):
            dest = folder / (name + "_" + mode)
            if not (dest / "summary.json").exists():
                evaluate_main(
                    [
                        "--engine",
                        str(out / "engine.json"),
                        "--queries",
                        str(out / "supervision/test.npz"),
                        "--point",
                        str(point),
                        "--out",
                        str(dest),
                        "--dataset-config",
                        str(out / "config.yaml"),
                    ]
                    + (["--breakdown"] if mode == "breakdown" else [])
                )
    save(
        folder / "complete.json",
        dict(
            complete=True,
            targets=selection["rows"],
            unavailable_targets=[r for r in selection["rows"] if not r["attained"]],
            qps="Run mhcr-qps with engine.json, supervision/test.npz and selected evaluation point JSON",
        ),
    )
    return (
        list(folder.glob("*.json"))
        + list(folder.glob("*/*.json"))
        + list(folder.glob("*/*.npz"))
    )


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--inputs", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--through", choices=STAGES, default="evaluate")
    p.add_argument("--threads", type=int, default=1)
    p.add_argument(
        "--timing-core",
        type=int,
        help="One allowed CPU for cost, calibration and evaluation",
    )
    a = p.parse_args(argv)
    from .cli import config

    c = config(a.config)
    ip = Path(a.inputs).resolve()
    inputs = yaml.safe_load(ip.read_text())
    for key in (
        "n",
        "d",
        "k",
        "pq_m",
        "seed",
        "partition_sample",
        "partition_iterations",
        "pq_sample",
        "pq_iterations",
        "build_chunk",
    ):
        if not isinstance(c.get(key), int) or (key != "seed" and c[key] <= 0):
            p.error("Dataset recipe needs a valid integer " + key)
    for stage in ("teacher", "student"):
        for key in (
            "training_rows",
            "hidden",
            "epochs",
            "batch",
            "lr",
            "weight_decay",
            "seed",
        ):
            if c.get(stage, {}).get(key) is None:
                p.error(stage + " recipe is missing " + key)
    if (
        c.get("by_residual") is not False
        or c.get("pq_bits") != 4
        or c.get("pq_block") != 32
    ):
        p.error("Expected non-residual PQ4 FastScan block32")
    if (
        c["student"].get("budget_unit") not in ("ms", "fraction")
        or c["teacher"].get("rank_gamma", 0) <= 0
    ):
        p.error("Invalid training objective configuration")
    for key in ("training", "calibration", "test", "base", "inverse_norm"):
        if key in inputs:
            inputs[key] = str((ip.parent / inputs[key]).resolve())
    if "base_shards" in inputs:
        inputs["base_shards"] = [
            str((ip.parent / x).resolve()) for x in inputs["base_shards"]
        ]
    if ("base" in inputs) == ("base_shards" in inputs):
        p.error("Inputs must specify exactly one of base or base_shards")
    if a.threads < 1:
        p.error("--threads must be positive")
    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    import fcntl

    lock = (out / ".lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    affinity = set(os.sched_getaffinity(0))
    if a.timing_core is not None and a.timing_core not in affinity:
        p.error("Timing core is outside allowed affinity")
    paths = [x for v in inputs.values() for x in (v if isinstance(v, list) else [v])]
    identity = dict(
        config=c,
        inputs=inputs,
        input_files=[file_identity(x) for x in paths],
        sources={
            str(x.relative_to(Path(__file__).parent)): sha(x)
            for x in Path(__file__).parent.rglob("*")
            if x.suffix in (".py", ".cpp")
        },
        numerical_versions=dict(
            numpy=np.__version__, torch=torch.__version__, faiss=faiss.__version__
        ),
        threads=a.threads,
        timing_core=a.timing_core,
    )
    if "base" in inputs:
        identity["gallery"] = gallery_identity(
            inputs["base"], c.get("base_format", "fp32")
        )
    bind(out / "protocol.json", identity)
    (out / "config.yaml").write_text(yaml.safe_dump(c, sort_keys=False))
    for stage in STAGES[: STAGES.index(a.through) + 1]:
        receipt = out / (stage + "_complete.json")
        if receipt.exists():
            for meta in read(receipt)["artifacts"]:
                if file_identity(meta["path"]) != meta:
                    raise ValueError(
                        "Completed stage artifact changed: " + meta["path"]
                    )
            continue
        timing = stage in ("cost", "calibrate", "evaluate")
        os.sched_setaffinity(
            0,
            (
                {a.timing_core if a.timing_core is not None else min(affinity)}
                if timing
                else affinity
            ),
        )
        count = 1 if timing else a.threads
        torch.set_num_threads(count)
        faiss.omp_set_num_threads(count)
        session = out / f"{stage}_session_{time.time_ns()}.json"
        start = time.time()
        save(out / "status.json", dict(stage=stage, pid=os.getpid(), started=start))
        try:
            artifacts = globals()[stage](c, inputs, out, count)
            value = dict(
                stage=stage,
                complete=True,
                seconds=time.time() - start,
                memory=memory(),
                affinity=sorted(os.sched_getaffinity(0)),
                threads=count,
                artifacts=[file_identity(x) for x in artifacts],
            )
            save(session, value)
            save(receipt, value)
        except BaseException:
            save(
                session,
                dict(
                    stage=stage,
                    complete=False,
                    seconds=time.time() - start,
                    memory=memory(),
                    error=traceback.format_exc(),
                ),
            )
            raise
        finally:
            os.sched_setaffinity(0, affinity)
            gc.collect()
    save(
        out / "status.json",
        dict(complete=True, through=a.through, pid=os.getpid(), time=time.time()),
    )


if __name__ == "__main__":
    main()
