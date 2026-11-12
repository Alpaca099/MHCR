"""Native shard boundaries, canonical values, resource lifetime and bad inputs."""

import json
import shutil
import numpy as np
import pytest
from mhcr.serving.fp16 import HalfBase, build_library, require_platform


def test_fp16_shards(tmp_path):
    try:
        require_platform()
    except RuntimeError as error:
        pytest.skip(str(error))
    if not shutil.which("g++"):
        pytest.skip("g++ is required for the optional native backend")
    library = build_library(tmp_path / "half.so")
    rng = np.random.default_rng(812)
    raw = rng.normal(size=(137, 768)).astype(np.float16)
    inverse = (1 / np.linalg.norm(raw.astype(np.float32), axis=1)).astype(np.float32)
    records = []
    for i, shard in enumerate((raw[:41], raw[41:])):
        path = tmp_path / f"shard {i}.npy"
        np.save(path, shard)
        records.append(
            f'{json.dumps(path.name)} {np.load(path, mmap_mode="r").offset} {len(shard)}'
        )
    np.save(tmp_path / "norm.npy", inverse)
    offset = np.load(tmp_path / "norm.npy", mmap_mode="r").offset
    manifest = tmp_path / "gallery.txt"
    manifest.write_text(f'137 768 2 "norm.npy" {offset}\n' + "\n".join(records) + "\n")
    expected = raw.astype(np.float32) * inverse[:, None]
    ids = np.array([136, 41, 40, 0, 41], np.int64)
    gallery = HalfBase(manifest, library)
    assert gallery.shape == expected.shape
    assert np.array_equal(gallery[ids], expected[ids])
    q = rng.normal(size=768).astype(np.float32)
    scores = gallery.scores(q, ids)
    np.testing.assert_allclose(scores, expected[ids] @ q, rtol=2e-5, atol=2e-6)
    with pytest.raises(ValueError):
        gallery.scores(q, np.array([-1], np.int64))
    with pytest.raises(ValueError):
        gallery.scores(q, np.array([137], np.int64))
    with pytest.raises(ValueError):
        gallery.scores(q, np.array([1.5]))
    gallery.close()
    gallery.close()
    with pytest.raises(RuntimeError):
        gallery[ids]
    reopened = HalfBase(manifest, library)
    assert scores.tobytes() == reopened.scores(q, ids).tobytes()
    reopened.close()
    exercise_native_engine(tmp_path, library, manifest, expected)
    manifest.write_text(manifest.read_text().replace("137 768 2", "138 768 2"))
    with pytest.raises(ValueError):
        HalfBase(manifest, library)
    with pytest.raises(FileExistsError):
        build_library(library)


def exercise_native_engine(tmp_path, library, manifest, base):
    """The same manifest must work through process QPS and evaluation CLIs."""
    import os
    import sys
    import subprocess
    import torch
    import faiss
    from mhcr.router import QueryCellRouter
    from mhcr.serving.engine import Engine

    torch.set_num_threads(1)
    faiss.omp_set_num_threads(1)
    index = faiss.IndexIVFPQ(
        faiss.IndexFlatIP(768), 768, 4, 16, 4, faiss.METRIC_INNER_PRODUCT
    )
    index.by_residual = False
    index.cp.niter = index.pq.cp.niter = 2
    index.train(base)
    index.add(base)
    index = faiss.IndexIVFPQFastScan(index, 32)
    faiss.write_index(index, str(tmp_path / "index.faiss"))
    model = QueryCellRouter(
        768, 16, 4, np.zeros(768, np.float32), arch="res_mlp", hidden=32
    )
    torch.save(dict(model=model.state_dict()), tmp_path / "model.pt")
    np.save(tmp_path / "cost.npy", np.ones(4, np.float32))
    engine_manifest = dict(
        index="index.faiss",
        base=manifest.name,
        base_format="fp16_shards",
        native_library=str(library),
        routing_variant="datacomp",
        model="model.pt",
        cost="cost.npy",
        top_k=100,
        expected_nlist=4,
        quantized=False,
    )
    (tmp_path / "engine.json").write_text(json.dumps(engine_manifest))
    point = dict(method="IVF", nprobe=4, rerank=len(base))
    (tmp_path / "point.json").write_text(json.dumps(point))
    queries = base[:4].copy()
    engine = Engine.from_manifest(tmp_path / "engine.json")
    outputs = [engine.search(q, point) for q in queries]
    truth = np.array([v[0] for v in outputs])
    for q, (ids, scores) in zip(queries, outputs):
        ii, ss, parts = engine.search(q, point, profile=True)
        assert np.array_equal(ids, ii) and np.array_equal(scores, ss)
        assert parts["refined_candidates"] == len(base)
    np.savez(tmp_path / "queries.npz", q=queries, gt=truth, query_ids=np.arange(4))
    cores = sorted(os.sched_getaffinity(0))[:2]
    common = [
        "--engine",
        str(tmp_path / "engine.json"),
        "--queries",
        str(tmp_path / "queries.npz"),
        "--point",
        str(tmp_path / "point.json"),
    ]
    subprocess.run(
        [
            sys.executable,
            "-m",
            "mhcr.benchmark.qps_cli",
            *common,
            "--out",
            str(tmp_path / "qps"),
            "--workers",
            ",".join(str(i + 1) for i in range(len(cores))),
            "--cores",
            ",".join(map(str, cores)),
            "--cycles",
            "1",
            "--warmup",
            "1",
            "--min-recall",
            "1",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=90,
    )
    report = json.loads((tmp_path / "qps/qps.json").read_text())
    assert report["complete"] and all(row["recall100"] == 1 for row in report["rows"])
    for mode in ("breakdown", "ablation"):
        flags = ["--breakdown"] if mode == "breakdown" else ["--ablation", "ivf"]
        if mode == "ablation":
            (tmp_path / "point.json").write_text(json.dumps(dict(point, probes=4)))
        subprocess.run(
            [
                sys.executable,
                "-c",
                "from mhcr.cli import evaluate_main; evaluate_main()",
                *common,
                "--out",
                str(tmp_path / mode),
                *flags,
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert (
            json.loads((tmp_path / mode / "summary.json").read_text())["recall100"] == 1
        )
    engine.base.close()
