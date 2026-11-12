"""Fresh CLI pipeline, exact GT, disjointness and FP16 build recovery."""

import json
import os
import subprocess
import sys
import numpy as np
import pytest
import yaml
import faiss
from mhcr.data import ShardGallery
from mhcr.preparation import half_manifest, exact_truth, prepare_queries
from mhcr.index.build import build


def fixture(tmp_path, half=False):
    rng = np.random.default_rng(723)
    x = rng.normal(size=(2048, 16)).astype(np.float32)
    x[:, 0] += np.linspace(-3, 3, len(x))
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    if half:
        np.save(tmp_path / "shard0.npy", x[:700].astype(np.float16))
        np.save(tmp_path / "shard1.npy", x[700:].astype(np.float16))
        inputs = dict(base_shards=["shard0.npy", "shard1.npy"])
    else:
        np.save(tmp_path / "base.npy", x)
        inputs = dict(base="base.npy")
    for name, n in [("training", 64), ("calibration", 4), ("test", 5)]:
        q = rng.normal(size=(n, 16)).astype(np.float32)
        q /= np.linalg.norm(q, axis=1, keepdims=True)
        np.save(tmp_path / (name + ".npy"), q)
        inputs[name] = name + ".npy"
    recipe = dict(
        training_rows=64,
        hidden=32,
        epochs=1,
        batch=16,
        lr=0.001,
        weight_decay=0.0001,
        seed=81,
    )
    c = dict(
        dataset="fixture",
        n=2048,
        d=16,
        k=16,
        pq_m=8,
        pq_bits=4,
        pq_block=32,
        top_k=100,
        by_residual=False,
        seed=73,
        partition_sample=2048,
        partition_iterations=2,
        pq_sample=2048,
        pq_iterations=2,
        build_chunk=1024,
        teacher=dict(recipe, rank_gamma=20.0, tau=0.07),
        student=dict(
            recipe,
            training_rows=32,
            dimension=8,
            budget_unit="fraction",
            budget_range=None,
            family="budget_tail",
        ),
        cost=dict(queries=16, probes=[1, 2, 4, 8, 16], repeats=2, rerank=100),
        preparation=dict(query_chunk=16, base_chunk=512),
        evaluation=dict(
            targets=[0.90, 0.95, 0.98, 0.99],
            penalties=[0.0, 1.0],
            probes=[1, 16],
            reranks=[2048],
        ),
    )
    if half:
        c.update(base_format="fp16_shards", routing_variant="datacomp")
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(c))
    (tmp_path / "inputs.yaml").write_text(yaml.safe_dump(inputs))
    return c, inputs


@pytest.mark.parametrize("half", [False, True])
def test_fresh_pipeline(tmp_path, half):
    if half:
        from mhcr.serving.fp16 import require_platform

        try:
            require_platform()
        except RuntimeError as e:
            pytest.skip(str(e))
    c, inputs = fixture(tmp_path, half)
    cmd = [
        sys.executable,
        "-m",
        "mhcr.pipeline",
        "--config",
        str(tmp_path / "config.yaml"),
        "--inputs",
        str(tmp_path / "inputs.yaml"),
        "--out",
        str(tmp_path / "run"),
    ]
    env = dict(
        os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1"
    )
    p = subprocess.run(cmd, env=env, text=True, capture_output=True, timeout=180)
    assert p.returncode == 0, p.stdout + p.stderr
    root = tmp_path / "run"
    assert json.loads((root / "status.json").read_text())["complete"]
    for stage in (
        "prepare",
        "build",
        "labels",
        "cost",
        "teacher",
        "student",
        "calibrate",
        "evaluate",
    ):
        assert json.loads((root / (stage + "_complete.json")).read_text())["complete"]
    with np.load(root / "supervision/test.npz") as z:
        base = (
            ShardGallery(root / "prepared/gallery.txt")
            if half
            else np.load(tmp_path / "base.npy")
        )
        expected = np.argsort(-(z["q"] @ base[:].T), axis=1)[:, :100]
        assert np.array_equal(expected, z["gt"])
    selection = json.loads((root / "calibration/selection.json").read_text())["rows"]
    assert all(r["attained"] for r in selection)
    # A coarse two-point grid must also produce real intermediate probe trials.
    measured = [
        json.loads(p.read_text())
        for p in (root / "calibration").glob("*.json")
        if p.name != "selection.json"
    ]
    assert any(r["point"].get("nprobe") not in (None, 1, 16) for r in measured)
    # Resume does not rewrite completed artifacts or rerun stages.
    before = {str(p): p.stat().st_mtime_ns for p in root.glob("*_complete.json")}
    again = subprocess.run(cmd, env=env, text=True, capture_output=True, timeout=30)
    assert again.returncode == 0, again.stdout + again.stderr
    assert before == {
        str(p): p.stat().st_mtime_ns for p in root.glob("*_complete.json")
    }
    # Changed source vectors must be rejected before consuming checkpoints.
    np.save(tmp_path / "test.npy", np.load(tmp_path / "test.npy")[::-1])
    bad = subprocess.run(cmd, env=env, text=True, capture_output=True, timeout=30)
    assert bad.returncode != 0 and "protocol changed" in bad.stderr


def test_fp16_build_resume(tmp_path):
    tmp_path = tmp_path / "向量 shards"
    tmp_path.mkdir()
    c, inputs = fixture(tmp_path, True)
    manifest = half_manifest(
        [tmp_path / x for x in inputs["base_shards"]], tmp_path / "storage"
    )
    base = ShardGallery(manifest)
    faiss.omp_set_num_threads(1)

    def stop(path):
        raise InterruptedError("checkpoint fixture")

    with pytest.raises(InterruptedError):
        build(base, tmp_path / "resume", c, "same-protocol", stop)
    resumed = build(base, tmp_path / "resume", c, "same-protocol")
    fresh = build(base, tmp_path / "fresh", c, "same-protocol")
    assert np.array_equal(faiss.serialize_index(resumed), faiss.serialize_index(fresh))


def test_split_overlap_and_gt_recovery(tmp_path):
    c, inputs = fixture(tmp_path)
    paths = {k: str(tmp_path / v) for k, v in inputs.items()}
    np.save(tmp_path / "test.npy", np.load(tmp_path / "calibration.npy"))
    with pytest.raises(ValueError, match="overlap"):
        prepare_queries(paths, c, tmp_path / "overlap")
    base = np.load(tmp_path / "base.npy")
    q = np.load(tmp_path / "training.npy")[:3]
    a = exact_truth(base, q, tmp_path / "gt", "identity", query_chunk=2, base_chunk=512)
    b = exact_truth(base, q, tmp_path / "gt", "identity", query_chunk=2, base_chunk=512)
    assert np.array_equal(a, b)
    with pytest.raises(ValueError, match="identity"):
        exact_truth(
            base, q[::-1], tmp_path / "gt", "identity", query_chunk=2, base_chunk=512
        )
