import importlib
import json
import numpy as np
import pytest
import torch
import faiss
from mhcr.router import QueryCellRouter
from mhcr.metrics import build_curves_physical, dedup_top100
from mhcr.calibrate import gt_rank_1x, calibrate_end_to_end
from mhcr.algorithm import training_loss
from mhcr.training import train_epochs, model_hash
from mhcr.teacher import train_teacher
from mhcr.projection import project_teacher
from mhcr.index.build import build
from mhcr.serving.engine import Engine
from mhcr.cli import recall100
from mhcr.algorithm import datacomp_gate

torch.set_num_threads(1)
faiss.omp_set_num_threads(1)


def test_imports():
    for name in [
        "training",
        "teacher",
        "projection",
        "data",
        "index.build",
        "backends.faiss_flat",
        "serving.engine",
        "cli",
    ]:
        importlib.import_module("mhcr." + name)


def test_metrics_and_large_partition():
    gt = np.arange(100).reshape(1, -1)
    curve, _ = build_curves_physical(np.array([[0]]), np.zeros((100, 1), int), gt, 1)
    assert curve[0, 0] == 1
    assert recall100(gt, gt)[0] == 1
    with pytest.raises(ValueError):
        recall100(gt[:, :50], gt)
    assert np.array_equal(dedup_top100(np.repeat(gt, 2, axis=1)), gt)
    ranks = gt_rank_1x(
        np.arange(65536).reshape(1, -1), np.arange(65436, 65536).reshape(1, -1)
    )
    assert ranks[0, -1] == 65535 and ranks.dtype == np.int32


def test_uniform_labels_and_gradient():
    logits = torch.tensor([[0.2, 0.4, 0.7]], requires_grad=True)
    cells = torch.tensor([[0] * 20 + [1] * 10 + [2] * 70])
    loss, _ = training_loss(
        logits, cells, torch.ones(3), torch.ones(3), torch.tensor([[1.0]]), "ce"
    )
    loss.backward()
    expected = logits.detach().softmax(1) - torch.tensor([[0.2, 0.1, 0.7]])
    assert torch.allclose(logits.grad, expected, atol=1e-6)


def test_end_to_end_calibration():
    class Fake:
        index = type("Index", (), {"ntotal": 200})()

        def search(self, q, point):
            return (
                np.arange(100) if point["penalty"] <= 0.2 else np.arange(100, 200)
            ), None

    out = calibrate_end_to_end(
        Fake(),
        np.zeros((1, 2)),
        np.arange(100).reshape(1, -1),
        dict(method="MHCR"),
        [0, 0.1, 0.2, 0.3],
        0.99,
    )
    assert out["penalty"] == 0.2


def test_datacomp_gate_budget_units():
    density = torch.tensor([[1.0, 2.0, 3.0]], requires_grad=True)
    cost = torch.tensor([10.0, 20.0, 30.0])
    g = datacomp_gate(density, cost, torch.tensor([[25.0]]))
    assert abs(float((g * cost).sum()) - 25) < 0.1
    g.sum().backward()
    assert torch.isfinite(density.grad).all()


def test_flat_backend_returns_100_after_extension():
    from mhcr.backends.faiss_flat import build_flat, search

    rng = np.random.default_rng(8)
    base = rng.normal(size=(120, 8)).astype(np.float32)
    faiss.normalize_L2(base)
    index, _ = build_flat(base, base[:2], np.array([0] * 20 + [1] * 100))

    class Route:
        def online(self, q):
            return (
                np.array([[0, 1]]),
                np.zeros((1, 2), np.float32),
                np.ones(1, np.int32),
                None,
            )

    found, budget = search(Route(), index, base[:1])
    assert (
        found.shape == (1, 100) and len(np.unique(found)) == 100 and np.all(found >= 0)
    )
    assert budget[0] == 2


def test_training_build_reload_search(tmp_path):
    rng = np.random.default_rng(9)
    base = rng.normal(size=(640, 8)).astype(np.float32)
    faiss.normalize_L2(base)
    q = base[:8].copy()
    c = dict(
        n=640,
        d=8,
        k=4,
        pq_m=4,
        pq_bits=4,
        pq_block=32,
        seed=9,
        partition_sample=640,
        partition_iterations=2,
        pq_sample=640,
        pq_iterations=2,
        build_chunk=320,
    )
    index = build(base, tmp_path / "built", c, "test-protocol")
    assignment = np.load(tmp_path / "built/assignment.npy")
    sizes = np.load(tmp_path / "built/sizes.npy")
    gt = np.argsort(-(q @ base.T), axis=1)[:, :100]
    cells = assignment[gt]
    centers = np.load(tmp_path / "built/centroids.npy")
    identity = dict(
        training_content_verified=True,
        historical_teacher_reused=False,
        teacher_training_protocol="synthetic-test",
    )
    tc = dict(seed=7, epochs=1, batch=4, lr=0.001, weight_decay=0.0001, rank_gamma=20.0)
    torch.manual_seed(7)
    teacher = QueryCellRouter(
        8, 8, 4, q.mean(0), arch="res_mlp", hidden=16, proto_init=centers
    )
    train_teacher(teacher, q, cells, tmp_path / "teacher", identity, tc)
    student = project_teacher(teacher, 4, 8, tmp_path / "projection.npy", identity)
    initial = {k: v.clone() for k, v in student.state_dict().items()}
    sc = dict(tc, budget_range=[0.2, 0.6], budget_unit="fraction")
    result = train_epochs(
        student,
        q,
        cells,
        np.ones(4, np.float32),
        sizes,
        tmp_path / "student",
        identity,
        sc,
        "budget_tail",
        teacher,
    )
    final = model_hash(student)
    student.load_state_dict(initial)
    resumed = train_epochs(
        student,
        q,
        cells,
        np.ones(4, np.float32),
        sizes,
        tmp_path / "student",
        identity,
        sc,
        "budget_tail",
        teacher,
    )
    assert (
        model_hash(student) == final
        and result["model_sha256"] == resumed["model_sha256"]
    )
    np.save(tmp_path / "base.npy", base)
    np.save(tmp_path / "cost.npy", np.ones(4, np.float32))
    torch.save(
        dict(
            model=student.state_dict(),
            model_config=dict(d_clip=8, d_r=4, K=4, arch="res_mlp", hidden=8, tau=0.07),
        ),
        tmp_path / "model.pt",
    )
    manifest = dict(
        index="built/index/fastscan.faiss",
        base="base.npy",
        model="model.pt",
        cost="cost.npy",
        top_k=100,
        quantized=False,
    )
    from mhcr.serving.engine import load_model

    torch.save(
        dict(state=student.state_dict(), binding=dict(tau=0.07)),
        tmp_path / "bound_model.pt",
    )
    assert model_hash(load_model(tmp_path / "bound_model.pt")) == model_hash(student)
    (tmp_path / "engine.json").write_text(json.dumps(manifest))
    engine = Engine.from_manifest(tmp_path / "engine.json")
    ii, ss = engine.search(q[0], dict(method="MHCR", penalty=0, rerank=640))
    assert len(ii) == len(ss) == 100 and len(set(ii)) == 100 and set(ii) == set(gt[0])
    loaded = Engine.from_manifest(tmp_path / "engine.json")
    jj, tt = loaded.search(q[0], dict(method="IVF", nprobe=4, rerank=640))
    assert np.array_equal(ii, jj) and np.array_equal(ss, tt)
    assert index.ntotal == 640
    # Preserve the caller's scan/refinement vector; normalize only routing features.
    scaled_ids, scaled_scores = engine.search(
        q[0] * 2, dict(method="MHCR", penalty=0, rerank=640)
    )
    assert np.array_equal(scaled_ids, ii)
    assert np.array_equal(scaled_scores, ss * 2)
    # Profiling reports actual work and preserves the search result.
    pi, ps, parts = engine.search(
        q[0], dict(method="MHCR", penalty=0, rerank=640), profile=True
    )
    assert np.array_equal(pi, ii) and np.array_equal(ps, ss)
    assert parts["scanned_entries"] == 640 and parts["refined_candidates"] == 640
    assert (
        abs(
            parts["total_ms"]
            - sum(
                parts[k]
                for k in ("route_prepare_ms", "scan_ms", "refine_validate_topk_ms")
            )
        )
        < 1e-8
    )
    # Exercise the installed-style process CLI using only this tiny index.
    import os, sys, subprocess

    cores = sorted(os.sched_getaffinity(0))[:2]
    np.savez(tmp_path / "queries.npz", q=q, gt=gt, query_ids=np.arange(len(q)))
    (tmp_path / "point.json").write_text(
        json.dumps(dict(method="MHCR", penalty=0, rerank=640))
    )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "mhcr.benchmark.qps_cli",
            "--engine",
            str(tmp_path / "engine.json"),
            "--queries",
            str(tmp_path / "queries.npz"),
            "--point",
            str(tmp_path / "point.json"),
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
            "0.99",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=90,
    )
    report = json.loads((tmp_path / "qps/qps.json").read_text())
    assert report["complete"] and all(row["recall100"] == 1 for row in report["rows"])
    assert (tmp_path / "qps/workers-1/worker-0.npz").exists()
    for name, flags in [
        ("breakdown", ["--breakdown"]),
        ("ablation", ["--ablation", "full"]),
    ]:
        subprocess.run(
            [
                sys.executable,
                "-c",
                "from mhcr.cli import evaluate_main; evaluate_main()",
                "--engine",
                str(tmp_path / "engine.json"),
                "--queries",
                str(tmp_path / "queries.npz"),
                "--point",
                str(tmp_path / "point.json"),
                "--out",
                str(tmp_path / name),
                *flags,
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        report = json.loads((tmp_path / name / "summary.json").read_text())
        assert report["recall100"] == 1 and report["measurement"] == name
    from mhcr.benchmark.ablation import AblationEngine

    ablation = AblationEngine(
        engine, engine.router, engine.index, centers, np.ones(4), "full"
    )
    raw_query = q[0] * 3
    expected = engine.search(raw_query, dict(method="MHCR", penalty=0, rerank=640))
    actual = ablation.search(raw_query, dict(penalty=0, rerank=640))
    assert np.array_equal(actual[0], expected[0])
    assert np.array_equal(actual[1], expected[1])
    with pytest.raises(AssertionError):
        AblationEngine(
            engine,
            engine.router,
            engine.index,
            centers,
            np.ones(4),
            "cosine_fixed",
            param=2,
        )
