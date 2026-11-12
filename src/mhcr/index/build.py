"""Gallery-only spherical partition and PQ4 construction with resumable snapshots."""

from ..storage import P, read, save, sha, atomic, memory
import os, time, uuid
import numpy as np, faiss


def checkpoint(index, path, protocol):
    assert not path.exists() and not path.with_suffix(".json").exists()
    tmp = path.with_name(path.name + ".partial-" + uuid.uuid4().hex)
    faiss.write_index(index, str(tmp))
    with tmp.open("rb") as f:
        os.fsync(f.fileno())
    os.rename(tmp, path)
    if hasattr(os, "O_DIRECTORY"):
        fd = os.open(path.parent, os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    s = path.stat()
    save(
        path.with_suffix(".json"),
        dict(
            protocol=protocol,
            n=index.ntotal,
            bytes=s.st_size,
            mtime_ns=s.st_mtime_ns,
            sha256=sha(path),
        ),
    )


def load(path, protocol):
    meta = read(path.with_suffix(".json"))
    s = path.stat()
    assert meta["protocol"] == protocol and (s.st_size, s.st_mtime_ns) == (
        meta["bytes"],
        meta["mtime_ns"],
    )
    # Hash only the selected recovery snapshot, not all older retained snapshots.
    assert sha(path) == meta["sha256"]
    index = faiss.read_index(str(path))
    assert index.ntotal == meta["n"]
    return index


def build(base, out, config, protocol, on_checkpoint=None):
    c = config
    out = P(out)
    root = out / "index"
    root.mkdir(parents=True, exist_ok=True)
    assert base.shape == (c["n"], c["d"]) and base.dtype == np.float32
    assert c["pq_bits"] == 4 and c["d"] % c["pq_m"] == 0 and c["pq_block"] == 32
    snaps = sorted(root.glob("rows-*.faiss"))
    if snaps:
        index = load(snaps[-1], protocol)
    else:
        cp = out / "centroids.npy"
        cm = out / "centroids.json"
        if cp.exists():
            meta = read(cm)
            assert meta["protocol"] == protocol and sha(cp) == meta["sha256"]
            centers = np.load(cp)
        else:
            rows = np.sort(
                np.random.default_rng(c["seed"]).choice(
                    c["n"], min(c["partition_sample"], c["n"]), False
                )
            )
            sample = np.array(base[rows], np.float32)
            assert np.isfinite(sample).all()
            faiss.normalize_L2(sample)
            km = faiss.Kmeans(
                c["d"],
                c["k"],
                niter=c["partition_iterations"],
                spherical=True,
                seed=c["seed"],
                verbose=True,
                gpu=False,
            )
            km.train(sample)
            centers = km.centroids.copy()
            atomic(cp, lambda f: np.save(f, centers))
            save(
                cm,
                dict(
                    protocol=protocol,
                    sha256=sha(cp),
                    training_rows=len(rows),
                    gallery_only=True,
                ),
            )
            del sample, km
        assert centers.shape == (c["k"], c["d"])
        quant = faiss.IndexFlatIP(c["d"])
        quant.add(centers)
        index = faiss.IndexIVFPQ(
            quant, c["d"], c["k"], c["pq_m"], 4, faiss.METRIC_INNER_PRODUCT
        )
        index.by_residual = False
        index.pq.cp.niter = c["pq_iterations"]
        index.pq.cp.seed = c["seed"]
        rows = np.sort(
            np.random.default_rng(c["seed"] + 1).choice(
                c["n"], min(c["pq_sample"], c["n"]), False
            )
        )
        sample = np.array(base[rows], np.float32)
        assert np.isfinite(sample).all()
        index.pq.train(sample)
        index.is_trained = True
        del sample
        checkpoint(index, root / "rows-000000000.faiss", protocol)
    assert (
        index.d == c["d"]
        and index.nlist == c["k"]
        and not index.by_residual
        and index.metric_type == faiss.METRIC_INNER_PRODUCT
    )
    for start in range(index.ntotal, c["n"], c["build_chunk"]):
        end = min(start + c["build_chunk"], c["n"])
        for lo in range(start, end, 32768):
            x = np.ascontiguousarray(base[lo : min(lo + 32768, end)], np.float32)
            assert np.isfinite(x).all()
            # Serving gallery already has canonical FP32 normalization; do not normalize twice.
            index.add(x)
        path = root / f"rows-{end:09d}.faiss"
        checkpoint(index, path, protocol)
        save(
            out / "status.json",
            dict(stage="build", rows=end, time=time.time(), memory=memory()),
        )
        if on_checkpoint:
            on_checkpoint(path)
    assert index.ntotal == c["n"]
    packed = root / "fastscan.faiss"
    fast = (
        load(packed, protocol)
        if packed.exists()
        else faiss.IndexIVFPQFastScan(index, 32)
    )
    from .assets import assignments

    assignment, sizes = assignments(fast)
    for cell in range(c["k"]):
        count = index.invlists.list_size(cell)
        assert count == sizes[cell]
        if count:
            assert np.array_equal(
                faiss.rev_swig_ptr(index.invlists.get_ids(cell), count),
                faiss.rev_swig_ptr(fast.invlists.get_ids(cell), count),
            )
    assert np.array_equal(
        fast.quantizer.reconstruct_n(0, c["k"]),
        index.quantizer.reconstruct_n(0, c["k"]),
    )
    if not packed.exists():
        checkpoint(fast, packed, protocol)
    for name, value in [("assignment.npy", assignment), ("sizes.npy", sizes)]:
        if (out / name).exists():
            assert np.array_equal(np.load(out / name), value)
        else:
            atomic(out / name, lambda f: np.save(f, value))
    done = dict(
        complete=True,
        protocol=protocol,
        n=c["n"],
        d=c["d"],
        k=c["k"],
        final_index=str(packed),
        pq_m=c["pq_m"],
        pq_bits=4,
        metric="inner_product_on_canonical_normalized_vectors",
        base_storage=c.get("base_format", "fp32"),
        all_membership_verified=True,
        no_historical_centroids_or_index=True,
        artifacts={
            name: sha(out / name)
            for name in ["assignment.npy", "sizes.npy", "centroids.npy"]
        },
        costs="Measure each build attempt separately; snapshots do not aggregate attempt time or peak RSS",
    )
    save(out / "build_complete.json", done)
    return fast


def load_index(path):
    """Load a final FastScan index and physical list sizes."""
    from pathlib import Path
    from types import SimpleNamespace

    p = Path(path)
    if p.is_dir():
        p = p / "fastscan.faiss"
    index = faiss.read_index(str(p))
    if (
        not isinstance(index, faiss.IndexIVFPQFastScan)
        or index.by_residual
        or index.metric_type != faiss.METRIC_INNER_PRODUCT
    ):
        raise ValueError("Expected non-residual IP IVF-PQ FastScan")
    sizes = np.array(
        [index.invlists.list_size(i) for i in range(index.nlist)], np.int64
    )
    return SimpleNamespace(index=index, sizes=sizes)
