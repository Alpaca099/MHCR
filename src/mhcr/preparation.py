"""Disjoint query cohorts, canonical FP16 storage and exact blocked GT100."""

from pathlib import Path
import hashlib
import json
import os
import numpy as np
from .data import load_array, validate_truth, gallery_identity
from .storage import atomic, read, save, sha


def array_file(path, value):
    path = Path(path)
    if path.exists():
        if not np.array_equal(np.load(path), value):
            raise ValueError(f"Existing artifact differs: {path.name}")
    else:
        atomic(path, lambda f: np.save(f, value))


def half_manifest(shards, out, inverse_norm=None, chunk=32768):
    """Create metadata and inverse norms only; retain original immutable shards."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    arrays = [np.load(p, mmap_mode="r", allow_pickle=False) for p in shards]
    if not arrays or any(a.ndim != 2 or a.dtype != np.float16 for a in arrays):
        raise ValueError("FP16 shards must be row-major .npy matrices")
    d = arrays[0].shape[1]
    if any(a.shape[1] != d or not a.flags.c_contiguous for a in arrays):
        raise ValueError("Inconsistent shard layout")
    n = sum(len(a) for a in arrays)
    norm_path = (
        Path(inverse_norm).resolve() if inverse_norm else out / "inverse_norm.npy"
    )
    if inverse_norm and not norm_path.exists():
        raise FileNotFoundError(norm_path)
    if not norm_path.exists():
        # A partial norm file is never consumed; interrupted preparation can restart.
        partial = out / "inverse_norm.partial.npy"
        norms = np.lib.format.open_memmap(
            partial, mode="w+", dtype=np.float32, shape=(n,)
        )
        pos = 0
        for a in arrays:
            for lo in range(0, len(a), chunk):
                x = np.asarray(a[lo : lo + chunk], np.float32)
                length = np.linalg.norm(x, axis=1)
                if not np.isfinite(length).all() or np.any(length <= 0):
                    raise ValueError("Nonfinite or zero gallery vector")
                norms[pos : pos + len(x)] = 1 / length
                pos += len(x)
        norms.flush()
        del norms
        with partial.open("rb") as f:
            os.fsync(f.fileno())
        os.replace(partial, norm_path)
    norms = np.load(norm_path, mmap_mode="r")
    if norms.shape != (n,) or norms.dtype != np.float32:
        raise ValueError("Inverse norm shape/dtype mismatch")
    lines = [
        f"{n} {d} {len(arrays)} {json.dumps(str(norm_path.resolve()), ensure_ascii=False)} {norms.offset}"
    ]
    for p, a in zip(shards, arrays):
        lines.append(
            f"{json.dumps(str(Path(p).resolve()), ensure_ascii=False)} {a.offset} {len(a)}"
        )
    path = out / "gallery.txt"
    content = "\n".join(lines) + "\n"
    if path.exists() and path.read_text() != content:
        raise ValueError("Gallery manifest changed")
    if not path.exists():
        atomic(path, lambda f: f.write(content.encode()))
    return path


def cohort(path, namespace, n, d):
    """A cohort is NPZ(q, query_ids, optional gt) or a normalized matrix."""
    path = Path(path)
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as z:
            q = np.asarray(z["q"], np.float32)
            ids = z["query_ids"] if "query_ids" in z else None
            gt = z["gt"] if "gt" in z else None
    else:
        q = np.asarray(load_array(path), np.float32)
        ids = gt = None
    if q.ndim != 2 or q.shape[1] != d or not len(q) or not np.isfinite(q).all():
        raise ValueError("Invalid query matrix")
    if not np.allclose(np.linalg.norm(q, axis=1), 1, atol=1e-4):
        raise ValueError("Queries must be normalized before preparing ground truth")
    # Generated IDs have disjoint ranges; explicit IDs must be globally unique.
    if ids is None:
        ids = np.arange(len(q), dtype=np.int64) + namespace * (1 << 48)
    if (
        ids.shape != (len(q),)
        or ids.dtype.kind not in "iu"
        or len(np.unique(ids)) != len(q)
    ):
        raise ValueError("Query IDs must be distinct integers")
    if gt is not None:
        validate_truth(gt, n)
        if len(gt) != len(q):
            raise ValueError("Query/GT row mismatch")
    return dict(q=np.ascontiguousarray(q), query_ids=ids, gt=gt)


def prepare_queries(inputs, config, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    cohorts = {
        name: cohort(inputs[name], i, config["n"], config["d"])
        for i, name in enumerate(("training", "calibration", "test"), 1)
    }

    def hashes(q):
        return [hashlib.sha256(row.tobytes()).digest() for row in q]

    cal, test = cohorts["calibration"], cohorts["test"]
    blocked = set(hashes(cal["q"]))
    th = hashes(test["q"])
    if (
        blocked.intersection(th)
        or np.intersect1d(cal["query_ids"], test["query_ids"]).size
    ):
        raise ValueError("Calibration and test overlap")
    blocked.update(th)
    blocked_ids = set(np.r_[cal["query_ids"], test["query_ids"]].tolist())
    train = cohorts["training"]
    keep = np.array(
        [
            i
            for i, h in enumerate(hashes(train["q"]))
            if h not in blocked and int(train["query_ids"][i]) not in blocked_ids
        ],
        np.int64,
    )
    count = config["teacher"]["training_rows"]
    if len(keep) < count:
        raise ValueError(
            f"Only {len(keep)} disjoint training rows; recipe needs {count}"
        )
    chosen = np.sort(
        np.random.default_rng(config["teacher"]["seed"]).choice(keep, count, False)
    )
    cohorts["training"] = {
        k: v[chosen] if v is not None else None for k, v in train.items()
    }
    for name, z in cohorts.items():
        target = out / (name + ".npz")
        payload = {k: v for k, v in z.items() if v is not None}
        if target.exists():
            with np.load(target) as old:
                if set(old.files) != set(payload) or any(
                    not np.array_equal(old[k], v) for k, v in payload.items()
                ):
                    raise ValueError("Prepared cohort differs on resume")
        else:
            atomic(target, lambda f: np.savez(f, **payload))
    array_file(out / "training_source_rows.npy", chosen)
    identity = dict(
        training_content_verified=True,
        historical_teacher_reused=False,
        excluded_training_rows=len(train["q"]) - len(keep),
        training_rows=len(chosen),
        inputs={k: sha(inputs[k]) for k in cohorts},
        outputs={k: sha(out / (k + ".npz")) for k in cohorts},
        definition="Exact normalized-vector content and explicit-ID disjointness; original source rows retained",
    )
    save(out / "identity.json", identity)
    return identity


def exact_truth(base, queries, out, protocol, query_chunk=32, base_chunk=32768):
    """Exact full-gallery IP search with bounded score tiles and query checkpoints.

    No approximate index is used. Ties are deterministically ordered by ID.
    Query blocks can be resumed; storage and query identity are caller-bound.
    """
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    binding = dict(
        protocol=protocol,
        queries_sha256=hashlib.sha256(queries.tobytes()).hexdigest(),
        shape=list(base.shape),
        query_chunk=query_chunk,
        base_chunk=base_chunk,
    )
    if (out / "protocol.json").exists() and read(out / "protocol.json") != binding:
        raise ValueError("GT input identity changed")
    save(out / "protocol.json", binding)
    if len(base) < 100:
        raise ValueError("Recall@100 needs at least100 gallery rows")
    result = np.empty((len(queries), 100), np.int64)
    for start in range(0, len(queries), query_chunk):
        q = np.ascontiguousarray(queries[start : start + query_chunk], np.float32)
        path = out / f"queries-{start:09d}.npz"
        if path.exists():
            meta = read(path.with_suffix(".json"))
            if meta["sha256"] != sha(path) or meta["protocol"] != protocol:
                raise ValueError("GT checkpoint identity mismatch")
            with np.load(path) as z:
                result[start : start + len(q)] = z["ids"]
            continue
        ids = np.empty((len(q), 0), np.int64)
        scores = np.empty((len(q), 0), np.float32)
        for lo in range(0, len(base), base_chunk):
            x = np.ascontiguousarray(
                base[lo : min(lo + base_chunk, len(base))], np.float32
            )
            values = q @ x.T
            candidates = np.broadcast_to(
                np.arange(lo, lo + len(x), dtype=np.int64), values.shape
            )
            values = np.concatenate((scores, values), axis=1)
            candidates = np.concatenate((ids, candidates), axis=1)
            width = min(100, values.shape[1])
            take = np.argpartition(-values, width - 1, axis=1)[:, :width]
            new_i = []
            new_s = []
            for j in range(len(q)):
                boundary = values[j, take[j]].min()
                better = np.flatnonzero(values[j] > boundary)
                ties = np.flatnonzero(values[j] == boundary)
                ties = ties[np.argsort(candidates[j, ties])[: width - len(better)]]
                ix = np.r_[better, ties]
                ix = ix[np.lexsort((candidates[j, ix], -values[j, ix]))]
                new_i.append(candidates[j, ix])
                new_s.append(values[j, ix])
            ids = np.asarray(new_i)
            scores = np.asarray(new_s)
        validate_truth(ids, len(base))
        atomic(path, lambda f: np.savez(f, ids=ids, scores=scores))
        save(path.with_suffix(".json"), dict(protocol=protocol, sha256=sha(path)))
        result[start : start + len(q)] = ids
    return result
