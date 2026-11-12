"""Portable prepared arrays; all ground truth contains 100 neighbors."""

from pathlib import Path
import struct
import shlex
import numpy as np
from .storage import atomic, read, save, sha


class ShardGallery:
    """Bounded canonical FP32 views of read-only FP16 storage, also used in builds."""

    dtype = np.dtype("float32")

    def __init__(self, path):
        path = Path(path).resolve()
        lines = [shlex.split(x) for x in path.read_text().splitlines() if x.strip()]
        n, d, count = map(int, lines[0][:3])
        if n <= 0 or d <= 0 or count != len(lines) - 1:
            raise ValueError("Invalid FP16 manifest")

        def mapped(name, dtype, offset, shape):
            p = Path(name)
            p = p if p.is_absolute() else path.parent / p
            if (
                offset < 0
                or p.stat().st_size
                != offset + int(np.prod(shape)) * np.dtype(dtype).itemsize
            ):
                raise ValueError("Invalid shard size or offset")
            return np.memmap(p, mode="r", dtype=dtype, offset=offset, shape=shape)

        self.norm = mapped(lines[0][3], "<f4", int(lines[0][4]), (n,))
        self.parts = [mapped(x[0], "<f2", int(x[1]), (int(x[2]), d)) for x in lines[1:]]
        self.offsets = np.r_[0, np.cumsum([len(x) for x in self.parts])]
        if self.offsets[-1] != n:
            raise ValueError("FP16 row count mismatch")
        self.shape = (n, d)

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, rows):
        if isinstance(rows, slice):
            rows = np.arange(*rows.indices(len(self)), dtype=np.int64)
        rows = np.asarray(rows)
        if (
            rows.ndim != 1
            or rows.dtype.kind not in "iu"
            or np.any(rows < 0)
            or np.any(rows >= len(self))
        ):
            raise ValueError("Expected valid one-dimensional gallery IDs")
        result = np.empty((len(rows), self.shape[1]), np.float32)
        shards = np.searchsorted(self.offsets, rows, side="right") - 1
        for i in np.unique(shards):
            mask = shards == i
            result[mask] = self.parts[i][rows[mask] - self.offsets[i]]
        result *= np.asarray(self.norm[rows])[:, None]
        return result


def open_gallery(path, base_format="fp32"):
    if base_format == "fp16_shards":
        return ShardGallery(path)
    if base_format != "fp32":
        raise ValueError("Unknown gallery format")
    a = load_array(path)
    if a.ndim != 2 or a.dtype != np.float32:
        raise ValueError("Expected FP32 gallery or explicit fp16_shards")
    return a


def file_identity(path):
    p = Path(path).resolve()
    s = p.stat()
    return dict(path=str(p), bytes=s.st_size, mtime_ns=s.st_mtime_ns)


def gallery_identity(path, base_format):
    result = dict(format=base_format, file=file_identity(path))
    if base_format == "fp16_shards":
        p = Path(path).resolve()
        lines = [shlex.split(x) for x in p.read_text().splitlines() if x.strip()]
        names = [lines[0][3]] + [x[0] for x in lines[1:]]
        result["manifest_sha256"] = sha(p)
        result["files"] = [file_identity(p.parent / x) for x in names]
    return result


def fbin(path, shape=None, dtype="<f4"):
    path = Path(path)
    with path.open("rb") as f:
        actual = struct.unpack("<II", f.read(8))
    if shape is not None and tuple(shape) != actual:
        raise ValueError("Unexpected binary shape")
    if path.stat().st_size != 8 + int(np.prod(actual)) * np.dtype(dtype).itemsize:
        raise ValueError("Invalid file size")
    return np.memmap(path, dtype=dtype, mode="r", offset=8, shape=actual)


def load_array(path):
    path = Path(path)
    return (
        np.load(path, mmap_mode="r", allow_pickle=False)
        if path.suffix == ".npy"
        else fbin(path, dtype="<u4" if path.suffix == ".ibin" else "<f4")
    )


def validate_truth(gt, n=None):
    if gt.ndim != 2 or gt.shape[1] != 100 or not np.issubdtype(gt.dtype, np.integer):
        raise ValueError("Expected integer Q x 100 ground truth")
    if np.any(gt < 0) or (n is not None and np.any(gt >= n)):
        raise ValueError("Invalid ground-truth ID")
    if np.any(np.diff(np.sort(gt, axis=1), axis=1) == 0):
        raise ValueError("Duplicate ground truth")


def label_rows(assignment, neighbors, path, protocol, n, chunk=4096):
    path = Path(path)
    validate_truth(neighbors, n)
    if assignment.shape != (n,) or np.any(assignment < 0):
        raise ValueError("Invalid assignments")
    if path.exists():
        meta = read(path.with_suffix(".json"))
        if meta["protocol"] != protocol or sha(path) != meta["sha256"]:
            raise ValueError("Label identity changed")
        return np.load(path, mmap_mode="r")
    result = np.empty(neighbors.shape, np.int32)
    for lo in range(0, len(neighbors), chunk):
        result[lo : lo + chunk] = assignment[neighbors[lo : lo + chunk]]
    atomic(path, lambda f: np.save(f, result))
    save(
        path.with_suffix(".json"),
        dict(protocol=protocol, sha256=sha(path), rows=len(result)),
    )
    return result
