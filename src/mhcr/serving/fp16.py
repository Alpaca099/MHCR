"""Linux AVX-512 FP16 shard reader and canonical FP32 candidate refinement.

Build explicitly: python -m mhcr.serving.fp16 --build /path/to/libmhcr_fp16.so
The generated library belongs with deployment artifacts, not source control.
"""

import ctypes as C
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import weakref
import numpy as np


def require_platform():
    if sys.platform != "linux":
        raise RuntimeError("FP16 native refinement requires Linux x86 with AVX-512")
    flags = Path("/proc/cpuinfo").read_text().split("flags", 1)[-1].splitlines()[0]
    if not {"avx512f", "avx512dq", "avx2", "fma", "f16c"} <= set(flags.split()):
        raise RuntimeError(
            "FP16 native refinement requires AVX-512F/DQ, AVX2, FMA and F16C"
        )


def build_library(output, compiler="g++"):
    require_platform()
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=".mhcr-fp16-", suffix=".so", dir=output.parent
    )
    os.close(fd)
    try:
        subprocess.run(
            [
                compiler,
                "-std=c++17",
                "-Ofast",
                "-DNDEBUG",
                "-mavx512f",
                "-mavx512dq",
                "-mavx2",
                "-mfma",
                "-mf16c",
                "-shared",
                "-fPIC",
                str(Path(__file__).with_suffix(".cpp")),
                "-o",
                temporary,
            ],
            check=True,
        )
        # Atomic publication without overwriting another concurrent build.
        os.link(temporary, output)
    finally:
        os.unlink(temporary)
    return output


class HalfBase:
    """Map all shards read-only; only selected candidates are decoded to FP32."""

    def __init__(self, manifest, library):
        require_platform()
        self.lib = C.CDLL(str(Path(library).resolve()))
        fp, ip = C.POINTER(C.c_float), C.POINTER(C.c_int64)
        self.lib.mhcr_half_error.restype = C.c_char_p
        self.lib.mhcr_half_open.argtypes = [C.c_char_p]
        self.lib.mhcr_half_open.restype = C.c_void_p
        self.lib.mhcr_half_close.argtypes = [C.c_void_p]
        self.lib.mhcr_half_close.restype = None
        for name in ("size", "dim"):
            fn = getattr(self.lib, "mhcr_half_" + name)
            fn.argtypes, fn.restype = [C.c_void_p], C.c_uint64
        self.lib.mhcr_half_read.argtypes = [C.c_void_p, ip, C.c_size_t, fp]
        self.lib.mhcr_half_read.restype = C.c_int
        self.lib.mhcr_half_scores.argtypes = [C.c_void_p, fp, ip, C.c_size_t, fp]
        self.lib.mhcr_half_scores.restype = C.c_int
        self.handle = self.lib.mhcr_half_open(os.fsencode(Path(manifest).resolve()))
        if not self.handle:
            raise ValueError(self.lib.mhcr_half_error().decode())
        self._release = weakref.finalize(self, self.lib.mhcr_half_close, self.handle)
        self.shape = (
            int(self.lib.mhcr_half_size(self.handle)),
            int(self.lib.mhcr_half_dim(self.handle)),
        )

    def close(self):
        self._release()

    def _ids(self, ids):
        if not self._release.alive:
            raise RuntimeError("FP16 gallery is closed")
        ids = np.asarray(ids)
        if ids.ndim != 1 or ids.dtype.kind not in "iu":
            raise ValueError("Expected a one-dimensional integer ID array")
        if np.any(ids < 0) or np.any(ids >= self.shape[0]):
            raise ValueError("Gallery ID out of range")
        return np.ascontiguousarray(ids, np.int64)

    def _check(self, code):
        if code:
            raise RuntimeError(self.lib.mhcr_half_error().decode())

    def __getitem__(self, ids):
        ids = self._ids(ids)
        out = np.empty((len(ids), self.shape[1]), np.float32)
        self._check(
            self.lib.mhcr_half_read(
                self.handle,
                ids.ctypes.data_as(C.POINTER(C.c_int64)),
                len(ids),
                out.ctypes.data_as(C.POINTER(C.c_float)),
            )
        )
        return out

    def scores(self, query, ids):
        ids = self._ids(ids)
        query = np.ascontiguousarray(query, np.float32)
        if (
            query.shape not in ((self.shape[1],), (1, self.shape[1]))
            or not np.isfinite(query).all()
        ):
            raise ValueError("Expected one finite query matching gallery dimension")
        out = np.empty(len(ids), np.float32)
        self._check(
            self.lib.mhcr_half_scores(
                self.handle,
                query.ctypes.data_as(C.POINTER(C.c_float)),
                ids.ctypes.data_as(C.POINTER(C.c_int64)),
                len(ids),
                out.ctypes.data_as(C.POINTER(C.c_float)),
            )
        )
        return out


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", required=True, type=Path)
    parser.add_argument("--compiler", default="g++")
    args = parser.parse_args()
    print(build_library(args.build, args.compiler))
