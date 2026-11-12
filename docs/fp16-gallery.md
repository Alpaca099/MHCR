# FP16 gallery storage

DataComp uses the complete 128M gallery in FP16 shards. Candidate vectors are
decoded as `float32(stored_fp16) * saved_float32_inverse_norm`; they are never
rounded back to FP16. Only the candidates for the current query are decoded.
The index and routing model remain unchanged.

Build the optional Linux x86 native library on the deployment host (GCC with
C++17, AVX-512F/DQ, AVX2, FMA and F16C required):

```bash
python -m mhcr.serving.fp16 --build artifacts/libmhcr_fp16.so
```

The command refuses to overwrite an existing library. Keep the generated binary
with deployment artifacts, and rebuild it for a different CPU or compiler.
The wheel includes the C++ source; no RoarGraph dependency is needed.

Set these fields in the engine manifest, alongside the index, model and cost:

```json
{
  "base": "gallery.txt",
  "base_format": "fp16_shards",
  "native_library": "artifacts/libmhcr_fp16.so",
  "routing_variant": "datacomp",
  "expected_nlist": 65536,
  "quantized": true,
  "top_k": 100
}
```

`gallery.txt` describes the physical layout without copying vector data:

```text
N D SHARD_COUNT "inverse_norm.npy" NORM_DATA_OFFSET_BYTES
"shard_000.npy" SHARD_DATA_OFFSET_BYTES ROWS
"shard_001.npy" SHARD_DATA_OFFSET_BYTES ROWS
```

Use actual numbers in place of the uppercase fields. Shards define contiguous
canonical IDs in manifest order. Paths may be absolute or relative to the
manifest. Offsets exclude array headers; for a NumPy file obtain the data offset
from `np.load(path, mmap_mode="r").offset`. Norms are little-endian FP32; shards
are little-endian FP16, row-major. File sizes and total row count are checked
when opening. The dimension must be a multiple of 16 and at most 1024.

The saved inverse norms are part of the dataset identity. Recomputing them, or
renormalizing after casting normalized vectors back to FP16, changes scores.
Supply the original normalized FP32 queries. The `datacomp` routing variant
preserves their values, the quantized eager encoder and the original Torch top-k
prefix accumulation; the default `stable` variant serves the other datasets.

Evaluation, breakdown and QPS use the same engine manifest. Each QPS process
opens read-only mappings of the same files; pages are shared by the OS, while
each process has its own index and router. Budget memory for those private
objects when choosing worker count. Warmup and residency are benchmark choices:
record page faults and distinguish warm-memory results from disk-backed runs.

Both `mhcr-build` and `mhcr-pipeline` accept this storage format. Fresh partition
and PQ training decode only their configured samples; index population and exact
GT use bounded blocks. Checkpoints retain canonical IDs across shard boundaries.
The complete corpus is never materialized as FP32.
