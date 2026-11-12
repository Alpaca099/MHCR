# Data assets

No datasets are redistributed. Supply embedding matrices and disjoint query
cohorts through the [pipeline input manifest](../repro/README.md).

- Gallery: normalized contiguous FP32 `.npy` or `.fbin` (two uint32 header values N,D).
- Training queries: normalized FP32 array; cell labels: integer Q x 100 `.npy`.
- Neighbor IDs: integer Q x 100 `.npy` or uint32 `.ibin`, ordered by descending
  exact inner product. Each row has 100 distinct canonical gallery IDs.
- Evaluation/calibration: `.npz` with `q`, `gt` (Q x 100), `query_ids`.
- Index: non-residual inner-product PQ4 FastScan, canonical single-assignment IDs.
- Costs: positive finite per-cell marginal costs, length K.

DataComp is the full 128,000,000-vector, 768-dimensional gallery with 65,536 cells.
Its build and query backends read FP16 shards and saved FP32 inverse norms
directly, decoding bounded blocks or selected candidates. See
[FP16 gallery storage](../docs/fp16-gallery.md) for the manifest and build command.
No full FP32 gallery copy or smaller replacement gallery is required.
