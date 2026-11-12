# Reproduction interfaces

Install the package before using these commands; no research checkout is needed.

## From vectors to results

Copy [inputs.example.yaml](inputs.example.yaml) and set four input paths:

```yaml
base: data/gallery.fbin
training: data/training.npz
calibration: data/calibration.npz
test: data/test.npz
```

Paths are relative to the input YAML. The gallery is a normalized FP32 `.npy`
or `.fbin` matrix. Query files are normalized FP32 `.npy`/`.fbin` matrices or
NPZ files containing `q`, optional integer `query_ids`, and optional `gt` (Q × 100).
Explicit query IDs must be globally unique across splits. With matrix-only
inputs, split-specific IDs are generated; content overlap is still checked.
If `gt` is absent, blocked exact full-gallery inner products generate top-100
neighbors, with query-block checkpoints. Supplying independently prepared exact
GT avoids repeating this expensive step; its input identity is recorded.

```bash
mhcr-pipeline --config repro/configs/cc3m.yaml --inputs inputs.yaml --out artifacts/cc3m --threads 8
```

The command prepares disjoint cohorts, builds a full-gallery FastScan index,
creates cell labels, measures scan costs, trains the teacher and student,
selects operating points on calibration queries and evaluates held-out queries.
It also writes latency breakdowns. Calibration refines observed recall brackets
around each target (up to eight steps per rerank budget and target by default);
every selected point is measured, not interpolated. No pretrained model, old index, cost array or
external research checkout is required. Targets unavailable in the supplied
calibration grid are recorded explicitly, never interpolated or labeled achieved.

Repeat the same command to resume verified stage/epoch/query checkpoints.
`--through build` stops after index construction; other stage names are shown by
`--help`. Inputs, recipe and package sources are bound to the run, so a different
recipe needs a new output directory. `--timing-core` pins scan-cost calibration
and evaluation to one allowed CPU; preparation/training use `--threads`.
Run benchmarks without competing heavy work on that CPU or its sibling.

For DataComp, use `datacomp128m.yaml` and either an existing FP16 manifest as
`base`, or list the original FP16 NumPy files:

```yaml
base_shards:
  - data/images_000.npy
  - data/images_001.npy
training: data/training.npz
calibration: data/calibration.npz
test: data/test.npz
```

The second form computes inverse norms and a manifest without copying the gallery.
An optional `inverse_norm: data/inverse_norm.npy` preserves pre-existing canonical
norms. Partition/PQ training read only their configured samples, index population
reads bounded chunks, and GT uses bounded score tiles. No full FP32 gallery is
created. Native FP16 query refinement requires Linux x86 AVX-512 and GCC;
[storage details](../docs/fp16-gallery.md) include the exact decoding convention.

The output contains `engine.json`, `built/`, `teacher/`, `student/`, `cost/`,
`supervision/`, `calibration/selection.json`, and `evaluation/`. Training budgets
are derived from the new measured costs and training-query cell coverage when
`student.budget_range` is null. The resolved recipe is saved separately.

QPS uses the same generated engine and a selected point, with explicit worker CPUs:

```bash
mhcr-qps --engine artifacts/cc3m/engine.json --queries artifacts/cc3m/supervision/test.npz --point artifacts/cc3m/evaluation/mhcr_0.950.json --workers 1,2 --cores 0,1 --cycles 3 --warmup 24 --min-recall 0.95 --out artifacts/cc3m/qps95
```

The minimum recall check applies to the test cohort; calibration success does
not guarantee that every new held-out cohort reaches the same numerical target.

## Individual stages and existing artifacts

Create `engine.json` with asset paths relative to the manifest:

```json
{
  "index": "index/fastscan.faiss",
  "base": "base.fbin",
  "model": "model.pt",
  "cost": "cost_ms.npy",
  "top_k": 100,
  "quantized": true
}
```

The default gallery is normalized FP32 with canonical IDs matching the index.
DataComp uses [FP16 shards with native refinement](../docs/fp16-gallery.md).
Load only
trusted model checkpoints. Query NPZ files contain `q` (Q × D), `gt` (Q × 100
image IDs in descending exact similarity order), and unique `query_ids` (Q). Supply normalized queries; routing
normalizes its own feature copy while scanning and refinement preserve the input.

`point.json` specifies the method and its calibrated operating point:

```json
{"method": "MHCR", "penalty": 0.001, "rerank": 5000}
```

These values illustrate the format; choose the penalty on calibration queries.
For IVF, use `method: "IVF"`, `nprobe`, and `rerank` instead.

## Commands

```bash
python repro/build_index.py --config recipe.yaml --base base.fbin --out built
python repro/train.py --stage teacher --config recipe.yaml --features teacher.npy --cells teacher_cells.npy --centroids built/centroids.npy --identity identity.json --out teacher
python repro/train.py --stage student --config recipe.yaml --features student.npy --cells student_cells.npy --teacher teacher/model.pt --cost cost_ms.npy --sizes built/sizes.npy --identity identity.json --out student
python repro/evaluate.py --engine engine.json --queries test.npz --point point.json --out evaluation
```

The same evaluation entry supports latency decomposition and serving ablations:

```bash
python repro/evaluate.py --engine engine.json --queries test.npz --point point.json --breakdown --out breakdown
python repro/evaluate.py --engine engine.json --queries test.npz --point point.json --ablation fixed_probe --out ablation
```

Use `--dataset-config repro/configs/cc12m.yaml` to assert gallery/index identity.
For fixed-probe ablations, point.json must include integer `probes`; cosine arms
also require a positive calibrated `temperature`. All arms require `rerank` and
adaptive arms use `penalty`. Supported arms: full, fixed_probe, cosine_adaptive,
cosine_fixed, ivf. Calibrate each arm separately to match Recall@100. The fixed
arms cannot use the penalty calibration option; supply their calibrated probes.

Breakdown raw records include routing/preparation, scan, refinement/validation/
top-100 time, selected cells, scanned entries and refined candidates. These are
phase timings and work counts, not a full instrumentation of distance arithmetic.
Profiling adds overhead; use ordinary evaluation for uninstrumented latency.

Optional end-to-end calibration:

```bash
python repro/evaluate.py --engine engine.json --queries test.npz --point point.json --calibration calibration.npz --penalties 0,0.001,0.002 --target 0.98 --out evaluation
```

Calibration selects the largest tested penalty meeting **mean returned-ID Recall@100**.
Test and calibration query IDs/content must be disjoint. Cell-coverage LCB helpers
in `calibrate.py` are analysis utilities, not this end-to-end operating-point selector.

Dataset YAMLs include the index, teacher and student recipes. `published_training`
retains selected-model query counts and reference budget ranges; new runs derive
budgets from their own index. See [configuration notes](configs/README.md).
The lower-level training entry accepts prepared feature/cell arrays and a provenance
receipt. The pipeline generates these receipts after checking split content and IDs.

`mhcr.benchmark.qps` and `mhcr.benchmark.ablation` expose callable helpers. They are not autonomous experiment
schedulers. QPS requires worker-private engines and explicit CPU/resource setup.

Throughput has its own command and implementation:

```bash
python repro/qps.py --engine engine.json --queries test.npz --point point.json --workers 1,2,4,8 --cores 8,9,10,11,12,13,14,15 --cycles 3 --warmup 24 --min-recall 0.95 --out qps_run
```

After installation, `mhcr-qps` provides the same interface. The default is
process-based `spawn`, with explicit `fork` available for validated environments.
See [the QPS protocol](../docs/qps.md) for memory tradeoffs, timing, raw records,
correctness auditing and failure handling.
