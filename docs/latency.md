# Latency measurement

`repro/evaluate.py` uses one PyTorch thread and one Faiss OpenMP thread. It runs
queries sequentially, without warmup, and records returned-ID Recall@100, mean
and P99 latency and per-query major/minor page faults. It does not pin the process
to a CPU; use `taskset -c <core>` when a pinned measurement is required.

Optional calibration runs before test queries and warms shared assets. Record
whether calibration was performed in the same process. This portable protocol
is not an exact replay of the historical benchmark's warmup or machine state.

Keep all queries, including page-faulting queries. Record CPU, software versions,
affinity, competing load, index/model identity and the selected point. Compare
methods on the same host with the same measurement procedure.

`mhcr.benchmark.qps.trial` measures closed-loop request throughput with private
engines per worker. Configure worker affinity and thread limits explicitly, then
use `audit_trial` to compare every response to a single-worker reference.
For the process-based benchmark, use `repro/qps.py`; see [QPS](qps.md).

Use `repro/evaluate.py --breakdown` with the same engine and operating point for
phase measurements. Raw records contain routing/preparation, scan and
refinement/validation/top-100 times. Selected-cell counts, scanned list entries
and refined candidate counts describe work volume, not complete arithmetic or
SIMD instruction counts. Report instrumented timings separately from ordinary
latency measurements.
