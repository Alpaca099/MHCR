# Audited QPS measurements

Run on Linux with a frozen operating point and an evaluation NPZ containing
`q`, `gt` (Q x 100), and unique `query_ids`. No tuning occurs in this command.

```bash
mhcr-qps --engine engine.json --queries test.npz --point point.json \
  --workers 1,2,4,8 --cores 8,9,10,11,12,13,14,15 \
  --cycles 3 --warmup 24 --min-recall 0.95 --out qps-run
```

Equivalent entry points are `python repro/qps.py` and
`python -m mhcr.benchmark.qps_cli`. Choose CPU IDs available in the current
affinity; each worker uses a distinct core and one library thread. Core IDs
alone do not guarantee distinct physical cores; inspect machine topology.

The default `--start-method spawn` loads a private engine in each child. Loading
and warmup are outside timing. This avoids inheriting initialized thread-library
state, but can require multiple full index copies. Budget RAM before increasing
workers. `--start-method fork` inherits a parent-loaded engine and shares pages
until writes occur. It is an explicit, environment-dependent option: initialized
native thread pools may deadlock, for MHCR as well as IVF. Test that environment
before formal measurements. The command does not silently switch methods.

Every child announces readiness after loading, allocation and warmup. The parent
then timestamps a common event release. Timing ends at the last captured response.
The request loop, copying returned IDs/scores and per-query page-fault sampling
are included; bulk result IPC, disk persistence and correctness auditing follow
all timed queries. This is closed-loop throughput, one request per worker,
without batching. Thread-based `mhcr.benchmark.qps.trial` remains a separate
library utility for measuring threaded deployments.

A single-worker reference pass precedes the requested trials. Every result must
match its reference IDs and float32 scores bit for bit; Recall@100 is recomputed
from GT and checked against `--min-recall`. This checks concurrent consistency,
not independent algorithm correctness. Speedup is reported only when a measured
one-worker point is present in `--workers`.

Outputs:

- `protocol.json`: arguments, point, input hashes, environment and measurement scope.
- `reference/`: the single-worker raw records and audit.
- `workers-W/worker-I.npz`: order, query IDs, returned IDs/scores, start/end times,
  major/minor faults for every captured query.
- `workers-W/trial.json` and `audit.json`: trial metadata, CPU time, process peak
  RSS, validity and audit outcome.
- `qps.json`: incrementally saved QPS, recall, mean/P99 latency and speedup.
- `status.json`: completion or the stage that failed.

Readiness and timed work have explicit deadlines (`--ready-timeout`,
`--run-timeout`). Failures invalidate the entire trial and stop subsequent trials.
Responsive failed workers return their captured records; killed or hung workers
can have missing records, explicitly listed. A pending or failed audit is not a
valid QPS result. All children started by the runner are reaped on failure.

Peak RSS includes loading/warmup. Shared pages count in each process's RSS;
adding worker RSS values does not measure unique physical memory. PSS/USS are
not collected by this command. Use external monitoring when comparing deployment
memory. Match CPU allocation, recall target, data, refinement settings, warmup and
timing policy across methods; disclose different process/thread deployments.

This runner is not a claim of bitwise reproduction of a historical QPS harness.
Compare its protocol with the archived experiment before reproducing a paper row.
