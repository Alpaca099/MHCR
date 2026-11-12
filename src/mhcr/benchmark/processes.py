"""Bounded process trials with explicit readiness, release and result collection."""

import multiprocessing as mp
from multiprocessing.connection import wait
import os
import resource
import time
import numpy as np
from .qps import TrialFailed


def _worker(
    w,
    engine,
    factory,
    queries,
    qids,
    point,
    cycles,
    core,
    warmup,
    configure,
    start,
    collect,
    conn,
    timeout,
):
    record = dict(worker=w)
    captured = 0
    before = resource.getrusage(resource.RUSAGE_SELF)
    try:
        os.sched_setaffinity(0, {core})
        if configure:
            configure(w)
        if factory:
            engine = factory()
        count = len(queries) * cycles
        order = np.concatenate(
            [
                np.random.default_rng(20261107 + w * 100003 + c).permutation(
                    len(queries)
                )
                for c in range(cycles)
            ]
        )
        record.update(
            order=order,
            qids=np.asarray(qids)[order],
            ids=np.empty((count, 100), np.int64),
            scores=np.empty((count, 100), np.float32),
        )
        for key in ("start_ns", "end_ns", "query_major_faults", "query_minor_faults"):
            record[key] = np.empty(count, np.int64)
        for i in range(warmup):
            engine.search(queries[i % len(queries)], point)
        conn.send(("ready", w, None))
        if not start.wait(timeout):
            raise TimeoutError("Release timeout")
        before = resource.getrusage(resource.RUSAGE_SELF)
        for i, position in enumerate(order):
            record["start_ns"][i] = time.perf_counter_ns()
            b = resource.getrusage(resource.RUSAGE_SELF)
            ids, scores = engine.search(queries[position], point)
            if (
                ids.shape != (100,)
                or scores.shape != (100,)
                or ids.dtype != np.int64
                or scores.dtype != np.float32
            ):
                raise ValueError(
                    "Expected int64 IDs and float32 scores, both length 100"
                )
            record["ids"][i], record["scores"][i] = ids, scores
            a = resource.getrusage(resource.RUSAGE_SELF)
            record["query_major_faults"][i] = a.ru_majflt - b.ru_majflt
            record["query_minor_faults"][i] = a.ru_minflt - b.ru_minflt
            record["end_ns"][i] = time.perf_counter_ns()
            captured = i + 1
        conn.send(("finished", w, None))
        # Bulk IPC happens only after every worker has finished timing.
        if not collect.wait(timeout):
            raise TimeoutError("Collection timeout")
        error = None
    except BaseException as exc:
        error = repr(exc)
    finally:
        a = resource.getrusage(resource.RUSAGE_SELF)
        record.update(
            major_faults=a.ru_majflt - before.ru_majflt,
            minor_faults=a.ru_minflt - before.ru_minflt,
            cpu_seconds=a.ru_utime + a.ru_stime - before.ru_utime - before.ru_stime,
            peak_rss_bytes=int(a.ru_maxrss * 1024),
        )
        for key, value in list(record.items()):
            if isinstance(value, np.ndarray):
                record[key] = value[:captured]
        try:
            conn.send(("record", w, dict(error=error, record=record)))
        finally:
            conn.close()


def trial_processes(
    engine,
    queries,
    qids,
    point,
    cycles,
    cores,
    configure_worker=None,
    *,
    engine_factory=None,
    start_method="fork",
    warmup=0,
    ready_timeout=600.0,
    run_timeout=900.0,
    collect_timeout=120.0,
):
    """Fork accepts a preloaded engine; spawn requires a picklable factory.

    Timeouts invalidate the whole trial. Records from killed/unresponsive workers
    may be missing; no uninitialized arrays are returned as completed queries.
    """
    cores = list(cores)
    if (
        not cores
        or len(set(cores)) != len(cores)
        or not set(cores) <= os.sched_getaffinity(0)
    ):
        raise ValueError("Distinct cores within the current affinity are required")
    if (
        cycles < 1
        or warmup < 0
        or len(queries) == 0
        or len(queries) != len(qids)
        or len(np.unique(qids)) != len(qids)
    ):
        raise ValueError("Invalid cohort, cycles or warmup")
    if min(ready_timeout, run_timeout, collect_timeout) <= 0:
        raise ValueError("Timeouts must be positive")
    if start_method not in ("fork", "spawn") or (
        start_method == "spawn" and engine_factory is None
    ):
        raise ValueError(
            "spawn requires an engine_factory; supported methods: spawn, fork"
        )
    ctx = mp.get_context(start_method)
    start = ctx.Event()
    collect = ctx.Event()
    procs = []
    readers = {}
    records = {}
    errors = {}
    begin = None
    failure = None

    def receive_until(stage, timeout):
        pending = set(range(len(cores)))
        deadline = time.monotonic() + timeout
        while pending:
            left = deadline - time.monotonic()
            if left <= 0:
                raise TimeoutError(stage + " timeout")
            for conn in wait(
                [c for c, w in readers.items() if w in pending], timeout=min(left, 0.1)
            ):
                kind, w, payload = conn.recv()
                if kind == "record":
                    records[w] = payload["record"]
                    if payload["error"]:
                        errors[w] = payload["error"]
                        raise RuntimeError(payload["error"])
                if kind != stage:
                    raise RuntimeError("Unexpected worker message: " + kind)
                pending.remove(w)
            for w in pending:
                if (
                    procs[w].exitcode is not None
                    and not next(c for c, i in readers.items() if i == w).poll()
                ):
                    raise RuntimeError(f"Worker {w} exited before {stage}")

    try:
        for w, core in enumerate(cores):
            recv, send = ctx.Pipe(False)
            p = ctx.Process(
                target=_worker,
                args=(
                    w,
                    engine,
                    engine_factory,
                    queries,
                    qids,
                    point,
                    cycles,
                    core,
                    warmup,
                    configure_worker,
                    start,
                    collect,
                    send,
                    max(ready_timeout, run_timeout, collect_timeout),
                ),
            )
            p.start()
            send.close()
            procs.append(p)
            readers[recv] = w
        receive_until("ready", ready_timeout)
        begin = time.perf_counter_ns()
        start.set()
        receive_until("finished", run_timeout)
        collect.set()
        receive_until("record", collect_timeout)
    except BaseException as exc:
        failure = repr(exc)
    finally:
        collect.set()
        # Drain responsive workers, including partial records, before terminating.
        deadline = time.monotonic() + min(collect_timeout, 2.0)
        while failure and time.monotonic() < deadline:
            active = [c for c, w in readers.items() if w not in records]
            if not active:
                break
            for conn in wait(active, timeout=0.05):
                try:
                    kind, w, payload = conn.recv()
                    if kind == "record":
                        records[w] = payload["record"]
                        if payload["error"]:
                            errors[w] = payload["error"]
                except (EOFError, OSError):
                    readers.pop(conn, None)
                    conn.close()
        for p in procs:
            if failure and p.is_alive():
                p.terminate()
            p.join(timeout=2)
            if p.is_alive():
                p.kill()
                p.join(timeout=2)
            if p.exitcode != 0:
                failure = failure or f"Worker exit {p.exitcode}"
        for conn in readers:
            conn.close()
    base = dict(
        workers=len(cores),
        cycles=cycles,
        point=dict(point),
        cores=cores,
        queries_per_worker=len(queries) * cycles,
        wall_start_ns=begin,
        start_method=start_method,
        warmup_per_worker=warmup,
        records=[records[w] for w in sorted(records)],
    )
    if failure:
        raise TrialFailed(
            dict(
                base,
                valid=False,
                qps=None,
                error=failure,
                worker_errors=errors,
                missing_worker_records=sorted(set(range(len(cores))) - records.keys()),
                observed_end_ns=time.perf_counter_ns(),
            )
        )
    finish = max(int(r["end_ns"][-1]) for r in records.values())
    seconds = (finish - begin) / 1e9
    return dict(
        base,
        wall_end_ns=finish,
        wall_seconds=seconds,
        completed_queries=len(cores) * len(queries) * cycles,
        qps=len(cores) * len(queries) * cycles / seconds,
        timing="Ready/warmup excluded; common event release to last captured response; bulk IPC excluded",
        workload="Closed-loop, one outstanding query per worker, no batching",
    )
