"""Closed-loop throughput with worker-private engines and raw outputs.

Configure worker affinity and thread limits before timing. Verify all outputs
against a single-worker reference. Never share a mutable index across workers.
"""

import concurrent.futures, threading, time, resource, os
import numpy as np


class TrialFailed(RuntimeError):
    """Invalid whole trial carrying all fully captured partial responses."""

    def __init__(self, evidence):
        super().__init__("Throughput trial failed; no QPS result is valid")
        self.evidence = evidence


def trial(engines, queries, qids, point, cycles, cores=None, configure_worker=None):
    assert engines and len({id(engine) for engine in engines}) == len(engines)
    assert (
        cycles >= 1
        and len(queries) == len(qids) > 0
        and len(np.unique(qids)) == len(qids)
    )
    workers = len(engines)
    count = len(queries) * cycles
    assert cores is None or len(cores) == workers
    ready = threading.Barrier(workers + 1, timeout=60)
    start = threading.Event()
    abort = threading.Event()
    partial = [None] * workers
    errors = [None] * workers

    def work(worker):
        # Preallocate outside throughput window. Preserve every returned output.
        ids = np.empty((count, 100), np.int64)
        scores = np.empty((count, 100), np.float32)
        starts = np.empty(count, np.int64)
        ends = np.empty(count, np.int64)
        majors = np.empty(count, np.int64)
        minors = np.empty(count, np.int64)
        order = np.concatenate(
            [
                np.random.default_rng(20261107 + worker * 100003 + cycle).permutation(
                    len(queries)
                )
                for cycle in range(cycles)
            ]
        )
        ordered_qids = np.asarray(qids)[order]
        captured = 0
        before = resource.getrusage(resource.RUSAGE_THREAD)
        try:
            if cores is not None:
                os.sched_setaffinity(0, {cores[worker]})
            if configure_worker is not None:
                configure_worker(worker)
            before = resource.getrusage(resource.RUSAGE_THREAD)
            ready.wait()
            assert start.wait(timeout=60)
            for i, position in enumerate(order):
                if abort.is_set():
                    raise RuntimeError("Another worker failed; entire trial invalid")
                starts[i] = time.perf_counter_ns()
                usage_before = resource.getrusage(resource.RUSAGE_THREAD)
                found, values = engines[worker].search(queries[position], point)
                # Output capture is included equally for both methods. No disk IO.
                assert found.shape == values.shape == (100,)
                assert found.dtype == np.dtype("int64") and values.dtype == np.dtype(
                    "float32"
                )
                ids[i] = found
                scores[i] = values
                usage_after = resource.getrusage(resource.RUSAGE_THREAD)
                majors[i] = usage_after.ru_majflt - usage_before.ru_majflt
                minors[i] = usage_after.ru_minflt - usage_before.ru_minflt
                ends[i] = time.perf_counter_ns()
                captured = i + 1
        except BaseException as error:
            errors[worker] = repr(error)
            abort.set()
            ready.abort()
            raise
        finally:
            after = resource.getrusage(resource.RUSAGE_THREAD)
            # Only fully captured responses, never uninitialized capacity.
            partial[worker] = dict(
                worker=worker,
                order=order[:captured],
                qids=ordered_qids[:captured],
                ids=ids[:captured],
                scores=scores[:captured],
                start_ns=starts[:captured],
                end_ns=ends[:captured],
                major_faults=after.ru_majflt - before.ru_majflt,
                minor_faults=after.ru_minflt - before.ru_minflt,
                query_major_faults=majors[:captured],
                query_minor_faults=minors[:captured],
                cpu_seconds=(after.ru_utime + after.ru_stime)
                - (before.ru_utime + before.ru_stime),
            )
        return partial[worker]

    begin = None
    failure = None
    records = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(work, i) for i in range(workers)]
        try:
            ready.wait()
            begin = time.perf_counter_ns()
            start.set()
        except BaseException as error:
            failure = repr(error)
            abort.set()
            start.set()
            ready.abort()
        for worker, future in enumerate(futures):
            try:
                records.append(future.result())
            except BaseException as error:
                failure = repr(error)
                errors[worker] = repr(error)
                abort.set()
                start.set()
                ready.abort()
    if failure is not None:
        raise TrialFailed(
            dict(
                valid=False,
                qps=None,
                workers=workers,
                cycles=cycles,
                point=dict(point),
                cores=cores,
                queries_per_worker=count,
                wall_start_ns=begin,
                observed_end_ns=time.perf_counter_ns(),
                error=failure,
                worker_errors=errors,
                records=[row for row in partial if row is not None],
                missing_worker_records=[
                    i for i, row in enumerate(partial) if row is None
                ],
            )
        )
    finish = max(int(row["end_ns"][-1]) for row in records)
    assert finish > begin and all(
        np.all(row["start_ns"] >= begin) and np.all(row["end_ns"] >= row["start_ns"])
        for row in records
    )
    seconds = (finish - begin) / 1e9
    completed = workers * count
    return dict(
        workers=workers,
        cycles=cycles,
        queries_per_worker=count,
        completed_queries=completed,
        point=dict(point),
        cores=cores,
        wall_start_ns=begin,
        wall_end_ns=finish,
        wall_seconds=seconds,
        qps=completed / seconds,
        timing="Common release to last captured response; includes request loop, result capture and per-request thread page-fault sampling; excludes setup/warmup/persistence/audit",
        workload="Closed-loop one outstanding query per worker, no query batching, fixed complete cohort cycles",
        records=records,
    )


def audit_trial(result, qids, truth, n, reference):
    """Outside timed interval: all IDs/scores agree with single-worker reference."""
    assert truth.shape == (len(qids), 100) and len(np.unique(qids)) == len(qids)
    workers = result["workers"]
    cycles = result["cycles"]
    count = result["queries_per_worker"]
    assert workers >= 1 and cycles >= 1 and count == len(qids) * cycles
    assert len(result["records"]) == workers
    assert sorted(row["worker"] for row in result["records"]) == list(range(workers))
    assert len(reference) == len(qids)
    begin = result["wall_start_ns"]
    finish = result["wall_end_ns"]
    assert finish > begin
    assert finish == max(int(row["end_ns"][-1]) for row in result["records"])
    completed = 0
    hits = 0
    for row in result["records"]:
        order = row["order"]
        assert len(order) == result["queries_per_worker"]
        assert order.dtype.kind in "iu" and np.all((order >= 0) & (order < len(qids)))
        assert np.array_equal(
            np.bincount(order, minlength=len(qids)),
            np.full(len(qids), result["cycles"]),
        )
        for cycle in range(cycles):
            assert np.array_equal(
                np.sort(order[cycle * len(qids) : (cycle + 1) * len(qids)]),
                np.arange(len(qids)),
            )
        assert np.array_equal(row["qids"], np.asarray(qids)[order])
        assert row["ids"].shape == row["scores"].shape == (len(order), 100)
        assert row["ids"].dtype == np.dtype("int64") and row[
            "scores"
        ].dtype == np.dtype("float32")
        starts = row["start_ns"]
        ends = row["end_ns"]
        assert starts.shape == ends.shape == (
            count,
        ) and starts.dtype == ends.dtype == np.dtype("int64")
        assert (
            np.all(starts >= begin)
            and np.all(ends <= finish)
            and np.all(ends >= starts)
        )
        assert np.all(
            starts[1:] >= ends[:-1]
        ), "Worker requests overlap or run out of order"
        assert (
            row["major_faults"] >= 0
            and row["minor_faults"] >= 0
            and row["cpu_seconds"] >= 0
        )
        for key, total in (
            ("query_major_faults", "major_faults"),
            ("query_minor_faults", "minor_faults"),
        ):
            faults = row[key]
            assert (
                faults.shape == (count,)
                and faults.dtype == np.dtype("int64")
                and np.all(faults >= 0)
            )
            assert int(faults.sum()) <= row[total]
        for i, position in enumerate(order):
            ids = row["ids"][i]
            scores = row["scores"][i]
            assert (
                len(np.unique(ids)) == 100
                and np.all((ids >= 0) & (ids < n))
                and np.isfinite(scores).all()
            )
            expected = reference[int(position)]
            assert (
                np.array_equal(ids, expected[0])
                and scores.tobytes()
                == np.asarray(expected[1], dtype=np.float32).tobytes()
            ), "Concurrent output differs from frozen single-worker result"
            hits += len(np.intersect1d(ids, truth[position]))
        completed += len(order)
    assert completed == result["completed_queries"]
    seconds = (result["wall_end_ns"] - result["wall_start_ns"]) / 1e9
    assert seconds == result["wall_seconds"] and result["qps"] == completed / seconds
    return dict(
        passed=True,
        completed_queries=completed,
        recall100=hits / (completed * 100),
        single_worker_output_bitwise_equal=True,
        all_queries_retained=True,
    )
