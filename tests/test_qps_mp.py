"""Harness tests that need neither torch nor faiss."""

import os
import numpy as np
import pytest

from mhcr.benchmark.qps import audit_trial
from mhcr.benchmark.processes import trial_processes


class FakeEngine:
    def search(self, query, point):
        base = int(query[0])
        ids = (np.arange(100, dtype=np.int64) + base) % 512
        return ids, ids.astype(np.float32)


def _cores(n):
    available = sorted(os.sched_getaffinity(0))
    if len(available) < n:
        pytest.skip("needs %d CPUs" % n)
    return available[:n]


def _query_set(n=32):
    q = np.arange(n, dtype=np.float32).reshape(n, 1)
    qids = np.arange(n, dtype=np.int64)
    return q, qids


def test_single_worker_reports_its_own_window():
    q, qids = _query_set()
    result = trial_processes(FakeEngine(), q, qids, {"method": "fake"}, 1, _cores(1))
    assert result["workers"] == 1 and result["completed_queries"] == len(q)
    row = result["records"][0]
    assert np.all(row["start_ns"] >= result["wall_start_ns"])
    assert result["wall_end_ns"] == row["end_ns"][-1]


def test_two_workers_and_audit_agree_with_reference():
    q, qids = _query_set()
    truth = np.stack([(np.arange(100) + int(v)) % 512 for v in q[:, 0]]).astype(
        np.int64
    )
    ref = trial_processes(FakeEngine(), q, qids, {"method": "fake"}, 1, _cores(1))
    reference = [None] * len(q)
    for row in ref["records"]:
        for i, position in enumerate(row["order"]):
            reference[int(position)] = (row["ids"][i].copy(), row["scores"][i].copy())
    result = trial_processes(FakeEngine(), q, qids, {"method": "fake"}, 1, _cores(2))
    audit = audit_trial(result, qids, truth, 512, reference)
    assert audit["passed"] and audit["recall100"] == 1.0


def test_duplicate_cores_are_rejected():
    q, qids = _query_set(4)
    with pytest.raises(ValueError):
        trial_processes(FakeEngine(), q, qids, {"method": "fake"}, 1, [0, 0])
