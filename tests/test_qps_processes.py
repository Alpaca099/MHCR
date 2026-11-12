import multiprocessing as mp
import os
import time
import numpy as np
import pytest
from mhcr.benchmark.processes import trial_processes
from mhcr.benchmark.qps import audit_trial, TrialFailed
from mhcr.benchmark.qps_cli import persist


class Fake:
    def search(self, q, point):
        if point.get("fail") and q[0] == 1:
            raise ValueError("injected")
        time.sleep(point.get("delay", 0.005))
        return np.arange(100, dtype=np.int64), np.ones(100, np.float32)


def factory():
    return Fake()


def slow_setup(w):
    time.sleep(0.25)


@pytest.mark.parametrize("method", ["fork", "spawn"])
def test_setup_excluded_and_audited(method):
    cores = sorted(os.sched_getaffinity(0))[:2]
    q = np.arange(3, dtype=np.float32).reshape(-1, 1)
    ids = np.arange(3)
    r = trial_processes(
        None,
        q,
        ids,
        {},
        1,
        cores,
        slow_setup,
        engine_factory=factory,
        start_method=method,
        ready_timeout=30,
    )
    assert r["wall_seconds"] < 0.2
    assert min(x["start_ns"][0] for x in r["records"]) >= r["wall_start_ns"]
    refs = [(np.arange(100), np.ones(100, np.float32))] * 3
    assert audit_trial(r, ids, np.tile(np.arange(100), (3, 1)), 100, refs)["passed"]


def test_failure_preserved_and_reaped(tmp_path):
    before = {p.pid for p in mp.active_children()}
    with pytest.raises(TrialFailed) as e:
        trial_processes(
            Fake(),
            np.arange(3).reshape(-1, 1),
            np.arange(3),
            {"fail": True},
            1,
            sorted(os.sched_getaffinity(0))[:1],
            ready_timeout=5,
            run_timeout=1,
        )
    persist(tmp_path, e.value.evidence)
    assert e.value.evidence["qps"] is None
    assert e.value.evidence["records"]
    assert {p.pid for p in mp.active_children()} == before
    assert (tmp_path / "trial.json").exists()


def test_hung_worker_is_reaped():
    before = {p.pid for p in mp.active_children()}
    with pytest.raises(TrialFailed):
        trial_processes(
            Fake(),
            np.zeros((1, 1)),
            np.array([0]),
            {"delay": 20},
            1,
            sorted(os.sched_getaffinity(0))[:1],
            run_timeout=0.1,
            collect_timeout=0.1,
        )
    assert {p.pid for p in mp.active_children()} == before


def test_duplicate_cores_rejected():
    c = min(os.sched_getaffinity(0))
    with pytest.raises(ValueError):
        trial_processes(Fake(), np.zeros((1, 1)), np.array([0]), {}, 1, [c, c])
