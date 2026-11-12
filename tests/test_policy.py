import numpy as np
from mhcr.policy import prefix_budget


def test_prefix_matches_bruteforce():
    rng = np.random.default_rng(0)
    n, k = 64, 37
    p = rng.random((n, k)).astype(np.float32)
    p /= p.sum(axis=1, keepdims=True)
    c = rng.integers(1, 500, size=(n, k)).astype(np.int64)
    predcov = np.cumsum(p, axis=1)
    scans = np.cumsum(c, axis=1)
    for lam in (0.0, 1e-4, 1e-2, 1.0):
        got = prefix_budget(predcov, scans, lam)
        for i in range(n):
            obj = [0.0] + list(predcov[i] - lam * scans[i])
            want = max(int(np.argmax(obj)), 1)
            assert got[i] == want


def test_prefix_is_monotone_in_penalty():
    rng = np.random.default_rng(1)
    p = rng.random((8, 50)).astype(np.float32)
    p /= p.sum(axis=1, keepdims=True)
    c = np.ones((8, 50), dtype=np.int64)
    predcov, scans = np.cumsum(p, axis=1), np.cumsum(c, axis=1)
    lengths = [
        prefix_budget(predcov, scans, lam).mean() for lam in (0.0, 1e-3, 1e-2, 1e-1)
    ]
    assert all(a >= b for a, b in zip(lengths, lengths[1:]))
