"""Explicit-posting IVF-Flat backend for the extracted CPU routers."""

import time

import faiss
import numpy as np
import torch

from ..metrics import dedup_top100


def build_flat(base, centroids, assignments, batch_size=8192):
    """Build normalized-IP IVF with one or two assignments per vector.

    Additional assignments use -1 for absent copies. Input vectors must already
    be normalized, as in the formal experiments. Returns physical list sizes.
    """
    centroids = np.ascontiguousarray(centroids, dtype=np.float32)
    assignments = np.asarray(assignments, dtype=np.int64)
    if assignments.ndim == 1:
        assignments = assignments[:, None]
    k, d = centroids.shape
    if assignments.shape[0] != len(base) or assignments.shape[1] not in (1, 2):
        raise ValueError("Expected N x 1 or N x 2 assignments.")
    if (
        np.any(assignments[:, 0] < 0)
        or np.any(assignments < -1)
        or np.any(assignments >= k)
    ):
        raise ValueError("Invalid assignment ID.")
    if assignments.shape[1] == 2 and np.any(assignments[:, 0] == assignments[:, 1]):
        raise ValueError("Secondary assignment must differ from primary.")
    quantizer = faiss.IndexFlatIP(d)
    quantizer.add(centroids)
    index = faiss.IndexIVFFlat(quantizer, d, k, faiss.METRIC_INNER_PRODUCT)
    index.is_trained = True
    for start in range(0, len(base), batch_size):
        end = min(start + batch_size, len(base))
        vectors = np.ascontiguousarray(base[start:end], dtype=np.float32)
        if not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-4):
            raise ValueError("Database vectors must be L2 normalized.")
        for column in range(assignments.shape[1]):
            valid = assignments[start:end, column] >= 0
            x = np.ascontiguousarray(vectors[valid])
            ids = np.ascontiguousarray(np.arange(start, end, dtype=np.int64)[valid])
            cells = np.ascontiguousarray(assignments[start:end, column][valid])
            if len(x):
                index.add_core(
                    len(x),
                    faiss.swig_ptr(x),
                    faiss.swig_ptr(ids),
                    faiss.swig_ptr(cells),
                )
    sizes = np.bincount(assignments[assignments >= 0], minlength=k)
    return index, sizes


def search(router, index, queries):
    """Return 100 unique IDs, including final deduplication in the call boundary.

    Uses the top-200 retrieval for at most two postings per vector.
    IVF-Flat ignores the coarse distances for candidate scoring.
    """
    queries = np.ascontiguousarray(queries, dtype=np.float32)
    cells, distances, budgets, _ = router.online(torch.from_numpy(queries))
    found = np.full((len(queries), 200), -1, dtype=np.int64)
    for budget in np.unique(budgets):
        selected = np.flatnonzero(budgets == budget)
        index.nprobe = int(budget)
        _, ids = index.search_preassigned(
            np.ascontiguousarray(queries[selected]),
            200,
            np.ascontiguousarray(cells[selected, :budget], dtype=np.int64),
            np.ascontiguousarray(distances[selected, :budget], dtype=np.float32),
        )
        found[selected] = ids
    unique = dedup_top100(found)
    for i in np.flatnonzero(np.any(unique < 0, axis=1)):
        # Extend only undersized results; at most two copies of each ID exist.
        width = int(budgets[i])
        while np.any(unique[i] < 0) and width < cells.shape[1]:
            width = min(cells.shape[1], max(width + 1, width * 2))
            index.nprobe = width
            _, ids = index.search_preassigned(
                queries[i : i + 1],
                200,
                np.ascontiguousarray(cells[i : i + 1, :width], dtype=np.int64),
                np.zeros((1, width), dtype=np.float32),
            )
            unique[i] = dedup_top100(ids)[0]
        budgets[i] = width
        if np.any(unique[i] < 0):
            raise ValueError(
                "Gallery or routing order cannot supply 100 unique results"
            )
    return unique, budgets


def timed_search(router, index, queries):
    """New measurement boundary; deliberately not a replay of published timing."""
    started = time.perf_counter()
    ids, budgets = search(router, index, queries)
    return ids, budgets, time.perf_counter() - started
