from pathlib import Path
import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


import sys

AF = sys.modules[__name__]
TOP_K = 100


def refinement_candidate(rows, method, target, tolerance=0.002):
    """Refine an observed recall bracket; every candidate still needs measuring."""
    key = "penalty" if method == "MHCR" else "nprobe"
    ordered = sorted(rows, key=lambda r: r["point"][key])
    if any(target <= r["recall100"] <= target + tolerance for r in ordered):
        return None
    for left, right in zip(ordered, ordered[1:]):
        a, b = left["recall100"], right["recall100"]
        if (a < target) == (b < target):
            continue
        lo, hi = left["point"][key], right["point"][key]
        if method == "IVF":
            value = (lo + hi) // 2
        else:
            value = math.sqrt(lo * hi) if lo > 0 else hi / 2
        if lo < value < hi:
            return dict(left["point"], **{key: value})
    return None


def calibrate_end_to_end(engine, queries, truth, point, penalties, target):
    """Select the largest tested penalty meeting mean returned-ID Recall@100."""
    from .cli import recall100
    from .data import validate_truth

    validate_truth(truth, engine.index.ntotal)
    if len(queries) != len(truth) or not 0 < target <= 1 or point["method"] != "MHCR":
        raise ValueError("Invalid calibration protocol")
    rows = []
    for penalty in sorted(set(penalties)):
        if not np.isfinite(penalty) or penalty < 0:
            raise ValueError("Penalties must be finite and nonnegative")
        found = np.array(
            [engine.search(q, dict(point, penalty=penalty))[0] for q in queries]
        )
        rows.append(
            dict(penalty=penalty, recall100=float(recall100(found, truth).mean()))
        )
    feasible = [r for r in rows if r["recall100"] >= target]
    if not feasible:
        raise ValueError("No tested penalty reaches the requested Recall@100")
    return dict(max(feasible, key=lambda r: r["penalty"]), target=target, rows=rows)


def normal_lcb95(values):
    values = np.asarray(values, dtype=np.float64)
    if len(values) < 2:
        return float(values.mean())
    return float(values.mean() - 1.96 * values.std(ddof=1) / np.sqrt(len(values)))


def gt_rank_1x(cells, gt_cells):
    """Rank (or the sentinel if absent) of each top-100 GT vector's primary cell."""
    q, sentinel = cells.shape
    if gt_cells.shape[1] != 100:
        raise ValueError("Expected 100 reference neighbors")
    ranks = np.full((q, gt_cells.shape[1]), sentinel, dtype=np.int32)
    for j in range(gt_cells.shape[1]):
        found = cells == gt_cells[:, j : j + 1]
        ranks[:, j] = np.where(found.any(axis=1), found.argmax(axis=1), sentinel)
    return ranks


def gt_rank_multi(cells, gt_cells):
    """Minimum probe rank across each GT vector's primary/extra LIRA cells."""
    q, sentinel = cells.shape
    if gt_cells.shape[1] != 100:
        raise ValueError("Expected 100 reference neighbors")
    ranks = np.full((q, gt_cells.shape[1]), sentinel, dtype=np.int32)
    for j in range(gt_cells.shape[1]):
        best = np.full(q, sentinel, dtype=np.int32)
        for member in range(gt_cells.shape[2]):
            cell = gt_cells[:, j, member]
            valid = cell >= 0
            found = cells == np.maximum(cell, 0)[:, None]
            rank = np.where(found.any(axis=1), found.argmax(axis=1), sentinel).astype(
                np.int32
            )
            best = np.minimum(best, np.where(valid, rank, sentinel))
        ranks[:, j] = best
    return ranks


def query_recall(gt_rank, budget):
    return (gt_rank < budget[:, None]).mean(axis=1)


def prefix_scans(cells, cell_sizes):
    return np.cumsum(cell_sizes[cells], axis=1, dtype=np.int64)


def choose_acr_lambda(predcov, scans, gt_rank, target, full_order=False):
    """Highest-cost-saving lambda whose calibration recall LCB reaches target."""
    lambdas = np.logspace(-10.0, -1.0, 220, dtype=np.float64)
    best = None
    for lam in lambdas:
        objective = np.concatenate(
            [np.zeros((len(predcov), 1), dtype=np.float32), predcov - lam * scans],
            axis=1,
        )
        budget = np.maximum(objective.argmax(axis=1), 1).astype(np.int32)
        if np.any(budget >= predcov.shape[1]) and not full_order:
            # The objective may want cells beyond the stored prefix. Do not
            # turn this into an optimistic clipped operating point.
            continue
        recall = query_recall(gt_rank, budget)
        lcb = normal_lcb95(recall)
        if lcb < target:
            continue
        candidate = {
            "lambda": float(lam),
            "calib_recall": float(recall.mean()),
            "calib_recall_lcb95": lcb,
            "calib_scan_postings": float(
                scans[np.arange(len(scans)), budget - 1].mean()
            ),
            "calib_mean_budget": float(budget.mean()),
        }
        if (
            best is None
            or candidate["calib_scan_postings"] < best["calib_scan_postings"]
        ):
            best = candidate
    if best is None:
        raise RuntimeError(f"ACR failed calibration target={target}")
    return best


def select_fixed(cells, physical_sizes, gt_rank, target):
    scans = AF.prefix_scans(cells, physical_sizes)
    best = None
    for budget in range(1, cells.shape[1] + 1):
        recall = AF.query_recall(gt_rank, np.full(len(cells), budget, dtype=np.int32))
        lcb = AF.normal_lcb95(recall)
        if lcb >= target:
            best = {
                "nprobe": budget,
                "calib_recall": float(recall.mean()),
                "calib_recall_lcb95": lcb,
                "calib_scan_postings": float(scans[:, budget - 1].mean()),
            }
            break
    if best is None:
        raise RuntimeError(
            f"fixed router failed target={target} within sentinel={cells.shape[1]}"
        )
    return best
