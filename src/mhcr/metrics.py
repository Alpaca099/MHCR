from pathlib import Path
import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

TOP_K = 100


def dedup_top100(I):
    """(Q,k) faiss ids (score-sorted, -1 pad) → (Q,100) unique top-100 (collapses SOAR duplicates)."""
    Q = I.shape[0]
    out = np.full((Q, TOP_K), -1, dtype=np.int64)
    for q in range(Q):
        seen = []
        for x in I[q]:
            if x < 0:
                continue
            if x not in seen:
                seen.append(int(x))
                if len(seen) == TOP_K:
                    break
        out[q, : len(seen)] = seen
    return out


def build_curves_physical(order, multi, gt_ids, K):
    """Recall by unique GT IDs; cost by every posting occurrence in probed lists."""
    Q = order.shape[0]
    if gt_ids.shape != (Q, TOP_K):
        raise ValueError("Expected exactly 100 ground-truth IDs per query")
    valid_multi = multi >= 0
    physical_sizes = np.bincount(multi[valid_multi], minlength=K).astype(np.int64)
    recall = np.empty((Q, K), np.float32)
    pscan = np.empty((Q, K), np.float64)
    ar = np.arange(K, dtype=np.int64)
    for q in range(Q):
        oq = order[q]
        cell_rank = np.empty(K, dtype=np.int64)
        cell_rank[oq] = ar
        gt_cells = multi[gt_ids[q]]
        gt_valid = gt_cells >= 0
        gt_rank = np.where(gt_valid, cell_rank[np.maximum(gt_cells, 0)], K).min(axis=1)
        recall[q] = (
            np.cumsum(np.bincount(gt_rank, minlength=K + 1)[:K], dtype=np.float64)
            / TOP_K
        )
        pscan[q] = np.cumsum(physical_sizes[oq], dtype=np.float64)
    return recall, pscan
