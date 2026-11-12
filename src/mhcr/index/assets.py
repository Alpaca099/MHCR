"""Extract canonical single-assignment IVF membership."""

import numpy as np, faiss


def assignments(index):
    result = np.full(index.ntotal, -1, np.int32)
    sizes = np.zeros(index.nlist, np.int64)
    for cell in range(index.nlist):
        count = index.invlists.list_size(cell)
        sizes[cell] = count
        if not count:
            continue
        ids = faiss.rev_swig_ptr(index.invlists.get_ids(cell), count)
        if (
            np.any(ids < 0)
            or np.any(ids >= index.ntotal)
            or len(np.unique(ids)) != count
            or np.any(result[ids] >= 0)
        ):
            raise ValueError("Expected unique canonical IDs")
        result[ids] = cell
    if np.any(result < 0):
        raise ValueError("Missing canonical IDs")
    return result, sizes
