"""Bitwise parity against the frozen reference engine.

Skipped unless MHCR_PARITY_DIR points at a directory holding
queries.npy, ref_ids.npy and ref_scores.npy from an independently archived
reference run, plus engine.json and point.json. Reference outputs must not be
generated from the implementation under test.
"""

import os
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")

D = os.environ.get("MHCR_PARITY_DIR")


@pytest.mark.skipif(not D, reason="set MHCR_PARITY_DIR to run the parity check")
def test_ids_and_scores_are_equal():
    d = Path(D)
    q = np.load(d / "queries.npy")
    ref_ids = np.load(d / "ref_ids.npy")
    ref_scores = np.load(d / "ref_scores.npy")
    import json
    from mhcr.serving.engine import Engine

    engine = Engine.from_manifest(d / "engine.json")
    point = json.loads((d / "point.json").read_text())
    results = [engine.search(query, point) for query in q]
    ids, scores = np.array([r[0] for r in results]), np.array([r[1] for r in results])
    assert np.array_equal(ids, ref_ids)
    assert np.array_equal(scores, ref_scores)
