import pytest
import numpy as np

torch = pytest.importorskip("torch")

from mhcr.router import ResidualMLP, QueryCellRouter  # noqa: E402


def test_mlp_shapes():
    m = ResidualMLP(d_in=32, d_hidden=64, d_out=16)
    x = torch.randn(5, 32)
    out = m(x)
    assert out.shape == (5, 16)


def test_prototypes_are_unit_norm():
    p = QueryCellRouter(32, 32, 16, np.zeros(32, np.float32))
    n = p.prototypes_normalized()
    assert torch.allclose(n.norm(dim=1), torch.ones(16), atol=1e-5)
