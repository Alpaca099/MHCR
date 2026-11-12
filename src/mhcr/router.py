"""Query encoders and learned cell anchors used by MHCR."""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class ResidualMLP(nn.Module):
    def __init__(self, d_in: int, d_hidden: int, d_out: int):
        super().__init__()
        self.linear = nn.Linear(d_in, d_out, bias=False)
        self.nonlinear = nn.Sequential(
            nn.Linear(d_in, d_hidden, bias=True),
            nn.LayerNorm(d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, d_out, bias=False),
        )
        if d_in == d_out:
            nn.init.eye_(self.linear.weight)
        else:
            nn.init.orthogonal_(self.linear.weight)
        nn.init.kaiming_normal_(self.nonlinear[0].weight, nonlinearity="relu")
        nn.init.zeros_(self.nonlinear[0].bias)
        nn.init.normal_(self.nonlinear[3].weight, std=0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x) + self.nonlinear(x)


class QueryCellRouter(nn.Module):
    def __init__(
        self,
        d_clip: int,
        d_r: int,
        K: int,
        query_mean: np.ndarray,
        arch: str = "linear",
        hidden: int = 1024,
        tau: float = 0.07,
        proto_init: np.ndarray | None = None,
    ):
        super().__init__()
        self.d_clip = d_clip
        self.d_r = d_r
        self.K = K
        self.arch = arch
        self.tau = tau

        self.register_buffer(
            "query_mean", torch.from_numpy(query_mean.astype(np.float32))
        )

        if arch == "linear":
            self.encoder = nn.Linear(d_clip, d_r, bias=False)
            if d_clip == d_r:
                nn.init.eye_(self.encoder.weight)
            else:
                nn.init.orthogonal_(self.encoder.weight)
        elif arch == "mlp":
            self.encoder = nn.Sequential(
                nn.Linear(d_clip, hidden, bias=True),
                nn.LayerNorm(hidden),
                nn.GELU(),
                nn.Linear(hidden, d_r, bias=False),
            )
            nn.init.kaiming_normal_(self.encoder[0].weight, nonlinearity="relu")
            nn.init.zeros_(self.encoder[0].bias)
            nn.init.orthogonal_(self.encoder[3].weight)
        elif arch == "res_mlp":
            self.encoder = ResidualMLP(d_clip, hidden, d_r)
        else:
            raise ValueError(f"unknown arch: {arch}")

        if proto_init is not None:
            proto = torch.from_numpy(proto_init.astype(np.float32))
        else:
            proto = torch.randn(K, d_r) * 0.02
        self.prototypes = nn.Parameter(proto)
        # Kept in the state dictionary for published checkpoint compatibility.
        self.register_buffer("proto_init", F.normalize(proto.clone(), dim=-1))

    def encode_query(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.encoder(x - self.query_mean), dim=-1)

    def prototypes_normalized(self) -> torch.Tensor:
        return F.normalize(self.prototypes, dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encode_query(x)
        return (h @ self.prototypes_normalized().T) / self.tau
