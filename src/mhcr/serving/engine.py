"""Explicit asset loader for worker-private FastScan engines."""

from pathlib import Path
from types import SimpleNamespace
import numpy as np, torch
from ..algorithm import Router
from ..data import load_array
from ..index.build import load_index
from ..router import QueryCellRouter
from ..storage import read
from .fastscan import Engine as RouteEngine


def load_model(path, config=None):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    state = ck.get("model", ck.get("state"))
    if state is None:
        raise ValueError("Checkpoint must contain model or state weights")
    c = config or ck.get("model_config")
    if c is None:
        if "prototypes" not in state:
            raise ValueError("Only learned free-anchor checkpoints are supported")
        k, dim = state["prototypes"].shape
        c = dict(
            d_clip=len(state["query_mean"]),
            d_r=dim,
            K=k,
            arch=ck.get("router_arch", "res_mlp"),
            tau=ck.get("tau", ck.get("binding", {}).get("tau", 0.07)),
            hidden=state["encoder.nonlinear.0.weight"].shape[0],
        )
    model = QueryCellRouter(query_mean=state["query_mean"].numpy(), **c)
    model.load_state_dict(state)
    return model.eval()


def load_gallery(config, directory):
    """Open the declared gallery without loading a second index in QPS parents."""

    def asset(key):
        p = Path(config[key])
        return p if p.is_absolute() else Path(directory) / p

    kind = config.get("base_format", "fp32")
    if kind == "fp16_shards":
        from .fp16 import HalfBase

        return HalfBase(asset("base"), asset("native_library"))
    if kind == "fp32":
        return load_array(asset("base"))
    raise ValueError("Unknown base_format")


class Engine(RouteEngine):
    @classmethod
    def from_manifest(cls, path):
        path = Path(path).resolve()
        c = read(path)

        def asset(key):
            p = Path(c[key])
            return p if p.is_absolute() else path.parent / p

        bundle = load_index(asset("index"))
        if "expected_nlist" in c and bundle.index.nlist != c["expected_nlist"]:
            raise ValueError("Index partition count differs from manifest")
        base_format = c.get("base_format", "fp32")
        base = load_gallery(c, path.parent)
        if base_format == "fp16_shards":
            if base.shape != (bundle.index.ntotal, bundle.index.d):
                base.close()
                raise ValueError("FP16 gallery/index dimensions differ")
        if base_format == "fp32" and (
            base.dtype != np.float32
            or base.shape != (bundle.index.ntotal, bundle.index.d)
            or not base.flags.c_contiguous
        ):
            raise ValueError(
                "Expected canonical normalized FP32 gallery matching index"
            )
        if c.get("top_k") != 100:
            raise ValueError("Manifest must set top_k=100")
        cost = np.load(asset("cost"), allow_pickle=False)
        if (
            cost.shape != (bundle.index.nlist,)
            or not np.isfinite(cost).all()
            or np.any(cost <= 0)
        ):
            raise ValueError("Invalid scan costs")
        model = load_model(asset("model"), c.get("model_config"))
        if model.K != bundle.index.nlist or model.d_clip != bundle.index.d:
            raise ValueError("Router/index dimensions differ")
        return cls(
            SimpleNamespace(
                index=bundle.index,
                base=base,
                sizes=bundle.sizes,
                router=Router(model, cost, quantized=c.get("quantized", True)),
                routing_variant=c.get("routing_variant", "stable"),
                base_format=base_format,
            )
        )
