"""Epoch-checkpointed student training.

Receives caller-verified filtered training vectors and cell labels only. It never
opens historical teacher weights or accesses calibration/test data.
"""

from pathlib import Path
import hashlib, json, os, time, uuid
import numpy as np
import torch
from .algorithm import training_loss


def atomic(path, writer):
    tmp = path.with_name(path.name + ".partial-" + uuid.uuid4().hex)
    with tmp.open("xb") as f:
        writer(f)
        f.flush()
        os.fsync(f.fileno())
    assert not path.exists()
    os.rename(tmp, path)
    if hasattr(os, "O_DIRECTORY"):
        fd = os.open(path.parent, os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def model_hash(model):
    h = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        a = value.detach().cpu().numpy()
        h.update(name.encode())
        h.update(str(a.shape).encode())
        h.update(str(a.dtype).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def array_hash(array):
    assert array.flags.c_contiguous
    h = hashlib.sha256()
    h.update(str(array.shape).encode())
    h.update(str(array.dtype).encode())
    h.update(memoryview(array).cast("B"))
    return h.hexdigest()


def train_epochs(
    model,
    features,
    cells,
    cost,
    sizes,
    out,
    identity,
    config,
    family="ce",
    teacher=None,
    on_checkpoint=None,
):
    if config.get("budget_unit") not in ("ms", "fraction"):
        raise ValueError("Explicit budget_unit must be ms or fraction")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    features = np.ascontiguousarray(features, np.float32)
    cells = np.ascontiguousarray(cells, np.int64)
    cost = np.ascontiguousarray(cost, np.float32)
    sizes = np.ascontiguousarray(sizes, np.float32)
    assert (
        features.ndim == 2 and cells.shape == (len(features), 100) and len(features) > 0
    )
    assert (
        np.isfinite(features).all()
        and np.isfinite(cost).all()
        and np.isfinite(sizes).all()
    )
    assert np.all(cost > 0) and np.all(sizes >= 0) and sizes.sum() > 0
    assert family in ("ce", "budget_mean", "budget_tail")
    assert (
        identity.get("training_content_verified") is True
        and identity.get("historical_teacher_reused") is False
    )
    if teacher is not None:
        assert identity.get(
            "teacher_training_protocol"
        ), "Teacher needs a verified clean training protocol"
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad_(False)
    manifest = dict(
        identity=identity,
        config=config,
        family=family,
        initial_model_sha256=model_hash(model),
        teacher_sha256=model_hash(teacher) if teacher is not None else None,
        features_sha256=array_hash(features),
        cells_sha256=array_hash(cells),
        cost_sha256=array_hash(cost),
        sizes_sha256=array_hash(sizes),
        source_sha256={
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [Path(__file__), Path(__file__).with_name("algorithm.py")]
        },
        objective="Uniform GT100 cell mass; CE or10*budget primary +0.1*(CE+0.25*clean-teacher CE); worst20pct missing mass, not P99 loss",
    )
    bound = out / "protocol.json"
    if bound.exists():
        assert json.loads(bound.read_text()) == manifest, "Training identity changed"
    else:
        atomic(
            bound,
            lambda f: f.write(json.dumps(manifest, sort_keys=True, indent=2).encode()),
        )
    protocol = hashlib.sha256(bound.read_bytes()).hexdigest()
    opt = torch.optim.AdamW(
        model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, config["epochs"])
    begin = 0
    checkpoints = sorted(out.glob("epoch-*.pt"))
    if checkpoints:
        ck = torch.load(checkpoints[-1], map_location="cpu", weights_only=False)
        assert ck["protocol"] == protocol
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        scheduler.load_state_dict(ck["scheduler"])
        torch.set_rng_state(ck["rng"])
        begin = ck["epoch"]
    ct = torch.from_numpy(cost)
    st = (
        ct
        if config.get("budget_unit") == "ms"
        else torch.from_numpy(sizes / sizes.sum())
    )
    for epoch in range(begin, config["epochs"]):
        started = time.time()
        model.train()
        order = np.random.default_rng(config["seed"] + epoch).permutation(len(features))
        generator = torch.Generator().manual_seed(config["seed"] + 10000 + epoch)
        loss_sum = 0.0
        for start in range(0, len(order), config["batch"]):
            chosen = order[start : start + config["batch"]]
            x = torch.from_numpy(features[chosen])
            target = torch.from_numpy(cells[chosen])
            budget = torch.empty((len(chosen), 1)).uniform_(
                *config["budget_range"], generator=generator
            )
            if teacher is not None:
                with torch.no_grad():
                    soft = torch.softmax(teacher(x), 1)
            else:
                soft = None
            loss, _ = training_loss(
                model(x),
                target,
                ct,
                st,
                budget,
                family,
                soft,
                config.get("gate_variant", "implicit32"),
            )
            assert torch.isfinite(loss)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            loss_sum += float(loss.detach()) * len(chosen)
        scheduler.step()
        path = out / f"epoch-{epoch+1:03d}.pt"
        ck = dict(
            protocol=protocol,
            epoch=epoch + 1,
            model=model.state_dict(),
            optimizer=opt.state_dict(),
            scheduler=scheduler.state_dict(),
            rng=torch.get_rng_state(),
            mean_loss=loss_sum / len(features),
            seconds=time.time() - started,
        )
        atomic(path, lambda f: torch.save(ck, f))
        if on_checkpoint is not None:
            on_checkpoint(path)
    result = dict(
        complete=True,
        protocol=protocol,
        epochs=config["epochs"],
        family=family,
        training_rows=len(features),
        final=f'epoch-{config["epochs"]:03d}.pt',
        model_sha256=model_hash(model),
    )
    done = out / "complete.json"
    if done.exists():
        assert json.loads(done.read_text()) == result
    else:
        atomic(done, lambda f: f.write(json.dumps(result, indent=2).encode()))
    return result
