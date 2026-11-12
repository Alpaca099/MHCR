"""Rank-exponential teacher supervision over top-100 neighbors."""

from .storage import P, read, sha
from .training import atomic, model_hash, array_hash
import json, time
import numpy as np, torch


def train_teacher(model, features, cells, out, identity, config, on_checkpoint=None):
    out = P(out)
    out.mkdir(parents=True, exist_ok=True)
    features = np.ascontiguousarray(features, np.float32)
    cells = np.ascontiguousarray(cells, np.int64)
    assert (
        features.ndim == 2 and cells.shape == (len(features), 100) and len(features) > 0
    )
    assert np.isfinite(features).all() and np.all((cells >= 0) & (cells < model.K))
    assert (
        identity.get("training_content_verified") is True
        and identity.get("historical_teacher_reused") is False
    )
    manifest = dict(
        identity=identity,
        config=config,
        initial_model_sha256=model_hash(model),
        features_sha256=array_hash(features),
        cells_sha256=array_hash(cells),
        source_sha256=sha(P(__file__)),
        objective="rank-exponential sparse GT100 cell cross entropy; gamma recorded in config",
    )
    pp = out / "protocol.json"
    if pp.exists():
        assert read(pp) == manifest
    else:
        atomic(pp, lambda f: f.write(json.dumps(manifest, indent=2).encode()))
    protocol = sha(pp)
    opt = torch.optim.AdamW(
        model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"]
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, config["epochs"])
    begin = 0
    snaps = sorted(out.glob("epoch-*.pt"))
    if snaps:
        ck = torch.load(snaps[-1], map_location="cpu", weights_only=False)
        assert ck["protocol"] == protocol
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        sched.load_state_dict(ck["scheduler"])
        torch.set_rng_state(ck["rng"])
        begin = ck["epoch"]
    weights = torch.exp(-torch.arange(100, dtype=torch.float32) / config["rank_gamma"])
    weights /= weights.sum()
    for epoch in range(begin, config["epochs"]):
        started = time.time()
        model.train()
        order = np.random.default_rng(config["seed"] + epoch).permutation(len(features))
        loss_sum = 0.0
        for lo in range(0, len(order), config["batch"]):
            chosen = order[lo : lo + config["batch"]]
            x = torch.from_numpy(features[chosen])
            labels = torch.from_numpy(cells[chosen])
            loss = (
                -(torch.log_softmax(model(x), dim=1).gather(1, labels) * weights)
                .sum(1)
                .mean()
            )
            assert torch.isfinite(loss)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            loss_sum += float(loss.detach()) * len(chosen)
        sched.step()
        path = out / f"epoch-{epoch+1:03d}.pt"
        ck = dict(
            protocol=protocol,
            epoch=epoch + 1,
            model=model.state_dict(),
            optimizer=opt.state_dict(),
            scheduler=sched.state_dict(),
            rng=torch.get_rng_state(),
            mean_loss=loss_sum / len(features),
            seconds=time.time() - started,
        )
        atomic(path, lambda f: torch.save(ck, f))
        if on_checkpoint:
            on_checkpoint(path)
    result = dict(
        complete=True,
        protocol=protocol,
        epochs=config["epochs"],
        training_rows=len(features),
        final=f'epoch-{config["epochs"]:03d}.pt',
        model_sha256=model_hash(model),
    )
    done = out / "complete.json"
    if done.exists():
        assert read(done) == result
    else:
        atomic(done, lambda f: f.write(json.dumps(result, indent=2).encode()))
    return result
