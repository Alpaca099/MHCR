"""PCA projection of a NEW clean teacher, matching historical MHCR256 design."""

from .storage import read, save, sha, atomic
import numpy as np, torch
from .router import QueryCellRouter
from .training import model_hash


def project_teacher(teacher, dimension, hidden, path, identity):
    assert (
        identity.get("training_content_verified") is True
        and identity.get("historical_teacher_reused") is False
    )
    assert (
        identity.get("teacher_training_protocol")
        and teacher.d_r > dimension > 0
        and teacher.arch == "res_mlp"
    )
    digest = model_hash(teacher)
    meta_path = path.with_suffix(".json")
    if path.exists():
        meta = read(meta_path)
        assert (
            meta["teacher_sha256"] == digest
            and meta["identity"] == identity
            and meta["dimension"] == dimension
            and sha(path) == meta["sha256"]
        )
        projection = np.load(path)
    else:
        with torch.no_grad():
            proto = teacher.prototypes_normalized().detach()
            _, vectors = torch.linalg.eigh(proto.T @ proto)
            projection = vectors.flip(1)[:, :dimension].contiguous().numpy()
            # Canonicalize eigenvector sign, then persist; tied subspaces need not be rerun on resume.
            signs = np.sign(
                projection[np.argmax(abs(projection), axis=0), np.arange(dimension)]
            )
            projection *= signs
        atomic(path, lambda f: np.save(f, projection))
        save(
            meta_path,
            dict(
                identity=identity,
                teacher_sha256=digest,
                dimension=dimension,
                sha256=sha(path),
                method="PCA of normalized learned prototypes; fixed persisted basis",
            ),
        )
    assert projection.shape == (teacher.d_r, dimension) and np.allclose(
        projection.T @ projection, np.eye(dimension), atol=3e-5
    )
    v = torch.from_numpy(np.ascontiguousarray(projection))
    proto = teacher.prototypes_normalized().detach()
    model = QueryCellRouter(
        teacher.d_clip,
        dimension,
        teacher.K,
        teacher.query_mean.numpy(),
        arch="res_mlp",
        hidden=hidden,
        tau=teacher.tau,
        proto_init=(proto @ v).numpy(),
    )
    with torch.no_grad():
        src = teacher.encoder
        dst = model.encoder
        dst.linear.weight.copy_(v.T @ src.linear.weight)
        if hidden > src.nonlinear[0].out_features:
            raise ValueError("Student hidden width cannot exceed teacher width")
        importance = src.nonlinear[3].weight.norm(dim=0) * src.nonlinear[0].weight.norm(
            dim=1
        )
        keep = (
            torch.topk(importance, hidden).indices.sort().values
            if hidden < src.nonlinear[0].out_features
            else torch.arange(hidden)
        )
        dst.nonlinear[0].weight.copy_(src.nonlinear[0].weight[keep])
        dst.nonlinear[0].bias.copy_(src.nonlinear[0].bias[keep])
        dst.nonlinear[1].weight.copy_(src.nonlinear[1].weight[keep])
        dst.nonlinear[1].bias.copy_(src.nonlinear[1].bias[keep])
        dst.nonlinear[3].weight.copy_(v.T @ src.nonlinear[3].weight[:, keep])
    return model.eval()
