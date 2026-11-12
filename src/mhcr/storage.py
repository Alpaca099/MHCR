"""Portable file utilities, independent of experiment servers."""

from pathlib import Path as P
import hashlib, json, os, sys, uuid


def read(path):
    return json.loads(P(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with P(path).open("rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def atomic(path, writer):
    path = P(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    tmp = path.with_name(path.name + ".partial-" + uuid.uuid4().hex)
    with tmp.open("xb") as f:
        writer(f)
        f.flush()
        os.fsync(f.fileno())
    os.link(tmp, path)
    tmp.unlink()
    if hasattr(os, "O_DIRECTORY"):
        fd = os.open(path.parent, os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def save(path, value):
    path = P(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".partial-" + uuid.uuid4().hex)
    with tmp.open("x") as f:
        json.dump(value, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def memory():
    try:
        import resource

        n = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return dict(peak_rss_bytes=int(n if sys.platform == "darwin" else n * 1024))
    except ImportError:
        return dict(peak_rss_bytes=None)
