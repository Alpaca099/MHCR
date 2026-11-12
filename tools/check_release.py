"""Validate public sources and optionally export a clean source archive."""

import argparse
from pathlib import Path
import re
import zipfile

ROOT = Path(__file__).resolve().parent.parent
FILES = {
    "README.md",
    "LICENSE",
    "NOTICE",
    "CITATION.cff",
    "pyproject.toml",
    "MANIFEST.in",
    ".gitignore",
    "docs/images/system-overview.png",
    "src/mhcr/serving/fp16.cpp",
}
TREES = {"src/mhcr", "repro", "docs", "data", "tests", "tools"}
TEXT = {".py", ".md", ".yaml", ".yml", ".toml", ".cff", ".in"}
IGNORED = {"__pycache__", ".pytest_cache", ".git", ".venv", "build", "dist"}
CHECKS = {
    "machine-specific path": re.compile(
        r"/mnt/|/data/ctcc|[A-Z]:[\\/](?:Users|data)[\\/]"
    ),
    "obsolete evaluation/configuration": re.compile(
        r"\btop[-_ ]?10\b|recall@10\b|datacomp10m", re.I
    ),
    "unfinished placeholder": re.compile(
        r"\bTODO\b|\bFIXME\b|to be pasted|after LIRA", re.I
    ),
    "credential assignment": re.compile(
        r"""(?:password|api_key|access_token)\s*=\s*["'][^"']+["']""", re.I
    ),
    "private key": re.compile(r"-----BEGIN .*PRIVATE KEY-----"),
}


def source_files():
    paths = []
    for p in sorted(ROOT.rglob("*")):
        rel = p.relative_to(ROOT)
        if any(part in IGNORED or part.endswith(".egg-info") for part in rel.parts):
            continue
        if p.is_symlink():
            raise ValueError(f"Symlink is not a distributable source: {rel}")
        if not p.is_file():
            continue
        if p.suffix in {".pyc", ".pyo"}:
            continue
        permitted = rel.as_posix() in FILES or (
            p.suffix in TEXT and any(rel.as_posix().startswith(t + "/") for t in TREES)
        )
        if not permitted:
            raise ValueError(f"Unexpected release file: {rel}")
        paths.append(p)
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archive", type=Path, help="New ZIP outside the source directory"
    )
    args = parser.parse_args()
    paths = source_files()
    failures = []
    for p in paths:
        if p.suffix == ".png":
            continue
        if p == Path(__file__).resolve():
            continue
        for line, text in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            for name, pattern in CHECKS.items():
                if pattern.search(text):
                    failures.append(f"{p.relative_to(ROOT)}:{line}: {name}")
    if failures:
        print("\n".join(failures))
        return 1
    if args.archive:
        target = args.archive.resolve()
        if target == ROOT or ROOT in target.parents or target.suffix != ".zip":
            parser.error("Archive must be a .zip outside the source directory")
        target.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(target, "x", compression=zipfile.ZIP_DEFLATED) as out:
            for p in paths:
                out.write(p, "mhcr/" + p.relative_to(ROOT).as_posix())
        print(f"Exported {len(paths)} source files to {target}")
    print(f"OK: {len(paths)} public source files; generated caches/metadata excluded")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
