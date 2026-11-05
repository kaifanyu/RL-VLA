"""Fetch only the assets and normalization metadata required by plate-25.

No SFT model weights are downloaded. Run using the isolated ManiSkill interpreter.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path

from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
ASSETS = ROOT / "datasets/benchmark_assets/maniskill3"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def install_archive(name, url, target, expected=None):
    """Resume official downloads and install only a complete, verified archive."""
    archive = ASSETS / "downloads" / name
    archive.parent.mkdir(parents=True, exist_ok=True)
    if not archive.is_file() or not zipfile.is_zipfile(archive):
        subprocess.run([
            "curl", "-4", "--fail", "--location", "--retry", "4", "--connect-timeout", "30",
            "--continue-at", "-", "--output", str(archive), url,
        ], check=True)
    checksum = sha256(archive)
    if expected and checksum != expected:
        raise ValueError(f"Archive checksum mismatch: {archive}")
    marker = target / ".installation.json"
    if marker.is_file():
        if json.loads(marker.read_text())["sha256"] != checksum:
            raise ValueError(f"Existing asset installation differs from archive: {target}")
        return {"url": url, "archive": str(archive), "sha256": checksum, "target": str(target)}
    if target.exists():
        raise FileExistsError(f"Unverified asset directory already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as source:
        roots = {Path(info.filename).parts[0] for info in source.infolist() if info.filename}
        if len(roots) != 1:
            raise ValueError(f"Expected one root directory in {archive}")
        if source.testzip() is not None:
            raise ValueError(f"Corrupt archive: {archive}")
        with tempfile.TemporaryDirectory(prefix=".maniskill-extract-", dir=target.parent) as staging:
            stage = Path(staging)
            for info in source.infolist():
                path = Path(info.filename)
                if path.is_absolute() or ".." in path.parts:
                    raise ValueError(f"Unsafe archive path: {info.filename}")
                relative = Path(*path.parts[1:])
                output = stage / relative
                if info.is_dir():
                    output.mkdir(parents=True, exist_ok=True)
                else:
                    output.parent.mkdir(parents=True, exist_ok=True)
                    with source.open(info) as inp, output.open("wb") as out:
                        shutil.copyfileobj(inp, out)
            (stage / ".installation.json").write_text(json.dumps({"url": url, "sha256": checksum}) + "\n")
            stage.rename(target)
    return {"url": url, "archive": str(archive), "sha256": checksum, "target": str(target)}


def main():
    ASSETS.mkdir(parents=True, exist_ok=True)
    os.environ["MS_ASSET_DIR"] = str(ASSETS / "upstream")
    repositories = [
        ("RLinf/maniskill_assets", "dataset", "c23fc1880ed7861686d4f995360101eaee4d18a0",
         ["carrot/**", "README.md"], ASSETS / "rlinf"),
        ("RLinf/RLinf-Pi05-ManiSkill-25Main-SFT", "model", "eb0e2c90726f7f0bdf99e8c9b6656917f0eb5030",
         ["assets/**", "physical-intelligence/**", "README.md"], ASSETS / "normalization"),
    ]
    records = []
    for repo, kind, revision, patterns, target in repositories:
        snapshot_download(repo, repo_type=kind, revision=revision,
                          allow_patterns=patterns, local_dir=target, max_workers=4)
        records.append({"repository": repo, "type": kind, "revision": revision,
                        "patterns": patterns, "path": str(target)})
    # URLs/paths/checksum are from the pinned ManiSkill asset registry. Keep the
    # archives persistent and resume rather than deleting/redownloading assets
    # on every setup invocation (the upstream -y downloader does that).
    records.append(install_archive(
        "bridge_v2_real2sim_dataset.zip",
        "https://huggingface.co/datasets/haosulab/ManiSkill_bridge_v2_real2sim/resolve/main/bridge_v2_real2sim_dataset.zip",
        ASSETS / "upstream/data/tasks/bridge_v2_real2sim_dataset",
        "618512a205b4528cafecdad14b1788ed1130879f3064deb406516ed5b9c5ba92"))
    records.append(install_archive(
        "widowx250s_v0.2.0.zip",
        "https://github.com/haosulab/ManiSkill-WidowX250S/archive/refs/tags/v0.2.0.zip",
        ASSETS / "upstream/data/robots/widowx"))
    files = []
    for path in sorted(ASSETS.rglob("*")):
        if path.is_file() and ".cache" not in path.parts and "downloads" not in path.relative_to(ASSETS).parts:
            files.append({"path": str(path.relative_to(ASSETS)), "bytes": path.stat().st_size,
                          "sha256": sha256(path)})
    (HERE / "assets_manifest.json").write_text(json.dumps({
        "repositories": records, "files": files,
        "policy_weights_downloaded": False,
        "normalization_note": "Dataset calibration from ManiSkill SFT assets; no SFT weights used.",
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
