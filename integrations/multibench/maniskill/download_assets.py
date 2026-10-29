"""Fetch only the assets and normalization metadata required by plate-25.

No SFT model weights are downloaded. Run using the isolated ManiSkill interpreter.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess

from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
ASSETS = ROOT / "datasets/benchmark_assets/maniskill3"


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
    for uid in ["bridge_v2_real2sim", "widowx250s"]:
        subprocess.run([os.sys.executable, "-m", "mani_skill.utils.download_asset", uid, "-y"], check=True)
    records.append({"source": "ManiSkill v3.0.0b22 built-in asset downloader",
                    "asset_ids": ["bridge_v2_real2sim", "widowx250s"],
                    "path": str(ASSETS / "upstream")})
    files = []
    for path in sorted(ASSETS.rglob("*")):
        if path.is_file() and ".cache" not in path.parts:
            files.append({"path": str(path.relative_to(ASSETS)), "bytes": path.stat().st_size,
                          "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    (HERE / "assets_manifest.json").write_text(json.dumps({
        "repositories": records, "files": files,
        "policy_weights_downloaded": False,
        "normalization_note": "Dataset calibration from ManiSkill SFT assets; no SFT weights used.",
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
