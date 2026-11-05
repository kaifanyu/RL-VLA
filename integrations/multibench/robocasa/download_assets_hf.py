"""Restore RoboCasa assets from pinned official HF publishers when Box is unavailable.

The source layout follows RLinf's official RoboCasa365 installation guide. This
does not download demonstration data or policy weights. Run with an interpreter
containing huggingface_hub (the supplied /opt/pi05 environment is sufficient).
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import os
import sys
import time
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
ASSETS = ROOT / "third_party/robocasa365/robocasa/models/assets"
CACHE = ROOT / "datasets/benchmark_assets/robocasa365/downloads"
SOURCES = [
    (
        "robocasa/robocasa-assets",
        "1b92c3d02ca4354984fec961357db0bff7b32166",
        ["textures.zip", "objaverse.zip"],
    ),
    (
        "nvidia/PhysicalAI-Robotics-Manipulation-Objects-Kitchen-MJCF",
        "420a04af939c34873e6839a586b70844baf28aab",
        ["fixtures_lightwheel/*.zip", "objects_lightwheel/*.zip"],
    ),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=["all", "robocasa", "nvidia"], default="all")
    parser.add_argument("--file", action="append", help="Optional exact filenames or glob patterns")
    parser.add_argument(
        "--include-optional",
        action="store_true",
        help="Also fetch AI-generated objects/textures unused by default kitchen tasks",
    )
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--transport", choices=["hub", "ranges"], default="hub")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--file-workers", type=int, default=3)
    args = parser.parse_args()
    os.environ.setdefault("HF_HOME", str(ROOT / ".cache/huggingface"))
    if args.transport == "hub":
        from huggingface_hub import snapshot_download
    else:
        sys.path.insert(0, str(ROOT / "scripts"))
        from download_verified_file import download, use_cached_ipv4

        use_cached_ipv4()

    records = []
    for repo, revision, patterns in SOURCES:
        if args.source != "all" and repo.split("/")[0] != args.source:
            continue
        patterns = list(args.file or patterns)
        if args.include_optional and repo.startswith("robocasa/") and not args.file:
            patterns += ["generative_textures.zip", "aigen_objs.zip"]
        folder = CACHE / repo.replace("/", "--")
        if args.transport == "hub":
            snapshot_download(
                repo,
                repo_type="dataset",
                revision=revision,
                allow_patterns=patterns,
                local_dir=folder,
                max_workers=3,
            )
        else:
            url = f"https://huggingface.co/api/datasets/{repo}/revision/{revision}?blobs=true"
            folder.mkdir(parents=True, exist_ok=True)
            metadata_path = folder / (revision + ".publisher-metadata.json")
            if metadata_path.exists():
                metadata = json.loads(metadata_path.read_text())
            else:
                for attempt in range(7):
                    try:
                        with urllib.request.urlopen(url, timeout=30) as response:
                            metadata = json.load(response)
                        break
                    except Exception:
                        if attempt == 6:
                            raise
                        time.sleep(min(20, 2**attempt))
                metadata_path.write_text(json.dumps(metadata) + "\n")
            if metadata["sha"] != revision:
                raise ValueError("Asset metadata does not match the pinned publisher revision")
            selected = [
                item
                for item in metadata["siblings"]
                if any(fnmatch.fnmatch(item["rfilename"], pattern) for pattern in patterns)
            ]

            def fetch(item, repo=repo, revision=revision, folder=folder):
                name = item["rfilename"]
                digest = item["lfs"]["sha256"]
                return download(
                    f"https://huggingface.co/datasets/{repo}/resolve/{revision}/{name}",
                    folder / name,
                    digest,
                    size=item["size"],
                    workers=args.workers,
                )

            with ThreadPoolExecutor(max_workers=args.file_workers) as pool:
                list(pool.map(fetch, selected))
        for archive in sorted(folder.rglob("*.zip")):
            relative = archive.relative_to(folder)
            if not any(fnmatch.fnmatch(str(relative), pattern) for pattern in patterns):
                continue
            start = time.monotonic()
            with archive.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            if relative.parts[0] == "fixtures_lightwheel":
                destination = ASSETS / "fixtures"
            elif relative.parts[0] == "objects_lightwheel":
                destination = ASSETS / "objects/lightwheel"
            elif archive.name in {"objaverse.zip", "aigen_objs.zip"}:
                destination = ASSETS / "objects"
            else:
                destination = ASSETS
            marker = archive.with_suffix(".extracted.json")
            if marker.exists():
                record = json.loads(marker.read_text())
                if record["sha256"] != digest or record["destination"] != str(destination):
                    raise ValueError(f"Archive or extraction destination changed: {archive}")
                records.append(record)
                continue
            destination.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(archive) as z:
                for member in z.infolist():
                    if (
                        not (destination / member.filename)
                        .resolve()
                        .is_relative_to(destination.resolve())
                    ):
                        raise ValueError(f"Unsafe archive member: {member.filename}")
                    if (member.external_attr >> 16) & 0o170000 == 0o120000:
                        raise ValueError(f"Unexpected archive symlink: {member.filename}")
                print(f"Extracting {relative} into {destination}", flush=True)
                z.extractall(destination)
                count = len(z.infolist())
                top_level = sorted({n.split("/")[0] for n in z.namelist()})
            record = {
                "repo_id": repo,
                "revision": revision,
                "file": str(relative),
                "source": f"https://huggingface.co/datasets/{repo}/resolve/{revision}/{relative}",
                "archive": str(archive),
                "sha256": digest,
                "bytes": archive.stat().st_size,
                "destination": str(destination),
                "members": count,
                "top_level": top_level,
                "elapsed_seconds": time.monotonic() - start,
            }
            marker.write_text(json.dumps(record, indent=2) + "\n")
            records.append(record)
    output = args.manifest or ROOT / "runs/setup/robocasa365/assets_hf_manifest.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "sources_documented_at": "https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/robocasa365.html",
                "reason": "Pinned upstream Box links returned HTTP 404 on this host",
                "optional_ai_assets_requested": args.include_optional,
                "default_asset_scope": "All default native task textures, fixtures, Objaverse and Lightwheel objects",
                "optional_features": "AI-generated objects/textures require --include-optional; disabled by default",
                "archives": records,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Asset restore complete: {output}", flush=True)


if __name__ == "__main__":
    main()
