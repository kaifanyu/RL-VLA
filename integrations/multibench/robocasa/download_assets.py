"""Download official RoboCasa assets without importing simulator dependencies.

The registry is read from the pinned upstream checkout. Archives are retained for
reproducibility and resumable installation; no demonstrations or policy weights
are downloaded here.
"""
from __future__ import annotations

import concurrent.futures
import hashlib
import json
from pathlib import Path
import time
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
ASSETS = ROOT / "third_party/robocasa365/robocasa/models/assets"
REGISTRY = {
    "textures": ASSETS,
    "generative_textures": ASSETS,
    "fixtures_lightwheel": ASSETS,
    "objaverse": ASSETS / "objects",
    "aigen_objs": ASSETS / "objects",
    "objects_lightwheel": ASSETS / "objects",
}


def one(name: str, parent: Path, links: dict) -> dict:
    cache = HERE / "assets/downloads"
    cache.mkdir(parents=True, exist_ok=True)
    archive = cache / f"{name}.zip"
    marker = cache / f"{name}.json"
    if marker.exists():
        return json.loads(marker.read_text())
    shared = links[name]
    url = shared.split("/s/")[0] + "/shared/static/" + shared.rsplit("/", 1)[1] + ".zip"
    started = time.time()
    if not archive.exists():
        temporary = archive.with_suffix(".partial")
        for attempt in range(3):
            try:
                print(f"Downloading {name}", flush=True)
                with urllib.request.urlopen(url, timeout=120) as src, temporary.open("wb") as dst:
                    while block := src.read(8 * 1024 * 1024):
                        dst.write(block)
                temporary.replace(archive)
                break
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(2)
    digest = hashlib.file_digest(archive.open("rb"), "sha256").hexdigest()
    parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        names = z.namelist()
        for member in names:
            if not (parent / member).resolve().is_relative_to(parent.resolve()):
                raise ValueError(f"Unsafe archive member {member}")
        print(f"Extracting {name}: {len(names)} members", flush=True)
        z.extractall(parent)
    record = dict(name=name, source=url, archive=str(archive), sha256=digest,
                  bytes=archive.stat().st_size, members=len(names),
                  top_level=sorted({n.split('/')[0] for n in names}),
                  destination=str(parent), elapsed_seconds=time.time()-started)
    marker.write_text(json.dumps(record, indent=2) + "\n")
    print(f"Completed {name}", flush=True)
    return record


def main() -> None:
    links = json.loads((ASSETS / "box_links/box_links_assets.json").read_text())
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        records = list(pool.map(lambda item: one(*item, links), REGISTRY.items()))
    (HERE / "assets/manifest.json").write_text(json.dumps(records, indent=2) + "\n")


if __name__ == "__main__":
    main()
