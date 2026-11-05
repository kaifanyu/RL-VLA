"""Prefetch pinned simulator wheels with verified ranged transfers on slow links."""

import argparse
import json
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from packaging.tags import sys_tags
from packaging.utils import parse_wheel_filename

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
from download_verified_file import download, use_cached_ipv4

PACKAGES = {
    "numpy": "2.2.5",
    "numba": "0.61.2",
    "scipy": "1.15.3",
    "mujoco": "3.3.1",
    "llvmlite": "0.44.0",
    "pygame": "2.6.1",
    "opencv-python": "5.0.0.93",
    "pillow": "13.0.0",
    "h5py": "3.16.0",
    "pyopengl": "3.1.10",
    "imageio-ffmpeg": "0.6.0",
}


def main():
    if sys.version_info[:2] != (3, 11):
        raise RuntimeError("Select wheels with a Python 3.11 interpreter")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requirements", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / ".cache/wheelhouse-robocasa")
    parser.add_argument("--file-workers", type=int, default=3)
    parser.add_argument("--range-workers", type=int, default=8)
    parser.add_argument("--request-size", type=int, default=0)
    args = parser.parse_args()
    packages = PACKAGES
    if args.requirements:
        packages = {}
        for line in args.requirements.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            name, separator, version = line.partition("==")
            if not separator or not name or not version or " " in line:
                raise ValueError(f"Expected one exact name==version pin: {line}")
            packages[name] = version
    use_cached_ipv4()
    tags = list(sys_tags())
    ranks = {tag: index for index, tag in enumerate(tags)}
    folder = args.output
    folder.mkdir(parents=True, exist_ok=True)

    def one(item):
        name, version = item
        cache = folder / "metadata" / f"{name}-{version}.json"
        if cache.is_file():
            metadata = json.loads(cache.read_text())
        else:
            for attempt in range(7):
                try:
                    with urllib.request.urlopen(
                        f"https://pypi.org/pypi/{name}/{version}/json", timeout=90
                    ) as r:
                        metadata = json.load(r)
                    break
                except Exception:
                    if attempt == 6:
                        raise
                    time.sleep(min(20, 2**attempt))
            cache.parent.mkdir(parents=True, exist_ok=True)
            temp = cache.with_suffix(".json.tmp")
            temp.write_text(json.dumps(metadata))
            temp.replace(cache)
        choices = []
        for wheel in metadata["urls"]:
            if wheel["packagetype"] != "bdist_wheel":
                continue
            supported = parse_wheel_filename(wheel["filename"])[3] & ranks.keys()
            if supported:
                choices.append((min(ranks[t] for t in supported), wheel))
        if choices:
            wheel = min(choices, key=lambda item: item[0])[1]
        else:
            sources = [item for item in metadata["urls"] if item["packagetype"] == "sdist"]
            if len(sources) != 1:
                raise RuntimeError(f"No compatible wheel or unique sdist for {name}=={version}")
            wheel = sources[0]
            print(f"Source build required for {name}=={version}", flush=True)
        download(
            wheel["url"],
            folder / wheel["filename"],
            wheel["digests"]["sha256"],
            size=wheel["size"],
            workers=args.range_workers,
            request_bytes=args.request_size,
        )
        return {"name": name, "version": version, **wheel}

    with ThreadPoolExecutor(max_workers=args.file_workers) as pool:
        records = list(pool.map(one, packages.items()))
    (folder / "manifest.json").write_text(json.dumps(records, indent=2) + "\n")


if __name__ == "__main__":
    main()
