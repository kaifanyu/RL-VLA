"""Download compatible large wheels from a uv lock with verified HTTP ranges."""

import argparse
import json
import re
import tomllib
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import unquote, urlsplit

from download_verified_file import download, use_cached_ipv4
from packaging.requirements import Requirement
from packaging.tags import sys_tags
from packaging.utils import parse_wheel_filename


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--range-workers", type=int, default=16)
    parser.add_argument("--request-size", type=int, default=0)
    parser.add_argument("--requirements", type=Path,
                        help="Limit downloads to packages in a frozen uv export")
    parser.add_argument("--small-only", action="store_true", help="Fetch artifacts smaller than 20 MiB")
    args = parser.parse_args()
    args.destination.mkdir(parents=True, exist_ok=True)
    compatible = set(sys_tags())
    required = None
    if args.requirements is not None:
        required = set()
        for line in args.requirements.read_text().splitlines():
            if not re.match(r"^[a-zA-Z0-9_.-]+==", line):
                continue
            requirement = Requirement(line.rstrip(" \\"))
            if requirement.marker is None or requirement.marker.evaluate():
                required.add((requirement.name, next(iter(requirement.specifier)).version))
    selected = {}
    for package in tomllib.loads(args.lock.read_text())["package"]:
        if required is not None and (package["name"], package["version"]) not in required:
            continue
        for wheel in package.get("wheels", []):
            name = unquote(urlsplit(wheel["url"]).path.rsplit("/", 1)[-1])
            if (wheel.get("size", 0) < 20*1024**2) != args.small_only:
                continue
            if compatible.intersection(parse_wheel_filename(name)[3]):
                selected.setdefault((package["name"], package["version"]),
                                    dict(wheel, filename=name, package=package["name"]))
        if args.small_only and not package.get("wheels") and package.get("sdist"):
            archive = package["sdist"]
            if archive.get("size", 0) < 20*1024**2:
                name = unquote(urlsplit(archive["url"]).path.rsplit("/", 1)[-1])
                selected.setdefault((package["name"], package["version"]),
                                    dict(archive, filename=name, package=package["name"]))
    rows = sorted(selected.values(), key=lambda row: -row["size"])
    manifest = "manifest-small.json" if args.small_only else "manifest.json"
    (args.destination / manifest).write_text(json.dumps(rows, indent=2) + "\n")
    use_cached_ipv4()

    def fetch(row):
        url = row["url"]
        if row["package"].startswith("nvidia-"):
            mirror = "https://download-r2.pytorch.org/whl/cu126/" + row["filename"]
            try:
                request = urllib.request.Request(mirror, headers={"Range": "bytes=0-0"})
                with urllib.request.urlopen(request, timeout=20) as response:
                    if response.headers.get("Content-Range") == f"bytes 0-0/{row['size']}":
                        url = mirror
            except OSError:
                pass
        download(url, args.destination / row["filename"],
                 row["hash"].removeprefix("sha256:"), size=row["size"],
                 workers=args.range_workers, request_bytes=args.request_size)

    print(f"Fetching {len(rows)} compatible large wheels from the original lock", flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        list(executor.map(fetch, rows))


if __name__ == "__main__":
    main()
