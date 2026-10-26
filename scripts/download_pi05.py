"""Download the official JAX pi05_libero checkpoint and verify GCS CRC32C hashes.

Uses public HTTPS, so no Google Cloud credentials or CLI are needed. Existing
valid files are reused; interrupted files are resumed using HTTP byte ranges.
Requires google-crc32c (composite GCS objects do not have MD5 hashes).
"""

import argparse
import base64
import json
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import google_crc32c

BUCKET = "openpi-assets"
PREFIX = "checkpoints/pi05_libero/"
ROOT = Path(__file__).resolve().parents[1]


def checksum(path):
    digest = google_crc32c.Checksum()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024**2):
            digest.update(block)
    return base64.b64encode(digest.digest()).decode()


def download(item, destination):
    relative = Path(item["name"].removeprefix(PREFIX))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Unsafe object name: {item['name']}")
    target = destination / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    size = int(item["size"])
    if target.is_file() and target.stat().st_size == size and checksum(target) == item["crc32c"]:
        print(f"Verified existing {relative}", flush=True)
        return
    partial = target.with_name(target.name + ".part")
    url = (f"https://storage.googleapis.com/{BUCKET}/"
           + urllib.parse.quote(item["name"], safe="/")
           + "?generation=" + item["generation"])
    for attempt in range(5):
        try:
            offset = partial.stat().st_size if partial.exists() else 0
            if offset < size:
                request = urllib.request.Request(url, headers={"Range": f"bytes={offset}-"})
                with urllib.request.urlopen(request, timeout=120) as response:
                    mode = "ab" if response.status == 206 else "wb"
                    with partial.open(mode) as stream:
                        while block := response.read(8 * 1024**2):
                            stream.write(block)
            if partial.stat().st_size != size or checksum(partial) != item["crc32c"]:
                partial.unlink()
                raise ValueError(f"Size or CRC32C mismatch: {relative}")
            partial.replace(target)
            print(f"Downloaded and verified {relative} ({size:,} bytes)", flush=True)
            return
        except Exception:
            if attempt == 4:
                raise
            time.sleep(2 ** attempt)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, default=ROOT / "checkpoints/openpi/pi05_libero")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--manifest", type=Path, default=ROOT / "runs/setup/pi05_download.json")
    args = parser.parse_args()
    items, page = [], None
    while True:
        query = {"prefix": PREFIX}
        if page:
            query["pageToken"] = page
        url = f"https://storage.googleapis.com/storage/v1/b/{BUCKET}/o?" + urllib.parse.urlencode(query)
        with urllib.request.urlopen(url, timeout=60) as response:
            listing = json.load(response)
        items.extend(item for item in listing.get("items", []) if not item["name"].endswith("/"))
        page = listing.get("nextPageToken")
        if not page:
            break
    if not items or any("crc32c" not in item for item in items):
        raise ValueError("Expected a nonempty checkpoint with GCS CRC32C checksums")
    print(f"Verifying {len(items)} objects, {sum(int(i['size']) for i in items):,} bytes", flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(lambda item: download(item, args.destination), items))
    manifest = {"source": f"gs://{BUCKET}/{PREFIX}", "verified": True,
                "destination": str(args.destination.resolve()), "objects": items}
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Checkpoint verified; manifest: {args.manifest}", flush=True)


if __name__ == "__main__":
    main()
