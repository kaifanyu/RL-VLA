"""Download an official JAX pi05 checkpoint and verify GCS CRC32C hashes.

Uses public HTTPS, so no Google Cloud credentials or CLI are needed. Existing
valid files are reused; interrupted files are resumed using HTTP byte ranges.
Requires google-crc32c (composite GCS objects do not have MD5 hashes).
"""

import argparse
import base64
import json
import shutil
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import google_crc32c

BUCKET = "openpi-assets"
ROOT = Path(__file__).resolve().parents[1]


def checksum(path):
    digest = google_crc32c.Checksum()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024**2):
            digest.update(block)
    return base64.b64encode(digest.digest()).decode()


def download_ranges(item, target, url, *, workers, chunk_bytes, request_bytes=0):
    """Resume bounded HTTP ranges, then verify the complete original GCS object."""
    size = int(item["size"])
    parts = target.with_name(target.name + ".parts")
    parts.mkdir(exist_ok=True)
    partial = target.with_name(target.name + ".part")
    ranges = [(start, min(start + chunk_bytes, size)) for start in range(0, size, chunk_bytes)]
    # Preserve complete ranges from the original downloader's contiguous partial.
    if partial.exists():
        with partial.open("rb") as stream:
            for start, end in ranges:
                if end > partial.stat().st_size:
                    break
                path = parts / f"{start:012d}"
                if not path.exists():
                    stream.seek(start)
                    path.write_bytes(stream.read(end-start))

    def fetch(bounds):
        start, end = bounds
        path = parts / f"{start:012d}"
        if path.exists() and path.stat().st_size == end-start:
            return path
        scratch = path.with_suffix(".tmp")
        for attempt in range(8):
            try:
                with scratch.open("wb") as output:
                    for left in range(start, end, request_bytes or chunk_bytes):
                        right = min(left + (request_bytes or chunk_bytes), end)
                        request = urllib.request.Request(url, headers={"Range": f"bytes={left}-{right-1}"})
                        with urllib.request.urlopen(request, timeout=90) as response:
                            expected = f"bytes {left}-{right-1}/{size}"
                            if response.status != 206 or response.headers.get("Content-Range") != expected:
                                raise ValueError("Server did not return the exact requested range")
                            data = response.read(right-left+1)
                            if len(data) != right-left:
                                raise ValueError("Incomplete HTTP range")
                            output.write(data)
                if scratch.stat().st_size != end-start:
                    raise ValueError("Incomplete range")
                scratch.replace(path)
                print(f"Range complete {target.name} {end:,}/{size:,}", flush=True)
                return path
            except Exception as exc:
                print(f"Range retry {target.name} {start}: {type(exc).__name__}: {exc}", flush=True)
                if attempt == 7:
                    raise
                time.sleep(min(2**attempt, 30))

    with ThreadPoolExecutor(max_workers=workers) as pool:
        paths = list(pool.map(fetch, ranges))
    assembled = target.with_name(target.name + ".assembling")
    with assembled.open("wb") as output:
        for path in paths:
            with path.open("rb") as stream:
                shutil.copyfileobj(stream, output, length=8*1024**2)
    if assembled.stat().st_size != size or checksum(assembled) != item["crc32c"]:
        raise ValueError(f"Reassembled object failed GCS CRC32C: {target}")
    assembled.replace(target)
    partial.unlink(missing_ok=True)
    shutil.rmtree(parts)


def download(item, destination, prefix, *, chunk_workers=1, chunk_bytes=16*1024**2, request_bytes=0):
    if not item["name"].startswith(prefix):
        raise ValueError(f"Object outside checkpoint prefix: {item['name']}")
    relative = Path(item["name"].removeprefix(prefix))
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
    if chunk_workers > 1 and size >= chunk_bytes:
        download_ranges(item, target, url, workers=chunk_workers, chunk_bytes=chunk_bytes,
                        request_bytes=request_bytes)
        print(f"Downloaded and verified {relative} ({size:,} bytes)", flush=True)
        return
    for attempt in range(5):
        try:
            # pi05_base includes an empty commit_success.txt marker.
            if size == 0:
                partial.write_bytes(b"")
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
    parser.add_argument("--checkpoint", choices=("pi05_libero", "pi05_base"), default="pi05_libero")
    parser.add_argument("--destination", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--chunk-workers", type=int, default=1,
                        help="Parallel ranges per large object; 1 preserves sequential resume")
    parser.add_argument("--chunk-mib", type=int, default=16)
    parser.add_argument("--request-kib", type=int, default=0,
                        help="Split each saved chunk into smaller HTTP requests; 0 uses the saved chunk size")
    parser.add_argument("--ipv4", action="store_true", help="Use IPv4 on hosts without an IPv6 route")
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()
    if min(args.workers, args.chunk_workers, args.chunk_mib) < 1 or args.request_kib < 0:
        parser.error("worker counts and chunk size must be positive")
    if args.ipv4:
        from download_verified_file import use_cached_ipv4
        use_cached_ipv4()
    prefix = f"checkpoints/{args.checkpoint}/"
    if args.destination is None:
        args.destination = ROOT / "checkpoints/openpi" / args.checkpoint
    if args.manifest is None:
        name = "pi05_download.json" if args.checkpoint == "pi05_libero" else "pi05_base_download.json"
        args.manifest = ROOT / "runs/setup" / name
    items, page = [], None
    while True:
        query = {"prefix": prefix}
        if page:
            query["pageToken"] = page
        url = f"https://storage.googleapis.com/storage/v1/b/{BUCKET}/o?" + urllib.parse.urlencode(query)
        for attempt in range(8):
            try:
                with urllib.request.urlopen(url, timeout=60) as response:
                    listing = json.load(response)
                break
            except Exception:
                if attempt == 7:
                    raise
                time.sleep(min(2**attempt, 30))
        items.extend(item for item in listing.get("items", []) if not item["name"].endswith("/"))
        page = listing.get("nextPageToken")
        if not page:
            break
    if not items or any("crc32c" not in item for item in items):
        raise ValueError("Expected a nonempty checkpoint with GCS CRC32C checksums")
    print(f"Verifying {len(items)} objects, {sum(int(i['size']) for i in items):,} bytes", flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(lambda item: download(item, args.destination, prefix,
                      chunk_workers=args.chunk_workers, chunk_bytes=args.chunk_mib*1024**2,
                      request_bytes=args.request_kib*1024), items))
    manifest = {"source": f"gs://{BUCKET}/{prefix}", "verified": True,
                "destination": str(args.destination.resolve()), "objects": items}
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Checkpoint verified; manifest: {args.manifest}", flush=True)


if __name__ == "__main__":
    main()
