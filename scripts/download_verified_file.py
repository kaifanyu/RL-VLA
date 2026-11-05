"""Resume a public HTTP file in bounded ranges and verify its SHA256.

Useful on links that stall during long wheel/checkpoint transfers. Partial
ranges remain next to the output across restarts. The final path is published
only after its complete size and SHA256 match the expected values.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import socket
import threading
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

_original_getaddrinfo = socket.getaddrinfo
_dns_cache = {}
_dns_lock = threading.Lock()


def rank_addresses(addresses):
    """Avoid a slow/unreachable first DNS answer stalling every connection."""

    def probe(entry):
        family, kind, protocol, _, address = entry
        if kind != socket.SOCK_STREAM:
            return 0.0, entry
        start = time.monotonic()
        try:
            with socket.socket(family, kind, protocol) as connection:
                connection.settimeout(2.0)
                connection.connect(address)
            return time.monotonic() - start, entry
        except OSError:
            return float("inf"), entry

    with ThreadPoolExecutor(max_workers=min(8, len(addresses))) as pool:
        ranked = list(pool.map(probe, addresses))
    return [entry for _, entry in sorted(ranked, key=lambda result: result[0])]


def use_cached_ipv4():
    def getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
        key = (host, port, type, proto, flags)
        with _dns_lock:
            if key not in _dns_cache:
                for attempt in range(5):
                    try:
                        _dns_cache[key] = rank_addresses(
                            _original_getaddrinfo(host, port, socket.AF_INET, type, proto, flags)
                        )
                        break
                    except socket.gaierror:
                        if attempt == 4:
                            raise
                        time.sleep(2**attempt)
            return _dns_cache[key]

    socket.getaddrinfo = getaddrinfo


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def read_bounded(response, limit, *, seconds=45):
    """Enforce a whole-body deadline even when a connection trickles bytes."""
    started = time.monotonic()
    chunks, count = [], 0
    while count < limit:
        if time.monotonic() - started > seconds:
            raise TimeoutError(f"Response body exceeded {seconds}s transfer deadline")
        data = response.read1(min(64 * 1024, limit - count))
        if not data:
            break
        chunks.append(data)
        count += len(data)
    return b"".join(chunks)


def download(
    url, output, expected_sha256, *, size=None, workers=8, chunk_size=1024**2, request_bytes=0
):
    if not re.fullmatch(r"[0-9a-fA-F]{64}", expected_sha256):
        raise ValueError("Expected a complete SHA256 hexadecimal digest")
    expected_sha256 = expected_sha256.lower()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if size is None:
        for attempt in range(8):
            try:
                with urllib.request.urlopen(
                    urllib.request.Request(url, headers={"Range": "bytes=0-0"}), timeout=90
                ) as response:
                    if response.status != 206:
                        raise ValueError("Server did not honor the size-probe range request")
                    size = int(response.headers["Content-Range"].split("/")[-1])
                    response.read()
                break
            except Exception:
                if attempt == 7:
                    raise
                time.sleep(min(2**attempt, 20))
    if size <= 0 or workers <= 0 or chunk_size <= 0 or request_bytes < 0:
        raise ValueError(
            "File size, workers, and chunk size must be positive; request_bytes must be nonnegative"
        )
    if output.exists():
        if output.stat().st_size == size and sha256(output) == expected_sha256:
            print(f"Verified existing {output.name}", flush=True)
            return output
        raise FileExistsError(f"Existing final output does not match expected size/hash: {output}")
    parts = output.with_name(output.name + ".parts")
    parts.mkdir(exist_ok=True)

    def fetch(index):
        start = index * chunk_size
        end = min(size, start + chunk_size) - 1
        path = parts / f"{index:08d}"
        if path.exists() and path.stat().st_size == end - start + 1:
            return
        for attempt in range(8):
            try:
                pieces = []
                transfer_size = request_bytes or chunk_size
                for request_start in range(start, end + 1, transfer_size):
                    request_end = min(end, request_start + transfer_size - 1)
                    range_url = url
                    parsed = urllib.parse.urlsplit(url)
                    if parsed.hostname == "huggingface.co" and "/resolve/" in parsed.path:
                        query = urllib.parse.parse_qsl(parsed.query, keep_blank_values=True)
                        query += [
                            ("range_start", str(request_start)),
                            ("range_attempt", str(attempt)),
                        ]
                        range_url = urllib.parse.urlunsplit(
                            parsed._replace(query=urllib.parse.urlencode(query))
                        )
                    request = urllib.request.Request(
                        range_url, headers={"Range": f"bytes={request_start}-{request_end}"}
                    )
                    with urllib.request.urlopen(request, timeout=20) as response:
                        expected_range = f"bytes {request_start}-{request_end}/{size}"
                        if (
                            response.status != 206
                            or response.headers.get("Content-Range") != expected_range
                        ):
                            raise ValueError(f"Server did not honor range {expected_range}")
                        piece = read_bounded(response, request_end - request_start + 2)
                    if len(piece) != request_end - request_start + 1:
                        raise ValueError(f"Incorrect byte count for subrange at {request_start}")
                    pieces.append(piece)
                data = b"".join(pieces)
                if len(data) != end - start + 1:
                    raise ValueError(f"Incorrect byte count for part {index}")
                temp = path.with_suffix(".tmp")
                temp.write_bytes(data)
                temp.replace(path)
                return
            except Exception as exc:
                print(
                    f"{output.name}: range {index} retry {attempt + 1}/8: {type(exc).__name__}: {exc}",
                    flush=True,
                )
                if attempt == 7:
                    raise
                time.sleep(min(2**attempt, 20))

    count = (size + chunk_size - 1) // chunk_size
    print(f"Downloading {output.name}: {size} bytes, {count} ranges, {workers} workers", flush=True)
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(fetch, index) for index in range(count)]
        for completed, future in enumerate(as_completed(futures), 1):
            future.result()
            if completed % 64 == 0 or completed == count:
                print(
                    f"{output.name}: {completed}/{count} ranges in {time.monotonic() - started:.1f}s",
                    flush=True,
                )
    temporary = output.with_name(output.name + ".assembling")
    with temporary.open("wb") as destination:
        for index in range(count):
            with (parts / f"{index:08d}").open("rb") as source:
                for block in iter(lambda: source.read(1024**2), b""):
                    destination.write(block)
    if temporary.stat().st_size != size or sha256(temporary) != expected_sha256:
        raise ValueError(f"Downloaded file failed final size/SHA256 validation: {temporary}")
    temporary.replace(output)
    print(f"Verified {output.name}: SHA256 {expected_sha256}", flush=True)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--size", type=int)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--chunk-size", type=int, default=1024**2)
    parser.add_argument(
        "--request-size",
        type=int,
        default=0,
        help="Subdivide each persistent chunk into HTTP ranges of this size; 0 uses chunk size",
    )
    args = parser.parse_args()
    use_cached_ipv4()
    download(
        args.url,
        args.output,
        args.sha256,
        size=args.size,
        workers=args.workers,
        chunk_size=args.chunk_size,
        request_bytes=args.request_size,
    )


if __name__ == "__main__":
    main()
