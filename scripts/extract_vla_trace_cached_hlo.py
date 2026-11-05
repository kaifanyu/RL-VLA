"""Read-only snapshot and CPU decoding of persistent JAX executable HLO."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import zlib


def varint(data, offset):
    value, shift = 0, 0
    while True:
        byte = data[offset]
        offset += 1
        value |= (byte & 127) << shift
        if byte < 128:
            return value, offset
        shift += 7
        if shift > 63:
            raise ValueError("Invalid protobuf varint")


def module_proto(data):
    """Unwrap the measured JAX0.5.3 GPU-executable first-field envelope."""
    for depth in range(8):
        tag, offset = varint(data, 0)
        size, offset = varint(data, offset)
        if tag != 10 or offset + size > len(data):
            raise ValueError("Unexpected native executable envelope")
        first = data[offset:offset + size]
        if first.startswith(b"jit_"):
            return data, depth
        data = first
    raise ValueError("HLO module not found within bounded native envelope")


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main(cache, output):
    from jaxlib import xla_client

    output.mkdir(parents=True, exist_ok=False)
    rows = []
    sources = sorted([*cache.glob("jit__lambda_*-cache"), *cache.glob("jit__encode_prefix*-cache")])
    for source in sources:
        before = source.stat()
        data = source.read_bytes()
        after = source.stat()
        if (before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns):
            raise ValueError("Live source changed during individual snapshot")
        (output / source.name).write_bytes(data)
        uncompressed = zlib.decompress(data)
        proto, depth = module_proto(uncompressed[4:])
        hlo = xla_client._xla.HloModule.from_serialized_hlo_module_proto(proto).to_string()
        destination = output / (source.name + ".optimized_hlo.txt")
        destination.write_text(hlo)
        configs, occurrences = [], {}
        for number, line in enumerate(hlo.splitlines(), 1):
            if "triton_gemm_config" not in line:
                continue
            match = re.search(r'op_name="([^"]+)"', line)
            name = match.group(1) if match else "missing_op_metadata"
            name = name[name.index("while/body/"):] if "while/body/" in name else name
            occurrences[name] = occurrences.get(name, 0) + 1
            config = json.loads(line.split("backend_config=", 1)[1])["fusion_backend_config"]["triton_gemm_config"]
            configs.append({"op": name, "occurrence": occurrences[name], "config": config, "line": number})
        rows.append({"source": str(source.resolve()), "source_sha256": sha(data), "source_mtime_ns": before.st_mtime_ns,
                     "source_mtime_utc": datetime.fromtimestamp(before.st_mtime, timezone.utc).isoformat(),
                     "bytes": len(data), "compilation_seconds": int.from_bytes(uncompressed[:4], "big"),
                     "envelope_depth": depth, "hlo": str(destination.resolve()), "hlo_sha256": sha(hlo.encode()),
                     "entry_signature": hlo.splitlines()[0], "gemm_configs": configs})
    summary = {"complete": True, "scope": "Sequential read-only snapshots; each file stable during its read. Not an atomic global or historical cache snapshot. CPU HLO decoding only: no executable deserialization into a GPU backend, no model, no shared-cache writes.",
               "created_utc": datetime.now(timezone.utc).isoformat(), "script_sha256": sha(Path(__file__).read_bytes()), "entries": rows}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"decoded": len(rows), "output": str(output)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    main(arguments.cache, arguments.output)
