"""Durable read-only contact overlay; preserves the original native sidecar."""

import argparse
import fcntl
import json
import time
from pathlib import Path

from rl_vla.diagnostics import contact_support_overlay_v1 as implementation
from rl_vla.diagnostics.contact_support_overlay_v1 import (
    DESTINATION,
    atomic,
    refresh,
    verify_overlay,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--poll-seconds", type=float, default=45)
    args = parser.parse_args()
    if args.poll_seconds <= 0:
        parser.error("poll seconds must be positive")
    destination = args.run.resolve() / DESTINATION
    destination.mkdir(parents=True, exist_ok=True)
    with (destination / "worker.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        sources = {
            str(path.resolve()): implementation.sha(path)
            for path in (
                Path(__file__),
                Path(implementation.__file__),
                Path(implementation.validation.__file__),
                implementation.LEGACY_SOURCE,
                implementation.ROOT / "src/rl_vla/diagnostics/contact_evidence.py",
            )
        }
        source_manifest = destination / "source_manifest.json"
        manifest = {"sources": sources, "code_hash": implementation.digest(sources)}
        if source_manifest.exists():
            if implementation.read(source_manifest) != manifest:
                raise ValueError(
                    "Contact overlay source changed; preserve lineage and coordinate a versioned repair"
                )
        else:
            atomic(source_manifest, manifest)
            snapshots = destination / "sources"
            snapshots.mkdir()
            for source in sources:
                path = Path(source)
                (snapshots / path.name).write_bytes(path.read_bytes())

        def guard_sources():
            if any(implementation.sha(path) != expected for path, expected in sources.items()):
                raise ValueError("Contact overlay source changed while worker was active")

        cache_path = destination / "cache.json"
        cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
        transient_count = 0
        while True:
            try:
                guard_sources()
                summary = refresh(args.run, cache)
                guard_sources()
                verify_overlay(args.run)
                transient_count = 0
                atomic(cache_path, cache)
                atomic(
                    destination / "status.json",
                    {
                        "state": "complete" if summary["complete"] else "waiting",
                        "resolved_cases": summary["resolved_cases"],
                        "native_cases": summary["completed_native_cases"],
                        "structural_na_cases": summary["structural_na_cases"],
                        "support_execution_adopted": summary["support_execution_adopted"],
                    },
                )
                print(
                    json.dumps(
                        {
                            "resolved_cases": summary["resolved_cases"],
                            "complete": summary["complete"],
                            "structural_na_cases": summary["structural_na_cases"],
                        }
                    ),
                    flush=True,
                )
                if summary["complete"] or not args.watch:
                    return
            except (json.JSONDecodeError, FileNotFoundError) as error:
                transient_count += 1
                atomic(
                    destination / "status.json",
                    {
                        "state": "waiting_for_atomic_producer" if transient_count < 3 else "failed",
                        "consecutive_transient_errors": transient_count,
                        "error": repr(error),
                    },
                )
                if transient_count >= 3 or not args.watch:
                    raise
            except BaseException as error:
                atomic(destination / "status.json", {"state": "failed", "error": repr(error)})
                raise
            time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
