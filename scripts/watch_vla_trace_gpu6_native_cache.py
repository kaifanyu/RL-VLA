"""Read-only provenance checks for the already-running native Object1 OOD queue."""

import argparse
import hashlib
import json
from pathlib import Path
import time


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def write(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def process(pid):
    path = Path("/proc") / str(pid)
    try:
        fields = (path / "stat").read_text().rsplit(") ", 1)[1].split()
        environment = dict(item.split("=", 1) for item in (path / "environ").read_text().split("\0") if "=" in item)
    except FileNotFoundError:
        return None
    keys = ("JAX_COMPILATION_CACHE_DIR", "XLA_FLAGS", "JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES")
    return {"pid": pid, "parent": int(fields[1]), "start_ticks": int(fields[19]),
            "environment": {key: environment.get(key) for key in keys},
            "command": (path / "cmdline").read_text().split("\0")[:-1]}


def verify_cache_environment(environment, cache):
    if (environment.get("JAX_COMPILATION_CACHE_DIR") != str(cache)
            or environment.get("XLA_FLAGS") not in (None, "")
            or environment.get("JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES") not in (None, "")):
        raise ValueError("Native OOD cache/profile environment differs from proved natural-key execution")


def verify_files(records):
    for path, digest in records.items():
        if sha(path) != digest:
            raise ValueError(f"Native cached executable or proof changed: {path}")


def run(root, pid, poll_seconds):
    output = root / "compiler_profile_integration/gpu6_native_ood_cache_binding_v1"
    output.mkdir(exist_ok=False)
    oracle = root / "compiler_profile_integration/gpu6_native_cache_oracle_v1/summary.json"
    proof = json.loads(oracle.read_text())
    if (proof.get("passed") is not True or proof.get("complete") is not True
            or proof.get("original_native_cache_compatibility_verified") is not True
            or proof.get("gpu") != 6 or len(proof.get("cases", [])) != 16):
        raise ValueError("Native GPU6 OOD requires passed complete natural-cache-key compatibility evidence")
    initial = process(pid)
    if initial is None:
        raise ValueError("Original native OOD parent is absent")
    cache = root.parents[1] / ".cache/jax-rl-vla"
    verify_cache_environment(initial["environment"], cache)
    files = {str(oracle): sha(oracle), str(Path(__file__).resolve()): sha(__file__)}
    binaries = {}
    identities = []
    for case in proof["cases"]:
        identities.append((case["role"], case["iteration"], case["state_id"], case["decision_index"]))
        if case.get("context_exact") is not True or any(value is not True for value in case["same_as_original_proved_oracle"].values()):
            raise ValueError("Natural GPU6 cache oracle has non-exact output/context evidence")
        files[case["summary"]] = case["summary_sha256"]
        files[case["original_gpu5_oracle_summary"]] = case["original_gpu5_oracle_summary_sha256"]
        for row in case["cached_executables"]:
            path = Path(row["path"]).resolve()
            if path.parent != cache or path.name != row["cache_key"] + "-cache":
                raise ValueError("Oracle executable is not under its exact natural shared-cache key")
            if str(path) in binaries and binaries[str(path)] != row["sha256"]:
                raise ValueError("Conflicting natural-cache binary hashes")
            binaries[str(path)] = row["sha256"]
    expected = {(role, iteration, state, decision) for role, iteration in
                (("baseline", None), ("trained", 25), ("trained", 50), ("trained", 100))
                for state in (100, 101) for decision in (0, 1)}
    if set(identities) != expected or len(identities) != len(expected) or len(binaries) != 3:
        raise ValueError("Native cache compatibility evidence lacks exact four-policy coverage")
    files.update(binaries)
    queue_source = root.parents[1] / "scripts/run_vla_trace_ood_evaluations.py"
    shell_source = root.parents[1] / "scripts/run_rldt.sh"
    files.update({str(queue_source): sha(queue_source), str(shell_source): sha(shell_source)})
    verify_files(files)
    write(output / "manifest.json", {"scope": "Read-only supplemental provenance; no queue/runtime/environment/cache modification",
          "task": "object_1", "parent": initial, "cache_root": str(cache), "oracle": str(oracle),
          "oracle_sha256": sha(oracle), "binary_sha256": binaries, "input_sha256": files,
          "checks": "Full SHA256 of all three cached binaries and source evidence at startup, every poll, and completion",
          "limitation": "Checks existing natural-key binary identity; does not force cache hits or certify unobserved future compiler shapes"})
    count = 0
    try:
        while True:
            verify_files(files)
            current = process(pid)
            if current and current["start_ticks"] != initial["start_ticks"]:
                current = None
            if current:
                verify_cache_environment(current["environment"], cache)
            queue_complete = root / "ood/object_1/queue_complete.json"
            complete = queue_complete.exists() and json.loads(queue_complete.read_text()).get("complete") is True
            if not current and not complete:
                raise ValueError("Original OOD parent exited before a complete queue marker")
            count += 1
            write(output / "status.json", {"passed": True, "complete": complete,
                  "state": "completed" if complete else "watching", "checks_performed": count,
                  "checked_unix_seconds": time.time(), "parent_identity_matches": current is not None,
                  "binary_sha256": binaries})
            if complete:
                write(output / "completion.json", {"passed": True, "complete": True,
                      "queue_complete_sha256": sha(queue_complete), "input_sha256": files,
                      "checks_performed": count})
                return
            time.sleep(poll_seconds)
    except BaseException as error:
        write(output / "status.json", {"passed": False, "complete": False, "state": "failed",
              "error": repr(error), "checks_performed": count, "checked_unix_seconds": time.time()})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--parent-pid", type=int, required=True)
    parser.add_argument("--poll-seconds", type=float, default=60)
    args = parser.parse_args()
    run(args.run.resolve(), args.parent_pid, args.poll_seconds)
