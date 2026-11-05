"""Freeze an environment-only diagnostic compiler profile from measured native proofs."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

from rl_vla.diagnostics import compiler_profile
from map_vla_trace_autotune import proto_fields, single


def freeze(run):
    run = run.resolve()
    output = run / "compiler_profiles" / compiler_profile.PROFILE_VERSION
    if output.exists():
        raise FileExistsError("Immutable compiler profile already exists")
    mapping = run / "compiler_review/canonical_autotune_v1"
    profile = mapping / "canonical_autotune.textproto"
    mapping_audit = json.loads((mapping / "mapping_audit.json").read_text())
    replay = run / "trained_autotune_replay_v2/summary.json"
    component = run / "trained_autotune_components_v1/summary.json"
    replay_value, component_value = (json.loads(path.read_text()) for path in (replay, component))
    if not mapping_audit["mapping_passed"] or not replay_value["all_exact"] or not component_value["passed"]:
        raise ValueError("Technical mapping/replay/component proof did not pass")
    if len(component_value["checks"]) != 15 or not all(row["exact"] and row["max_abs_difference"] == 0 for row in component_value["checks"].values()):
        raise ValueError("Expected all fifteen exact native-component comparisons")
    sources = {profile, replay, component, mapping / "mapping_audit.json", mapping / "all_expert_gemm_audit.json",
               run / "compiler_review/replay_v2_all_expert_gemm_audit.json", Path(__file__).resolve(),
               Path(compiler_profile.__file__).resolve(), run / "trained_autotune_components_v1/executed_source.py",
               run / "trained_autotune_replay_v2/executed_source.py"}
    for audit in (mapping_audit, json.loads((mapping / "all_expert_gemm_audit.json").read_text()),
                  json.loads((run / "compiler_review/replay_v2_all_expert_gemm_audit.json").read_text())):
        for path, expected in audit["sources"].items():
            if compiler_profile.sha256(path) != expected:
                raise ValueError(f"Technical source changed: {path}")
            sources.add(Path(path))
    for path, expected in component_value["source_input_sha256"].items():
        if compiler_profile.sha256(path) != expected:
            raise ValueError(f"Native source/input changed: {path}")
        sources.add(Path(path))
    devices = {single(row["children"], "device")["value"] for row in proto_fields(profile.read_text())
               if row["name"] == "results"}
    if len(devices) != 1:
        raise ValueError("Expected one exact autotune device descriptor")
    descriptor = devices.pop()
    if not descriptor.startswith("CUDA: 8.0, Cores: 70,"):
        raise ValueError(f"Unexpected measured compiler device: {descriptor}")
    query = ["nvidia-smi", "--query-gpu=index,name,uuid,memory.total,compute_cap", "--format=csv,noheader"]
    raw = subprocess.check_output(query, text=True)
    gpu_rows = []
    for row in raw.splitlines():
        index, name, uuid, memory, capability = (part.strip() for part in row.split(","))
        gpu_rows.append({"index": int(index), "reported_name": name, "uuid": uuid,
                         "memory_mib": int(memory.split()[0]), "compute_capability": capability})
    if len(gpu_rows) != 8 or any(row["memory_mib"] != 65536 or row["compute_capability"] != "8.0" for row in gpu_rows):
        raise ValueError("Measured GPUs differ from the verified device class")
    native_python = run.parents[1] / "third_party/openpi/.venv/bin/python"
    versions = json.loads(subprocess.check_output([str(native_python), "-c", "import importlib.metadata,json; print(json.dumps({p:importlib.metadata.version(p) for p in ('jax','jaxlib')}))"], text=True))
    if versions != {"jax": "0.5.3", "jaxlib": "0.5.3"}:
        raise ValueError(f"Unexpected native compiler packages: {versions}")
    manifest = {"schema_version": 1, "profile_id": compiler_profile.PROFILE_VERSION,
                "created_utc": datetime.now(timezone.utc).isoformat(), "versions": versions,
                "autotune_file": {"name": "autotune.textproto", "sha256": compiler_profile.sha256(profile), "bytes": profile.stat().st_size},
                "environment_helper": {"path": str(Path(compiler_profile.__file__).resolve()), "sha256": compiler_profile.sha256(compiler_profile.__file__)},
                "isolated_cache_root": str(run / "compiler_caches" / compiler_profile.PROFILE_VERSION),
                "startup_environment": {"XLA_FLAGS": "--xla_gpu_load_autotune_results_from=<this immutable profile>",
                                        "JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES": "", "JAX_COMPILATION_CACHE_DIR": "<distinct descendant of isolated_cache_root>"},
                "device": {"autotune_descriptor": descriptor, "compute_capability": "8.0", "streaming_multiprocessor_count": 70,
                           "memory_gib": 64, "inventory_command": query, "gpu_inventory": gpu_rows,
                           "terminology": "70 is the reported streaming-multiprocessor count; the architecture compute capability is8.0, notSM70."},
                "sources_sha256": {str(path.resolve()): compiler_profile.sha256(path) for path in sorted(sources)},
                "technical_coverage": {"task": "object_0", "iteration": 25, "validation_state_id": 100,
                                       "recorded_native_action_replay": "five sampler graphs exact to actual recorded actions",
                                       "component_checks": 15, "component_tolerance": "bit exact",
                                       "all_requested_checkpoint_gate_complete": False},
                "scope": "Immutable compiler choices and child environment only. No native arithmetic, training source, shared cache or global environment is modified. Full requested-checkpoint and worker-deployment fidelity remain separate acceptance gates.",
                "constant_key_scope": mapping_audit["constant_scope"]}
    output.mkdir(parents=True)
    (output / "autotune.textproto").write_bytes(profile.read_bytes())
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    compiler_profile.load_profile(output)
    print(json.dumps({"profile": str(output), "manifest_sha256": compiler_profile.sha256(output / "manifest.json"),
                      "profile_sha256": manifest["autotune_file"]["sha256"]}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    freeze(parser.parse_args().run)
