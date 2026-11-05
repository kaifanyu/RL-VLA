#!/usr/bin/env python
"""Wrap the canonical Oct31 Object 0 states as a FlashSAC state set (no simulator, no regeneration).

The states are copied byte for byte, so FlashSAC uses the same train/validation/test ids as
RLDT and DICE-RL. Refuses to overwrite an existing state set.
"""
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rl_vla.pro_residual.spec import load_config, load_state_array, resolve_case, sha256_file, state_set_request

CANONICAL = ROOT / "runs/vla_trace_objects_20261031/tasks/object_0"
CANONICAL_SHA256 = "9f4d57d3368a345730395ff5955b4c6c377ed01a3f17c607cbfbaa3523219af2"
OUTPUT = ROOT / "runs/pro_state_sets/object_0_canonical_oct31"
CONFIG = ROOT / "configs/flashsac_object_0.toml"


def main():
    if OUTPUT.exists():
        raise SystemExit(f"{OUTPUT} already exists")
    source = CANONICAL / "pro_states.npy"
    if sha256_file(source) != CANONICAL_SHA256:
        raise SystemExit("Canonical pro_states.npy does not match the audited hash")
    task = json.loads((CANONICAL / "protocol.json").read_text())["tasks"]["object_0"]
    generation = json.loads((CANONICAL / "generation.json").read_text())
    splits = {"train": task["train_state_ids"], "validation": task["validation_state_ids"],
              "test": task["test_state_ids"]}
    config = load_config(CONFIG)
    case = resolve_case(dict(config["task"], state_set=None))
    states = load_state_array(str(source))
    offset = int(generation["published_test_start"])
    if sorted(sum(splits.values(), [])) != list(range(len(states))):
        raise SystemExit("Canonical split ids do not partition the state array")
    if splits["test"] != list(range(offset, len(states))):
        raise SystemExit("Canonical test ids are not the published tail")
    if task["name"] != case["name"]:
        raise SystemExit("Canonical task differs from the FlashSAC case")
    counts = {key: len(value) for key, value in splits.items()}
    request = state_set_request(case, generation["generation_seed"], generation["split_seed"], counts)
    OUTPUT.mkdir(parents=True)
    shutil.copyfile(source, OUTPUT / "pro_states.npy")
    manifest = {
        "format": "rl-vla-pro-state-set-v1", "request": request,
        "states_file": str(OUTPUT / "pro_states.npy"), "states_sha256": sha256_file(OUTPUT / "pro_states.npy"),
        "state_count": int(len(states)), "state_vector_dim": int(states.shape[1]),
        "published_offset": offset, "splits": splits,
        "method": "Byte copy of the canonical Oct31 RLDT/DICE-RL Object 0 states; not regenerated.",
        "canonical_source": {"directory": str(CANONICAL),
                             "protocol_sha256": sha256_file(CANONICAL / "protocol.json"),
                             "generation_sha256": sha256_file(CANONICAL / "generation.json")},
    }
    (OUTPUT / "state_set.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: manifest[key] for key in ("states_sha256", "state_count", "published_offset")}, indent=2))


if __name__ == "__main__":
    main()
