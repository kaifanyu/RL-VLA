"""Prepare DICE configs for one explicitly selected LIBERO-PRO protocol task."""

import argparse
import hashlib
import json
import re
from pathlib import Path

from rl_vla.envs.libero_pro import validate_splits

ROOT = Path(__file__).resolve().parents[1]


def prepare(protocol_path, task_key, output, instruction, source_id, source_split, replay):
    protocol_path = protocol_path.resolve()
    plan = json.loads(protocol_path.read_text())
    if plan.get("version") != 2 or task_key not in plan.get("tasks", {}):
        raise ValueError("Select an existing version-2 protocol and explicit task key")
    task = plan["tasks"][task_key]
    validate_splits(task, 2)
    if not instruction.strip() or not source_id.strip() or not source_split.strip():
        raise ValueError("Instruction, demonstration source ID, and source split are required")
    template = (ROOT / "configs/dice_pi05_libero_pro.toml").read_text()
    protocol_hash = hashlib.sha256(protocol_path.read_bytes()).hexdigest()
    provenance = {
        "task": {
            "task_key": task_key,
            "task_suite": task["suite"],
            "task_id": task["task_id"],
            "instruction": instruction,
        },
        "environment": {
            "adapter": "rl_vla.envs.libero_pro:make_env",
            "protocol_id": f"sha256:{protocol_hash}",
            "protocol_sha256": protocol_hash,
            "protocol_version": plan["version"],
            "split": "train",
        },
        "demonstration": {"source_id": source_id, "source_split": source_split},
    }
    replacements = {
        'protocol = ""': f"protocol = {json.dumps(str(protocol_path))}",
        'task_key = ""': f"task_key = {json.dumps(task_key)}",
        'path = ""': f"path = {json.dumps(str(replay.resolve()))}",
    }
    for old, new in replacements.items():
        if template.count(old) != 1:
            raise ValueError(f"DICE PRO template must contain exactly one {old!r}")
        template = template.replace(old, new)
    output.mkdir(parents=True, exist_ok=True)
    destinations = [output / f"{split}.toml" for split in ("train", "validation", "test")]
    existing = [str(path) for path in destinations if path.exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite existing DICE configs: {existing}")
    results = []
    for split, destination in zip(("train", "validation", "test"), destinations, strict=True):
        text = re.sub(
            r'^split = "train"$', f'split = "{split}"', template,
            count=1, flags=re.MULTILINE,
        )
        if f'split = "{split}"' not in text:
            raise ValueError("DICE PRO template is missing its split setting")
        text += "\n[data.provenance.task]\n"
        text += f"task_key = {json.dumps(task_key)}\n"
        text += f"task_suite = {json.dumps(task['suite'])}\n"
        text += f"task_id = {task['task_id']}\n"
        text += f"instruction = {json.dumps(instruction)}\n"
        text += "\n[data.provenance.environment]\n"
        text += f"adapter = {json.dumps(provenance['environment']['adapter'])}\n"
        text += f"protocol_id = {json.dumps(provenance['environment']['protocol_id'])}\n"
        text += f"protocol_sha256 = {json.dumps(protocol_hash)}\n"
        text += f"protocol_version = {plan['version']}\n"
        text += 'split = "train"\n'
        text += "\n[data.provenance.demonstration]\n"
        text += f"source_id = {json.dumps(source_id)}\n"
        text += f"source_split = {json.dumps(source_split)}\n"
        with destination.open("x") as stream:
            stream.write(text)
        results.append(str(destination))
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--task-key", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--source-id", required=True)
    parser.add_argument("--source-split", required=True)
    parser.add_argument("--replay", required=True, type=Path)
    args = parser.parse_args()
    outputs = prepare(
        args.protocol,
        args.task_key,
        args.output.resolve(),
        args.instruction,
        args.source_id,
        args.source_split,
        args.replay,
    )
    print(json.dumps({"configs": outputs, "training_started": False}, indent=2))


if __name__ == "__main__":
    main()
