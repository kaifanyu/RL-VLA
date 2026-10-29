"""Compare actual saved RLDT expert weights against the original base on CPU."""
import argparse
import ast
import hashlib
import json
from pathlib import Path

import numpy as np


def read_tensor(path, name):
    """Read a small safetensors tensor directly without loading either full model."""
    with Path(path).open("rb") as stream:
        length = int.from_bytes(stream.read(8), "little")
        header = json.loads(stream.read(length))
        spec = header[name]
        start, end = spec["data_offsets"]
        stream.seek(8 + length + start)
        data = stream.read(end - start)
    if spec["dtype"] == "BF16":
        array = (np.frombuffer(data, dtype="<u2").astype("<u4") << 16).view("<f4")
    else:
        dtype = {"F32": "<f4", "F16": "<f2"}[spec["dtype"]]
        array = np.frombuffer(data, dtype=dtype).astype(np.float32)
    return array.reshape(spec["shape"])


def verify(directory, output):
    from rl_vla.rldt_pi05 import is_expert_parameter

    directory = Path(directory)
    metadata = json.loads((directory / "final/rldt_pi05.json").read_text())
    base = Path(metadata["checkpoint"])
    tree = json.loads((base / "params/_METADATA").read_text())["tree_metadata"]
    paths = [ast.literal_eval(key)[1:] for key in tree]
    paths = [key[:-1] if key[-1] == "value" else key for key in paths]
    converted = base.parent / "pi05_base_rlinf"
    conversion = json.loads((converted / "conversion_manifest.json").read_text())
    if Path(conversion["source_checkpoint"]).resolve() != base.resolve() or conversion["conversion_dtype"] != "bfloat16":
        raise ValueError("Small-tensor baseline comparison needs the verified conversion of this exact base")
    rows = [json.loads(line) for line in (directory / "metrics.jsonl").read_text().splitlines() if line]
    finite_metrics = bool(rows) and all(np.isfinite(list(row["metrics"].values())).all() for row in rows)
    count, compared, changed, maximum, finite = 0, 0, 0, 0., True
    examples = []
    with np.load(directory / "final/expert.npz", allow_pickle=False) as archive:
        expected = {"/".join(map(str, key)) for key in paths if is_expert_parameter(key)}
        if set(archive.files) != expected:
            raise ValueError("Saved trainable keys differ from the complete action expert")
        for key in archive.files:
            after = archive[key]
            finite &= bool(np.isfinite(after).all())
            count += after.size
            projection, leaf = key.split("/")[0], key.split("/")[-1]
            if projection in {"action_in_proj", "action_out_proj"} and leaf in {"kernel", "bias"}:
                torch_name = projection + (".weight" if leaf == "kernel" else ".bias")
                before = read_tensor(converted / "model.safetensors", torch_name)
                if leaf == "kernel":
                    before = before.T
                if before.shape != after.shape:
                    raise ValueError(f"Parameter shape mismatch: {key}")
                n = int(np.count_nonzero(before != after))
                compared += after.size
                changed += n
                maximum = max(maximum, float(np.max(np.abs(after - before))))
                examples.append({"parameter": key, "compared_values": after.size, "changed_values": n})
    frozen_base_matches = all(
        hashlib.sha256((base / relative).read_bytes()).hexdigest() == metadata[key]
        for key, relative in [("checkpoint_metadata_sha256", "params/_METADATA"),
                              ("checkpoint_manifest_sha256", "params/manifest.ocdbt")])
    summary = json.loads((directory / "summary.json").read_text())
    positive_grad = any(row["metrics"].get("actor_grad_norm", 0) > 0 for row in rows)
    positive_update = any(row["metrics"].get("actor_update_norm", 0) > 0 for row in rows)
    expected_motor = {"maniskill": 7, "robotwin": 14, "robocasa": 12}[metadata["calibration"]["benchmark"]]
    dimensions_correct = metadata["horizon"] == 50 and metadata["action_dim"] == 32 and metadata["motor_dim"] == expected_motor
    passed = (count == 430_098_464 and changed > 0 and finite and finite_metrics and positive_grad
              and positive_update and frozen_base_matches and summary["actor_updates"] >= 1
              and summary["critic_updates"] >= 1 and dimensions_correct
              and metadata["model_config"] == "pi05_base" and not metadata["projection_only"])
    result = {"passed": bool(passed), "smoke_passed": bool(passed), "benchmark": metadata["calibration"]["benchmark"],
              "trainable_parameters": count, "compared_parameters": compared,
              "changed_parameters_in_compared_tensors": changed, "max_abs_change": maximum,
              "delta_scope": "Four complete input/output projection tensors; finite checks cover every saved expert tensor; global update metric covers full expert.",
              "changed_examples": examples, "finite_saved_parameters": finite,
              "finite_metrics": bool(finite_metrics), "positive_gradient": positive_grad,
              "positive_update": positive_update, "frozen_base_provenance_unchanged": frozen_base_matches,
              "vlm_frozen": frozen_base_matches, "dimensions_correct": dimensions_correct,
              "frozen_parameters": metadata["frozen_parameters"],
              "actor_updates": summary["actor_updates"], "critic_updates": summary["critic_updates"],
              "base_checkpoint": str(base), "saved_checkpoint": str(directory / "final"),
              "frozen_verification": "Saved/optimized keys contain only the action expert; inference restores VLM from unchanged original base. No task-trained VLM is exported.",
              "model_config": metadata["model_config"], "horizon": metadata["horizon"],
              "particles": json.loads((directory / "config.json").read_text())["rldt"]["particles"],
              "training_summary": summary}
    episodes = directory / "trajectories/episodes.jsonl"
    if episodes.exists():
        resets = [json.loads(line).get("metadata", {}).get("reset", {})
                  for line in episodes.read_text().splitlines() if line]
        result["recorded_resets"] = resets
        devices = [reset["render_device"] for reset in resets if "render_device" in reset]
        if devices:
            result["render_devices"] = devices
            result["render_affinity_verified"] = all(device.get("affinity_verified", False)
                                                     for device in devices)
    Path(output).write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    if not passed:
        raise RuntimeError("Native RLDT real update verification failed")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    verify(args.smoke, args.output)
