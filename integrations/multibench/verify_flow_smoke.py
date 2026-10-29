"""Verify a real one-update Flow-SDE run against its unchanged base weights."""
import argparse
import json
from pathlib import Path

import yaml

from pi_rl.verify_smoke import verify


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    model = config["actor"]["model"]
    initial = Path(model["model_path"]) / "model.safetensors"
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    verify(initial, args.checkpoint, args.metrics, output)
    result = json.loads(output.read_text())
    environment = config["env"]["train"]
    progress_path = Path(environment["recording_directory"]) / "progress.json"
    progress = json.loads(progress_path.read_text())
    benchmark = model["benchmark"]
    dimensions = {"robotwin": 14, "robocasa": 12, "maniskill": 7}
    settings = model["openpi"]
    contract = {
        "benchmark": benchmark,
        "native_action_dim": progress["native_action_dim"],
        "action_horizon": settings["action_horizon"],
        "discrete_state_input": settings["discrete_state_input"],
        "pi05": settings["pi05"],
        "noise_method": settings["noise_method"],
        "real_environment_decisions": progress["decisions"],
        "real_controller_steps": progress["primitive_steps"],
    }
    contract_passed = bool(
        contract["native_action_dim"] == dimensions[benchmark] == model["action_dim"]
        and contract["action_horizon"] == 50
        and contract["discrete_state_input"] is True
        and contract["pi05"] is True
        and contract["noise_method"] == "flow_sde"
        and contract["real_environment_decisions"] >= 2
        and contract["real_controller_steps"] >= 2
    )
    result.update(passed=bool(result["smoke_passed"] and contract_passed),
                  benchmark=benchmark, contract=contract,
                  contract_passed=contract_passed, config=str(Path(args.config).resolve()),
                  metrics=str(Path(args.metrics).resolve()), progress=str(progress_path),
                  render_device=progress.get("render_device"))
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    if not result["passed"]:
        raise RuntimeError("Flow-SDE smoke failed the base-policy/native-environment contract")
    print(json.dumps({"passed": True, "benchmark": benchmark, "report": str(output)}), flush=True)


if __name__ == "__main__":
    main()
