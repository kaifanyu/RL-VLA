"""Real-model task-specific smoke with complete frozen/expert tensor audit."""

import argparse
import hashlib
import json
from pathlib import Path

import jax
import numpy as np

from rl_vla import rldt


def tensor_hashes(state):
    return {"/".join(map(str, path)): hashlib.sha256(np.asarray(value.value).tobytes()).hexdigest()
            for path, value in state.flat_state().items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = rldt.load_config(args.config)
    config["train"].update(iterations=1, collect_decisions=2, critic_steps=1, actor_steps=1,
                           critic_batch_size=2, actor_batch_size=1, critic_trajectories=1)
    config["rldt"].update(particles=8, bootstrap_samples=1, critic_warmup_iterations=0)
    config["recording"] = {"enabled": False}
    flow = rldt.make_flow(config["policy"])
    frozen_before = tensor_hashes(flow.frozen)
    original_make_flow = rldt.make_flow
    rldt.make_flow = lambda _: flow
    try:
        summary = rldt.train(config, str(args.output.resolve()))
    finally:
        rldt.make_flow = original_make_flow
    frozen_after = tensor_hashes(flow.frozen)
    trained = flow.load_params(args.output/"final")
    drift = {}
    for path, value in flow.params.flat_state().items():
        before = np.asarray(value.value, dtype=np.float64)
        after = np.asarray(trained.flat_state()[path].value, dtype=np.float64)
        drift["/".join(map(str, path))] = float(np.linalg.norm(after-before))
    expert_changed = any(v > 0 for p, v in drift.items() if any(n.endswith("_1") for n in p.split("/")))
    frozen_identical = frozen_before == frozen_after
    passed = expert_changed and frozen_identical and summary["actor_updates"] == 1
    report = {"passed": passed, "expert_1_changed": expert_changed,
              "frozen_backbone_identical": frozen_identical,
              "frozen_before": frozen_before, "frozen_after": frozen_after,
              "expert_tensor_delta_l2": drift, "summary": summary,
              "devices": [str(d) for d in jax.devices()],
              "scope": "Technical smoke only, not a learning/performance result"}
    (args.output/"verification.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    if not passed:
        raise RuntimeError("Expert/frozen-boundary verification failed")
    print(json.dumps({"passed": passed, "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
