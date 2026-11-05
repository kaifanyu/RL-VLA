"""Real-model VLM-LoRA identity, gradient, frozen-weight, and reload gate."""
import argparse
import copy
import hashlib
import json
from pathlib import Path

import jax
import numpy as np
from flax import nnx

from rl_vla import rldt
from rl_vla.rldt_lora import is_lora_target
from rl_vla.rldt_pi05 import Pi05Flow


def hashes(state):
    return {"/".join(map(str, p)): hashlib.sha256(np.asarray(v.value).tobytes()).hexdigest()
            for p, v in state.flat_state().items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = rldt.load_config(args.config)
    config["train"].update(iterations=1, collect_decisions=2, critic_steps=1, actor_steps=1,
                           critic_batch_size=2, actor_batch_size=1, critic_trajectories=1)
    config["rldt"].update(bootstrap_samples=1, critic_warmup_iterations=0)
    config["recording"] = {"enabled": False}
    print("Loading VLM-LoRA policy for native identity and update checks", flush=True)
    flow = rldt.make_flow(config["policy"])
    assert flow.lora_target == "vlm"
    assert all(is_lora_target(p, "vlm") for p in flow.lora_paths)
    assert not any(is_lora_target(p, "action_expert") for p in flow.lora_paths)
    frozen_before = hashes(flow.frozen)
    env = rldt.make_primitive_env(config["env"])
    try:
        raw, _ = env.reset(seed=0)
    finally:
        env.close()
    context = flow.prepare([raw])
    key = jax.random.PRNGKey(100000)
    sample = jax.jit(lambda p, c, k, f: flow.sample(p, c, k, 10, frozen=f)[0])
    initial = np.asarray(sample(flow.params, context, key, flow.frozen))

    # A separate unadapted model view shares immutable base arrays. This checks
    # actual motor sampling against pi05_libero before any optimizer update.
    base = copy.copy(flow)
    base.__class__ = Pi05Flow
    base.params = nnx.State({})
    base.lora_rank = 0
    base.lora_paths = ()
    base._encode_jit = jax.jit(base._encode_prefix)
    base_context = base.prepare([raw])
    base_sample = jax.jit(lambda p, c, k, f: base.sample(p, c, k, 10, frozen=f)[0])
    original = np.asarray(base_sample(base.params, base_context, key, base.frozen))
    identity_error = float(np.max(np.abs(initial-original)))
    np.testing.assert_allclose(initial, original, atol=2e-5, rtol=2e-5)
    print(f"Zero-adapter native identity passed: max action difference {identity_error}", flush=True)

    original_factory = rldt.make_flow
    rldt.make_flow = lambda _: flow
    try:
        summary = rldt.train(config, str(args.output.resolve()))
    finally:
        rldt.make_flow = original_factory
    trained = flow.load_params(args.output / "final")
    delta = {"/".join(map(str, p)): float(np.linalg.norm(
        np.asarray(trained.flat_state()[p].value, dtype=np.float64)
        - np.asarray(v.value, dtype=np.float64))) for p, v in flow.params.flat_state().items()}
    updated = np.asarray(sample(trained, context, key, flow.frozen))
    action_change = float(np.max(np.abs(updated-initial)))
    frozen_after = hashes(flow.frozen)
    metrics = json.loads((args.output / "metrics.jsonl").read_text().splitlines()[-1])["metrics"]
    passed = (frozen_before == frozen_after and any(x > 0 for x in delta.values())
              and metrics["actor_grad_norm"] > 0 and metrics["actor_update_norm"] > 0
              and action_change > 0 and np.isfinite(updated).all()
              and summary["actor_updates"] == 1)
    report = dict(passed=bool(passed), target="vlm", frozen_backbone_identical=frozen_before == frozen_after,
                  action_expert_frozen=True, image_encoder_frozen=True,
                  zero_adapter_action_max_difference=identity_error,
                  trained_adapter_action_max_difference=action_change,
                  adapter_tensor_delta_l2=delta, frozen_before=frozen_before, frozen_after=frozen_after,
                  actor_metrics=metrics, summary=summary, policy_metadata=flow.metadata,
                  scope="Technical smoke only, not a learning/performance result")
    (args.output / "verification.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(json.dumps({k:v for k,v in report.items() if k not in {
        "frozen_before", "frozen_after", "adapter_tensor_delta_l2", "summary", "policy_metadata"}}, indent=2), flush=True)
    if not passed:
        raise RuntimeError("VLM-LoRA real update/frozen-weight/reload verification failed")


if __name__ == "__main__":
    main()
