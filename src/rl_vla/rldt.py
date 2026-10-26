"""RLDT online flow fine-tuning. Run via scripts/run_rldt.sh; see docs/rldt.md."""

import argparse
import copy
import json
import math
import subprocess
import time
import tomllib
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax

from rl_vla.rldt_env import make_primitive_env
from rl_vla.rldt_math import actor_loss, svgd_field
from rl_vla.rldt_replay import TrajectoryReplay, lambda_returns


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def load_config(path):
    with Path(path).open("rb") as stream:
        config = tomllib.load(stream)
    if config.get("algorithm") != "rldt":
        raise ValueError("Expected algorithm='rldt'; residual SAC/PPO configs are separate")
    for section in ("policy", "env", "train", "rldt"):
        if section not in config:
            raise ValueError(f"Missing [{section}]")
    if config["policy"]["kind"] not in {"toy", "pi05"}:
        raise ValueError("policy.kind must be toy or pi05")
    for name in ("iterations", "collect_decisions", "critic_steps", "actor_steps",
                 "critic_batch_size", "actor_batch_size", "critic_trajectories",
                 "replay_capacity", "save_every"):
        if int(config["train"][name]) < 1:
            raise ValueError(f"train.{name} must be positive")
    r = config["rldt"]
    for name in ("ode_steps", "particles", "execute_steps", "bootstrap_samples"):
        if int(r[name]) < (2 if name == "particles" else 1):
            raise ValueError(f"Invalid rldt.{name}")
    for name in ("gamma", "target_momentum"):
        if not 0 < float(r[name]) <= 1:
            raise ValueError(f"rldt.{name} must be in (0,1]")
    if not 0 <= float(r["td_lambda"]) <= 1:
        raise ValueError("td_lambda must be in [0,1]")
    if not 0 <= float(r.get("fresh_fraction", .33)) <= 1:
        raise ValueError("fresh_fraction must be in [0,1]")
    for name in ("actor_lr", "critic_lr", "max_grad_norm"):
        if not math.isfinite(float(r[name])) or float(r[name]) <= 0:
            raise ValueError(f"rldt.{name} must be finite and positive")
    for name in ("kappa0", "consistency_weight", "constraint_weight"):
        if not math.isfinite(float(r[name])) or float(r[name]) < 0:
            raise ValueError(f"rldt.{name} must be finite and nonnegative")
    if r.get("kernel_mode", "official") not in {"paper", "official"}:
        raise ValueError("kernel_mode must be paper or official")
    for name in ("actor_warmup_steps", "critic_warmup_iterations"):
        if int(r.get(name, 0)) < 0:
            raise ValueError(f"rldt.{name} must be nonnegative")
    if int(r["critic_width"]) < 1:
        raise ValueError("critic_width must be positive")
    if r.get("reference", "moving") not in {"moving", "pretrained"}:
        raise ValueError("reference must be moving or pretrained")
    if config["train"]["replay_capacity"] < config["train"]["collect_decisions"]:
        raise ValueError("replay_capacity must fit one collection window")
    return config


def make_flow(config):
    if config["kind"] == "pi05":
        from rl_vla.rldt_pi05 import Pi05Flow

        return Pi05Flow.from_config(config)
    from rl_vla.rldt_toy import ToyFlow

    return ToyFlow(config)


def init_critic(key, feature_dim, action_dim, width):
    """Independent SiLU double-Q networks in fp32."""
    keys = iter(jax.random.split(key, 8))
    twins = []
    for _ in range(2):
        layers = []
        for n_in, n_out in zip(
            [feature_dim + action_dim, width, width, width], [width, width, width, 1],
            strict=True,
        ):
            layers.append({"w": jax.random.normal(next(keys), (n_in, n_out))
                           * np.sqrt(1 / n_in), "b": jnp.zeros(n_out)})
        twins.append(layers)
    return twins


def critic(params, features, actions):
    features = jnp.asarray(features, jnp.float32)
    # Frozen VLM features have a stable per-observation scale; no running stats.
    features = (features - features.mean(-1, keepdims=True)) / jnp.sqrt(
        features.var(-1, keepdims=True) + 1e-5)
    values = []
    for network in params:
        h = jnp.concatenate((features, actions.reshape((actions.shape[0], -1))), -1)
        for i, layer in enumerate(network):
            h = h @ layer["w"] + layer["b"]
            if i < len(network) - 1:
                h = jax.nn.silu(h)
        values.append(h[:, 0])
    return jnp.stack(values, -1)


class Learner:
    def __init__(self, flow, feature_dim, config):
        self.flow, self.config = flow, config
        self.r = config["rldt"]
        self.steps = int(self.r["ode_steps"])
        self.k = int(self.r["particles"])
        self.execute = int(self.r["execute_steps"])
        if not 1 <= self.execute <= flow.horizon:
            raise ValueError("execute_steps must fit the checkpoint action horizon")
        self.mask = jnp.zeros((flow.horizon, flow.action_dim)).at[
            :self.execute, :flow.motor_dim].set(1)
        self.key = jax.random.PRNGKey(int(config["train"]["seed"]))
        self.params = flow.params
        self.reference = jax.tree.map(lambda x: x.copy(), self.params)
        warmup = int(self.r.get("actor_warmup_steps", 32))
        lr = float(self.r["actor_lr"])
        schedule = optax.linear_schedule(lr / max(1, warmup), lr, max(1, warmup))
        self.actor_tx = optax.chain(optax.clip_by_global_norm(
            float(self.r.get("max_grad_norm", 1))), optax.adam(schedule))
        self.actor_opt = self.actor_tx.init(self.params)
        self.q = init_critic(self.next_key(), feature_dim, self.execute * flow.motor_dim,
                             int(self.r["critic_width"]))
        self.target_q = jax.tree.map(lambda x: x.copy(), self.q)
        self.q_tx = optax.adam(float(self.r["critic_lr"]))
        self.q_opt = self.q_tx.init(self.q)
        # Pass even frozen weights dynamically: closing over a multi-GB VLM or
        # frozen expert would embed its arrays as enormous XLA constants.
        sample_jit = jax.jit(lambda p, c, k, f: flow.sample(
            p, c, k, self.steps, frozen=f))
        self.sample = lambda p, c, k: sample_jit(p, c, k, flow.frozen)
        velocity_jit = jax.jit(lambda p, c, x, t, f: flow.velocity(p, c, x, t, frozen=f))
        self.velocity = lambda p, c, x, t: velocity_jit(p, c, x, t, flow.frozen)
        self._actor_update = jax.jit(self._update_actor)
        self._critic_update = jax.jit(self._update_critic)

    def next_key(self):
        self.key, key = jax.random.split(self.key)
        return key

    def active(self, actions):
        return actions[..., :self.execute, :self.flow.motor_dim]

    def _update_critic(self, q, target, state, data, indices):
        next_q = critic(target, data["next_features"], data["next_actions"])
        values = next_q.min(-1).reshape((-1, int(self.r["bootstrap_samples"]))).mean(-1)
        if self.r.get("clip_bootstrap", True):
            values = jnp.clip(values, 0, 1)
        returns = jax.lax.stop_gradient(lambda_returns(
            data["rewards"], data["discounts"], values, data["trace"], self.r["td_lambda"]))

        def loss_fn(params):
            qs = critic(params, data["features"][indices], data["actions"][indices])
            return jnp.mean(jnp.square(qs - returns[indices, None]))

        loss, grads = jax.value_and_grad(loss_fn)(q)
        updates, state = self.q_tx.update(grads, state, q)
        q = optax.apply_updates(q, updates)
        rho = float(self.r["target_momentum"])
        target = jax.tree.map(lambda old, new: rho * old + (1 - rho) * new, target, q)
        return q, target, state, {"critic_loss": loss, "target_mean": returns.mean()}

    def _update_actor(self, params, state, context, x, times, clean, field,
                      constraint_x, constraint_t, reference_v, frozen):
        batch = x.shape[0] // self.k

        def loss_fn(p):
            v = self.flow.velocity(p, context, x, times, frozen=frozen)
            cv = self.flow.velocity(p, context, constraint_x, constraint_t, frozen=frozen)
            shape = (batch, self.k, self.flow.horizon, self.flow.action_dim)
            return actor_loss(
                v.reshape(shape), x.reshape(shape), times.reshape(batch, self.k),
                clean.reshape(shape), field.reshape(shape), reference_v.reshape(shape),
                num_steps=self.steps,
                consistency_weight=float(self.r["consistency_weight"]),
                constraint_weight=float(self.r["constraint_weight"]),
                action_mask=self.mask, constraint_velocity=cv.reshape(shape),
            )

        (_, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        updates, state = self.actor_tx.update(grads, state, params)
        params = optax.apply_updates(params, updates)
        return params, state, metrics | {"actor_grad_norm": optax.global_norm(grads),
                                        "actor_update_norm": optax.global_norm(updates)}

    def actor_step(self, observations):
        base_context = self.flow.prepare(observations)
        context = self.flow.repeat_context(base_context, self.k)
        clean, path = self.sample(self.params, context, self.next_key())
        count = clean.shape[0]
        indices = jax.random.randint(self.next_key(), (count,), 0, self.steps)
        times = 1 - indices.astype(jnp.float32) / self.steps
        x = path[indices, jnp.arange(count)]
        v = self.velocity(self.params, context, x, times)
        queries = x - times[:, None, None] * v
        actions = self.active(clean)
        features = context["features"]
        # Ensemble mean for transport, unlike the target critic minimum.
        grad = jax.grad(lambda a: critic(self.q, features, a).mean(-1).sum())(actions)
        shape = (len(observations), self.k, -1)
        field, transport_metrics = svgd_field(
            actions.reshape(shape), self.active(queries).reshape(shape), grad.reshape(shape),
            kappa0=float(self.r["kappa0"]), kernel_mode=self.r.get("kernel_mode", "official"),
            bandwidth_floor=1e-4, temperature_mode="scaled",
        )
        full_field = jnp.zeros_like(clean).at[:, :self.execute, :self.flow.motor_dim].set(
            field.reshape(actions.shape))
        constraint_t = 1 - jax.random.randint(
            self.next_key(), (count,), 0, self.steps).astype(jnp.float32) / self.steps
        noise = jax.random.normal(self.next_key(), clean.shape)
        constraint_x = constraint_t[:, None, None] * noise + (
            1 - constraint_t[:, None, None]) * clean
        ref_v = self.velocity(self.reference, context, constraint_x, constraint_t)
        self.params, self.actor_opt, metrics = self._actor_update(
            self.params, self.actor_opt, context, x, times, clean, full_field,
            constraint_x, constraint_t, ref_v, self.flow.frozen,
        )
        return metrics | transport_metrics

    def critic_data(self, segments):
        """Cache current-policy bootstrap actions once per outer iteration."""
        rows, trace = [], []
        for segment in segments:
            rows.extend(segment)
            trace.extend([1.] * (len(segment) - 1) + [0.])
        b = int(self.r["bootstrap_samples"])
        next_features, next_actions = [], []
        for row in rows:
            if row["terminated"]:
                next_features.extend([row["features"]] * b)
                next_actions.extend([np.zeros_like(row["action"])] * b)
                continue
            context = self.flow.repeat_context(self.flow.prepare([row["next_obs"]]), b)
            actions, _ = self.sample(self.params, context, self.next_key())
            next_features.extend(np.asarray(context["features"]))
            next_actions.extend(np.asarray(self.active(actions)))
        return {k: jnp.asarray(v) for k, v in {
            "features": np.stack([r["features"] for r in rows]),
            "actions": np.stack([r["action"] for r in rows]),
            "rewards": np.array([r["reward"] for r in rows], np.float32),
            "discounts": np.array([r["discount"] for r in rows], np.float32),
            "trace": np.array(trace, np.float32),
            "next_features": np.stack(next_features), "next_actions": np.stack(next_actions),
        }.items()}

    def state(self):
        return (self.params, self.reference, self.q, self.target_q,
                self.actor_opt, self.q_opt, self.key)

    def save(self, directory):
        self.flow.save(directory, self.params)
        leaves, _ = jax.tree.flatten(self.state())
        np.savez(Path(directory) / "learner.npz",
                 **{f"arr_{i}": np.asarray(x) for i, x in enumerate(leaves)})

    def restore(self, directory):
        manifest_path = Path(directory).parent / "manifest.json"
        if manifest_path.exists() and hasattr(self.flow, "metadata"):
            saved = json.loads(manifest_path.read_text()).get("policy_metadata", {})
            for key in ("model_config", "projection_only", "checkpoint_metadata_sha256",
                        "checkpoint_manifest_sha256", "normalization_assets_sha256"):
                if saved.get(key) != self.flow.metadata.get(key):
                    raise ValueError(f"Warm-start base checkpoint {key} does not match")
        template, tree = jax.tree.flatten(self.state())
        with np.load(Path(directory) / "learner.npz", allow_pickle=False) as data:
            if len(data.files) != len(template):
                raise ValueError("Checkpoint architecture/config mismatch")
            leaves = [jnp.asarray(data[f"arr_{i}"], dtype=x.dtype)
                      for i, x in enumerate(template)]
        if any(x.shape != y.shape for x, y in zip(template, leaves, strict=True)):
            raise ValueError("Checkpoint shapes do not match")
        (self.params, self.reference, self.q, self.target_q,
         self.actor_opt, self.q_opt, self.key) = jax.tree.unflatten(tree, leaves)


def execute_chunk(env, actions, gamma):
    actions = np.asarray(actions, np.float32)
    if not np.isfinite(actions).all():
        raise FloatingPointError("Nonfinite policy actions")
    reward, raw_return = 0., 0.
    for i, action in enumerate(actions):
        raw, r, terminated, truncated, info = env.step(np.clip(action, env.low, env.high))
        if not np.isfinite(r):
            raise FloatingPointError("Nonfinite environment reward")
        if "final_observation" in info or "final_obs" in info:
            raise ValueError("Auto-reset environments are unsupported")
        reward += gamma**i * r
        raw_return += r
        if terminated or truncated:
            break
    count = i + 1
    return raw, reward, terminated, truncated, dict(info) | {
        "executed_steps": count, "raw_return": raw_return,
        "discount": gamma**count * (1 - terminated),
    }


def train(config, output, init_from=None):
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "config.json", config)
    print("Loading flow policy...", flush=True)
    flow = make_flow(config["policy"])
    settings, r = config["train"], config["rldt"]
    seed = int(settings["seed"])
    print("Starting primitive environment...", flush=True)
    env = make_primitive_env(config["env"])
    start = time.monotonic()
    counters = {"iterations": 0, "decisions": 0, "primitive_steps": 0,
                "episodes": 0, "successes": 0, "actor_updates": 0, "critic_updates": 0}
    try:
        raw, _ = env.reset(seed=seed)
        print("Encoding initial observation and initializing optimizers...", flush=True)
        context = flow.prepare([raw])
        learner = Learner(flow, context["features"].shape[-1], config)
        if init_from:
            saved_config = json.loads((Path(init_from).parent / "config.json").read_text())
            for section in ("policy", "env", "rldt"):
                if saved_config[section] != config[section]:
                    raise ValueError(f"Warm-start {section} differs from saved run")
            learner.restore(init_from)
        manifest = {"config": config, "jax": jax.__version__,
                    "devices": [str(d) for d in jax.devices()], "init_from": init_from,
                    "trainable_parameters": sum(x.size for x in jax.tree.leaves(learner.params)),
                    "frozen_vlm": config["policy"]["kind"] == "pi05",
                    "policy_metadata": getattr(flow, "metadata", {}),
                    "paper": "rldt.pdf", "official_reference_commit":
                    "3ab73e775efd0f041ad28170c616146be86bb6ea"}
        try:
            manifest["openpi_commit"] = subprocess.check_output(
                ["git", "-C", "third_party/openpi", "rev-parse", "HEAD"], text=True).strip()
        except (OSError, subprocess.CalledProcessError):
            manifest["openpi_commit"] = None
        write_json(out / "manifest.json", manifest)
        replay = TrajectoryReplay(int(settings["replay_capacity"]), seed)
        rng = np.random.default_rng(seed)
        with (out / "metrics.jsonl").open("w") as log:
            for iteration in range(1, int(settings["iterations"]) + 1):
                print(f"RLDT iteration {iteration}: collecting rollouts", flush=True)
                fresh, segment = [], []
                for _ in range(int(settings["collect_decisions"])):
                    context = flow.prepare([raw])
                    actions, _ = learner.sample(learner.params, context, learner.next_key())
                    motor_actions = flow.decode(actions, [raw])[0, :learner.execute]
                    following, reward, terminated, truncated, info = execute_chunk(
                        env, motor_actions, float(r["gamma"]))
                    segment.append({"obs": raw, "next_obs": following,
                        "features": np.asarray(context["features"])[0],
                        "action": np.asarray(learner.active(actions))[0], "reward": reward,
                        "discount": info["discount"], "terminated": terminated,
                        "truncated": truncated, "executed_steps": info["executed_steps"]})
                    counters["decisions"] += 1
                    counters["primitive_steps"] += info["executed_steps"]
                    if terminated or truncated:
                        replay.add(segment)
                        fresh.append(segment)
                        segment = []
                        counters["episodes"] += 1
                        counters["successes"] += int(info.get("is_success", False))
                        raw, _ = env.reset(seed=seed + counters["episodes"])
                    else:
                        raw = following
                if segment:
                    replay.add(segment)
                    fresh.append(segment)
                segments = replay.sample(int(settings["critic_trajectories"]),
                    balanced=r.get("balanced_replay", True), fresh=fresh,
                    fresh_fraction=float(r.get("fresh_fraction", .33)))
                print(f"RLDT iteration {iteration}: preparing critic targets", flush=True)
                data = learner.critic_data(segments)
                metrics = {}
                for _ in range(int(settings["critic_steps"])):
                    indices = jnp.asarray(rng.integers(
                        len(data["rewards"]), size=int(settings["critic_batch_size"])))
                    learner.q, learner.target_q, learner.q_opt, metrics = learner._critic_update(
                        learner.q, learner.target_q, learner.q_opt, data, indices)
                    counters["critic_updates"] += 1
                if iteration > int(r.get("critic_warmup_iterations", 2)):
                    print(f"RLDT iteration {iteration}: updating flow policy", flush=True)
                    rows = [row for seg in segments for row in seg]
                    for _ in range(int(settings["actor_steps"])):
                        observations = [rows[i]["obs"] for i in rng.integers(
                            len(rows), size=int(settings["actor_batch_size"]))]
                        metrics = metrics | learner.actor_step(observations)
                        counters["actor_updates"] += 1
                if r.get("reference", "moving") == "moving":
                    learner.reference = jax.tree.map(lambda x: x.copy(), learner.params)
                metrics = {k: float(v) for k, v in metrics.items()}
                if not all(np.isfinite(v) for v in metrics.values()):
                    raise FloatingPointError(f"Nonfinite training metrics: {metrics}")
                counters["iterations"] = iteration
                record = counters | {"elapsed_seconds": time.monotonic() - start,
                                      "replay_decisions": replay.size, "metrics": metrics}
                log.write(json.dumps(record, allow_nan=False) + "\n")
                log.flush()
                print(json.dumps(record, allow_nan=False), flush=True)
                if iteration % int(settings["save_every"]) == 0:
                    learner.save(out / f"iteration_{iteration:06d}")
        learner.save(out / "final")
        summary = counters | {"elapsed_seconds": time.monotonic() - start,
                              "checkpoint": str(out / "final")}
        write_json(out / "summary.json", summary)
        return summary
    finally:
        env.close()


def evaluate(config, output, checkpoint=None, episodes=5, seed=100000):
    path = Path(output)
    if path.exists() or path.with_suffix(".episodes.jsonl").exists():
        raise FileExistsError(path)
    print("Loading flow policy and evaluation checkpoint...", flush=True)
    flow = make_flow(config["policy"])
    params = flow.load_params(checkpoint) if checkpoint else flow.params
    env = make_primitive_env(config["env"])
    r = config["rldt"]
    sample = jax.jit(lambda p, c, key, frozen: flow.sample(
        p, c, key, int(r["ode_steps"]), frozen=frozen)[0])
    results = []
    path.parent.mkdir(parents=True, exist_ok=True)
    episode_log = path.with_suffix(".episodes.jsonl").open("x")
    try:
        for episode in range(episodes):
            print(f"Evaluation episode {episode + 1}/{episodes}: resetting environment", flush=True)
            # Match each episode's noise independently of earlier episode lengths.
            key = jax.random.PRNGKey(seed + episode)
            raw, reset_info = env.reset(seed=seed + episode)
            steps, total = 0, 0.
            for decision in range(10000):
                key, subkey = jax.random.split(key)
                actions = sample(params, flow.prepare([raw]), subkey, flow.frozen)
                motor = flow.decode(actions, [raw])[0, :int(r["execute_steps"])]
                raw, _, terminated, truncated, info = execute_chunk(env, motor, float(r["gamma"]))
                steps += info["executed_steps"]
                total += info["raw_return"]
                if terminated or truncated:
                    break
                if (decision + 1) % 10 == 0:
                    print(f"Evaluation episode {episode + 1}: {steps} primitive steps", flush=True)
            else:
                raise RuntimeError("Evaluation exceeded 10000 decisions without episode boundary")
            results.append({"seed": seed + episode, "reset": reset_info,
                            "success": bool(info.get("is_success", False)),
                            "return": total, "primitive_steps": steps})
            episode_log.write(json.dumps(results[-1], allow_nan=False) + "\n")
            episode_log.flush()
            print(json.dumps(results[-1]), flush=True)
        result = {"config": config, "checkpoint": checkpoint, "episodes": results,
                  "success_rate": float(np.mean([e["success"] for e in results]))}
        path.parent.mkdir(parents=True, exist_ok=True)
        write_json(path, result)
        return result
    finally:
        episode_log.close()
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["train", "evaluate", "preflight", "export"])
    parser.add_argument("--config", required=True)
    parser.add_argument("--output")
    parser.add_argument("--checkpoint")
    parser.add_argument("--init-from", help="Warm-start learner; starts fresh replay and environments")
    parser.add_argument("--smoke", action="store_true", help="Tiny real updates; no performance claim")
    parser.add_argument("--particles", type=int, help="Override SVGD particle count (at least 2)")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=100000)
    args = parser.parse_args()
    config = load_config(args.config)
    if args.smoke:
        config = copy.deepcopy(config)
        config["train"].update(iterations=1, collect_decisions=2, critic_steps=1, actor_steps=1,
                               critic_batch_size=2, actor_batch_size=1, critic_trajectories=1)
        config["rldt"].update(particles=2, bootstrap_samples=1, critic_warmup_iterations=0)
    if args.particles is not None:
        if args.particles < 2:
            parser.error("--particles must be at least 2")
        config["rldt"]["particles"] = args.particles
    if args.command == "preflight":
        flow = make_flow(config["policy"])
        result = {"devices": [str(d) for d in jax.devices()], "horizon": flow.horizon,
                  "internal_action_dim": flow.action_dim, "motor_dim": flow.motor_dim,
                  "trainable_parameters": sum(x.size for x in jax.tree.leaves(flow.params))}
    else:
        if not args.output:
            parser.error("--output is required")
        if args.command == "train":
            result = train(config, args.output, args.init_from)
        elif args.command == "evaluate":
            if args.episodes < 1:
                parser.error("--episodes must be positive")
            result = evaluate(config, args.output, args.checkpoint, args.episodes, args.seed)
        else:
            if not args.checkpoint or config["policy"]["kind"] != "pi05":
                parser.error("export requires --checkpoint and policy.kind=pi05")
            flow = make_flow(config["policy"])
            flow.export_native(args.output, flow.load_params(args.checkpoint))
            result = {"native_checkpoint": args.output}
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
