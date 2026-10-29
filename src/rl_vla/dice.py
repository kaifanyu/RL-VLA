"""Standalone DICE pi0.5 runner. Default command checks setup only.

Training requires the explicit `train` subcommand; no existing CLI is changed.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
import tomllib
from collections import deque
from pathlib import Path

import numpy as np

UPSTREAM = "98d936e52523878f0172e606ea32403176497100"


def config_from(path):
    with open(path, "rb") as stream:
        c = tomllib.load(stream)
    if c.get("algorithm") != "dice":
        raise ValueError("Expected a separate algorithm='dice' configuration")
    for section in ("policy", "env", "train", "dice", "data"):
        if section not in c:
            raise ValueError(f"Missing [{section}]")
    if c["policy"]["model_config"] != "pi05_libero":
        raise ValueError("Only pi05_libero observation/action mapping is supported")
    d, t = c["dice"], c["train"]
    if type(t["replay_max_mb"]) is not int or t["replay_max_mb"] < 1:
        raise ValueError("replay_max_mb must be a positive integer")
    for key in ("execute_steps", "ode_steps", "particles", "selection_samples", "width"):
        if type(d[key]) is not int or d[key] < 1:
            raise ValueError(f"dice.{key} must be a positive integer")
    for key in ("episodes", "batch_size", "replay_capacity", "updates_per_episode", "save_every"):
        if type(t[key]) is not int or t[key] < 1:
            raise ValueError(f"train.{key} must be a positive integer")
    for key in ("actor_lr", "critic_lr", "max_grad_norm"):
        if not np.isfinite(d[key]) or d[key] <= 0:
            raise ValueError(f"Invalid dice.{key}")
    if not 0 < d["gamma"] <= 1 or not 0 < d["target_tau"] <= 1:
        raise ValueError("gamma/target_tau must lie in (0,1]")
    if not np.isfinite(d["bc_weight"]) or d["bc_weight"] < 0:
        raise ValueError("Invalid bc_weight")
    if not np.isfinite(d["underestimation_threshold"]) or d["underestimation_threshold"] > 0:
        raise ValueError("Underestimation threshold must be finite and nonpositive")
    for key in ("filter_warmup", "selection_warmup"):
        if type(d[key]) is not int or d[key] < 0:
            raise ValueError(f"Invalid dice.{key}")
    if d["execute_steps"] > 10 or t["batch_size"] > t["replay_capacity"]:
        raise ValueError("Invalid chunk horizon or replay capacity")
    if c["data"]["mode"] not in ("demonstrations", "online_only_ablation"):
        raise ValueError("Explicitly select demonstrations or online_only_ablation")
    if not 0 < c["data"].get("expert_fraction", 0.5) < 1:
        raise ValueError("expert_fraction must lie in (0,1)")
    end = c["data"].get("expert_fraction_end", c["data"].get("expert_fraction", 0.5))
    duration = c["data"].get("expert_fraction_decay_decisions", 1)
    if not np.isfinite(end) or not 0 <= end <= 1:
        raise ValueError("expert_fraction_end must lie in [0,1]")
    if type(duration) is not int or duration < 1:
        raise ValueError("expert_fraction_decay_decisions must be a positive integer")
    if d.get("selection_strategy", "max_q_min") not in ("max_q_min", "max_q_std"):
        raise ValueError("Invalid action selection strategy")
    return c


def check(c):
    """Read-only check: no JAX initialization, model load, reset, or optimizer."""
    checkpoint = Path(c["policy"]["checkpoint"])
    missing = [
        str(checkpoint / p) for p in ("params/_METADATA", "assets") if not (checkpoint / p).exists()
    ]
    python = Path(c["env"]["python"])
    if not python.exists():
        missing.append(str(python))
    revisions = {}
    for name, expected in (
        ("dice_rl_upstream", UPSTREAM),
        ("openpi", "215abfb217dbac7d5f1273282331b9b1866c0479"),
    ):
        checkout = Path("third_party") / name
        if not checkout.exists():
            missing.append(str(checkout))
            continue
        revision = subprocess.check_output(
            ["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True
        ).strip()
        revisions[name] = revision
        if revision != expected:
            missing.append(f"{name}: expected {expected}, found {revision}")
    data_ready = (
        c["data"]["mode"] == "online_only_ablation"
        or Path(c["data"].get("path", "MISSING")).is_file()
    )
    result = {
        "setup_ok": not missing,
        "missing": missing,
        "training_data_ready": data_ready,
        "data_mode": c["data"]["mode"],
        "upstream_commit": revisions.get("dice_rl_upstream"),
        "openpi_commit": revisions.get("openpi"),
        "checkpoint": str(checkpoint.resolve()),
        "training_started": False,
    }
    print(json.dumps(result, indent=2))
    return result


def resource_check():
    """Host-relative limits; GPU query respects the scheduler's visible devices."""
    from rl_vla.dice_runtime import check_resources

    check_resources()


def tensor(x):
    import torch

    return torch.as_tensor(np.asarray(x).copy(), dtype=torch.float32)


def proposals(prior, observations, count):
    records = [prior.propose(raw, count)[:3] for raw in observations]
    return tuple(tensor(np.stack([record[i] for record in records])) for i in range(3))


def replay_bytes(value):
    if isinstance(value, np.ndarray):
        return value.nbytes
    if isinstance(value, dict):
        return sum(replay_bytes(x) for x in value.values())
    if isinstance(value, (list, tuple)):
        return sum(replay_bytes(x) for x in value)
    return 0


def load_demonstrations(c, prior):
    """Explicit normalized transition format, not arbitrary pickle/checkpoints."""
    if c["data"]["mode"] == "online_only_ablation":
        return []
    from openpi_client import msgpack_numpy

    path = Path(c["data"]["path"])
    if path.stat().st_size > c["train"]["replay_max_mb"] * 1024**2:
        raise ValueError("Demonstration archive exceeds configured replay byte limit")
    payload = msgpack_numpy.unpackb(path.read_bytes())
    if payload["format"] != "dice-pi05-transitions-v1":
        raise ValueError("Unsupported demonstration format; see docs/dice.md")
    for key in ("checkpoint_metadata_sha256", "normalization_assets_sha256"):
        if payload["metadata"].get(key) != prior.metadata[key]:
            raise ValueError("Demonstration normalization/checkpoint fingerprint mismatch")
    if payload["metadata"].get("execute_steps") != prior.execute_steps:
        raise ValueError("Demonstration execution horizon mismatch")
    if payload["metadata"].get("gamma") != c["dice"]["gamma"]:
        raise ValueError("Demonstration return discount mismatch")
    records = payload["transitions"]
    if not records:
        raise ValueError("Demonstration replay is empty")
    for r in records:
        for key in ("observation", "next_observation"):
            if not isinstance(r[key], dict) or "openpi" not in r[key]:
                raise ValueError("Demonstrations require images, proprioception, and prompt")
        a = np.asarray(r["actions"], np.float32)
        if a.shape != (prior.action_dim,) or not np.isfinite(a).all():
            raise ValueError("Expected flattened normalized executed action prefix")
        for key in ("rewards", "returns", "discounts"):
            if not np.isfinite(r[key]):
                raise ValueError("Non-finite demonstration reward/return/discount")
        if not 0 <= r["discounts"] <= 1:
            raise ValueError("Invalid demonstration discount")
        r["features"] = prior.propose(r["observation"], 1)[0][0]
        r["expert"] = 1.0
    if replay_bytes(records) > c["train"]["replay_max_mb"] * 1024**2:
        raise ValueError("Demonstration replay exceeds byte limit")
    return records


def save(path, learner, c, prior, counters):
    import torch

    temporary = path.with_suffix(".tmp")
    torch.save(
        {
            "format": "dice-pi05-v1",
            "config": c,
            "prior": prior.metadata,
            "counters": counters,
            "learner": learner.state_dict(),
        },
        temporary,
    )
    temporary.replace(path)


def run(c, output, *, validate=False, resume=None):
    import torch

    from rl_vla.dice_core import DICE, discounted_returns
    from rl_vla.dice_pi05 import FrozenPi05
    from rl_vla.dice_runtime import expert_fraction, save_training
    from rl_vla.rldt_env import make_primitive_env

    setup = check(c)
    if setup is not None and not setup["setup_ok"]:
        raise RuntimeError("DICE setup is incomplete; see check report")
    if (
        not validate
        and c["data"]["mode"] == "demonstrations"
        and not Path(c["data"]["path"]).is_file()
    ):
        raise FileNotFoundError("Demonstration replay is required; see docs/dice.md")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.manual_seed(c["train"]["seed"])
    rng = np.random.default_rng(c["train"]["seed"])
    resource_check()
    prior = FrozenPi05(c["policy"], {**c["dice"], "seed": c["train"]["seed"]})
    env = make_primitive_env(c["env"])
    replay = deque(maxlen=c["train"]["replay_capacity"])
    decisions = 0
    try:
        if validate:
            raw, _ = env.reset(seed=c["train"]["seed"])
            f, z, p, full = prior.propose(raw, 1)
            # No learner or optimizer is constructed during policy validation.
            decoded = prior.decode(p[0], full[0], raw)
            expected = prior.flow.decode(full, [raw])[0, : prior.execute_steps]
            if not np.array_equal(decoded, expected):
                raise RuntimeError("Zero residual changes the base policy action")
            result = {
                "policy_validation": "passed",
                "features_shape": list(f.shape),
                "noise_shape": list(z.shape),
                "actions_shape": list(decoded.shape),
                "environment_steps": 0,
                "optimizer_updates": 0,
                "prior": prior.metadata,
            }
            (output / "validation.json").write_text(json.dumps(result, indent=2) + "\n")
            print(json.dumps(result, indent=2))
            return
        learner = DICE(prior.feature_dim, prior.noise_dim, prior.action_dim, c["dice"])
        demos = load_demonstrations(c, prior)
        start_episode = 0
        if resume:
            from rl_vla.dice_runtime import restore_training

            start_episode, decisions = restore_training(resume, learner, c, prior, replay, rng)
        metadata = {
            "config": c,
            "prior": prior.metadata,
            "upstream": UPSTREAM,
            "adaptation": "frozen-VLM features; normalized executed-prefix; finite-horizon MC",
            "demonstrations": len(demos),
            "torch": torch.__version__,
        }
        (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        for episode in range(start_episode, c["train"]["episodes"]):
            raw, reset_info = env.reset(seed=c["train"]["seed"] + episode)
            primitive_steps, clipped = 0, 0
            trajectory, terminated, truncated = [], False, False
            while not (terminated or truncated):
                resource_check()
                count = (
                    1
                    if decisions <= c["dice"]["selection_warmup"]
                    else c["dice"]["selection_samples"]
                )
                f, z, p, full = prior.propose(raw, count)
                action, index = learner.select(
                    tensor(f),
                    tensor(z),
                    tensor(p),
                    strategy=c["dice"].get("selection_strategy", "max_q_min"),
                )
                chunk = prior.decode(action.numpy(), full[index], raw)
                reward, n = 0.0, 0
                for primitive in chunk:
                    command = np.clip(primitive, env.low, env.high)
                    clipped += int(np.any(command != primitive))
                    next_raw, r, terminated, truncated, _info = env.step(command)
                    primitive_steps += 1
                    reward += c["dice"]["gamma"] ** n * r
                    n += 1
                    if terminated or truncated:
                        break
                trajectory.append(
                    {
                        "observation": raw,
                        "next_observation": next_raw,
                        "features": f[index],
                        "actions": action.numpy().copy(),
                        "rewards": reward,
                        "discounts": 0.0 if terminated or truncated else c["dice"]["gamma"] ** n,
                        "expert": 0.0,
                    }
                )
                raw = next_raw
                decisions += 1
                if (
                    replay_bytes(trajectory) + replay_bytes(demos)
                    > c["train"]["replay_max_mb"] * 1024**2
                ):
                    raise RuntimeError("Current episode exceeds replay memory budget")
            returns = discounted_returns(
                [r["rewards"] for r in trajectory], [r["discounts"] for r in trajectory]
            )
            for record, value in zip(trajectory, returns, strict=True):
                record["returns"] = value
                replay.append(record)
            while (
                replay
                and replay_bytes(list(replay)) + replay_bytes(demos)
                > c["train"]["replay_max_mb"] * 1024**2
            ):
                replay.popleft()
            last = {}
            if len(replay) >= c["train"]["batch_size"]:
                for _ in range(c["train"]["updates_per_episode"]):
                    resource_check()
                    ratio = expert_fraction(c["data"], decisions)
                    rows = []
                    for _ in range(c["train"]["batch_size"]):
                        pool = demos if demos and rng.random() < ratio else replay
                        rows.append(pool[int(rng.integers(len(pool)))])
                    batch = {
                        key: tensor(np.stack([row[key] for row in rows]))
                        for key in (
                            "features",
                            "actions",
                            "rewards",
                            "discounts",
                            "returns",
                            "expert",
                        )
                    }
                    prop = proposals(
                        prior, [r["observation"] for r in rows], c["dice"]["particles"]
                    )
                    next_prop = proposals(
                        prior, [r["next_observation"] for r in rows], c["dice"]["particles"]
                    )
                    last = learner.update(batch, prop, next_prop, step=decisions)
            counters = {"episodes": episode + 1, "decisions": decisions, "updates": learner.updates}
            with (output / "metrics.jsonl").open("a") as stream:
                stream.write(
                    json.dumps(
                        {
                            **counters,
                            **last,
                            "success": bool(_info.get("is_success", terminated)),
                            "initial_state_id": reset_info.get("initial_state_id"),
                            "primitive_steps": primitive_steps,
                            "clipped_commands": clipped,
                            "expert_fraction": expert_fraction(c["data"], decisions),
                            "return": returns[0],
                            "timestamp": time.time(),
                        }
                    )
                    + "\n"
                )
            if (episode + 1) % c["train"]["save_every"] == 0:
                save(output / "latest.pt", learner, c, prior, counters)
                save_training(output / "resume.pt", learner, c, prior, counters, replay, rng)
        save(output / "final.pt", learner, c, prior, counters)
        save_training(output / "resume.pt", learner, c, prior, counters, replay, rng)
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("check", "validate-policy", "train", "evaluate"),
        nargs="?",
        default="check",
    )
    parser.add_argument("--config", default="configs/dice_pi05_libero.toml")
    parser.add_argument("--output")
    parser.add_argument("--checkpoint", help="DICE checkpoint for evaluation")
    parser.add_argument("--resume", help="Trusted local episode-boundary resume.pt")
    parser.add_argument("--baseline", action="store_true", help="Evaluate the unedited prior")
    parser.add_argument("--episodes", type=int, default=50, help="Evaluation episode count")
    parser.add_argument("--seed", type=int, default=100000, help="First evaluation reset seed")
    args = parser.parse_args()
    c = config_from(args.config)
    if args.command == "check":
        result = check(c)
        if not result["setup_ok"]:
            raise SystemExit(1)
    else:
        if not args.output:
            parser.error("--output must be a new directory")
        if args.command == "evaluate":
            from rl_vla.dice_eval import evaluate

            if bool(args.checkpoint) == args.baseline:
                parser.error("evaluate requires exactly one of --checkpoint or --baseline")
            evaluate(
                c, args.output, checkpoint=args.checkpoint, episodes=args.episodes, seed=args.seed
            )
        else:
            if args.resume and args.command != "train":
                parser.error("--resume is only valid for train")
            run(c, args.output, validate=args.command == "validate-policy", resume=args.resume)


if __name__ == "__main__":
    main()
