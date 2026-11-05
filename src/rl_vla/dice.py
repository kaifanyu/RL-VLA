"""Standalone DICE pi0.5 runner. Default command checks setup only.

Training requires the explicit `train` subcommand; no existing CLI is changed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
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
    if c.get("official_code_commit", UPSTREAM) != UPSTREAM:
        raise ValueError("Official source pin differs from the audited implementation")
    for section in ("policy", "env", "train", "dice", "data"):
        if section not in c:
            raise ValueError(f"Missing [{section}]")
    if c["policy"]["model_config"] != "pi05_libero":
        raise ValueError("Only pi05_libero observation/action mapping is supported")
    if c["policy"].get("lora_rank", 0):
        raise ValueError("DICE freezes the complete prior; LoRA policies are a separate adaptation")
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
    for key in (
        "critic_ensemble_size",
        "k_train",
        "k_bootstrap",
        "k_collection",
        "actor_update_freq",
        "target_update_freq",
        "n_step",
    ):
        if key in d and (type(d[key]) is not int or d[key] < 1):
            raise ValueError(f"Invalid dice.{key}")
    if d.get("calibration_mode", "replay") not in ("replay", "candidate"):
        raise ValueError("Invalid calibration_mode")
    if d.get("discount_mode", "primitive") not in ("primitive", "chunk"):
        raise ValueError("Invalid discount_mode")
    if d.get("delay_clock", "optimizer") not in ("optimizer", "collection"):
        raise ValueError("Invalid delay_clock")
    if d.get("activation", "Mish") not in ("Mish", "GELU"):
        raise ValueError("Invalid activation")
    if not np.isfinite(d.get("reward_scale", 1.0)) or d.get("reward_scale", 1.0) <= 0:
        raise ValueError("Invalid reward_scale")
    for key in ("gradient_steps_per_chunk", "max_steps"):
        if key in t and (type(t[key]) is not int or t[key] < 1):
            raise ValueError(f"Invalid train.{key}")
    if (
        type(t.get("reference_parallel_envs", 1)) is not int
        or t.get("reference_parallel_envs", 1) < 1
    ):
        raise ValueError("reference_parallel_envs must be positive")
    if d.get("lr_scheduler", False) and not (
        0 <= d["lr_warmup_steps"] < d["lr_cycle_steps"]
        and 0 <= d["min_lr"] <= min(d["actor_lr"], d["critic_lr"])
    ):
        raise ValueError("Invalid learning-rate schedule")
    if "profile" in c:
        for old, new in (("particles", "k_train"), ("selection_samples", "k_collection")):
            if d[old] != d[new]:
                raise ValueError(f"Legacy {old} must match explicit {new}")
    evaluation_k = c.get("evaluation", {}).get("k", 1)
    if type(evaluation_k) is not int or evaluation_k < 1:
        raise ValueError("Evaluation K must be a positive integer")
    if "profile" in c:
        required_labels = {
            "no_demonstration_replay": c["data"]["mode"] == "online_only_ablation",
            "no_filtered_regularization": not d.get("filter_enabled", True),
            "no_residual_regularization": d["bc_weight"] == 0,
            "single_sample_training": d["k_train"] == 1,
            "single_sample_bootstrap": d["k_bootstrap"] == 1,
            "no_value_guided_collection": d["k_collection"] == 1,
        }
        for label, required in required_labels.items():
            if required and label not in c.get("ablations", []):
                raise ValueError(f"Disabled component requires ablation label: {label}")
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
    if hasattr(prior, "propose_many"):
        return tuple(tensor(x) for x in prior.propose_many(observations, count))
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


def validate_replay_provenance(provenance, expected):
    """Require complete replay lineage and match it to the selected environment."""
    if not isinstance(provenance, dict) or not isinstance(expected, dict):
        raise TypeError("Demonstration provenance is missing or malformed")
    required = {
        "task": ("task_key", "task_suite", "task_id", "instruction"),
        "environment": ("adapter", "split"),
        "demonstration": ("source_id", "source_split"),
    }
    for section, keys in required.items():
        actual_section = provenance.get(section)
        expected_section = expected.get(section)
        if not isinstance(actual_section, dict) or not isinstance(expected_section, dict):
            raise TypeError(f"Demonstration provenance is missing {section} identity")
        optional = ("protocol_id", "protocol_sha256", "protocol_version")
        checked_keys = keys + tuple(
            key for key in optional if key in actual_section or key in expected_section
        )
        for key in checked_keys:
            actual = actual_section.get(key)
            target = expected_section.get(key)
            if key not in actual_section or key not in expected_section:
                raise ValueError(f"Demonstration provenance is unverifiable: {section}.{key}")
            if actual != target:
                label = "task identity" if section == "task" else f"{section} identity"
                raise ValueError(
                    f"Demonstration {label} mismatch at {key}: {actual!r} != {target!r}"
                )
    if (
        not provenance["task"]["task_key"]
        or not provenance["task"]["task_suite"]
        or type(provenance["task"]["task_id"]) is not int
        or provenance["task"]["task_id"] < 0
    ):
        raise ValueError("Demonstration task identity is invalid")
    if not provenance["task"]["instruction"]:
        raise ValueError("Demonstration instruction is missing")
    if not provenance["environment"]["adapter"]:
        raise ValueError("Demonstration environment adapter identity is missing")
    if not provenance["environment"]["split"]:
        raise ValueError("Demonstration environment split is missing")
    environment = provenance["environment"]
    protocol_keys = ("protocol_id", "protocol_sha256", "protocol_version")
    if any(key in environment for key in protocol_keys):
        if not all(key in environment for key in protocol_keys):
            raise ValueError("Demonstration protocol provenance is incomplete")
        protocol_hash = environment["protocol_sha256"]
        if (
            not isinstance(protocol_hash, str)
            or re.fullmatch(r"[0-9a-f]{64}", protocol_hash) is None
        ):
            raise ValueError("Demonstration protocol SHA-256 is missing or invalid")
        if environment["protocol_id"] != f"sha256:{protocol_hash}":
            raise ValueError("Demonstration protocol identifier does not match its SHA-256")
        if type(environment["protocol_version"]) is not int:
            raise ValueError("Demonstration protocol version is invalid")
    if (
        not provenance["demonstration"]["source_id"]
        or not provenance["demonstration"]["source_split"]
    ):
        raise ValueError("Demonstration source/split provenance is missing")


def config_provenance(c):
    provenance = c["data"].get("provenance")
    validate_replay_provenance(provenance, provenance)
    environment = provenance["environment"]
    if environment["adapter"] != c["env"].get("factory"):
        raise ValueError("Configured replay adapter differs from the environment factory")
    kwargs = c["env"].get("kwargs", {})
    if environment["split"] != kwargs.get("split", "train"):
        raise ValueError("Configured replay split differs from the demonstration environment split")
    if c["env"].get("factory") == "rl_vla.envs.libero_pro:make_env":
        if not all(
            key in environment for key in ("protocol_id", "protocol_sha256", "protocol_version")
        ):
            raise ValueError("LIBERO-PRO replay requires verified protocol provenance")
        protocol_path = Path(kwargs.get("protocol", ""))
        protocol_bytes = protocol_path.read_bytes()
        digest = hashlib.sha256(protocol_bytes).hexdigest()
        if digest != environment["protocol_sha256"]:
            raise ValueError("Configured demonstration protocol fingerprint mismatch")
        plan = json.loads(protocol_bytes)
        task = plan["tasks"].get(kwargs.get("task_key"))
        if task is None:
            raise ValueError("Configured LIBERO-PRO task_key is absent from protocol")
        if (
            provenance["task"]["task_key"],
            provenance["task"]["task_suite"],
            provenance["task"]["task_id"],
        ) != (kwargs["task_key"], task["suite"], task["task_id"]):
            raise ValueError("Configured replay task identity differs from LIBERO-PRO protocol")
        if environment["protocol_version"] != plan.get("version"):
            raise ValueError("Configured replay protocol version mismatch")
    else:
        if (provenance["task"]["task_suite"], provenance["task"]["task_id"]) != (
            kwargs.get("task_suite"),
            kwargs.get("task_id"),
        ):
            raise ValueError("Configured replay task identity differs from environment settings")
    return provenance


def validate_live_task(observation, info, provenance):
    expected_task = provenance["task"]
    if observation.get("openpi", {}).get("prompt") != expected_task["instruction"]:
        raise ValueError("Live environment instruction differs from demonstration provenance")
    if (info.get("task_suite"), info.get("task_id")) != (
        expected_task["task_suite"],
        expected_task["task_id"],
    ):
        raise ValueError("Live environment task identity differs from demonstration provenance")


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
    for key, default in (("discount_mode", "primitive"), ("n_step", 1), ("reward_scale", 1.0)):
        if payload["metadata"].get(key, default) != c["dice"].get(key, default):
            raise ValueError(f"Demonstration {key} mismatch; re-export replay for this profile")
    expected_provenance = config_provenance(c)
    validate_replay_provenance(payload.get("provenance"), expected_provenance)
    records = payload["transitions"]
    if not records:
        raise ValueError("Demonstration replay is empty")
    for r in records:
        validate_replay_provenance(r.get("provenance"), expected_provenance)
        for key in ("observation", "next_observation"):
            if not isinstance(r[key], dict) or "openpi" not in r[key]:
                raise ValueError("Demonstrations require images, proprioception, and prompt")
            if r[key]["openpi"].get("prompt") != expected_provenance["task"]["instruction"]:
                raise ValueError("Demonstration instruction does not match task provenance")
        a = np.asarray(r["actions"], np.float32)
        if a.shape != (prior.action_dim,) or not np.isfinite(a).all():
            raise ValueError("Expected flattened normalized executed action prefix")
        if "profile" in c:
            from rl_vla.dice_replay import action_mask

            executed = r.get("executed_steps")
            if type(executed) is not int or not 1 <= executed <= prior.execute_steps:
                raise ValueError("Profile replay requires actual executed_steps")
            expected_mask = action_mask(prior.action_dim, executed, prior.execute_steps)
            if not np.array_equal(r.get("action_mask"), expected_mask):
                raise ValueError("Profile replay requires a verified executed-prefix action mask")
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


def update_replay(c, learner, prior, replay, demos, rng, *, collection_step, commands, count):
    """Train only on completed MC-labelled episodes; called per collected chunk."""
    from rl_vla.dice_runtime import expert_fraction
    from rl_vla.dice_replay import sample_rows

    last = {}
    if len(replay) < c["train"]["batch_size"]:
        return last
    schedule_step = commands if c["data"].get("schedule_units") == "commands" else collection_step
    ratio = expert_fraction(c["data"], schedule_step)
    for _ in range(count):
        resource_check()
        rows = sample_rows(
            demos,
            replay,
            c["train"]["batch_size"],
            ratio,
            rng,
            fixed_split=c["data"].get("fixed_split", False),
        )
        batch = {
            key: tensor(np.stack([r[key] for r in rows]))
            for key in ("features", "actions", "rewards", "discounts", "returns", "expert")
        }
        if all("action_mask" in r for r in rows):
            batch["action_mask"] = tensor(np.stack([r["action_mask"] for r in rows]))
        prop = proposals(
            prior,
            [r["observation"] for r in rows],
            c["dice"].get("k_train", c["dice"]["particles"]),
        )
        nxt = proposals(
            prior,
            [r["next_observation"] for r in rows],
            c["dice"].get("k_bootstrap", c["dice"].get("k_train", c["dice"]["particles"])),
        )
        last = learner.update(
            batch, prop, nxt, step=collection_step // c["train"].get("reference_parallel_envs", 1)
        )
    return last


def trim_replay(c, replay, demos):
    while (
        replay
        and replay_bytes(list(replay)) + replay_bytes(demos) > c["train"]["replay_max_mb"] * 1024**2
    ):
        replay.popleft()


def run(c, output, *, validate=False, resume=None):
    from rl_vla.attempt_accounting import reject_restart
    reject_restart(c, resume)
    import torch

    from rl_vla.dice_core import DICE
    from rl_vla.dice_replay import action_mask, chunk_discount, chunk_reward, finish_episode
    from rl_vla.dice_pi05 import FrozenPi05
    from rl_vla.dice_runtime import expert_fraction, save_training, source_manifest
    from rl_vla.rldt_env import make_primitive_env

    if c["env"].get("factory") == "rl_vla.envs.libero_pro:make_env":
        kwargs = c["env"].get("kwargs", {})
        if not kwargs.get("protocol") or not kwargs.get("task_key"):
            raise ValueError("LIBERO-PRO training requires explicit protocol and task_key")
        if kwargs.get("split") != "train":
            raise ValueError("DICE training must use the LIBERO-PRO train split")
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
    decisions = total_commands = 0
    counters = {}
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
        learner = DICE(prior.feature_dim, prior.noise_dim, prior.action_dim, c["dice"], **({"device": c["train"]["learner_device"]} if "learner_device" in c["train"] else {}))
        demos = load_demonstrations(c, prior)
        if demos:
            provenance = config_provenance(c)
            probe_observation, probe_info = env.reset(seed=c["train"]["seed"])
            validate_live_task(probe_observation, probe_info, provenance)
        start_episode = 0
        if resume:
            from rl_vla.dice_runtime import restore_training

            start_episode, decisions = restore_training(resume, learner, c, prior, replay, rng)
            restored = torch.load(resume, map_location="cpu", weights_only=False)
            counters = restored["counters"]
            total_commands = restored["counters"].get("controller_steps", 0)
            if c["train"].get("max_steps") and "controller_steps" not in restored["counters"]:
                raise ValueError("Budgeted resume requires saved controller counter")
        metadata = {
            "config": c,
            "prior": prior.metadata,
            "upstream": UPSTREAM,
            "adaptation": "frozen-VLM features; normalized executed-prefix; finite-horizon MC",
            "demonstrations": len(demos),
            "torch": torch.__version__,
            "sources": source_manifest(),
        }
        (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        if "profile" in c and not resume:
            save(
                output / "step_000000.pt",
                learner,
                c,
                prior,
                {"episodes": 0, "decisions": 0, "controller_steps": 0, "updates": 0},
            )
        for episode in range(start_episode, c["train"]["episodes"]):
            if total_commands >= c["train"].get("max_steps", float("inf")):
                break
            budget_stop = False
            last = {}
            per_chunk_updates = "gradient_steps_per_chunk" in c["train"]
            raw, reset_info = env.reset(seed=c["train"]["seed"] + episode)
            primitive_steps, clipped = 0, 0
            trajectory, terminated, truncated = [], False, False
            while not (terminated or truncated):
                resource_check()
                selection_clock = decisions // c["train"].get("reference_parallel_envs", 1)
                count = (
                    1
                    if selection_clock <= c["dice"]["selection_warmup"]
                    or (per_chunk_updates and learner.updates == 0)
                    else c["dice"].get("k_collection", c["dice"]["selection_samples"])
                )
                f, z, p, full = prior.propose(raw, count)
                action, index = learner.select(
                    tensor(f),
                    tensor(z),
                    tensor(p),
                    strategy=c["dice"].get("selection_strategy", "max_q_min"),
                )
                chunk = prior.decode(action.numpy(), full[index], raw)
                primitive_rewards, n = [], 0
                for primitive in chunk:
                    if total_commands >= c["train"].get("max_steps", float("inf")):
                        budget_stop = True
                        break
                    command = np.clip(primitive, env.low, env.high)
                    clipped += int(np.any(command != primitive))
                    next_raw, r, terminated, truncated, _info = env.step(command)
                    primitive_steps += 1
                    primitive_rewards.append(float(r))
                    total_commands += 1
                    n += 1
                    if total_commands in c["train"].get("milestones", []):
                        # Snapshot immediately after this command; pending MC labels
                        # are excluded. Collection and the episode continue unchanged.
                        save(
                            output / f"step_{total_commands:06d}.pt",
                            learner,
                            c,
                            prior,
                            {
                                "episodes": episode,
                                "decisions": decisions + 1,
                                "controller_steps": total_commands,
                                "updates": learner.updates,
                                "pending_episode_commands": primitive_steps,
                                "snapshot_phase": "post_command_pre_episode_updates",
                            },
                        )
                    if terminated or truncated:
                        break
                if not n:
                    break
                reward = chunk_reward(
                    primitive_rewards,
                    c["dice"]["gamma"],
                    c["dice"].get("discount_mode", "primitive"),
                    c["dice"].get("reward_scale", 1.0),
                )
                trajectory.append(
                    {
                        "observation": raw,
                        "next_observation": next_raw,
                        "features": f[index],
                        "actions": action.numpy().copy(),
                        "rewards": reward,
                        "discounts": chunk_discount(
                            c["dice"]["gamma"],
                            n,
                            terminated or truncated,
                            c["dice"].get("discount_mode", "primitive"),
                        ),
                        "action_mask": action_mask(prior.action_dim, n, c["dice"]["execute_steps"]),
                        "executed_steps": n,
                        "collection_commands": total_commands,
                        "terminated": bool(terminated),
                        "truncated": bool(truncated),
                        "expert": 0.0,
                    }
                )
                raw = next_raw
                decisions += 1
                if terminated or truncated:
                    if per_chunk_updates:
                        replay.extend(finish_episode(trajectory, c["dice"].get("n_step", 1)))
                        trim_replay(c, replay, demos)
                if per_chunk_updates:
                    # Rational serial credit G/E; no catch-up updates for warmup.
                    g = c["train"]["gradient_steps_per_chunk"]
                    e = c["train"].get("reference_parallel_envs", 1)
                    count = (decisions * g) // e - ((decisions - 1) * g) // e
                    last = update_replay(
                        c,
                        learner,
                        prior,
                        replay,
                        demos,
                        rng,
                        collection_step=decisions - 1,
                        commands=total_commands,
                        count=count,
                    )
                if total_commands >= c["train"].get("max_steps", float("inf")):
                    budget_stop = not (terminated or truncated)
                    break
                if (
                    replay_bytes(trajectory) + replay_bytes(demos)
                    > c["train"]["replay_max_mb"] * 1024**2
                ):
                    raise RuntimeError("Current episode exceeds replay memory budget")
            # A command-budget interruption is not a task terminal. Never invent
            # a zero tail for its MC calibration or contaminate n-step replay.
            if budget_stop:
                (output / "partial_trajectory.json").write_text(
                    json.dumps(
                        {
                            "commands": primitive_steps,
                            "chunks": len(trajectory),
                            "reason": "budget_stop",
                            "included_in_mc_replay": False,
                        }
                    )
                    + "\n"
                )
                records, returns = [], []
            else:
                records = finish_episode(trajectory, c["dice"].get("n_step", 1))
                returns = [r["returns"] for r in records]
                if not per_chunk_updates:
                    replay.extend(records)
            trim_replay(c, replay, demos)
            if not per_chunk_updates and not budget_stop:
                last = update_replay(
                    c,
                    learner,
                    prior,
                    replay,
                    demos,
                    rng,
                    collection_step=decisions - 1,
                    commands=total_commands,
                    count=c["train"]["updates_per_episode"],
                )
            counters = {"episodes": episode + 1, "decisions": decisions, "updates": learner.updates}
            if "profile" in c:
                counters.update(
                    controller_steps=total_commands,
                    actor_updates=learner.actor_updates,
                    target_updates=learner.target_updates,
                )
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
                            "expert_fraction": expert_fraction(
                                c["data"],
                                total_commands
                                if c["data"].get("schedule_units") == "commands"
                                else decisions,
                            ),
                            "return": returns[0] if returns else None,
                            "budget_stop": budget_stop,
                            "controller_steps": total_commands,
                            "gradient_updates_per_collected_chunk": learner.updates
                            / max(decisions, 1),
                            "gradient_updates_per_controller_command": learner.updates
                            / max(total_commands, 1),
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
    parser.add_argument(
        "--split", choices=("validation", "test", "source"), help="Evaluation split"
    )
    parser.add_argument(
        "--evaluation-k", type=int, help="Override evaluation K only; never K_train"
    )
    parser.add_argument("--train-seed", type=int, help="Override the local experiment seed")
    parser.add_argument(
        "--diagnostic",
        action="store_true",
        help="Allow an evaluation subset; never selection eligible",
    )
    args = parser.parse_args()
    c = config_from(args.config)
    if args.train_seed is not None:
        c["train"]["seed"] = args.train_seed
    if args.split:
        if args.command != "evaluate":
            parser.error("--split is only valid for evaluate")
        c["env"].setdefault("kwargs", {})["split"] = args.split
    if args.evaluation_k is not None:
        if args.command != "evaluate" or args.evaluation_k < 1:
            parser.error("--evaluation-k requires evaluate and a positive K")
        c.setdefault("evaluation", {})["k"] = args.evaluation_k
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
                c,
                args.output,
                checkpoint=args.checkpoint,
                episodes=args.episodes,
                seed=args.seed,
                diagnostic=args.diagnostic,
            )
        else:
            if args.resume and args.command != "train":
                parser.error("--resume is only valid for train")
            run(c, args.output, validate=args.command == "validate-policy", resume=args.resume)


if __name__ == "__main__":
    main()
