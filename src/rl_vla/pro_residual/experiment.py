"""Experiment directory policy, resolved configuration and shared construction."""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import tarfile
import time
from pathlib import Path

import numpy as np

from .residual import ActionSpec, ObservationBuilder, episode_seed, original_seed
from .rollout import write_json
from .spec import (
    ORIGINAL_SUITE_INDEX,
    ROOT,
    build_split,
    config_hash,
    published_index,
    resolve_case,
    resolve_original_task,
)

SUBDIRS = ("config", "checkpoints", "videos", "metadata", "logs")


def resolve_run_dir(path: str | Path) -> Path:
    path = Path(path)
    return (path if path.is_absolute() else ROOT / path).resolve()


def prepare_new_run(run_dir: Path) -> None:
    """Never overwrite: the directory must be absent or empty."""
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"{run_dir} already exists and is not empty. Use --resume to continue it, "
                              "or choose a new --output directory.")
    for name in SUBDIRS:
        (run_dir / name).mkdir(parents=True, exist_ok=True)


def git_state() -> dict:
    def run(*args):
        try:
            return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True, stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    status = run("status", "--porcelain") or ""
    return {"commit": run("rev-parse", "HEAD"), "uncommitted_paths": len(status.splitlines())}


class Experiment:
    def __init__(self, config: dict, run_dir: Path):
        self.config = config
        self.run_dir = run_dir
        self.case = resolve_case(config["task"])
        self.contract = config.get("contract", {})
        for key, actual in (("states_sha256", self.case["states_sha256"]), ("bddl_sha256", self.case["bddl_sha256"])):
            if self.contract.get(key) and self.contract[key] != actual:
                raise ValueError(f"Case {key} {actual} differs from the benchmark contract")
        self.original = resolve_original_task(self.case)
        residual = config["residual"]
        self.spec = ActionSpec(residual["scale"], residual["action_low"], residual["action_high"])
        self.execute_steps = int(config["base"]["execute_steps"])
        self.gamma = float(residual["gamma"])
        self.base = None
        self.builder = None
        self.split = None

    # -- files -----------------------------------------------------------------------
    def write_config(self) -> None:
        write_json(self.run_dir / "config/resolved_config.json",
                   {"config": self.config, "config_hash": config_hash(self.config),
                    "created_unix_time": time.time(), "host": platform.node(), "git": git_state(),
                    "cpu_affinity": sorted(os.sched_getaffinity(0)),
                    "environment": {k: os.environ.get(k) for k in ("CUDA_VISIBLE_DEVICES", "MUJOCO_GL",
                                                                   "XLA_PYTHON_CLIENT_MEM_FRACTION")}})
        write_json(self.run_dir / "config/task_identity.json", {
            "label": self.config["task"].get("label"),
            "label_resolution": self.config["task"].get("label_resolution"),
            "case": self.case, "original_task_for_retention": self.original,
            "action_spec": self.spec.describe(), "timing": self.timing()})
        self.write_provenance()

    def write_provenance(self) -> None:
        """Keep what a later reproduction needs: exact source, packages and launch command."""
        config_dir = self.run_dir / "config"
        with tarfile.open(config_dir / "source_snapshot.tar.gz", "w:gz") as archive:
            for relative in ("src/rl_vla", "scripts/run_flashsac_pro.sh", "scripts/env.sh"):
                archive.add(ROOT / relative, arcname=relative,
                            filter=lambda item: None if "__pycache__" in item.name else item)
        listing = ("import importlib.metadata as m, json; print(json.dumps(sorted("
                   "f'{d.metadata[\"Name\"]}=={d.version}' for d in m.distributions())))")
        packages = {}
        for name, python in (("learner", sys.executable), ("simulator", self.config["simulator"]["pro_python"])):
            python = python if Path(python).is_absolute() else str(ROOT / python)
            packages[name] = {"python": python,
                              "packages": json.loads(subprocess.check_output([python, "-c", listing], text=True))}
        write_json(config_dir / "python_packages.json", packages)
        vendors = {}
        for name in ("FlashSAC", "openpi", "libero_pro"):
            source = str(ROOT / "third_party" / name)
            status = subprocess.run(["git", "-C", source, "status", "--porcelain"], text=True, capture_output=True)
            commit = subprocess.run(["git", "-C", source, "rev-parse", "HEAD"], text=True, capture_output=True)
            vendors[name] = {"commit": commit.stdout.strip() or None,
                             "uncommitted_paths": len(status.stdout.splitlines())}
        write_json(config_dir / "vendor_commits.json", vendors)
        write_json(config_dir / "launch.json", {"argv": sys.argv, "cwd": os.getcwd(),
                                                "executable": sys.executable})

    def ensure_split(self, create: bool) -> dict:
        path = self.run_dir / "config/split_manifest.json"
        fresh = build_split(self.config, self.case, self.original)
        if path.exists():
            saved = json.loads(path.read_text())
            for key in ("case_id", "states_sha256", "train", "validation", "test", "seed"):
                if saved[key] != fresh[key]:
                    raise ValueError(f"Saved split manifest differs in {key}; refusing to continue")
            self.split = saved
        elif create:
            write_json(path, fresh)
            self.split = fresh
        else:
            raise FileNotFoundError(f"Missing {path}")
        return self.split

    def check_resume_config(self) -> None:
        saved = json.loads((self.run_dir / "config/resolved_config.json").read_text())
        if saved["config_hash"] != config_hash(self.config):
            raise ValueError("Configuration differs from the saved run (task, base, residual, flashsac or fixed "
                             "train settings). Only budgets/logging may change on --resume.")

    def timing(self) -> dict:
        return {"base_query_every_env_steps": self.execute_steps,
                "base_chunk_length": int(self.config["base"]["action_horizon"]),
                "base_actions_executed_per_query": self.execute_steps,
                "residual_acts_every_env_steps": 1, "execute_steps": self.execute_steps,
                "control_timestep_s": 0.05, "replay_transition": "one executed environment step",
                "gamma_per_env_step": self.gamma}

    # -- heavy objects -----------------------------------------------------------------
    def load_base(self):
        from .base import FrozenPi05Base, FrozenPi05Flow

        checkpoint = Path(self.config["base"]["checkpoint"])
        checkpoint = checkpoint if checkpoint.is_absolute() else ROOT / checkpoint
        settings = {"model_config": self.config["base"]["model_config"],
                    "with_features": bool(self.config["residual"]["use_vlm_features"]),
                    "noise_protocol": self.config["base"]["noise_protocol"]}
        if self.config["base"].get("backend") == "pi05_flow":
            self.base = FrozenPi05Flow(checkpoint, ode_steps=int(self.config["base"]["ode_steps"]), **settings)
        else:
            self.base = FrozenPi05Base(checkpoint, **settings)
        self.builder = ObservationBuilder(self.base.state_q01, self.base.state_q99, self.execute_steps,
                                          self.base.feature_dim)
        return self.base

    def make_learner(self, device: str):
        from rl_vla.flashsac import FlashSAC

        settings = dict(self.config["flashsac"])
        settings.setdefault("batch_size", self.config["train"]["batch_size"])
        return FlashSAC(self.builder.dim, self.builder.dim, self.spec.action_dim, settings, device=device)

    # -- episode definitions -------------------------------------------------------------
    def max_steps(self, source: str) -> int | None:
        value = self.config["simulator"].get("max_steps")
        return int(value) if value else None

    def pro_make_request(self) -> dict:
        return {"kind": "pro_case", "case": self.case, "output": str(self.run_dir / "generated_states"),
                "max_steps": self.max_steps("pro")}

    def retention_simulator(self) -> str:
        return "pro" if self.contract.get("retention_env") == "source_protocol" else "original"

    def original_make_request(self) -> dict:
        if self.contract.get("retention_env") == "source_protocol":
            return {"kind": "source_protocol", "suite": self.original["suite"], "task_id": self.original["task_id"],
                    "bddl": self.original["bddl"], "states": self.original["init_file"],
                    "max_steps": self.max_steps("original") or self.original["horizon"]}
        return {"kind": "original", "suite": self.original["suite"], "task_id": self.original["task_id"],
                "name": self.original["name"],
                "max_steps": self.max_steps("original") or self.original["horizon"]}

    def evaluation_episode(self, split: str, initial_state_id: int) -> dict:
        """Seeds and noise keys identical to the existing evaluation pipelines."""
        start = self.contract.get("eval_seed_start")
        if split == "retention":
            if start is not None:  # Shared contract: as RLDT's ordered source split.
                seed = int(start) + self.split_states("retention").index(initial_state_id)
            else:
                seed = original_seed(ORIGINAL_SUITE_INDEX[self.original["suite"]], self.original["task_id"],
                                     initial_state_id)
            key = f"{self.original['suite']}_task{self.original['task_id']:02d}_init{initial_state_id:03d}:{seed}"
            return {"source": "original", "seed": seed, "noise_key": key, "make": self.original_make_request(),
                    "slug": self.original["slug"]}
        published = self.published_index(initial_state_id)
        if start is not None and split in ("validation", "test"):
            # Shared contract: seed start + i for the i-th state of the ordered split, as RLDT and DICE-RL.
            index = self.split[split].index(initial_state_id)
            seed = int(start) + index
            key = f"{self.case['case_id']}_{split}{index:03d}:{seed}"
        elif published is not None:
            # Published PRO state: the robustness queue's seed and noise key, so base-only
            # episodes reproduce the queue (with the osmesa renderer).
            seed = episode_seed(self.case["case_id"], published)
            key = f"{self.case['case_id']}_init{published:03d}:{seed}"
        else:
            namespace = f"{self.case['case_id']}:generated:{self.case['states_sha256'][:16]}"
            seed = episode_seed(namespace, initial_state_id)
            key = f"{self.case['case_id']}_gen{initial_state_id:03d}:{seed}"
        return {"source": "pro", "seed": seed, "noise_key": key, "make": self.pro_make_request(),
                "slug": self.case["slug"], "published_index": published}

    def published_index(self, initial_state_id: int) -> int | None:
        """Index in the published PRO state file (queue episodes), or None for generated states."""
        return published_index(self.case, initial_state_id)

    def split_states(self, split: str) -> list[int]:
        if split == "retention":
            if not self.original.get("available"):
                raise ValueError("No original LIBERO task is available for the retention check")
            return list(self.split["retention"]["initial_state_ids"])
        return list(self.split[split])


def resolve_device(name: str) -> str:
    import torch

    if name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return name


def actor_fn(learner, device: str, deterministic: bool, generator=None):
    import torch

    def act(observation: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            tensor = torch.as_tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
            action, _, _ = learner.actor.sample(tensor, deterministic=deterministic, training=False)
        return action.squeeze(0).cpu().numpy().astype(np.float32)

    return act


def resource_check(config: dict, *, storage: bool = False) -> dict:
    """Refuse to add load near the robustness queue's limits; never touch other processes."""
    train = config["train"]
    command = ["nvidia-smi", "--query-gpu=memory.used,temperature.gpu", "--format=csv,noheader,nounits"]
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")[0].strip()
    if visible:  # Watch the GPU this run uses, not the host's first GPU.
        command.append(f"--id={visible}")
    output = subprocess.check_output(command, text=True, timeout=10)
    memory, temperature = (float(x) for x in output.strip().splitlines()[0].split(","))
    record = {"gpu_memory_gib": memory / 1024, "gpu_temperature_c": temperature}
    if memory / 1024 >= float(train["max_gpu_memory_gib"]) or temperature >= float(train["max_gpu_temperature_c"]):
        raise RuntimeError(f"GPU pressure {record}; stopping this run only")
    if storage:
        used = int(subprocess.check_output(["du", "-sb", "/workspace"], text=True, timeout=600).split()[0])
        record["workspace_gib"] = used / 2**30
        if used / 2**30 >= float(train["max_workspace_gib"]):
            raise RuntimeError(f"Workspace usage {used / 2**30:.1f} GiB exceeds the configured limit")
    return record
