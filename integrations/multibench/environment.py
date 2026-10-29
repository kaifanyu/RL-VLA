"""Frozen paired episode schedules over independently installed native simulators."""
from __future__ import annotations

import hashlib
import importlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SPECS = {
    "robotwin": {"module": "adapter", "python": ".venv-robotwin", "max_steps": 800},
    "robocasa": {"module": "environment", "python": ".venv-robocasa365", "max_steps": 750},
    "maniskill": {"module": "environment", "python": ".venv-maniskill3", "max_steps": 80},
}


def process_config(benchmark, protocol, split):
    spec = SPECS[benchmark]
    paths = [ROOT / "integrations/multibench/core", ROOT / "integrations", ROOT / "third_party/rlinf", ROOT / "third_party/openpi/packages/openpi-client/src"]
    if benchmark == "robocasa":
        paths += [ROOT / "third_party/robocasa365", ROOT / "third_party/robocasa365/dependencies/robosuite"]
    env = {"PYTHONPATH": os.pathsep.join(map(str, paths)), "OMP_NUM_THREADS": "1",
           "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "NUMBA_NUM_THREADS": "1",
           "PYTHONUNBUFFERED": "1"}
    if benchmark == "robocasa":
        env.update(MUJOCO_GL="osmesa", PYOPENGL_PLATFORM="osmesa", LP_NUM_THREADS="1", NUMBA_DISABLE_JIT="1")
    else:
        # SAPIEN's Vulkan renderer needs the assigned GPU; JAX in this process stays on CPU.
        env.update(PYOPENGL_PLATFORM="egl")
    return {"kind": "external", "factory": "multibench.environment:make_env",
            "python": str(ROOT / spec["python"] / "bin/python"),
            "gpu_rendering": benchmark != "robocasa", "process_env": env,
            "kwargs": {"benchmark": benchmark, "protocol": str(Path(protocol).resolve()), "split": split}}


def state_id(benchmark, seed):
    if benchmark == "maniskill":
        return int.from_bytes(hashlib.sha256(str(seed).encode()).digest()[:8], "little") % 802816
    return int(seed)


def make_env(benchmark, protocol=None, split="train", seed=0, episode_ids=None, max_episode_steps=None):
    import gymnasium as gym
    spec = SPECS[benchmark]
    if split not in {"train", "validation", "test"}:
        raise ValueError(f"Unknown episode split: {split}")
    seeds, expected = None, None
    if protocol:
        definition = json.loads(Path(protocol).read_text())["tasks"][benchmark]
        seeds = definition[f"{split}_seeds"]
        expected = definition[f"{split}_state_ids"]
        if not seeds or len(seeds) != len(expected) or len(set(seeds)) != len(seeds):
            raise ValueError("Invalid frozen reset schedule")
        max_episode_steps = definition["max_steps"]
    horizon = int(max_episode_steps or spec["max_steps"])
    module = importlib.import_module(f"multibench.{benchmark}.{spec['module']}")
    if benchmark == "maniskill":
        native = module.make_env(episode_ids=episode_ids, max_episode_steps=horizon)
    elif benchmark == "robocasa":
        native = module.make_env(split="target", max_steps=horizon, seed=seed)
    else:
        native = module.make_env(max_steps=horizon)

    class ScheduledEnv(gym.Wrapper):
        def __init__(self, env):
            super().__init__(env)
            self.cursor = 0
            self.controller_steps = 0

        def reset(self, *, seed=None, options=None):
            backend_seed = int(seeds[self.cursor % len(seeds)] if seeds else seed)
            raw, info = self.env.reset(seed=backend_seed, options=options)
            if expected is not None and info["initial_state_id"] != expected[self.cursor % len(expected)]:
                raise RuntimeError("Simulator reset differs from the frozen initial-state schedule")
            info = dict(info, collector_seed=seed, backend_seed=backend_seed, split=split,
                        schedule_index=self.cursor, benchmark=benchmark)
            self.cursor += 1
            self.controller_steps = 0
            return raw, info

        def step(self, action):
            raw, reward, terminated, truncated, info = self.env.step(action)
            self.controller_steps += 1
            success = bool(info["is_success"])
            if float(reward) != float(success):
                raise RuntimeError("Benchmark reward is not native binary task success")
            # Success is terminal for every learner, including tasks whose native Gym
            # termination flag describes only a time limit.
            terminated = bool(terminated or success)
            truncated = bool((truncated or self.controller_steps >= horizon) and not terminated)
            return raw, reward, terminated, truncated, dict(info, controller_steps=self.controller_steps)

    return ScheduledEnv(native)
