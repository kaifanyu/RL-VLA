"""RLinf vector API over the three benchmark adapters shared with native RLDT.

Derived from the existing controlled-environment bridge; the terminal padding,
auto-reset, discounted primitive rewards, and rollout recording are preserved.

The first integration deliberately supports one environment and one worker. A
terminal action stops real execution immediately; remaining tensor slots are
padding, not additional environment interactions. Use chunk-level PPO rewards
and advantages, auto_reset=True, and ignore_terminations=False.
"""

import copy
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import ClassVar

import numpy as np
import torch

from rl_vla.rldt_env import ProcessEnv
from rl_vla.trajectory_recording import TrajectoryRecorder


def _plain(value):
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


class ControlledEnv:
    metadata: ClassVar[dict] = {"render_modes": [], "render_fps": 20}

    def __init__(self, cfg, num_envs, seed_offset, total_num_processes, worker_info):
        if num_envs != 1 or total_num_processes != 1 or seed_offset != 0:
            raise ValueError("Controlled benchmark currently requires one environment and one worker")
        if not cfg.get("auto_reset", True) or cfg.get("ignore_terminations", False):
            raise ValueError("Controlled benchmark requires auto_reset=True and ignore_terminations=False")
        self.cfg, self.num_envs = cfg, 1
        self.seed_offset, self.total_num_processes = seed_offset, total_num_processes
        self.worker_info = worker_info
        self.seed = int(cfg.get("seed", 0))
        self.auto_reset, self.ignore_terminations = True, False
        self.protocol_path = Path(cfg["protocol"])
        protocol_bytes = self.protocol_path.read_bytes()
        self.protocol = json.loads(protocol_bytes)
        self.benchmark, self.split = str(cfg["benchmark"]), str(cfg["split"])
        self.task_key = self.benchmark
        self.task = self.protocol["tasks"][self.task_key]
        self.max_steps = int(self.task["max_steps"])
        if int(cfg.get("max_episode_steps", self.max_steps)) != self.max_steps:
            raise ValueError("max_episode_steps differs from the frozen benchmark protocol")
        self.is_eval = self.split != "train"
        self.evaluation_size = len(self.task[f"{self.split}_state_ids"])
        self.primitive_gamma = float(cfg.get("primitive_gamma", 1.0))
        if not 0 < self.primitive_gamma <= 1:
            raise ValueError("primitive_gamma must be in (0, 1]")
        from multibench.environment import process_config
        self._process_config = process_config(self.benchmark, self.protocol_path, self.split)
        self.env = ProcessEnv(self._process_config)
        self._is_start = True
        self._closed = False
        self.current_raw_obs = None
        self._elapsed_steps = np.zeros(1, np.int32)
        self.episode_return = 0.
        self.episode_count = 0
        self.reset_count = 0
        self.total_steps = 0
        self.total_decisions = 0
        self.total_successes = 0
        self._boundary = False
        self._evaluation_exhausted = False
        self._policy_trace = None
        self.recorder = None
        self.recording_root = cfg.get("recording_directory")
        if self.recording_root:
            self.recording_root = Path(self.recording_root)
            self.recording_root.mkdir(parents=True, exist_ok=True)
        if self.recording_root and cfg.get("record_trajectories", True):
            self.recorder = TrajectoryRecorder(self.recording_root, {
                "algorithm": "RLinf PPO", "config": _plain(cfg),
                "protocol_sha256": hashlib.sha256(protocol_bytes).hexdigest(),
                "primitive_environment": self._process_config,
                "normalized_actions_note": "Empty unless set_policy_trace is called by the "
                "rollout worker. decoded_actions always contain the received motor chunk.",
            })

    @property
    def is_start(self):
        return self._is_start

    @is_start.setter
    def is_start(self, value):
        self._is_start = bool(value)

    @property
    def elapsed_steps(self):
        return self._elapsed_steps.copy()

    @property
    def info_logging_keys(self):
        return []

    @staticmethod
    def _wrap_obs(raw):
        policy = raw["openpi"]
        def image(name):
            value = policy.get(name)
            return None if value is None else torch.from_numpy(np.asarray(value).copy())[None]
        return {
            "main_images": image("observation/image"),
            "wrist_images": image("observation/wrist_image"),
            "extra_view_images": image("observation/right_wrist_image"),
            "states": torch.from_numpy(np.asarray(policy["observation/state"], np.float32).copy())[None],
            "task_descriptions": [str(policy["prompt"])],
        }

    def reset(self, env_idx=None, reset_state_ids=None):
        if env_idx is not None and np.asarray(env_idx).reshape(-1).tolist() != [0]:
            raise ValueError("Only environment index 0 is available")
        if reset_state_ids is not None:
            raise ValueError("Reset states are owned by the frozen benchmark split")
        if self._closed:
            raise RuntimeError("Environment is closed")
        if self.is_eval and self._is_start and self.reset_count:
            # RLinf reuses the evaluator between checkpoints. A fresh primitive
            # process restarts the exact frozen enumeration and matched seeds.
            if self.recorder is not None and self.recorder.active:
                self.recorder.end_episode({}, reason="evaluation_restart")
            self.env.close()
            self.env = ProcessEnv(self._process_config)
            self.reset_count = 0
            self._evaluation_exhausted = False
        elif self.recorder is not None and self.recorder.active:
            self.recorder.end_episode({}, reason="explicit_reset")
        seed = self.seed + self.reset_count
        self.current_raw_obs, self.reset_info = self.env.reset(seed=seed)
        self.current_seed = seed
        self.reset_count += 1
        self._is_start = False
        self._boundary = False
        self._elapsed_steps[:] = 0
        self.episode_return = 0.
        return self._wrap_obs(self.current_raw_obs), {"reset": copy.deepcopy(self.reset_info)}

    def set_policy_trace(self, normalized_actions, metadata=None):
        """Optional rollout-worker hook for full normalized chunks/sampling identity."""
        self._policy_trace = (np.asarray(normalized_actions).copy(), dict(metadata or {}))

    def _episode_info(self, success):
        length = int(self._elapsed_steps[0])
        return {"episode": {
            "success_once": torch.tensor([bool(success)]),
            "return": torch.tensor([self.episode_return], dtype=torch.float32),
            "episode_len": torch.tensor([length], dtype=torch.int64),
            "reward": torch.tensor([self.episode_return / max(length, 1)], dtype=torch.float32),
        }}

    def _record_progress(self):
        if self.recording_root is None:
            return
        temporary = self.recording_root / "progress.json.tmp"
        temporary.write_text(json.dumps({
            "decisions": self.total_decisions, "primitive_steps": self.total_steps,
            "episodes": self.episode_count, "successes": self.total_successes,
            "task_key": self.task_key, "split": self.split,
            "native_action_dim": len(self.env.low),
            "render_device": getattr(self, "reset_info", {}).get("render_device"),
        }, allow_nan=False) + "\n")
        temporary.replace(self.recording_root / "progress.json")

    def chunk_step(self, chunk_actions):
        if self.current_raw_obs is None:
            raise RuntimeError("reset() must precede chunk_step()")
        if isinstance(chunk_actions, torch.Tensor):
            chunk_actions = chunk_actions.detach().cpu().float().numpy()
        actions = np.asarray(chunk_actions, np.float32)
        if actions.ndim != 3 or actions.shape[0] != 1 or actions.shape[-1] != len(self.env.low):
            raise ValueError(f"Expected motor action shape [1, chunk_steps, {len(self.env.low)}]")
        chunk_size = actions.shape[1]
        if chunk_size < 1 or not np.isfinite(actions).all():
            raise ValueError("Action chunks must be nonempty and finite")
        rewards = torch.zeros((1, chunk_size), dtype=torch.float32)
        terminations = torch.zeros((1, chunk_size), dtype=torch.bool)
        truncations = torch.zeros_like(terminations)
        if self._evaluation_exhausted:
            obs = self._wrap_obs(self.current_raw_obs)
            return [obs] * chunk_size, rewards, terminations, truncations, [{}] * chunk_size
        if self._boundary:
            raise RuntimeError("An episode boundary requires reset()")
        if self.recorder is not None:
            if not self.recorder.active:
                self.recorder.start_episode(self.current_raw_obs, {
                    "seed": self.current_seed, "reset": self.reset_info,
                })
            normalized, trace = self._policy_trace or (np.empty((0,), np.float32), {})
            self.recorder.begin_chunk(self.current_raw_obs, normalized, actions[0], {
                "decision": self.total_decisions, "normalized_actions_available":
                self._policy_trace is not None, "planned_execute_steps": chunk_size, **trace,
            })
        self._policy_trace = None
        obs_list, infos_list = [], []
        terminated = truncated = False
        raw_rewards = []
        for index in range(chunk_size):
            command = np.clip(actions[0, index], self.env.low, self.env.high)
            raw, reward, terminated, truncated, info = self.env.step(command)
            if "final_observation" in info or "final_obs" in info:
                raise ValueError("Primitive environment must not autoreset")
            if not np.isfinite(reward):
                raise FloatingPointError("Nonfinite environment reward")
            self.current_raw_obs = raw
            self._elapsed_steps[0] += 1
            self.total_steps += 1
            self.episode_return += reward
            raw_rewards.append(float(reward))
            rewards[0, index] = self.primitive_gamma ** index * float(reward)
            if self.recorder is not None:
                self.recorder.record_step(command, raw, reward, terminated, truncated, info)
            obs_list.append(self._wrap_obs(raw))
            infos_list.append(self._episode_info(info.get("is_success", False)))
            if terminated or truncated:
                break
        executed_steps = len(raw_rewards)
        self.total_decisions += 1
        if self.recorder is not None:
            self.recorder.end_chunk({"executed_steps": executed_steps,
                                     "discounted_reward": float(rewards.sum()), "info": info})
        # Padding repeats the actual final observation and has zero reward.
        # No action is sent to the simulator after an episode boundary.
        while len(obs_list) < chunk_size:
            obs_list.append(obs_list[-1])
            infos_list.append(copy.deepcopy(infos_list[-1]))
        infos_list[-1]["executed_steps"] = torch.tensor([executed_steps], dtype=torch.int64)
        infos_list[-1]["primitive_step_mask"] = (
            torch.arange(chunk_size)[None, :] < executed_steps)
        if terminated or truncated:
            self._boundary = True
            terminations[0, -1], truncations[0, -1] = bool(terminated), bool(truncated)
            success = bool(info.get("is_success", False))
            summary = {"seed": self.current_seed, "reset": self.reset_info,
                       "success": success, "return": self.episode_return,
                       "primitive_steps": int(self._elapsed_steps[0]),
                       "terminated": bool(terminated), "truncated": bool(truncated)}
            self.episode_count += 1
            self.total_successes += int(success)
            if self.recorder is not None:
                self.recorder.end_episode(summary)
            final_obs, final_info = obs_list[-1], copy.deepcopy(infos_list[-1])
            if self.is_eval and self.reset_count >= self.evaluation_size:
                self._evaluation_exhausted = True
                reset_obs, reset_info = final_obs, {}
            else:
                reset_obs, reset_info = self.reset()
            obs_list[-1] = reset_obs
            infos_list[-1] = reset_info | {
                "final_observation": final_obs, "final_info": final_info,
                "_final_info": torch.tensor([True]),
                "_final_observation": torch.tensor([True]), "_elapsed_steps": torch.tensor([True]),
                "executed_steps": torch.tensor([executed_steps], dtype=torch.int64),
                "primitive_step_mask": torch.arange(chunk_size)[None, :] < executed_steps,
            }
        self._record_progress()
        return obs_list, rewards, terminations, truncations, infos_list

    def step(self, actions=None, auto_reset=True):
        if not auto_reset:
            raise ValueError("Use chunk_step for controlled execution without within-chunk resets")
        if isinstance(actions, torch.Tensor):
            actions = actions.detach().cpu().float().numpy()
        obs, rewards, terminations, truncations, infos = self.chunk_step(np.asarray(actions)[:, None])
        return obs[-1], rewards[:, 0], terminations[:, 0], truncations[:, 0], infos[-1]

    def update_reset_state_ids(self):
        """RLinf calls this at iteration boundaries; never reset training here."""

    def close(self):
        if not self._closed:
            try:
                if self.recorder is not None:
                    self.recorder.close(reason="environment_closed")
            finally:
                self.env.close()
                self._closed = True


def install():
    """Select this adapter inside only the new campaign's RLinf processes.

    Retaining the existing pro_controlled dispatch name preserves upstream's
    identity action handling and recording hooks without editing its source.
    """
    import rlinf_bridge.environment
    rlinf_bridge.environment.ControlledProEnv = ControlledEnv
