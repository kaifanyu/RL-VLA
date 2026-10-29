"""Sequential fresh development states through the existing benchmark adapter."""

import hashlib
import json
import subprocess
from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from rl_vla.robustness import load_states, make_case_env


def checked_case(ready_file):
    case = json.loads(Path(ready_file).read_text())
    if not case.get("ready") or case.get("split") != "development_only":
        raise ValueError("Only validated fresh development cells may enter this screen")
    for filename, expected in ((case["states"], case["states_sha256"]),
                               (case["real_bddl"], case["bddl_sha256"])):
        if hashlib.sha256(Path(filename).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Prepared screen asset changed: {filename}")
    states = load_states(case["states"])
    if len(states) != case["episodes"] or len(case["reset_checks"]) != case["episodes"]:
        raise ValueError("All development states must have validated resets")
    if len(np.unique(states, axis=0)) != len(states):
        raise ValueError("Development states are duplicated")
    return case


class SequentialCaseEnv(gym.Env):
    def __init__(self, *, ready_file, output):
        super().__init__()
        self.case = checked_case(ready_file)
        self.output = Path(output)
        self.index, self.native = 0, None
        self.action_space = spaces.Box(-1., 1., (7,), dtype=np.float32)
        # Import the fork only inside its isolated simulator interpreter.
        import libero.libero as libero_package

        imported = Path(libero_package.__file__).resolve()
        commit = subprocess.check_output(
            ["git", "-C", str(imported.parent), "rev-parse", "HEAD"], text=True).strip()
        if commit != self.case["source_commit"]:
            raise RuntimeError("Simulator imported a different benchmark fork/revision")

    def reset(self, *, seed=None, options=None):
        if options:
            raise ValueError("Screen state order is fixed")
        if seed is None:
            raise ValueError("Screen reset requires an explicit seed")
        if self.index >= self.case["episodes"]:
            raise ValueError("Screen requested more than its frozen development states")
        if self.native is not None:
            self.native.close()
        self.native, state_index = make_case_env(self.case, seed, self.output, self.index)
        raw, info = self.native.reset(seed=seed, options={"initial_state_id": state_index})
        self.index += 1
        return raw, info | {
            "screen_case_id": self.case["case_id"], "benchmark": self.case["benchmark"],
            "experiment_split": "development_only",
            "development_states_sha256": self.case["states_sha256"],
        }

    def step(self, action):
        if self.native is None:
            raise RuntimeError("reset must precede screen step")
        return self.native.step(action)

    def close(self):
        if self.native is not None:
            self.native.close()
            self.native = None


def make_env(*, ready_file, output):
    return SequentialCaseEnv(ready_file=ready_file, output=output)
