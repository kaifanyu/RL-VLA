"""Simulator child process (LIBERO-PRO or original LIBERO) and its parent client.

The child runs in the simulator interpreter (``.venv-robustness`` with the PRO fork
on PYTHONPATH, or ``.venv`` for original LIBERO), so OpenPI's JAX environment and
the simulator stacks stay separate, exactly as for the robustness queue / RLDT.
PRO environments are built with ``rl_vla.robustness.make_case_env`` (the queue's
own constructor, seeding and state loading); original LIBERO with ``LiberoEnv``.

Messages use ``rl_vla.rldt_env``'s length-prefixed msgpack framing. Each reply
carries the policy-facing observation (``state`` and ``openpi`` dict, already
rotated/resized exactly as the queue does) plus a ``diag`` dict that is used ONLY
for logging/recording: simulated time, raw end-effector pose, the upright full
resolution external camera frame and simulator object poses.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from rl_vla.rldt_env import _receive, _send

ROOT = Path(__file__).resolve().parents[3]


# ----------------------------------------------------------------------------- child
class _Worker:
    def __init__(self):
        self.env = None
        self.raw = None

    def _hook(self, env):
        original = env._observation

        def observation(raw):  # Keep the raw robosuite dict for diagnostics only.
            self.raw = raw
            return original(raw)

        env._observation = observation

    def make(self, request):
        from rl_vla.runtime import seed_everything

        if self.env is not None:
            self.env.close()
            self.env = None
        seed = int(request["seed"])
        if request["kind"] == "pro_case":
            from rl_vla.robustness import make_case_env

            env, state_index = make_case_env(request["case"], seed, Path(request["output"]),
                                             int(request["initial_state_id"]))
        elif request["kind"] == "original":
            from libero.libero import benchmark

            from rl_vla.envs.libero import LiberoEnv

            name = benchmark.get_benchmark_dict()[request["suite"]]().get_task(int(request["task_id"])).name
            if request.get("name") and name != request["name"]:
                raise ValueError(f"Original task id {request['task_id']} is {name!r}, expected {request['name']!r}")
            seed_everything(seed, 1)  # Same order as rl_vla.baseline: seed, then reset.
            env = LiberoEnv(task_suite=request["suite"], task_id=int(request["task_id"]),
                            max_steps=int(request["max_steps"]), camera_resolution=256,
                            resize_size=224, settle_steps=10)
            state_index = int(request["initial_state_id"])
        elif request["kind"] == "source_protocol":
            # rl_vla.envs.libero_pro.make_env(split="source"): PRO fork, BDDL instruction as prompt.
            from libero.libero.envs import OffScreenRenderEnv

            from rl_vla.envs.libero import LiberoEnv
            from rl_vla.robustness import load_states

            seed_everything(seed, 1)
            max_steps = int(request["max_steps"])
            backend = OffScreenRenderEnv(bddl_file_name=request["bddl"], camera_heights=256, camera_widths=256,
                                         horizon=max(1000, max_steps + 11))
            env = LiberoEnv(task_suite=request["suite"], task_id=int(request["task_id"]), max_steps=max_steps,
                            camera_resolution=256, resize_size=224, settle_steps=10, backend=backend,
                            initial_states=load_states(request["states"]),
                            task_description=backend.language_instruction)
            state_index = int(request["initial_state_id"])
        else:
            raise ValueError(f"Unknown environment kind {request['kind']!r}")
        if request.get("max_steps") is not None:
            env.max_steps = int(request["max_steps"])
        self._hook(env)
        self.env = env
        inner = env.backend.env
        return {"state_index": int(state_index), "task_description": env.task_description,
                "max_steps": int(env.max_steps), "settle_steps": int(env.settle_steps),
                "low": env.action_space.low, "high": env.action_space.high,
                "control_timestep": float(inner.control_timestep),
                "control_freq": float(inner.control_freq),
                "available_initial_states": len(env.initial_states),
                "controller": type(env.backend.robots[0].controller).__name__,
                "controller_output_max": np.asarray(env.backend.robots[0].controller.output_max).tolist(),
                "goal_state": [list(map(str, g)) for g in inner.parsed_problem["goal_state"]],
                "objects": sorted(getattr(inner, "objects_dict", {}).keys())}

    def _diag(self, record: bool, objects: bool):
        raw, inner = self.raw, self.env.backend.env
        diag = {"sim_time": float(inner.sim.data.time),
                "eef_pos": np.asarray(raw["robot0_eef_pos"], np.float64),
                "eef_quat_xyzw": np.asarray(raw["robot0_eef_quat"], np.float64),
                "gripper_qpos": np.asarray(raw["robot0_gripper_qpos"], np.float64)}
        if record:
            # Raw robosuite renders are upside down; rotate 180 degrees for viewing.
            diag["agentview"] = np.ascontiguousarray(np.asarray(raw["agentview_image"])[::-1, ::-1])
        if objects:
            diag["objects"] = {name: np.concatenate([np.asarray(raw[f"{name}_pos"], np.float64),
                                                     np.asarray(raw[f"{name}_quat"], np.float64)])
                               for name in sorted(getattr(inner, "objects_dict", {}))
                               if f"{name}_pos" in raw and f"{name}_quat" in raw}
        return diag

    def accounting_phase(self, request):
        os.environ["RL_VLA_ATTEMPT_PHASE"] = request["phase"]
        return {"phase": request["phase"]}

    def reset(self, request):
        observation, info = self.env.reset(seed=int(request["seed"]),
                                           options={"initial_state_id": int(request["initial_state_id"])})
        return {"obs": observation, "info": info,
                "diag": self._diag(request.get("record", False), request.get("objects", False))}

    def step(self, request):
        observation, reward, terminated, truncated, info = self.env.step(np.asarray(request["action"], np.float32))
        info = {k: v for k, v in info.items() if isinstance(v, (bool, int, float, str, np.generic))}
        return {"obs": observation, "reward": float(reward), "terminated": bool(terminated),
                "truncated": bool(truncated), "info": info,
                "diag": self._diag(request.get("record", False), request.get("objects", False))}


def worker_main():
    output = os.fdopen(os.dup(sys.stdout.fileno()), "wb")
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())  # Library prints go to stderr.
    worker = _Worker()
    try:
        _send(output, {"ready": True, "python": sys.executable})
        while True:
            request = _receive(sys.stdin.buffer)
            command = request["command"]
            if command == "close":
                break
            try:
                result = getattr(worker, command)(request)
            except Exception:  # noqa: BLE001 - forward the traceback to the parent
                import traceback

                result = {"error": traceback.format_exc()}
            _send(output, result)
    finally:
        if worker.env is not None:
            worker.env.close()
        output.close()


# ---------------------------------------------------------------------------- parent
def simulator_environment(simulator_config: dict, source: str) -> tuple[str, dict]:
    """Interpreter and environment variables of a simulator child ('pro' or 'original')."""
    if source == "pro":
        python = simulator_config["pro_python"]
        pythonpath = str(ROOT / "third_party/libero_pro")
        libero_config = simulator_config["pro_libero_config"]
    elif source == "original":
        python = simulator_config["original_python"]
        pythonpath = ""
        libero_config = simulator_config["original_libero_config"]
    else:
        raise ValueError("source must be 'pro' or 'original'")
    python = str((ROOT / python) if not Path(python).is_absolute() else python)
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES="", LIBERO_CONFIG_PATH=libero_config,
               PYTHONPATH=pythonpath + (os.pathsep + str(ROOT / "src")))
    for key, value in (("MUJOCO_GL", "osmesa"), ("PYOPENGL_PLATFORM", "osmesa"),
                       ("NUMBA_DISABLE_JIT", "1"), ("OMP_NUM_THREADS", "1"),
                       ("LP_NUM_THREADS", "1"), ("OPENBLAS_NUM_THREADS", "1"),
                       ("MKL_NUM_THREADS", "1")):
        env.setdefault(key, value)
    # osmesa (CPU) is the robustness-queue protocol; egl renders on the GPU. The renderer
    # is set explicitly so the inherited scripts/env.sh value cannot override the config.
    renderer = simulator_config.get("renderer", "osmesa")
    env["MUJOCO_GL"] = env["PYOPENGL_PLATFORM"] = renderer
    if renderer == "egl":
        # CUDA_VISIBLE_DEVICES is empty for the worker, which robosuite's EGL device lookup
        # cannot parse; name the EGL device explicitly instead.
        env["MUJOCO_EGL_DEVICE_ID"] = str(simulator_config.get("egl_device_id", 0))
    return python, env


class SimClient:
    """Owns one simulator child. ``source`` is 'pro' or 'original'."""

    def __init__(self, simulator_config: dict, source: str, log_path: Path | None = None):
        self.source = source
        python, env = simulator_environment(simulator_config, source)
        self.environment = {k: env.get(k) for k in ("LIBERO_CONFIG_PATH", "PYTHONPATH", "MUJOCO_GL",
                                                    "PYOPENGL_PLATFORM", "MUJOCO_EGL_DEVICE_ID", "NUMBA_DISABLE_JIT",
                                                    "LP_NUM_THREADS")}
        self.python = python
        self._log = open(log_path, "ab") if log_path else subprocess.DEVNULL  # noqa: SIM115
        self.process = subprocess.Popen([python, "-m", "rl_vla.pro_residual.sim", "--worker"],
                                        stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        stderr=self._log, env=env, cwd=str(ROOT))
        self._reply()

    def _reply(self):
        result = _receive(self.process.stdout)
        if isinstance(result, dict) and "error" in result:
            raise RuntimeError("Simulator worker error:\n" + result["error"])
        return result

    def call(self, command: str, **kwargs):
        _send(self.process.stdin, {"command": command, **kwargs})
        return self._reply()

    def close(self):
        if self.process.poll() is None:
            try:
                _send(self.process.stdin, {"command": "close"})
                self.process.wait(timeout=30)
            except (BrokenPipeError, subprocess.TimeoutExpired):
                self.process.terminate()
                self.process.wait(timeout=10)
        for stream in (self.process.stdin, self.process.stdout):
            if stream:
                stream.close()
        if self._log is not subprocess.DEVNULL:
            self._log.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true", required=True)
    parser.parse_args()
    worker_main()
