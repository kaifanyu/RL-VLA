"""Primitive environment IPC: keep OpenPI and the existing simulator venv isolated."""

import argparse
import json
import os
import struct
import subprocess
import sys

import numpy as np


def _send(stream, value):
    from openpi_client import msgpack_numpy

    payload = msgpack_numpy.packb(value)
    stream.write(struct.pack("!Q", len(payload)) + payload)
    stream.flush()


def _read_exact(stream, count):
    result = bytearray()
    while len(result) < count:
        block = stream.read(count - len(result))
        if not block:
            raise EOFError("Simulator worker exited; see its stderr")
        result.extend(block)
    return bytes(result)


def _receive(stream):
    from openpi_client import msgpack_numpy

    length = struct.unpack("!Q", _read_exact(stream, 8))[0]
    if length > 256 * 1024**2:
        raise ValueError("Unexpectedly large environment message")
    return msgpack_numpy.unpackb(_read_exact(stream, length))


class ProcessEnv:
    """Local child, no network service, no shared GPU policy state."""

    def __init__(self, config):
        config = dict(config)
        python = config.pop("python")
        environment = os.environ.copy()
        simulator_env = config.pop("process_env", {})
        gpu_rendering = config.pop("gpu_rendering", False)
        if not gpu_rendering:
            environment["CUDA_VISIBLE_DEVICES"] = ""
        environment.update({str(k): str(v) for k, v in simulator_env.items()})
        environment["JAX_PLATFORMS"] = "cpu"
        environment["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
        self.process = subprocess.Popen(
            [python, "-m", "rl_vla.rldt_env", "--worker", json.dumps(config)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, env=environment,
        )
        result = self._reply()
        self.low = np.asarray(result["low"], np.float32)
        self.high = np.asarray(result["high"], np.float32)

    def _reply(self):
        result = _receive(self.process.stdout)
        if isinstance(result, dict) and "error" in result:
            raise RuntimeError(result["error"])
        return result

    def reset(self, *, seed):
        _send(self.process.stdin, {"command": "reset", "seed": seed})
        return self._reply()

    def step(self, action):
        _send(self.process.stdin, {"command": "step", "action": action})
        return self._reply()

    def close(self):
        if self.process.poll() is None:
            try:
                _send(self.process.stdin, {"command": "close"})
                self.process.wait(timeout=15)
            except (BrokenPipeError, subprocess.TimeoutExpired):
                self.process.terminate()
                self.process.wait(timeout=5)
        self.process.stdin.close()
        self.process.stdout.close()


def make_primitive_env(config):
    if config.get("python"):
        return ProcessEnv(config)
    from rl_vla.envs import make_env

    env = make_env(config)
    env.low, env.high = env.action_space.low, env.action_space.high
    return env


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", required=True)
    args = parser.parse_args()
    # LIBERO and Gym print during import/reset. Reserve the original stdout fd
    # exclusively for binary messages, redirect every library print to stderr.
    output = os.fdopen(os.dup(sys.stdout.fileno()), "wb")
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    env = None
    try:
        config = json.loads(args.worker)
        from rl_vla.envs import make_env

        env = make_env(config)
        _send(output, {"low": env.action_space.low, "high": env.action_space.high})
        while True:
            request = _receive(sys.stdin.buffer)
            if request["command"] == "close":
                break
            if request["command"] == "reset":
                result = env.reset(seed=request["seed"])
            elif request["command"] == "step":
                result = env.step(np.asarray(request["action"], np.float32))
            else:
                raise ValueError("Unknown simulator command")
            _send(output, result)
    except Exception:  # noqa: BLE001 - send worker traceback across the IPC boundary
        import traceback

        _send(output, {"error": traceback.format_exc()})
    finally:
        if env is not None:
            env.close()
        output.close()


if __name__ == "__main__":
    main()
