"""Lossless, bounded-memory rollout recording, without pickle or video codecs.

Each decision is an atomic compressed NPZ containing the initial observation and
every subsequent primitive-step observation. Read with ``np.load(...,
allow_pickle=False)`` and parse ``metadata_json.item()`` to recover the mapping
from observation array names to their original nested dictionary keys.
"""

import json
import time
from pathlib import Path

import numpy as np


def _json_value(value):
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _json_text(value):
    return json.dumps(_json_value(value), allow_nan=False, sort_keys=True)


def _write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(_json_text(value) + "\n")
    temporary.replace(path)


def _flatten_observation(observation, prefix=()):
    leaves = {}
    if not isinstance(observation, dict):
        raise TypeError("Recorded observations must be nested dictionaries")
    for key, value in observation.items():
        path = prefix + (str(key),)
        if isinstance(value, dict):
            leaves.update(_flatten_observation(value, path))
        else:
            array = np.asarray(value)
            if array.dtype.hasobject:
                raise TypeError(f"Object arrays cannot be recorded: {path}")
            leaves[path] = array.copy()
    return leaves


def _observation_arrays(observations):
    paths = sorted(observations[0])
    if any(set(observation) != set(paths) for observation in observations):
        raise ValueError("Observation keys changed within a recorded chunk")
    schema = {f"observation_{index:03d}": list(path) for index, path in enumerate(paths)}
    arrays = {name: np.stack([observation[tuple(path)] for observation in observations])
              for name, path in schema.items()}
    return arrays, schema


def _write_npz(path, arrays, metadata):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays, metadata_json=np.asarray(_json_text(metadata)))
    temporary.replace(path)


class TrajectoryRecorder:
    """Stream full primitive-step policy observations into one file per decision.

    This captures policy camera inputs, not unprocessed simulator cameras or
    simulator internal state. Reset metadata plus the frozen experiment state
    file identify the simulator initial condition. Episodes may span multiple
    training iterations and thus multiple behavior-policy parameter versions.
    """

    def __init__(self, directory, metadata):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.episode = None
        self.chunk = None
        self.episode_count = 0
        self.total_steps = 0
        self.total_chunks = 0
        _write_json(self.directory / "manifest.json", {
            "schema_version": 1, "format": "compressed-npz-json",
            "created_unix_ns": time.time_ns(), "metadata": metadata,
            "observation_semantics": "All primitive-step policy inputs, including reset and final "
            "observations. LIBERO images are rotated/resized policy inputs, not raw camera frames.",
            "chunk_semantics": "Observations have length executed_steps+1; action/reward/boundary "
            "arrays have length executed_steps. executed_actions are clipped commands actually "
            "sent to env.step. Consecutive chunks duplicate their shared boundary observation.",
            "simulator_state_semantics": "Reset identity refers to the frozen experiment state "
            "file. Internal simulator states are not included unless provided in reset metadata.",
        })

    @property
    def active(self):
        return self.episode is not None

    def start_episode(self, observation, metadata):
        if self.active:
            raise RuntimeError("Finish the recorded episode before starting another")
        self.episode_count += 1
        directory = self.directory / f"episode_{self.episode_count:06d}"
        directory.mkdir()
        self.episode = {"episode": self.episode_count, "metadata": metadata,
                        "started_unix_ns": time.time_ns(), "chunks": 0,
                        "primitive_steps": 0, "complete": False, "reason": "recording"}
        self.episode_directory = directory
        arrays, schema = _observation_arrays([_flatten_observation(observation)])
        _write_npz(directory / "reset.npz", arrays, {"observation_schema": schema})
        _write_json(directory / "episode.json", self.episode)

    def begin_chunk(self, observation, normalized_actions, decoded_actions, metadata):
        if not self.active or self.chunk is not None:
            raise RuntimeError("A chunk requires an active episode and no unfinished chunk")
        self.chunk = {
            "observations": [_flatten_observation(observation)],
            "normalized_actions": np.asarray(normalized_actions).copy(),
            "decoded_actions": np.asarray(decoded_actions).copy(),
            "metadata": metadata, "actions": [], "rewards": [], "terminated": [],
            "truncated": [], "infos": [], "timestamps": [],
        }

    def record_step(self, action, observation, reward, terminated, truncated, info):
        if self.chunk is None:
            raise RuntimeError("begin_chunk must precede recording primitive steps")
        self.chunk["observations"].append(_flatten_observation(observation))
        self.chunk["actions"].append(np.asarray(action).copy())
        self.chunk["rewards"].append(float(reward))
        self.chunk["terminated"].append(bool(terminated))
        self.chunk["truncated"].append(bool(truncated))
        # Snapshot metadata too: some environments reuse mutable info dictionaries.
        self.chunk["infos"].append(json.loads(_json_text(info)))
        self.chunk["timestamps"].append(time.time_ns())

    def end_chunk(self, metadata):
        if self.chunk is None or not self.chunk["actions"]:
            raise RuntimeError("Cannot finish a chunk without executed actions")
        chunk = self.chunk
        arrays, schema = _observation_arrays(chunk["observations"])
        arrays.update(
            normalized_actions=chunk["normalized_actions"],
            decoded_actions=chunk["decoded_actions"],
            executed_actions=np.stack(chunk["actions"]),
            rewards=np.asarray(chunk["rewards"], np.float64),
            terminated=np.asarray(chunk["terminated"], bool),
            truncated=np.asarray(chunk["truncated"], bool),
            step_unix_ns=np.asarray(chunk["timestamps"], np.int64),
        )
        count = len(chunk["actions"])
        chunk_index = self.episode["chunks"] + 1
        _write_npz(self.episode_directory / f"chunk_{chunk_index:06d}.npz", arrays, {
            "episode": self.episode_count, "chunk": chunk_index,
            "first_primitive_step": self.episode["primitive_steps"],
            "executed_steps": count, "observation_schema": schema,
            "decision": chunk["metadata"], "outcome": metadata, "step_infos": chunk["infos"],
        })
        self.episode["chunks"] += 1
        self.episode["primitive_steps"] += count
        self.total_chunks += 1
        self.total_steps += count
        self.chunk = None
        _write_json(self.episode_directory / "episode.json", self.episode)

    def end_episode(self, summary, *, reason="environment_boundary"):
        if not self.active or self.chunk is not None:
            raise RuntimeError("Finish the active chunk before finishing the episode")
        self.episode.update(summary=summary, reason=reason,
                            complete=reason == "environment_boundary",
                            finished_unix_ns=time.time_ns())
        _write_json(self.episode_directory / "episode.json", self.episode)
        with (self.directory / "episodes.jsonl").open("a") as stream:
            stream.write(_json_text(self.episode) + "\n")
        directory = str(self.episode_directory)
        self.episode = None
        return directory

    def close(self, *, reason="interrupted"):
        if self.chunk is not None:
            if self.chunk["actions"]:
                self.end_chunk({"partial_chunk": True, "reason": reason})
            else:
                self.chunk = None
        if self.active:
            self.end_episode({}, reason=reason)
        _write_json(self.directory / "summary.json", {
            "episodes": self.episode_count, "chunks": self.total_chunks,
            "primitive_steps": self.total_steps, "reason": reason,
        })
