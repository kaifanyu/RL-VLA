"""LIBERO-PRO tasks with frozen, disjoint initial-state splits for RLDT."""

import hashlib
import json
from pathlib import Path

from rl_vla.envs.libero import LiberoEnv
from rl_vla.robustness import load_states


def checked_file(record):
    path = Path(record["path"])
    if hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
        raise ValueError(f"Experiment asset changed: {path}")
    return path


class SplitLiberoEnv(LiberoEnv):
    """Sample only train states; enumerate evaluation states exactly once."""

    def __init__(self, *, state_ids, split, **kwargs):
        self.state_ids = tuple(state_ids)
        self.split = split
        self.evaluation_index = 0
        if not self.state_ids or len(set(self.state_ids)) != len(self.state_ids):
            raise ValueError("State IDs must be nonempty and unique")
        super().__init__(**kwargs)
        for state_id in self.state_ids:
            self._validate_state_id(state_id)

    def reset(self, *, seed=None, options=None):
        if options:
            raise ValueError("Experiment state selection is fixed by its split")
        if self.split == "train":
            # A separate RNG prevents backend/global RNG changes affecting selection.
            import numpy as np

            state_id = int(np.random.default_rng(seed).choice(self.state_ids))
        else:
            if self.evaluation_index >= len(self.state_ids):
                raise ValueError("Evaluation requested more episodes than frozen states")
            state_id = self.state_ids[self.evaluation_index]
        observation, info = super().reset(seed=seed, options={"initial_state_id": state_id})
        if self.split != "train":
            self.evaluation_index += 1
        return observation, info | {"experiment_split": self.split}


def make_env(*, protocol, task_key, split):
    if split not in {"train", "heldout", "source"}:
        raise ValueError(f"Unknown experiment split: {split}")
    plan = json.loads(Path(protocol).read_text())
    task = plan["tasks"][task_key]
    train_ids, heldout_ids = task["train_state_ids"], task["heldout_state_ids"]
    if set(train_ids) & set(heldout_ids):
        raise ValueError("Training and held-out state IDs overlap")
    if set(train_ids) | set(heldout_ids) != set(range(task["pro_state_count"])):
        raise ValueError("Training and held-out splits do not cover the PRO states")

    import libero.libero as libero_package
    from libero.libero.envs import OffScreenRenderEnv

    expected = Path(plan["pro_source"])
    if not Path(libero_package.__file__).resolve().is_relative_to(expected):
        raise RuntimeError("Use .venv-pro: a different LIBERO fork was imported")
    condition = task["source" if split == "source" else "pro"]
    bddl = checked_file(condition["bddl"])
    states = load_states(checked_file(condition["states"]))
    state_ids = task[f"{split}_state_ids"]
    backend = OffScreenRenderEnv(
        bddl_file_name=str(bddl), camera_heights=256, camera_widths=256,
        horizon=max(1000, task["max_steps"] + 11),
    )
    try:
        return SplitLiberoEnv(
            state_ids=state_ids, split=split, task_suite=task["suite"],
            task_id=task["task_id"], max_steps=task["max_steps"],
            camera_resolution=256, resize_size=224, settle_steps=10,
            backend=backend, initial_states=states,
            task_description=backend.language_instruction,
        )
    except Exception:
        backend.close()
        raise
