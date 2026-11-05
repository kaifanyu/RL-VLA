"""Deterministic pixel controls. Native attention knockouts live in pi05_attention."""

import gymnasium as gym
import numpy as np


def replace_region(image, mask, mode="background_fill", *, background_mask=None, block=14):
    image, mask = np.asarray(image), np.asarray(mask, dtype=bool)
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3 or mask.shape != image.shape[:2]:
        raise ValueError("Expected HWC uint8 RGB and a matching mask")
    result = image.copy()
    if mode == "none":
        return result
    if mode == "background_fill":
        background = ~mask if background_mask is None else np.asarray(background_mask, bool) & ~mask
        if not background.any():
            raise ValueError("No background pixels available")
        result[mask] = np.median(image[background], axis=0).astype(np.uint8)
    elif mode == "mosaic":
        if block < 1:
            raise ValueError("Mosaic block must be positive")
        # Compute each block from the original, then apply only masked pixels.
        mosaic = image.copy()
        for y in range(0, image.shape[0], block):
            for x in range(0, image.shape[1], block):
                tile = image[y:y + block, x:x + block]
                mosaic[y:y + block, x:x + block] = np.rint(tile.mean(axis=(0, 1))).astype(np.uint8)
        result[mask] = mosaic[mask]
    else:
        raise ValueError(f"Unknown replacement mode: {mode}")
    return result


def matched_area_control(mask, allowed, *, seed):
    """Random compact region with exactly the same area within allowed background."""
    mask, allowed = np.asarray(mask, bool), np.asarray(allowed, bool)
    if mask.shape != allowed.shape or mask.ndim != 2:
        raise ValueError("Masks must have the same 2D shape")
    count = int(mask.sum())
    candidates = np.flatnonzero(allowed & ~mask)
    if len(candidates) < count:
        raise ValueError("Insufficient allowed pixels for the matched-area control")
    result = np.zeros_like(mask)
    if count:
        center = np.unravel_index(np.random.default_rng(seed).choice(candidates), mask.shape)
        yy, xx = np.indices(mask.shape)
        distance = ((yy - center[0]) ** 2 + (xx - center[1]) ** 2).ravel()[candidates]
        result.ravel()[candidates[np.argsort(distance, kind="stable")[:count]]] = True
    return result


def action_delta(reference, intervention):
    a, b = np.asarray(reference, float), np.asarray(intervention, float)
    if a.shape != b.shape or a.ndim != 2 or not np.isfinite([a, b]).all():
        raise ValueError("Expected matched finite [time,action] arrays")
    difference = b - a
    return {"l2": float(np.linalg.norm(difference)), "rms": float(np.sqrt(np.mean(difference**2))),
            "per_step_l2": np.linalg.norm(difference, axis=-1).tolist(),
            "xyz_l2": float(np.linalg.norm(difference[:, :3]))}


class DiagnosticMaskEnv(gym.Env):
    """Attach current visible instance masks only at policy chunk boundaries."""

    def __init__(self, env, mask_every=5, region_spec=None):
        if mask_every < 1:
            raise ValueError("mask_every must be positive")
        self.env = env
        self.mask_every = mask_every
        self.region_spec=region_spec
        from rl_vla.diagnostics.sensor_masks import SensorMaskCapture
        self.sensor_capture=SensorMaskCapture(env)

    def __getattr__(self, name):
        return getattr(self.env, name)

    def _attach(self, observation):
        masks, metadata = self.sensor_capture.masks(observation)
        if self.region_spec is not None:
            from rl_vla.diagnostics.old_target import project_old_target_region
            old_masks,old_metadata=project_old_target_region(self.env,**self.region_spec,allow_invisible=True,
                                                            camera_transforms=self.sensor_capture.transforms)
            masks.update(old_masks)
            metadata["old_target_region"]=old_metadata
        return dict(observation, diagnostic_masks=masks, diagnostic_mask_metadata=metadata)

    def reset(self, **kwargs):
        self.sensor_capture.enabled=True
        observation, info = self.env.reset(**kwargs)
        self.sensor_capture.enabled=False
        return self._attach(observation), info

    def step(self, action):
        self.sensor_capture.enabled=(self.env.steps+1)%self.mask_every==0
        observation, reward, terminated, truncated, info = self.env.step(action)
        if self.sensor_capture.enabled:
            observation = self._attach(observation)
        self.sensor_capture.enabled=False
        return observation, reward, terminated, truncated, info

    def close(self):
        self.env.close()


def make_env(*, protocol, task_key, split, mask_every=5, limit=None):
    """ProcessEnv-compatible diagnostics factory; leaves policy pixels unchanged."""
    import json
    from pathlib import Path

    from rl_vla.diagnostics.observation_bank import TARGETS
    from rl_vla.envs.libero_pro import make_env as make_pro_env
    env = make_pro_env(protocol=protocol, task_key=task_key, split=split)
    if limit is not None:
        if split == "train" or not 1 <= limit <= len(env.state_ids):
            raise ValueError("A positive diagnostic subset is allowed only for evaluation")
        # A telemetry decorator forwards attributes only for reads.
        underlying = env.env if hasattr(env, "initial_z") else env
        underlying.state_ids = underlying.state_ids[:limit]
    definition=json.loads(Path(protocol).read_text())["tasks"][task_key]
    return DiagnosticMaskEnv(env, mask_every=mask_every,region_spec={"source_bddl":definition["source"]["bddl"]["path"],
                                                                  "target":TARGETS[task_key]+"_1"})
