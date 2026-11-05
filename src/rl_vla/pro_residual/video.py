"""MP4 writing and side-by-side paired comparison videos (no simulator, no JAX)."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def write_mp4(path: Path, frames, fps: float, *, lossless: bool = False) -> None:
    import imageio.v2 as imageio

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".tmp.mp4")
    params = ["-threads", "1"] + (["-crf", "0", "-preset", "veryslow"] if lossless else [])
    kwargs = {"fps": float(fps), "codec": "libx264", "macro_block_size": 1, "ffmpeg_params": params}
    if not lossless:
        kwargs["quality"] = 8
    imageio.mimwrite(temporary, [np.asarray(f, np.uint8) for f in frames], **kwargs)
    temporary.replace(path)


def read_mp4(path: Path) -> list[np.ndarray]:
    import imageio.v2 as imageio

    with imageio.get_reader(path) as reader:
        return [np.asarray(frame) for frame in reader]


def _font(size: int):
    from PIL import ImageFont

    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Older Pillow
        return ImageFont.load_default()


def _panel(frame, lines, color, scale: int):
    from PIL import Image, ImageDraw

    image = Image.fromarray(np.asarray(frame, np.uint8))
    if scale != 1:
        image = image.resize((image.width * scale, image.height * scale), Image.NEAREST)
    header = 18 * len(lines) + 8
    canvas = Image.new("RGB", (image.width, image.height + header), (20, 20, 20))
    canvas.paste(image, (0, header))
    draw = ImageDraw.Draw(canvas)
    font = _font(14)
    for i, (text, emphasis) in enumerate(lines):
        draw.text((6, 4 + 18 * i), text, fill=color if emphasis else (235, 235, 235), font=font)
    return np.asarray(canvas)


def comparison_frames(left: dict, right: dict, scale: int = 2) -> list[np.ndarray]:
    """Align two episodes by environment step; hold the shorter episode's last frame.

    Each side: {"frames": [...], "label": str, "success": bool, "steps": int, "max_steps": int}.
    Frame k shows the observation after k executed environment steps (k = 0: initial scene).
    """
    total = max(len(left["frames"]), len(right["frames"]))
    output = []
    for k in range(total):
        panels = []
        for side in (left, right):
            index = min(k, len(side["frames"]) - 1)
            finished = k >= len(side["frames"]) - 1
            outcome = "SUCCESS" if side["success"] else "FAIL (timeout)" if side.get("timeout") else "FAIL"
            color = (90, 220, 110) if side["success"] else (240, 90, 90)
            lines = [(side["label"], False),
                     (f"env step {index}/{side['max_steps']}" + ("  [ended, holding last frame]" if finished and k > index else ""), False),
                     (f"final: {outcome} at step {side['steps']}", True)]
            panels.append(_panel(side["frames"][index], lines, color, scale))
        gap = np.full((panels[0].shape[0], 6, 3), 255, np.uint8)
        output.append(np.concatenate([panels[0], gap, panels[1]], axis=1))
    return output


def _pad_height(panel, height: int):
    if panel.shape[0] >= height:
        return panel
    pad = np.full((height - panel.shape[0], panel.shape[1], 3), 20, np.uint8)
    return np.concatenate([panel, pad], axis=0)


def episode_frames(frames, policy_inputs: dict, *, max_steps: int, scale: int = 2) -> list[np.ndarray]:
    """One video per episode: simulator agentview | policy wrist input.

    Frame k is the scene after k executed environment steps. The wrist panel shows the most
    recent wrist image pi0.5 consumed at or before env step k (it is queried every
    execute_steps steps), exactly as fed to the model, so it updates in steps rather than
    every frame. The policy front image is the same camera as agentview (rotated) and is
    kept only in the policy-inputs npz.
    """
    env_steps = np.asarray(policy_inputs["env_step"])
    output = []
    for k, frame in enumerate(frames):
        later = np.nonzero(env_steps <= k)[0]
        i = int(later[-1]) if later.size else 0
        call, at = int(policy_inputs["call"][i]), int(env_steps[i])
        panels = [_panel(frame, [("agentview (simulator camera)", False), (f"env step {k}/{max_steps}", False)],
                         (235, 235, 235), scale),
                  _panel(policy_inputs["wrist"][i], [("policy input: wrist (as fed to pi0.5)", False),
                                                     (f"inference #{call} at env step {at}", False)],
                         (235, 235, 235), scale)]
        height = max(p.shape[0] for p in panels)
        panels = [_pad_height(p, height) for p in panels]
        gap = np.full((height, 6, 3), 255, np.uint8)
        output.append(np.concatenate([panels[0], gap, panels[1]], axis=1))
    return output
