"""Prepare and run the isolated GPU-7 wine-to-cabinet RLDT experiment."""
from __future__ import annotations

import argparse
import copy
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
PROMPT = "Pick the wine from the counter and place it in the cabinet."


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def toml(path, mapping):
    lines = []
    def emit(data, prefix=""):
        if prefix:
            lines.append(f"[{prefix}]")
        for key, value in data.items():
            if not isinstance(value, dict):
                lines.append(f"{key} = {json.dumps(value)}")
        lines.append("")
        for key, value in data.items():
            if isinstance(value, dict):
                emit(value, f"{prefix}.{key}" if prefix else key)
    emit(mapping)
    Path(path).write_text("\n".join(lines))


def prepare(campaign, gpu=7):
    import tomllib
    campaign = Path(campaign).resolve()
    campaign.mkdir(parents=True, exist_ok=False)
    source = campaign / "source"
    (source / "integrations").mkdir(parents=True)
    shutil.copytree(ROOT / "integrations/multibench", source / "integrations/multibench",
                    ignore=shutil.ignore_patterns("__pycache__", "downloads", "numba_cache", "smoke", "validation_gpu"))
    (source / "scripts").mkdir()
    shutil.copy2(ROOT / "scripts/env.sh", source / "scripts/env.sh")
    for name in ("third_party", "checkpoints", ".venv", ".venv-robocasa365", ".cache"):
        (source / name).symlink_to(ROOT / name, target_is_directory=True)
    (source / "runs").mkdir()
    task = campaign / "task"
    task.mkdir()
    definition = {"task_id": "robocasa/PickPlaceCounterToCabinet", "max_steps": 750,
                  "native_kwargs": {"obj_groups": "wine", "expected_prompt": PROMPT}}
    for split, seeds in (("train", list(range(1000, 1100))),
                         ("validation", list(range(2000, 2005))), ("test", list(range(3000, 3005)))):
        definition[f"{split}_seeds"] = seeds
        definition[f"{split}_state_ids"] = seeds
    protocol = {"schema": 1, "tasks": {"robocasa": definition},
                "validation_iterations": [25, 50, 100], "online_validation": True,
                "base_checkpoint": "pi05_base", "task_sft": False,
                "split_note": "Disjoint seeds in native target kitchens, target object restricted to wine; custom task evaluation.",
                "selection": "Most validation successes; earliest checkpoint breaks ties. Test excluded from selection."}
    write(task / "protocol.json", protocol)
    config = tomllib.loads((ROOT / "configs/rldt_pi05_libero.toml").read_text())
    config["policy"] = {"kind": "pi05", "checkpoint": str(ROOT / "checkpoints/openpi/pi05_base"),
                        "model_config": "pi05_base", "benchmark": "robocasa", "projection_only": False}
    config["train"].update(seed=0, iterations=100, collect_decisions=32, critic_steps=64, actor_steps=4,
                           critic_batch_size=64, actor_batch_size=1, critic_trajectories=2,
                           replay_capacity=4096, save_every=25)
    config["rldt"].update(ode_steps=10, execute_steps=5, particles=8, actor_lr=0.00002,
                          actor_warmup_steps=32, critic_warmup_iterations=2, bootstrap_samples=2)
    config["recording"] = {"enabled": False}
    paths = [source / "integrations/multibench/core", source / "integrations",
             ROOT / "third_party/openpi/packages/openpi-client/src",
             ROOT / "third_party/robocasa365", ROOT / "third_party/robocasa365/dependencies/robosuite"]
    for split in ("train", "validation", "test"):
        cfg = copy.deepcopy(config)
        cfg["env"] = {"kind": "external", "factory": "multibench.environment:make_env",
                      "python": str(ROOT / ".venv-robocasa365/bin/python"), "gpu_rendering": False,
                      "process_env": {"PYTHONPATH": os.pathsep.join(map(str, paths)), "MUJOCO_GL": "osmesa",
                                      "PYOPENGL_PLATFORM": "osmesa", "LP_NUM_THREADS": "4", "NUMBA_DISABLE_JIT": "1"},
                      "kwargs": {"benchmark": "robocasa", "protocol": str(task / "protocol.json"), "split": split}}
        toml(task / f"{split}.toml", cfg)
    write(campaign / "experiment.json", {"method": "rldt", "benchmark": "robocasa", "task_key": "robocasa",
          "task_directory": str(task), "protocol_file": str(task / "protocol.json"), "protocol": protocol,
          "record_evaluation": False})
    gpu_uuid = subprocess.check_output(["nvidia-smi", "-i", str(gpu), "--query-gpu=uuid", "--format=csv,noheader"], text=True).strip()
    write(campaign / "launch.json", {"gpu_index": gpu, "gpu_uuid": gpu_uuid, "cpus": list(range(16, 24)),
          "source": str(source), "task_prompt": PROMPT, "stages": ["verify", "baseline_validation", "baseline_test", "training", "selected_test"],
          "training_iterations": 100, "max_training_controller_steps": 16000,
          "evaluation_episodes_per_split": 5, "trainable_scope": "Full action expert, frozen VLM",
          "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat()})
    files = [p for p in source.rglob("*") if p.is_file() and not any(parent.is_symlink() for parent in p.parents)]
    write(campaign / "source_manifest.json", {str(p.relative_to(source)): sha(p) for p in files})
    write(campaign / "queue_status.json", {"state": "prepared", "stage": None})
    print(campaign)


def verify(campaign):
    """Native reset/step, sampler parity, real full-expert optimizer and checkpoint gates."""
    import jax
    import jax.numpy as jnp
    import numpy as np
    from flax import nnx
    from openpi.models.model import Observation
    from rl_vla import rldt
    from rl_vla.rldt_env import ProcessEnv
    config = rldt.load_config(campaign / "task/train.toml")
    flow = rldt.make_flow(config["policy"])
    assert len(jax.devices()) == 1 and jax.devices()[0].platform == "gpu"
    assert (flow.horizon, flow.action_dim, flow.motor_dim) == (50, 32, 12)
    assert flow.metadata["trainable_parameters"] == 430098464
    env = ProcessEnv(config["env"])
    states = []
    initial_hashes = None
    try:
        for seed in (0, 1):
            raw, info = env.reset(seed=seed)
            assert raw["openpi"]["prompt"] == PROMPT and info["target_object_group"] == "wine"
            assert raw["state"].shape == (16,)
            for key in ("observation/image", "observation/wrist_image", "observation/right_wrist_image"):
                assert raw["openpi"][key].shape == (224, 224, 3)
                assert np.std(raw["openpi"][key]) > 1
            states.append(info)
            hashes = {name: hashlib.sha256(np.asarray(value).tobytes()).hexdigest()
                      for name, value in raw["openpi"].items() if name != "prompt"}
            if initial_hashes is None:
                initial_hashes = hashes
                from PIL import Image
                Image.fromarray(np.concatenate([raw["openpi"][name] for name in
                    ("observation/image", "observation/wrist_image", "observation/right_wrist_image")], axis=1)).save(campaign / "initial_wine_cameras.png")
            for _ in range(2):
                _, reward, terminated, truncated, step_info = env.step(np.zeros(12, np.float32))
                assert reward == float(step_info["is_success"])
    finally:
        env.close()
    env = ProcessEnv(config["env"])
    try:
        repeated, _ = env.reset(seed=0)
        assert initial_hashes == {name: hashlib.sha256(np.asarray(value).tobytes()).hexdigest()
                                  for name, value in repeated["openpi"].items() if name != "prompt"}
    finally:
        env.close()
    context = flow.prepare([raw])
    transformed = flow._transform_observation(raw)
    observation = Observation.from_dict(jax.tree.map(lambda a: jnp.asarray(np.stack([a])), transformed))
    key = jax.random.PRNGKey(123)
    def native(p, f, obs, k):
        model = nnx.merge(flow.graphdef, p, f)
        return model.sample_actions(k, obs, num_steps=10)
    native_action = jax.jit(native)(flow.params, flow.frozen, observation, key)
    adapted_action = jax.jit(lambda p, f, c, k: flow.sample(p, c, k, 10, frozen=f)[0])(
        flow.params, flow.frozen, context, key)
    difference = np.abs(np.asarray(native_action) - np.asarray(adapted_action))
    # Same mixed-precision model state, independent upstream sample_actions path.
    parity = {"max_abs": float(difference.max()), "mean_abs": float(difference.mean()), "atol": 0.02, "rtol": 0.02}
    np.testing.assert_allclose(adapted_action, native_action, atol=.02, rtol=.02)
    write(campaign / "preflight.json", {"passed": True, "devices": [str(x) for x in jax.devices()],
          "native_resets": states, "repeat_reset_identical": True,
          "sampler_parity": parity, "policy_metadata": flow.metadata})
    del native_action, adapted_action, observation
    jax.clear_caches()
    before = {"/".join(map(str, key)): np.asarray(value.value).copy()
              for key, value in flow.params.flat_state().items() if key[0] in {"action_in_proj", "action_out_proj"}}
    cfg = copy.deepcopy(config)
    cfg["train"].update(iterations=1, collect_decisions=2, critic_steps=1, actor_steps=1,
                         critic_batch_size=2, actor_batch_size=1, critic_trajectories=1)
    cfg["rldt"].update(bootstrap_samples=1, critic_warmup_iterations=0)
    original = rldt.make_flow
    rldt.make_flow = lambda _: flow
    try:
        summary = rldt.train(cfg, campaign / "checks/smoke")
    finally:
        rldt.make_flow = original
    metrics = [json.loads(line) for line in (campaign / "checks/smoke/metrics.jsonl").read_text().splitlines()]
    last = metrics[-1]["metrics"]
    assert last["actor_grad_norm"] > 0 and last["actor_update_norm"] > 0
    changed = 0
    with np.load(campaign / "checks/smoke/final/expert.npz", allow_pickle=False) as archive:
        expected = {"/".join(map(str, key)) for key in flow.params.flat_state()}
        assert set(archive.files) == expected
        count = 0
        for name in archive.files:
            value = archive[name]
            assert np.isfinite(value).all(), name
            count += value.size
            if name in before:
                changed += int(np.count_nonzero(value != before[name]))
        assert count == 430098464 and changed > 0
    restored = flow.load_params(campaign / "checks/smoke/final")
    restored_actions = jax.jit(lambda p, f, c, k: flow.sample(p, c, k, 10, frozen=f)[0])(
        restored, flow.frozen, context, key)
    assert np.isfinite(np.asarray(restored_actions)).all()
    write(campaign / "verification.json", {"passed": True, "particles": 8, "full_expert_parameters": count,
          "changed_projection_values": changed, "finite_saved_expert": True,
          "checkpoint_reload_inference_finite": True, "sampler_parity": parity,
          "summary": summary, "metrics": last,
          "note": "Software and optimizer validation; not evidence of task learning."})


def run(campaign):
    campaign = Path(campaign).resolve()
    launch = json.loads((campaign / "launch.json").read_text())
    source = Path(launch["source"])
    lock = (ROOT / ".cache" / f"rldt-wine-gpu{launch['gpu_index']}.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    os.sched_setaffinity(0, launch["cpus"])
    for relative, digest in json.loads((campaign / "source_manifest.json").read_text()).items():
        if sha(source / relative) != digest:
            raise RuntimeError(f"Frozen source changed: {relative}")
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=launch["gpu_uuid"], ROBOCASA_RENDER_THREADS="4",
               XLA_PYTHON_CLIENT_MEM_FRACTION="0.65", XLA_PYTHON_CLIENT_PREALLOCATE="false",
               OPENPI_DATA_HOME=str(ROOT / ".cache/openpi"), HF_HOME=str(ROOT / ".cache/huggingface"),
               PYTHONUNBUFFERED="1")
    env.pop("JAX_PLATFORMS", None)
    env.pop("XLA_FLAGS", None)
    command = ["bash", str(source / "integrations/multibench/run.sh"), "native"]
    status = {"state": "waiting_for_gpu", "gpu_index": launch["gpu_index"], "gpu_uuid": launch["gpu_uuid"],
              "pid": os.getpid(), "completed_stages": []}
    def update(**changes):
        status.update(changes, updated_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
        write(campaign / "queue_status.json", status)
    update()
    while subprocess.check_output(["nvidia-smi", "-i", launch["gpu_uuid"],
              "--query-compute-apps=pid", "--format=csv,noheader"], text=True).strip():
        time.sleep(30)
    def stage(name, args):
        update(state="running", stage=name, log=str(campaign / f"{name}.log"))
        with (campaign / f"{name}.log").open("x") as log:
            proc = subprocess.Popen(command + args, cwd=source, env=env, stdout=log, stderr=subprocess.STDOUT)
            update(child_pid=proc.pid)
            code = proc.wait()
            if code:
                raise RuntimeError(f"{name} exited {code}; see {name}.log")
        status["completed_stages"].append(name)
        update()
    def evaluation(name, split, checkpoint=None):
        args = ["-m", "multibench.evaluate", "--method", "rldt", "--config", str(campaign / "task/validation.toml"),
                "--protocol", str(campaign / "task/protocol.json"), "--benchmark", "robocasa", "--split", split,
                "--episodes", "5", "--seed", "100000" if split == "validation" else "200000",
                "--output", str(campaign / f"{name}.json")]
        if checkpoint:
            args += ["--checkpoint", checkpoint]
        stage(name, args)
    try:
        stage("verify", ["-m", "multibench.robocasa.wine_campaign", "verify", "--campaign", str(campaign)])
        if not json.loads((campaign / "verification.json").read_text())["passed"]:
            raise RuntimeError("GPU verification failed")
        evaluation("baseline_validation", "validation")
        evaluation("baseline_test", "test")
        stage("training", ["-m", "rl_vla.rldt", "train", "--config", str(campaign / "task/train.toml"),
                           "--output", str(campaign / "training")])
        from rl_vla.checkpoint_validation import select_checkpoint
        selected = select_checkpoint(campaign)
        evaluation("selected_test", "test", selected["checkpoint"])
        update(state="completed", stage=None, selected_checkpoint=selected)
    except BaseException as exc:
        update(state="failed", error=repr(exc))
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "run", "verify"])
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=7)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.campaign, args.gpu)
    elif args.command == "run":
        run(args.campaign)
    else:
        verify(args.campaign.resolve())
