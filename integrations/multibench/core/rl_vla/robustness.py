"""Isolated LIBERO-Plus / PRO rollout worker and pinned evaluation plans."""
import argparse
import ast
import hashlib
import io
import json
import pickle
import re
import subprocess
import time
import zipfile
from pathlib import Path

import numpy as np

from rl_vla.base_policy import OpenPIBasePolicy
from rl_vla.baseline import HORIZONS, EvaluationClient, RecordingEnv, ResourceGuard, atomic_json
from rl_vla.config import load_config
from rl_vla.env_adapter import ResidualChunkEnv
from rl_vla.envs.libero import LiberoEnv
from rl_vla.runtime import seed_everything

ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT.parent / "datasets/benchmark_assets"
SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
PRO_CATEGORIES = {"object": "Object", "swap": "Position", "lan": "Language",
                  "task": "Task", "env": "Environment"}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def latin1_bytes(value, encoding):
    if not isinstance(value, str) or encoding != "latin1":
        raise pickle.UnpicklingError("Only protocol-2 Latin-1 byte reconstruction is allowed")
    return value.encode("latin1")


class NumpyOnlyUnpickler(pickle.Unpickler):
    """Read the NumPy-only ZIP format written by PRO's state generator."""
    def find_class(self, module, name):
        allowed = {
            ("numpy", "ndarray"): np.ndarray, ("numpy", "dtype"): np.dtype,
            ("numpy.core.multiarray", "_reconstruct"): np.core.multiarray._reconstruct,
            ("numpy.core.multiarray", "scalar"): np.core.multiarray.scalar,
            ("numpy.core.numeric", "_frombuffer"): np.core.numeric._frombuffer,
            ("_codecs", "encode"): latin1_bytes,
        }
        key = (module.replace("numpy._core", "numpy.core"), name)
        if key not in allowed:
            raise pickle.UnpicklingError(f"Unexpected class in initial states: {key}")
        return allowed[key]


def load_states(path):
    path = Path(path)
    if path.suffix == ".npy":
        values = np.load(path, allow_pickle=False)
    elif zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            if set(names) == {"archive/data.pkl", "archive/version"}:
                values = NumpyOnlyUnpickler(io.BytesIO(archive.read("archive/data.pkl"))).load()
            else:
                values = load_torch_states(path)
    else:
        values = load_torch_states(path)
    values = np.asarray(values)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    if values.ndim != 2 or not len(values) or not np.isfinite(values).all():
        raise ValueError(f"Invalid initial states: {path}")
    return values


def load_torch_states(path):
    import torch

    allowed = [np.core.multiarray._reconstruct, np.core.multiarray.scalar,
               np.ndarray, np.dtype, type(np.dtype(np.float64)), type(np.dtype(np.float32))]
    with torch.serialization.safe_globals(allowed):
        values = torch.load(path, map_location="cpu", weights_only=True)
    return values.numpy() if isinstance(values, torch.Tensor) else values


def task_map(repository):
    tree = ast.parse((repository / "libero/libero/benchmark/libero_suite_task_map.py").read_text())
    return ast.literal_eval(tree.body[0].value)


def plus_state_path(source, suite, name):
    """Match pinned Plus Benchmark.get_task_init_states, including layout states."""
    filename = name + ".pruned_init"
    root = source / "libero/libero/init_files"
    if "_language_" in filename:
        filename = filename.split("_language_")[0] + ".pruned_init"
    elif "_view_" in filename:
        filename = filename.split("_view_")[0] + ".pruned_init"
    else:
        if "_table_" in filename:
            filename = re.sub(r"_table_\d+", "", filename)
        if "_tb_" in filename:
            filename = re.sub(r"_tb_\d+", "", filename)
        if "_light_" in filename:
            filename = filename.split("_light_")[0] + ".pruned_init"
        if "_add_" in name or "_level" in name:
            root /= "libero_newobj"
    return root / suite / filename


def build_plan():
    cases = []
    for benchmark in ("plus", "pro"):
        source = ROOT / "third_party" / f"libero_{benchmark}"
        mapping = task_map(source)
        commit = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
        if benchmark == "plus":
            classification = json.loads((source / "libero/libero/benchmark/task_classification.json").read_text())
            for suite in SUITES:
                labels = {item["name"]: item for item in classification[suite]}
                for task_id, name in enumerate(mapping[suite]):
                    label = labels[name]
                    bddl = source / "libero/libero/bddl_files" / suite / (name + ".bddl")
                    real_bddl = bddl.with_name(name.split("_view_")[0] + ".bddl")
                    states = plus_state_path(source, suite, name)
                    cases.append({"benchmark": benchmark, "suite": suite, "task_id": task_id,
                                  "name": name, "category": label["category"],
                                  "difficulty": label["difficulty_level"], "episodes": 1,
                                  "bddl": str(bddl), "real_bddl": str(real_bddl),
                                  "states": str(states), "source_commit": commit})
        else:
            for suffix, category in PRO_CATEGORIES.items():
                for suite in SUITES:
                    for task_id, name in enumerate(mapping[f"{suite}_{suffix}"]):
                        base = ASSETS / "libero_pro"
                        if suffix == "env":
                            base = ASSETS / "libero_pro_generated"
                        bddl = base / "bddl_files" / f"{suite}_{suffix}" / (name + ".bddl")
                        states = None if suffix == "env" else str(
                            base / "init_files" / f"{suite}_{suffix}" / (name + ".pruned_init"))
                        cases.append({"benchmark": benchmark, "suite": suite, "task_id": task_id,
                                      "name": name, "category": category, "difficulty": None,
                                      "episodes": 50, "bddl": str(bddl), "real_bddl": str(bddl),
                                      "states": states, "source_commit": commit,
                                      "state_origin": "upstream_constructor_generated_seeded" if suffix == "env"
                                      else "published_initial_states"})
    state_counts = {}
    for case in cases:
        key = f"{case['benchmark']}:{case['suite']}:{case['category']}:{case['name']}"
        case["case_id"] = f"{case['benchmark']}_{case['suite']}_{hashlib.sha256(key.encode()).hexdigest()[:16]}"
        case["bddl_sha256"] = sha256(case["real_bddl"])
        if case["states"]:
            case["states_sha256"] = sha256(case["states"])
            if case["states"] not in state_counts:
                state_counts[case["states"]] = len(load_states(case["states"]))
            case["available_initial_states"] = state_counts[case["states"]]
            if case["available_initial_states"] < case["episodes"]:
                raise ValueError(f"Not enough states for {case['case_id']}")
    return {"version": 1, "model_config": "pi05_libero", "cases": cases,
            "horizons": {suite: HORIZONS[suite] for suite in SUITES},
            "camera_resolution": 256, "resize_size": 224, "settle_steps": 10,
            "action_horizon": 10, "execute_steps": 5, "request_pause": 0.25,
            "noise_protocol": "numpy-pcg64-float32-standard-normal-v1",
            "plus_hf_revision": "dd2bd61b7d9a6fef1abc52d606e983b41886a149",
            "pro_hf_revision": "c86fc3b8293185a6f373677018ff3e37f8391602",
            "pro_environment_note": "Missing from published HF bundle; generated with pinned "
            "upstream EnvironmentReplacePerturbator, which hardcodes living_room_table. "
            "Five Long tasks already use this environment; these are unchanged controls. "
            "Initial states generated at upstream constructor sampling point, with saved seeds."}


def episode_seed(case_id, index):
    return int.from_bytes(hashlib.sha256(f"{case_id}:{index}".encode()).digest()[:4], "little") % (2**31 - 1)


def make_case_env(case, seed, output, index):
    # A worker imports exactly one fork, selected by the supervisor's PYTHONPATH.
    from libero.libero.envs import OffScreenRenderEnv

    seed_everything(seed, 1)
    backend = OffScreenRenderEnv(bddl_file_name=case["bddl"], camera_heights=256,
                                camera_widths=256, horizon=max(1000, HORIZONS[case["suite"]] + 11))
    try:
        state_index = index
        if case["states"]:
            states = load_states(case["states"])
        else:
            # Same state sampling point as PRO's published generate_init_states.py:
            # a freshly constructed environment, before calling reset or stepping.
            path = output / "generated_states" / case["case_id"] / f"{index:03d}.npy"
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.exists():
                temporary = path.with_suffix(".tmp")
                with temporary.open("wb") as stream:
                    np.save(stream, np.asarray(backend.get_sim_state()).reshape(1, -1), allow_pickle=False)
                temporary.replace(path)
            states, state_index = load_states(path), 0
        if state_index >= len(states):
            raise ValueError(f"Insufficient published states for {case['case_id']}: {len(states)}")
        # BDDL is authoritative: filename-derived prompts discard PRO language
        # changes and can append Plus camera/noise parameters to the instruction.
        env = LiberoEnv(task_suite=case["suite"], task_id=case["task_id"],
                        max_steps=HORIZONS[case["suite"]], backend=backend, initial_states=states,
                        task_description=backend.language_instruction)
        return env, state_index
    except Exception:
        backend.close()
        raise


def run_case(args):
    case = json.loads(args.case.read_text())
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    client = None if args.smoke else EvaluationClient("127.0.0.1", args.port,
        ResourceGuard(output / "resources.jsonl"), 0.25)
    if client is not None:
        server = json.loads((output / "server_manifest.json").read_text())
        if client.metadata["checkpoint_sha256"] != server["checkpoint_sha256"]:
            raise ValueError("Worker connected to a different checkpoint")
    template = load_config(ROOT / "configs/libero_sac.toml")
    count = min(case["episodes"], args.limit) if args.limit else case["episodes"]
    try:
        for index in range(args.start_index, count):
            episode_id = f"{case['case_id']}_init{index:03d}"
            path = output / "episodes" / (episode_id + ".json")
            if path.exists() and not args.smoke:
                saved = json.loads(path.read_text())
                if saved["case_id"] != case["case_id"] or saved["initial_state_id"] != index:
                    raise ValueError("Saved episode does not match the case")
                continue
            seed = episode_seed(case["case_id"], index)
            atomic_json(output / "heartbeat.json", {"episode_id": episode_id, "started": time.time()})
            began = time.perf_counter()
            native, state_index = make_case_env(case, seed, output, index)
            recording = RecordingEnv(native)
            recording.frames = [] if args.video and index == 0 else None
            env = recording
            try:
                if client is not None:
                    client.episode_key = f"{episode_id}:{seed}"
                    policy = OpenPIBasePolicy(7, 10, client=client, host="127.0.0.1", port=args.port)
                    env = ResidualChunkEnv(recording, policy, template["chunk"])
                _, reset = env.reset(seed=seed, options={"initial_state_id": state_index})
                image = recording.last_observation["openpi"]["observation/image"]
                if image.std() == 0:
                    raise ValueError("Simulator returned a blank image")
                if (args.video and index == 0) or args.smoke:
                    from PIL import Image
                    target = output / "observations" / episode_id
                    target.mkdir(parents=True, exist_ok=True)
                    Image.fromarray(image).save(target / "front.png")
                    Image.fromarray(recording.last_observation["openpi"]["observation/wrist_image"]).save(target / "wrist.png")
                if args.smoke:
                    result = env.step(np.array([0, 0, 0, 0, 0, 0, -1], dtype=np.float32))
                    atomic_json(output / "smoke" / f"{case['case_id']}.json", {
                        "case": case, "prompt": native.task_description, "reset": reset,
                        "goal": native.backend.env.parsed_problem["goal_state"],
                        "camera_positions": native.backend.sim.model.cam_pos.tolist(),
                        "native_perturbation": {name: getattr(native.backend.env, name, None)
                            for name in ("horizon_view", "vertical_view", "scale_factor",
                                         "end_point_rot", "end_point_vertical", "init_state")},
                        "sensor_noise_code": getattr(native.backend, "noise", 0),
                        "robot_model": type(native.backend.robots[0].robot_model).__name__,
                        "robot_initial_joint_target": native.backend.robots[0].robot_model.init_qpos.tolist(),
                        "robot_state": recording.last_observation["state"].tolist(),
                        "image_std": float(image.std()), "reward": result[1]})
                    print(f"SMOKE PASSED {case['category']} {case['case_id']}", flush=True)
                    return
                steps, decisions, total, projected = 0, 0, 0.0, 0.0
                while True:
                    _, _, terminated, truncated, info = env.step(np.zeros(env.action_dim, dtype=np.float32))
                    steps += info["executed_steps"]
                    decisions += 1
                    total += info["raw_return"]
                    projected += info["projection_fraction"] * info["executed_steps"]
                    atomic_json(output / "heartbeat.json", {"episode_id": episode_id,
                                "updated": time.time(), "primitive_steps": steps})
                    if terminated or truncated:
                        break
                success = bool(info["is_success"])
                if success != bool(terminated) or total != float(success):
                    raise RuntimeError("Task success, termination and sparse reward disagree")
                row = {"episode_id": episode_id, "case_id": case["case_id"],
                       "benchmark": case["benchmark"], "suite": case["suite"],
                       "task_id": case["task_id"], "task_name": case["name"],
                       "category": case["category"], "difficulty": case["difficulty"],
                       "state_origin": case.get("state_origin", "published_initial_states"),
                       "initial_state_id": index, "seed": seed, "task_description": native.task_description,
                       "success": success, "return": total, "primitive_steps": steps,
                       "decisions": decisions, "terminated": terminated, "truncated": truncated,
                       "projection_fraction": projected / steps,
                       "inference_latency_ms": client.latencies.copy(),
                       "elapsed_seconds": time.perf_counter() - began, "reset_info": reset}
                atomic_json(path, row)
                if recording.frames:
                    import imageio.v2 as imageio
                    target = output / "videos" / f"{episode_id}.mp4"
                    target.parent.mkdir(parents=True, exist_ok=True)
                    imageio.mimwrite(target, recording.frames, fps=10, codec="libx264", quality=6,
                                    ffmpeg_params=["-threads", "1"])
                print(f"{episode_id} success={success} steps={steps} seconds={row['elapsed_seconds']:.1f}", flush=True)
            finally:
                env.close()
    finally:
        if client is not None:
            client.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    run_case(args)


if __name__ == "__main__":
    main()
