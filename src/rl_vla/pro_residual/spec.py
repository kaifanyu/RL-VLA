"""Configuration, task identity resolution and the reproducible split manifest.

The task is resolved from the pinned LIBERO-PRO source task map with the same
case-id hashing used by ``rl_vla.robustness.build_plan``, then cross-checked
against the frozen robustness ``protocol.json`` when that file is present.
"""

from __future__ import annotations

import ast
import copy
import hashlib
import json
import re
import subprocess
import tomllib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]


def _existing(*candidates: Path) -> Path:
    """First existing path; hosts differ on whether assets sit in or beside the repository."""
    return next((path for path in candidates if path.exists()), candidates[0])


ASSETS = _existing(ROOT.parent / "datasets/benchmark_assets", ROOT / "datasets/benchmark_assets")
PRO_CATEGORIES = {"object": "Object", "swap": "Position", "lan": "Language",
                  "task": "Task", "env": "Environment"}
HORIZONS = {"libero_spatial": 220, "libero_object": 280, "libero_goal": 300, "libero_10": 520}
SUITE_SHORT = {"libero_spatial": "spatial", "libero_object": "object", "libero_goal": "goal",
               "libero_10": "long"}
ORIGINAL_SUITE_INDEX = {"libero_spatial": 0, "libero_object": 1, "libero_goal": 2, "libero_10": 3}
ACTION_NAMES = ("x", "y", "z", "rx", "ry", "rz", "gripper")
NOISE_PROTOCOLS = ("numpy-pcg64-float32-standard-normal-v1", "jax-prngkey-split-normal-v1")

DEFAULTS = {
    "experiment": "flashsac_spatial_01",
    "output_dir": "runs/flashsac_spatial_01",
    "task": {"benchmark": "pro", "suite": "libero_spatial", "category": None, "task_id": None,
             "expected_case_id": None, "label": None, "label_resolution": None,
             "frozen_protocol": "runs/evaluation/pi05_libero_robustness_20260929/protocol.json",
             "state_set": None},
    "split": {"seed": 20261029, "train": 30, "validation": 10, "test": 10,
              "retention_initial_states": 10, "duplicate_l2_threshold": 1e-6},
    "simulator": {"pro_python": ".venv-robustness/bin/python", "original_python": ".venv/bin/python",
                  "pro_libero_config": str(_existing(ROOT.parent / ".cache/rl-vla-libero-pro", ROOT / ".cache/libero-pro")),
                  "original_libero_config": str(_existing(ROOT.parent / ".cache/rl-vla-libero", ROOT / ".cache/libero")),
                  "max_steps": None, "renderer": "osmesa"},
    "base": {"checkpoint": "checkpoints/openpi/pi05_libero", "model_config": "pi05_libero",
             "action_horizon": 10, "execute_steps": 5,
             "noise_protocol": "numpy-pcg64-float32-standard-normal-v1",
             "backend": "openpi_policy", "ode_steps": 10},
    "residual": {"scale": [0.1, 0.1, 0.1, 0.05, 0.05, 0.05, 0.0],
                 "action_low": [-1.0] * 7, "action_high": [1.0] * 7,
                 "use_vlm_features": True, "eval_deterministic": True, "gamma": 0.99},
    "flashsac": {},
    "train": {"seed": 0, "device": "auto", "total_env_steps": 50_000, "learning_starts": 2_000,
              "batch_size": 256, "updates_per_step": 1, "replay_capacity": 100_000,
              "log_every_steps": 200, "checkpoint_every_steps": 5_000,
              "validate_every_steps": 10_000, "validate_at_steps": None, "validate_at_start": True,
              "validation_states": "all", "save_replay": True, "torch_threads": 2,
              "max_gpu_memory_gib": 46.0, "max_gpu_temperature_c": 77.0,
              "max_workspace_gib": 185.0},
    # Shared benchmark contract. The defaults keep the historical FlashSAC behavior.
    "contract": {"name": None, "eval_seed_start": None, "train_reset": "legacy", "budget": "episode",
                 "retention_env": "original",
                 "states_sha256": None, "bddl_sha256": None, "base_parameter_sha256": None},
    "record": {"train_every_episodes": 25, "max_train_recordings": 8, "validation_videos": False,
               "policy_inputs": "npz", "object_poses": True},
}


def deep_merge(base: dict, override: dict) -> dict:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_config(path: str | Path) -> dict:
    with Path(path).open("rb") as stream:
        raw = tomllib.load(stream)
    if raw.get("algorithm") != "flashsac_pro_residual":
        raise ValueError("Expected algorithm = 'flashsac_pro_residual'")
    config = deep_merge(DEFAULTS, raw)
    validate_config(config)
    config["config_path"] = str(Path(path).resolve())
    return config


def validate_config(config: dict) -> None:
    task = config["task"]
    if task["benchmark"] != "pro":
        raise ValueError("Only benchmark = 'pro' is implemented")
    if task["category"] not in PRO_CATEGORIES.values():
        raise ValueError(f"task.category must be one of {sorted(PRO_CATEGORIES.values())}")
    if config["simulator"].get("renderer", "osmesa") not in ("osmesa", "egl"):
        raise ValueError("simulator.renderer must be 'osmesa' (queue protocol) or 'egl' (GPU rendering)")
    if task["category"] == "Environment":
        raise ValueError("PRO Environment cases use locally generated per-episode states; the "
                         "split manifest needs those states first. Not supported yet.")
    if not isinstance(task["task_id"], int) or task["task_id"] < 0:
        raise ValueError("task.task_id must be a nonnegative integer")
    base = config["base"]
    if not 1 <= int(base["execute_steps"]) <= int(base["action_horizon"]):
        raise ValueError("Require 1 <= base.execute_steps <= base.action_horizon")
    residual = config["residual"]
    for key in ("scale", "action_low", "action_high"):
        values = np.asarray(residual[key], dtype=np.float64)
        if values.shape != (7,) or not np.isfinite(values).all():
            raise ValueError(f"residual.{key} must contain seven finite values")
    scale = np.asarray(residual["scale"], np.float64)
    low, high = np.asarray(residual["action_low"]), np.asarray(residual["action_high"])
    if np.any(scale < 0) or not np.any(scale > 0):
        raise ValueError("residual.scale must be nonnegative with at least one active dimension")
    if np.any(low >= high) or np.any(low < -1) or np.any(high > 1):
        raise ValueError("residual bounds must satisfy -1 <= low < high <= 1 (LIBERO action space)")
    if not 0 < float(residual["gamma"]) <= 1:
        raise ValueError("residual.gamma must lie in (0, 1]")
    if base["noise_protocol"] not in NOISE_PROTOCOLS:
        raise ValueError(f"base.noise_protocol must be one of {NOISE_PROTOCOLS}")
    if base.get("backend", "openpi_policy") not in ("openpi_policy", "pi05_flow"):
        raise ValueError("base.backend must be 'openpi_policy' or 'pi05_flow'")
    if base.get("backend") == "pi05_flow" and base["noise_protocol"] != NOISE_PROTOCOLS[1]:
        raise ValueError(f"base.backend = 'pi05_flow' requires base.noise_protocol = {NOISE_PROTOCOLS[1]!r}")
    contract = config.get("contract", {})
    if contract.get("budget", "episode") not in ("episode", "strict"):
        raise ValueError("contract.budget must be 'episode' or 'strict'")
    if contract.get("train_reset", "legacy") not in ("legacy", "seed_plus_episode"):
        raise ValueError("contract.train_reset must be 'legacy' or 'seed_plus_episode'")
    split = config["split"]
    if min(split["train"], split["validation"], split["test"]) < 1:
        raise ValueError("Each split needs at least one initial state")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def config_hash(config: dict, ignore=("train", "record", "config_path", "output_dir")) -> str:
    """Hash of settings that must not change across a resume (budgets may)."""
    kept = {k: v for k, v in config.items() if k not in ignore}
    kept["train_fixed"] = {k: config["train"][k] for k in ("seed", "batch_size", "replay_capacity",
                                                          "updates_per_step", "learning_starts")}
    return hashlib.sha256(json.dumps(kept, sort_keys=True, default=str).encode()).hexdigest()


def _task_map(repository: Path) -> dict:
    tree = ast.parse((repository / "libero/libero/benchmark/libero_suite_task_map.py").read_text())
    return ast.literal_eval(tree.body[0].value)


def task_slug(case: dict) -> str:
    """Stable short slug, e.g. ``pro_spatial_task08`` (benchmark/suite/category/task id)."""
    return f"{case['benchmark']}_{SUITE_SHORT[case['suite']]}_{case['category'].lower()}{case['task_id']:02d}"


def resolve_case(task: dict) -> dict:
    """Return the PRO case exactly as the robustness plan defines it."""
    suffix = {v: k for k, v in PRO_CATEGORIES.items()}[task["category"]]
    suite, task_id = task["suite"], int(task["task_id"])
    source = ROOT / "third_party/libero_pro"
    names = _task_map(source)[f"{suite}_{suffix}"]
    if not 0 <= task_id < len(names):
        raise ValueError(f"task_id {task_id} outside {suite}_{suffix} (n={len(names)})")
    name = names[task_id]
    base = ASSETS / ("libero_pro_generated" if suffix == "env" else "libero_pro")
    bddl = base / "bddl_files" / f"{suite}_{suffix}" / (name + ".bddl")
    states = base / "init_files" / f"{suite}_{suffix}" / (name + ".pruned_init")
    commit = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    key = f"pro:{suite}:{task['category']}:{name}"
    case = {"benchmark": "pro", "suite": suite, "task_id": task_id, "name": name,
            "category": task["category"], "difficulty": None, "episodes": 50,
            "bddl": str(bddl), "real_bddl": str(bddl), "states": str(states),
            "source_commit": commit, "state_origin": "published_initial_states",
            "case_id": f"pro_{suite}_{hashlib.sha256(key.encode()).hexdigest()[:16]}",
            "bddl_sha256": sha256_file(bddl), "states_sha256": sha256_file(states),
            "perturbation_suffix": suffix, "task_map_key": f"{suite}_{suffix}"}
    case["available_initial_states"] = len(load_state_array(str(states)))
    if task.get("expected_case_id") and case["case_id"] != task["expected_case_id"]:
        raise ValueError(f"Resolved {case['case_id']}, config expects {task['expected_case_id']}")
    protocol = Path(task.get("frozen_protocol") or "")
    if not protocol.is_absolute():
        protocol = ROOT / protocol
    case["frozen_protocol"] = str(protocol) if protocol.is_file() else None
    if protocol.is_file():
        frozen = [c for c in json.loads(protocol.read_text())["cases"] if c["case_id"] == case["case_id"]]
        if len(frozen) != 1:
            raise ValueError("Resolved case is absent from the frozen robustness protocol")
        mismatched = [k for k, v in frozen[0].items() if case.get(k) != v]
        if mismatched:
            raise ValueError(f"Resolved case differs from frozen protocol in {mismatched}")
        case["matches_frozen_protocol"] = True
    if task.get("state_set"):
        apply_state_set(case, task["state_set"])
    case["slug"] = task_slug(case)
    case["horizon"] = HORIZONS[suite]
    return case


def state_set_request(case: dict, generation_seed: int, split_seed: int, counts: dict) -> dict:
    """Inputs of a generated state set (see rl_vla.pro_residual.states)."""
    return {"case_id": case["case_id"], "name": case["name"], "bddl": case["bddl"],
            "bddl_sha256": case["bddl_sha256"], "published_states": case["states"],
            "published_states_sha256": case["states_sha256"], "pro_commit": case["source_commit"],
            "generation_seed": int(generation_seed), "split_seed": int(split_seed),
            "train": int(counts["train"]), "validation": int(counts["validation"]), "test": int(counts["test"])}


def load_state_set(directory: str | Path) -> tuple[Path, dict]:
    directory = Path(directory)
    directory = directory if directory.is_absolute() else ROOT / directory
    manifest = json.loads((directory / "state_set.json").read_text())
    if manifest.get("format") != "rl-vla-pro-state-set-v1":
        raise ValueError(f"{directory} is not a PRO state set")
    return directory, manifest


def apply_state_set(case: dict, directory: str | Path) -> None:
    """Replace the published states by a generated train/validation + published test array."""
    directory, manifest = load_state_set(directory)
    request = manifest["request"]
    for key, expected in (("case_id", case["case_id"]), ("bddl_sha256", case["bddl_sha256"]),
                          ("published_states_sha256", case["states_sha256"]),
                          ("pro_commit", case["source_commit"])):
        if request[key] != expected:
            raise ValueError(f"State set {directory} was built for a different {key}")
    states_path = directory / "pro_states.npy"
    if sha256_file(states_path) != manifest["states_sha256"]:
        raise ValueError(f"{states_path} does not match its manifest")
    states = load_state_array(str(states_path))
    offset = int(manifest["published_offset"])
    if not np.array_equal(states[offset:], load_state_array(case["states"])):
        raise ValueError("The state set's test block is not the published PRO state file")
    case.update({
        "published_states": case["states"], "published_states_sha256": case["states_sha256"],
        "states": str(states_path), "states_sha256": manifest["states_sha256"],
        "available_initial_states": int(len(states)),
        "state_origin": "generated train/validation (RLDT controlled-PRO protocol) + published PRO test",
        "state_set": {"directory": str(directory), "manifest_sha256": sha256_file(directory / "state_set.json"),
                      "generation_seed": request["generation_seed"], "split_seed": request["split_seed"],
                      "published_offset": offset, "splits": manifest["splits"],
                      "simulator_versions": manifest.get("simulator_versions")}})


def published_index(case: dict, initial_state_id: int) -> int | None:
    """Index into the published PRO state file, or None for a generated state."""
    state_set = case.get("state_set")
    if not state_set:
        return int(initial_state_id)
    offset = int(state_set["published_offset"])
    return int(initial_state_id) - offset if initial_state_id >= offset else None


def resolve_original_task(case: dict) -> dict:
    """The unperturbed LIBERO task with the same task name (retention check)."""
    source = ROOT / "third_party/openpi/third_party/libero"
    names = _task_map(source)[case["suite"]]
    if case["name"] not in names:
        return {"available": False, "reason": "no original task with the same name"}
    task_id = names.index(case["name"])
    config = Path(DEFAULTS["simulator"]["original_libero_config"]) / "config.yaml"
    text = config.read_text()
    try:  # Some hosts write this config as JSON, which is also valid YAML.
        paths = json.loads(text)
    except json.JSONDecodeError:
        paths = dict(line.split(": ", 1) for line in text.splitlines() if ": " in line)
    bddl = Path(paths["bddl_files"]) / case["suite"] / (case["name"] + ".bddl")
    init = Path(paths["init_states"]) / case["suite"] / (case["name"] + ".pruned_init")
    return {"available": True, "benchmark": "libero", "suite": case["suite"], "task_id": task_id,
            "name": case["name"], "bddl": str(bddl), "bddl_sha256": sha256_file(bddl),
            "init_file": str(init), "init_sha256": sha256_file(init),
            "slug": f"libero_{SUITE_SHORT[case['suite']]}_orig{task_id:02d}",
            "horizon": HORIZONS[case["suite"]],
            "seed_protocol": "rl_vla.baseline: 100000 + suite_index*100000 + task_id*1000 + init"}


def load_state_array(path: str) -> np.ndarray:
    from rl_vla.robustness import load_states  # Restricted NumPy-only/torch-safe reader.

    return load_states(path)


def build_split(config: dict, case: dict, original: dict) -> dict:
    """Seeded split of one PRO case's published initial states, with overlap checks."""
    split = config["split"]
    states = load_state_array(case["states"])
    count = len(states)
    need = split["train"] + split["validation"] + split["test"]
    if need > count:
        raise ValueError(f"Split requests {need} initial states; the case has {count}")
    state_set = case.get("state_set")
    if state_set:
        # Fixed roles from the generated state set: generated train, generated validation,
        # published PRO states as the held-out test split (RLDT controlled-PRO protocol).
        groups = {k: list(state_set["splits"][k]) for k in ("train", "validation", "test")}
        for key in groups:
            if len(groups[key]) != int(split[key]):
                raise ValueError(f"split.{key} = {split[key]} but the state set defines {len(groups[key])}")
        unused = sorted(set(range(count)) - set().union(*map(set, groups.values())))
        split_seed, method = int(state_set["split_seed"]), (
            "state set: generated states permuted with default_rng(split_seed); ids [0, train) train, "
            "[train, train+validation) validation, published PRO states test")
    else:
        order = np.random.default_rng(int(split["seed"])).permutation(count).tolist()
        groups = {"test": sorted(order[:split["test"]]),
                  "validation": sorted(order[split["test"]:split["test"] + split["validation"]]),
                  "train": sorted(order[split["test"] + split["validation"]:need])}
        unused = sorted(order[need:])
        split_seed, method = int(split["seed"]), "numpy default_rng(seed).permutation; test, then validation, then train"
    train, validation, test = groups["train"], groups["validation"], groups["test"]
    threshold = float(split["duplicate_l2_threshold"])
    checks = {}
    for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")):
        dist = np.linalg.norm(states[groups[a]][:, None] - states[groups[b]][None], axis=-1)
        checks[f"{a}_vs_{b}"] = {"min_l2": float(dist.min()), "pairs_below_threshold": int((dist <= threshold).sum()),
                                 "exact_equal_pairs": int((dist == 0).sum())}
    pairwise = np.linalg.norm(states[:, None] - states[None], axis=-1)
    np.fill_diagonal(pairwise, np.inf)
    state_hashes = [hashlib.sha256(np.ascontiguousarray(row).tobytes()).hexdigest()[:16] for row in states]
    hash_groups = {name: {state_hashes[i] for i in ids} for name, ids in groups.items()}
    if any(hash_groups[a] & hash_groups[b] for a, b in (("train", "validation"), ("train", "test"), ("validation", "test"))):
        raise ValueError("Identical initial-state vectors appear in more than one split")
    if any(v["pairs_below_threshold"] for v in checks.values()):
        raise ValueError(f"Near-duplicate initial states cross splits: {checks}")
    retention = dict(original)
    if original.get("available"):
        original_states = load_state_array(original["init_file"])
        retention["initial_state_ids"] = list(range(min(int(split["retention_initial_states"]), len(original_states))))
        retention["available_initial_states"] = len(original_states)
        if original_states.shape[1] == states.shape[1]:
            dist = np.linalg.norm(states[train][:, None] - original_states[retention["initial_state_ids"]][None], axis=-1)
            retention["min_l2_to_pro_train"] = float(dist.min())
        else:
            retention["min_l2_to_pro_train"] = None
            retention["state_dimension_note"] = (f"original states have dim {original_states.shape[1]}, "
                                                 f"PRO states {states.shape[1]}: different scenes")
        retention["role"] = "evaluation only (retention); never used for training or selection"
    return {
        "version": 1,
        "case_id": case["case_id"], "task_name": case["name"], "category": case["category"],
        "states_file": case["states"], "states_sha256": case["states_sha256"],
        "available_initial_states": count, "state_vector_dim": int(states.shape[1]),
        "unit": "initial_state",
        "separates": ("initial states (object/robot configurations) of ONE fixed PRO perturbation "
                      "configuration. Task, BDDL, instruction and perturbation type are shared by all "
                      "splits; this split does not test generalization across tasks or perturbations."),
        "seed": split_seed, "method": method,
        "state_set": state_set,
        "train": train, "validation": validation, "test": test,
        "unused": unused,
        "roles": {"train": "residual training rollouts only",
                  "validation": "periodic evaluation and checkpoint selection",
                  "test": "held-out; never used by training or checkpoint selection"},
        "state_vector_sha256_prefix": {str(i): h for i, h in enumerate(state_hashes)},
        "overlap_checks": {"threshold_l2": threshold, **checks,
                           "min_pairwise_l2_all_states": float(pairwise.min()),
                           "identical_vectors_across_splits": 0},
        "retention": retention,
        "evaluation_seed_protocol": ("PRO: seed = rl_vla.robustness.episode_seed(case_id, init); base noise "
                                     "key = f'{case_id}_init{init:03d}:{seed}' and call index, identical to "
                                     "the robustness queue, so base-only episodes are directly comparable."
                                     + (" With a state set, init is the published index (id - published_offset) "
                                        "for test states; generated states use episode_seed(case_id + "
                                        "':generated:' + states_sha256[:16], id) and key "
                                        "f'{case_id}_gen{id:03d}:{seed}'." if state_set else "")),
    }


def safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", text)
