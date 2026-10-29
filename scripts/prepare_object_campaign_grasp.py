"""Freeze Object2 PRO swap and a six-item Object1 permutation, without policy filtering."""

import argparse
import json
import multiprocessing
import os
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from prepare_pro_suite_grasp import (
    DATASET,
    PRO_COMMIT,
    REVISION,
    ROOT,
    check_task,
    collect_attempts,
    generate_attempt,
    record,
    verify_record,
    write_json,
)

from rl_vla.envs.libero_pro import validate_splits
from rl_vla.robustness import load_states

TASKS = {
    "object_2": {"task_id": 2, "name": "pick_up_the_salad_dressing_and_place_it_in_the_basket",
                 "generation_seed": 2026280200, "split_seed": 20262802, "generated_count": 150},
    "object_1_hard": {"task_id": 1, "name": "pick_up_the_cream_cheese_and_place_it_in_the_basket",
                      "generation_seed": 2026281200, "split_seed": 20262812, "generated_count": 200},
}
ITEMS = ("cream_cheese_1", "alphabet_soup_1", "milk_1", "tomato_sauce_1", "butter_1", "orange_juice_1")


def permute_object1(text):
    """One six-cycle through original regions; retain geometry, basket and goal."""
    assignments = dict(re.findall(r"\(On (\w+) (floor_\w+)\)", text))
    regions = [assignments[item] for item in ITEMS]
    changes = {item: {"before": regions[i], "after": regions[(i - 1) % len(ITEMS)]}
               for i, item in enumerate(ITEMS)}
    for item, change in changes.items():
        text, count = re.subn(rf"\(On {item} {change['before']}\)",
                             f"(On {item} {change['after']})", text)
        if count != 1:
            raise ValueError(f"Expected exactly one initial assignment: {item}")
    return text, changes


def prepare_assets(directory, key):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "assets.json"
    if path.exists():
        assets = json.loads(path.read_text())
        for field in ("bddl", "published_states", "source_bddl", "source_states", "screen_states"):
            verify_record(assets[field])
        return assets
    spec = TASKS[key] | {"key": key, "suite": "libero_object", "max_steps": 280}
    ready = json.loads((ROOT / "runs/benchmark_screen_20261027/cases" /
                       f"pro_libero_object_swap_{spec['task_id']:02d}/ready.json").read_text())
    if ready["source_commit"] != PRO_COMMIT or ready["dataset_revision"] != REVISION:
        raise ValueError("Screen assets are not the pinned PRO version")
    original = ROOT / "third_party/openpi/third_party/libero/libero/libero"
    source_bddl = original / f"bddl_files/libero_object/{spec['name']}.bddl"
    source_states = original / f"init_files/libero_object/{spec['name']}.pruned_init"
    assets = {"spec": spec, "name": spec["name"], "pro_source": str(ROOT / "third_party/libero_pro"),
              "pro_commit": PRO_COMMIT, "dataset": DATASET, "dataset_revision": REVISION,
              "published_states": ready["published_states"], "bddl": ready["bddl_record"],
              "source_bddl": record(source_bddl), "source_states": record(source_states),
              "screen_states": ready["states_record"]}
    if key == "object_1_hard":
        text, changes = permute_object1(source_bddl.read_text())
        custom = directory / f"{spec['name']}.bddl"
        custom.write_text(text)
        assets.update(bddl=record(custom), permutation=changes,
                      condition="Custom Position/six-item-cycle",
                      condition_suite="local_libero_object_six_cycle",
                      test_scope="50 newly generated custom six-item-layout states excluded from training and selection")
    else:
        assets.update(condition="Position/swap", condition_suite="libero_object_swap",
                      test_scope="50 published PRO states excluded from training and selection")
    for field in ("bddl", "published_states", "source_bddl", "source_states", "screen_states"):
        verify_record(assets[field])
    write_json(path, assets)
    return assets


def freeze(directory, assets, generated, accepted, rejected):
    spec = assets["spec"]
    order = np.random.default_rng(spec["split_seed"]).permutation(len(generated))
    states = np.asarray(generated)[order]
    if spec["key"] == "object_2":
        states = np.concatenate([states, load_states(assets["published_states"]["path"])])
    if len(states) != 200:
        raise ValueError("Expected 100 train, 50 validation, 50 test states")
    state_path = directory / "pro_states.npy"
    np.save(state_path, states, allow_pickle=False)
    write_json(directory / "generation.json", {
        "method": "Fresh seeded simulator constructors; reject invalid/duplicate/initially solved states; no policy filtering",
        "accepted": [accepted[i] for i in order], "rejected": rejected,
        "generated_state_count": len(generated), "generation_seed": spec["generation_seed"],
        "split_seed": spec["split_seed"], "generator": record(Path(__file__)),
        "excluded_states": [assets[field] for field in ("published_states", "source_states", "screen_states")],
        "test_origin": "published_PRO" if spec["key"] == "object_2" else "locally_generated_custom_layout"})
    task = {field: spec[field] for field in ("suite", "task_id", "name", "max_steps")}
    task.update(condition=assets["condition"], condition_suite=assets["condition_suite"],
                test_scope=assets["test_scope"], pro_state_count=200,
                train_state_ids=list(range(100)), validation_state_ids=list(range(100, 150)),
                test_state_ids=list(range(150, 200)), source_state_ids=list(range(50)),
                pro={"bddl": assets["bddl"], "states": record(state_path)},
                source={"bddl": assets["source_bddl"], "states": assets["source_states"]})
    validate_splits(task, 2)
    protocol = directory / "protocol.json"
    plan = {"version": 2, "purpose": "User-selected Object2 zero-screen / Object1 harder-layout adaptation",
            "pro_source": assets["pro_source"], "pro_commit": PRO_COMMIT,
            "dataset": DATASET, "dataset_revision": REVISION, "training_seed": 0,
            "evaluation_seed": 100000, "validation_iterations": [25, 50, 100],
            "online_validation": True,
            "selection": "Highest validation success count among trained checkpoints; ties earliest iteration",
            "scope": "Single training seed; disjoint states within a fixed condition. Object1 is a local custom layout.",
            "tasks": {spec["key"]: task}}
    tail = "[train]" + (ROOT / "configs/rldt_pi05_libero.toml").read_text().split("[train]", 1)[1]
    for split in ("train", "validation", "test", "source"):
        config = f'''algorithm = "rldt"

[policy]
kind = "pi05"
checkpoint = "{ROOT / 'checkpoints/openpi/pi05_libero'}"
model_config = "pi05_libero"
projection_only = false

[env]
kind = "external"
factory = "rl_vla.envs.libero_pro:make_env"
python = "{ROOT / '.venv-pro/bin/python'}"

[env.kwargs]
protocol = "{protocol}"
task_key = "{spec['key']}"
split = "{split}"

'''
        (directory / f"{split}.toml").write_text(config + tail + "\n[recording]\nenabled = true\n")
    write_json(protocol, plan)


def generate(directory, assets, workers):
    if (directory / "protocol.json").exists():
        return
    attempts = directory / "generation_attempts"
    attempts.mkdir(exist_ok=True)
    context = {"assets": record(directory / "assets.json"), "generator": record(Path(__file__))}
    context_path = attempts / "context.json"
    if context_path.exists() and json.loads(context_path.read_text()) != context:
        raise ValueError("Generation inputs changed")
    write_json(context_path, context)
    excluded = np.concatenate([load_states(assets[field]["path"])
                               for field in ("published_states", "source_states", "screen_states")])
    spec = assets["spec"]
    ordered = []
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        for start in range(0, 1000, workers):
            paths = [attempts / f"{i:04d}.npz" for i in range(start, min(start + workers, 1000))]
            jobs = [pool.submit(generate_attempt, assets["bddl"]["path"], spec["generation_seed"] + i,
                                excluded.shape[1:], str(path))
                    for i, path in enumerate(paths, start=start) if not path.exists()]
            for future in as_completed(jobs):
                future.result()
            ordered.extend(paths)
            generated, accepted, rejected = collect_attempts(ordered, excluded, count=spec["generated_count"])
            write_json(directory / "generation_progress.json", {
                "accepted": len(generated), "target": spec["generated_count"], "rejected": rejected})
            print(f"{spec['key']}: generated {len(generated)}/{spec['generated_count']}", flush=True)
            if len(generated) == spec["generated_count"]:
                freeze(directory, assets, generated, accepted, rejected)
                return
            if len(rejected) >= 20 and not generated:
                raise RuntimeError("Repeated simulator generation failures")
    raise RuntimeError("Insufficient valid unique states")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if not 1 <= args.workers <= 8:
        parser.error("Use 1..8 workers within the retained allocation")
    os.environ["LIBERO_CONFIG_PATH"] = str(ROOT / ".cache/libero-pro")
    for key in TASKS:
        directory = args.output.resolve() / key
        assets = prepare_assets(directory, key)
        generate(directory, assets, args.workers)
        check_task(directory, args.workers)


if __name__ == "__main__":
    main()
