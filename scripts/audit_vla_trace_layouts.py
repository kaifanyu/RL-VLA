"""Verify every frozen initial object location against its native BDDL sampler.

This read-only geometry audit disables rendering, installs saved MuJoCo states,
and calls forward kinematics. It never steps a policy or modifies frozen tasks.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
from prepare_pro_suite_grasp import record, verify_record, write_json

from rl_vla.robustness import load_states


def audit(directory):
    from libero.libero.envs.env_wrapper import ControlEnv

    protocol = directory / "protocol.json"
    plan = json.loads(protocol.read_text())
    key, = plan["tasks"]
    task = plan["tasks"][key]
    definitions, rows, sampler_contracts = {}, [], {}
    for condition in ("pro", "source"):
        bddl = task[condition]["bddl"]["path"]
        states = load_states(task[condition]["states"]["path"])
        env = ControlEnv(bddl_file_name=bddl, use_camera_obs=False, has_offscreen_renderer=False)
        try:
            parsed = env.env.parsed_problem
            definitions[condition] = parsed
            sampler_contracts[condition] = {}
            for object_name, sampler in env.env.placement_initializer.samplers.items():
                obj, = sampler.mujoco_objects
                radius = float(obj.horizontal_radius) if sampler.ensure_object_boundary_in_range else 0.0
                # Pinned NumPy's legacy uniform accepts reversed low/high. The
                # basket's radius exceeds half its BDDL region width, producing
                # reversed bounds. Preserve and disclose that upstream behavior.
                effective = [[*sorted((x[0] + radius, x[1] - radius)),
                              *sorted((y[0] + radius, y[1] - radius))]
                             for x, y in zip(sampler.x_ranges, sampler.y_ranges, strict=True)]
                sampler_contracts[condition][object_name] = {
                    "x_ranges": sampler.x_ranges, "y_ranges": sampler.y_ranges,
                    "boundary_radius": radius, "effective_xy_minmax": effective,
                    "reversed_bounds": any(x[0] + radius > x[1] - radius
                                           or y[0] + radius > y[1] - radius
                                           for x, y in zip(sampler.x_ranges, sampler.y_ranges, strict=True))}
            goal = parsed["goal_state"]
            target = next(name for name in parsed["obj_of_interest"] if name != "basket_1")
            if goal != [["in", target, "basket_1_contain_region"]]:
                raise ValueError(f"Unexpected task goal: {goal}")
            splits = ("source",) if condition == "source" else ("train", "validation", "test")
            for split in splits:
                for state_id in task[f"{split}_state_ids"]:
                    env.set_state(states[state_id])
                    env.sim.forward()
                    if env.check_success():
                        raise ValueError(f"Initially satisfied goal: {key}/{split}/{state_id}")
                    positions = {}
                    for predicate, object_name, region_name in parsed["initial_state"]:
                        if predicate != "on":
                            raise ValueError(f"Unrecognized placement predicate: {predicate}")
                        region = parsed["regions"][region_name]
                        if region["target"] != "floor":
                            raise ValueError(f"Unrecognized placement reference: {region}")
                        position = env.sim.data.body_xpos[env.env.obj_body_id[object_name]].copy()
                        local_xy = position[:2] - np.asarray(env.env.workspace_offset)[:2]
                        declared = any(bounds[0] - 1e-6 <= local_xy[0] <= bounds[2] + 1e-6
                                       and bounds[1] - 1e-6 <= local_xy[1] <= bounds[3] + 1e-6
                                       for bounds in region["ranges"])
                        effective = sampler_contracts[condition][object_name]["effective_xy_minmax"]
                        within = any(bounds[0] - 1e-6 <= local_xy[0] <= bounds[1] + 1e-6
                                     and bounds[2] - 1e-6 <= local_xy[1] <= bounds[3] + 1e-6
                                     for bounds in effective)
                        if not within:
                            raise ValueError(f"Object outside native sampler support: {key}/{split}/{state_id} "
                                             f"{object_name} {local_xy} {effective}")
                        positions[object_name] = {"world_xyz": position.tolist(), "region": region_name,
                                                  "inside_declared_bddl_region": declared,
                                                  "inside_native_sampler_support": True}
                    rows.append({"split": split, "state_id": state_id, "target": target,
                                 "initial_goal": False, "objects": positions})
        finally:
            env.close()
    original = {name: region for pred, name, region in definitions["source"]["initial_state"] if pred == "on"}
    swap = {name: region for pred, name, region in definitions["pro"]["initial_state"] if pred == "on"}
    changes = {name: {"source": original[name], "pro": region} for name, region in swap.items()
               if original[name] != region}
    if len(changes) != 2 or set(original) != set(swap):
        raise ValueError(f"Expected exact two-object official swap: {changes}")
    payload = {"passed": True, "states_checked": len(rows), "protocol": record(protocol),
               "auditor": record(Path(__file__)), "swapped_objects": changes,
               "bddl_definitions": definitions, "native_sampler_contracts": sampler_contracts, "checks": rows,
               "native_sampler_caveat": "The pinned sampler subtracts object radius from BDDL ranges. For basket_1 the radius exceeds the half-width and NumPy accepts reversed uniform bounds; its effective support is wider than the written BDDL region. We preserve official states and generator behavior, disclose both bounds, and do not filter those valid native samples.",
               "method": "Raw saved state before settling; forward kinematics only; every object's body origin within actual pinned sampler XY support to 1e-6 m; exact target/basket goal; no policy or outcome filtering"}
    path = directory / "placement_audit.json"
    if path.exists():
        if json.loads(path.read_text()) != payload:
            raise ValueError("Refusing to overwrite existing placement audit")
    else:
        write_json(path, payload)
    print(f"{key}: checked {len(rows)} frozen states and exact official swap {list(changes)}", flush=True)


def finalize(directory):
    setup = json.loads((directory / "setup_check.json").read_text())
    split = json.loads((directory / "split_audit.json").read_text())
    placement = json.loads((directory / "placement_audit.json").read_text())
    plan = json.loads((directory / "protocol.json").read_text())
    if setup["states_checked"] != 250 or placement["states_checked"] != 250:
        raise ValueError("Incomplete reset or placement coverage")
    if not split["passed"] or not placement["passed"]:
        raise ValueError("Failed preparation audit")
    if plan["validation_iterations"] != [25, 50, 100] or plan["online_validation"] is not True:
        raise ValueError("Incorrect validation schedule")
    task, = plan["tasks"].values()
    for condition in ("pro", "source"):
        for item in task[condition].values():
            verify_record(item)
    files = [directory / name for name in ("assets.json", "protocol.json", "generation.json", "pro_states.npy",
             "train.toml", "validation.toml", "test.toml", "source.toml", "setup_check.json", "split_audit.json",
             "placement_audit.json", "reset_checks/initial_montage.png", "reset_checks/initial_montage.pdf")]
    payload = {"passed": True, "states_reset_checked": 250, "states_placement_checked": 250,
               "split_overlap_count": 0, "files": [record(path) for path in files],
               "source_and_task_assets": {condition: task[condition] for condition in ("pro", "source")},
               "generator": record(Path(__file__).with_name("prepare_vla_trace_objects.py")),
               "auditor": record(Path(__file__))}
    destination = directory / "preparation_complete.json"
    if destination.exists():
        if json.loads(destination.read_text()) != payload:
            raise ValueError("Frozen preparation completion record differs")
    else:
        write_json(destination, payload)
    print(f"{directory.name}: preparation_complete.json committed", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks-root", type=Path, required=True)
    parser.add_argument("--tasks", nargs="+", default=["object_0", "object_1", "object_2"])
    parser.add_argument("--wait-and-finalize", action="store_true",
                        help="Wait for generated protocol, audit placements, then publish readiness after reset checks")
    args = parser.parse_args()
    for task_key in args.tasks:
        directory = args.tasks_root.resolve() / task_key
        if args.wait_and_finalize:
            while not (directory / "protocol.json").exists():
                time.sleep(5)
        audit(directory)
        if args.wait_and_finalize:
            while not (directory / "setup_check.json").exists():
                time.sleep(5)
            finalize(directory)
