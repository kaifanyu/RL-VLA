"""Read-only native Panda IK and collision audit of frozen custom layouts.

This is a kinematic feasibility check, not a policy rollout or proof that an
RL policy can plan a collision-free trajectory. No policy/model is loaded.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from rl_vla.diagnostics.observation_bank import make_ood_env
from rl_vla.envs.libero import SETTLING_ACTION

TARGETS = {"object_0": "alphabet_soup_1", "object_1": "cream_cheese_1", "object_2": "salad_dressing_1"}
VARIANTS = ("both_new_regions", "continuous_new_offsets", "target_moved_distractor_fixed", "distractor_moved_target_fixed")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")
    temporary.replace(path)


def settle_without_render(env, state, backend_seed):
    wrapper, backend = env.backend, env.backend.env
    backend.hard_reset = False
    for name, observable in backend._observables.items():
        if "image" in name:
            observable.set_enabled(False)
    wrapper.seed(backend_seed)
    wrapper.reset()
    wrapper.set_init_state(state.copy())
    for _ in range(10):
        wrapper.step(SETTLING_ACTION.tolist())
    if wrapper.check_success():
        raise ValueError("Initial goal unexpectedly satisfied")
    return backend.sim.get_state().flatten().copy()


def arm_metadata(backend):
    robot = backend.robots[0]
    sim = backend.sim
    joint_ids = [sim.model.joint_name2id(name) for name in robot.robot_model.joints]
    return {"qpos": np.asarray(robot._ref_joint_pos_indexes),
            "dofs": np.asarray(robot._ref_joint_vel_indexes),
            "limits": sim.model.jnt_range[joint_ids].copy(),
            "site": robot.gripper.important_sites["grip_site"],
            "robot_geoms": {sim.model.geom_name2id(name) for name in robot.robot_model.contact_geoms+robot.gripper.contact_geoms},
            "gripper_geoms": {sim.model.geom_name2id(name) for name in robot.gripper.contact_geoms}}


def robot_contacts(backend, arm, target, initial_pairs):
    target_geoms = {backend.sim.model.geom_name2id(name) for name in backend.objects_dict[target].contact_geoms}
    collisions = []
    for index in range(backend.sim.data.ncon):
        contact = backend.sim.data.contact[index]
        a, b = int(contact.geom1), int(contact.geom2)
        pair = tuple(sorted((a, b)))
        if float(contact.dist) >= -.002 or pair in initial_pairs:
            continue
        if not {a, b}.intersection(arm["robot_geoms"]):
            continue
        if (a in arm["gripper_geoms"] and b in target_geoms) or (b in arm["gripper_geoms"] and a in target_geoms):
            continue
        collisions.append({"geom1": backend.sim.model.geom_id2name(a),
                           "geom2": backend.sim.model.geom_id2name(b), "penetration_m": -float(contact.dist)})
    return collisions


def solve(backend, arm, target_position, target_rotation):
    sim = backend.sim
    lo, hi = arm["limits"].T
    for iteration in range(250):
        sim.forward()
        position = np.asarray(sim.data.get_site_xpos(arm["site"]))
        rotation = np.asarray(sim.data.get_site_xmat(arm["site"]))
        dp = target_position-position
        dr = Rotation.from_matrix(target_rotation@rotation.T).as_rotvec()
        if np.linalg.norm(dp) <= .003 and np.linalg.norm(dr) <= .06:
            break
        jp = sim.data.get_site_jacp(arm["site"]).reshape(3, -1)[:, arm["dofs"]]
        jr = sim.data.get_site_jacr(arm["site"]).reshape(3, -1)[:, arm["dofs"]]
        jac = np.concatenate([jp, .35*jr])
        error = np.concatenate([dp, .35*dr])
        update = jac.T@np.linalg.solve(jac@jac.T + .015**2*np.eye(6), error)
        update *= min(1., .12/max(np.max(np.abs(update)), 1e-12))
        sim.data.qpos[arm["qpos"]] = np.clip(sim.data.qpos[arm["qpos"]]+update, lo+1e-5, hi-1e-5)
        sim.data.qvel[arm["dofs"]] = 0.
    sim.forward()
    dp = target_position-np.asarray(sim.data.get_site_xpos(arm["site"]))
    dr = Rotation.from_matrix(target_rotation@np.asarray(sim.data.get_site_xmat(arm["site"])).T).as_rotvec()
    return {"iterations": iteration+1, "position_error_m": float(np.linalg.norm(dp)),
            "orientation_error_rad": float(np.linalg.norm(dr)),
            "joint_positions": sim.data.qpos[arm["qpos"]].tolist(),
            "joint_limits_satisfied": bool(np.all(sim.data.qpos[arm["qpos"]] >= lo) and np.all(sim.data.qpos[arm["qpos"]] <= hi)),
            "converged": bool(np.linalg.norm(dp) <= .003 and np.linalg.norm(dr) <= .06)}


def audit(root, task, *, limit=None):
    destination = root/"diagnostic_layouts"/task
    final = destination/("kinematic_audit.json" if limit is None else "kinematic_audit_pilot.json")
    if final.exists():
        raise FileExistsError(final)
    rows = []
    contracts = {}
    all_equivalent = True
    for variant in VARIANTS:
        protocol = destination/variant/"protocol.json"
        plan = json.loads(protocol.read_text())
        contracts[str(protocol)] = digest(protocol)
        states_path = Path(plan["states"]["path"])
        contracts[str(states_path)] = plan["states"]["sha256"]
        states = np.load(states_path, allow_pickle=False)
        env = make_ood_env(protocol=str(protocol))
        try:
            _, reset = env.reset(seed=100000)
            backend = env.backend.env
            native = backend.sim.get_state().flatten().copy()
            accelerated = settle_without_render(env, states[0], reset["backend_seed"])
            error = float(np.max(np.abs(native-accelerated)))
            if error > 1e-10:
                raise RuntimeError(f"Render-free native settling differs from original reset by {error}")
            arm = arm_metadata(backend)
            target = TARGETS[task]
            initial_rotation = np.asarray(backend.sim.data.get_site_xmat(arm["site"])).copy()
            for index, state in enumerate(states[:limit] if limit is not None else states):
                seed = 100000+index
                backend_seed = int(np.random.default_rng(seed).integers(0, 2**31-1))
                settled = settle_without_render(env, state, backend_seed)
                sim = backend.sim
                initial_pairs = {tuple(sorted((int(sim.data.contact[i].geom1), int(sim.data.contact[i].geom2))))
                                 for i in range(sim.data.ncon)}
                center = sim.data.body_xpos[backend.obj_body_id[target]].copy()
                attempts = []
                for yaw in (0., np.pi/2, -np.pi/2, np.pi):
                    sim.set_state_from_flattened(settled)
                    sim.forward()
                    rotation = Rotation.from_rotvec([0., 0., yaw]).as_matrix()@initial_rotation
                    poses = []
                    for height in (.15, .10, .05, .02, 0.):
                        goal = center + [0., 0., height]
                        result = solve(backend, arm, goal, rotation)
                        result.update(target_position=goal.tolist(), height_above_object_origin_m=height,
                                      collisions=robot_contacts(backend, arm, target, initial_pairs))
                        poses.append(result)
                        if not result["converged"] or result["collisions"]:
                            break
                    passed = len(poses) == 5 and all(p["converged"] and not p["collisions"] for p in poses)
                    attempts.append({"yaw_radians": yaw, "passed": passed, "poses": poses})
                    if passed:
                        break
                row = {"variant": variant, "state_id": index, "passed": passed,
                       "target": target, "settled_object_origin": center.tolist(),
                       "render_free_reset_max_abs_difference": error, "attempts": attempts}
                rows.append(row)
                print(json.dumps({"task": task, "variant": variant, "state": index,
                                  "passed": passed, "yaw_attempts": len(attempts)}), flush=True)
                write(destination/"kinematic_progress.json", {"completed": len(rows), "passed": sum(x["passed"] for x in rows),
                      "last": row, "total": (limit or 50)*4})
        finally:
            env.close()
    for path, expected in contracts.items():
        if digest(path) != expected:
            raise ValueError("Frozen OOD states/protocol changed during kinematic audit")
    report = {"passed": all_equivalent and all(row["passed"] for row in rows), "task": task,
              "states_checked": len(rows), "policy_used": False, "technical_only": limit is not None,
              "method": "Native MuJoCo Panda site Jacobians, damped least-squares six-dimensional IK, native joint limits; four predeclared planar gripper yaws, five approach heights including object body origin; no model or policy outcomes",
              "position_tolerance_m": .003, "orientation_tolerance_rad": .06,
              "collision_penetration_tolerance_m": .002,
              "collision_rule": "Native robot contact geometries; ignore initial structural contacts and permitted gripper-target contact; reject other new penetrations over2mm",
              "limitation": "Five kinematic waypoint checks establish local pose feasibility; they do not certify continuous trajectory collision avoidance or successful grasp dynamics.",
              "source_sha256": digest(Path(__file__)), "frozen_inputs": contracts, "states": rows}
    write(final, report)
    print(json.dumps({"audit": str(final), "passed": report["passed"], "states": len(rows)}), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--task", choices=list(TARGETS), required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    audit(args.run.resolve(), args.task, limit=args.limit)
