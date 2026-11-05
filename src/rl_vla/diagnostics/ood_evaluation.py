"""Strict readiness, immutable enumeration, and paired merges for custom OOD layouts."""

import json
from pathlib import Path

VARIANTS = ("both_new_regions", "continuous_new_offsets", "target_moved_distractor_fixed",
            "distractor_moved_target_fixed")
POLICIES = {"pi05_libero": None, "rldt_25": 25, "rldt_50": 50, "rldt_100": 100}
SOURCE_FACTORY = "rl_vla.diagnostics.observation_bank:make_ood_env"
SHARD_FACTORY = "rl_vla.diagnostics.ood_evaluation:make_env"


def read_json(path):
    try:
        return json.loads(Path(path).read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def readiness(campaign, task):
    """Never use a saved checkpoint or tiny intervention smoke as a queue gate."""
    campaign = Path(campaign)
    method = campaign / "methods/rldt" / task
    paths = {"selection": method / "selection.json", "main": method / "status.json",
             "checkpoint_audit": method / "checkpoint_audit.json",
             "interventions": campaign / "interventions" / task / "queue_complete.json",
             "kinematics": campaign / "diagnostic_layouts" / task / "kinematic_audit.json",
             "layouts": campaign / "diagnostic_layouts" / task / "summary.json"}
    data = {name: read_json(path) for name, path in paths.items()}
    missing = [name for name, value in data.items() if value is None]
    if data["main"] is not None:
        if data["main"].get("state") == "failed":
            raise RuntimeError("Main task experiment failed; preserving OOD queue without evaluating")
        if (data["main"].get("stage"), data["main"].get("state")) != ("action_evaluations_complete", "completed"):
            missing.append("main_action_evaluations_complete")
    if data["selection"] is not None:
        if data["selection"].get("iteration") not in (25, 50, 100) or not data["selection"].get("locked"):
            raise ValueError("Checkpoint selection is not the locked predeclared selection")
        if data["selection"].get("test_results_seen") is not False:
            raise ValueError("Selection record does not certify selection before final tests")
    if data["checkpoint_audit"] is not None and data["checkpoint_audit"].get("passed") is not True:
        missing.append("checkpoint_audit_passed")
    if data["kinematics"] is not None and data["kinematics"].get("passed") is not True:
        missing.append("kinematic_audit_passed")
    if data["interventions"] is not None:
        gate = data["interventions"]
        if gate.get("state") == "failed" or (gate.get("complete") is True and gate.get("passed") is not True):
            raise RuntimeError("Closed-loop intervention queue failed")
        if gate.get("complete") is not True:
            missing.append("interventions_complete")
        elif gate.get("task") != task or gate.get("technical_only") is not False:
            raise ValueError("Intervention gate belongs to another task or is only technical smoke")
    if data["layouts"] is not None:
        layout = data["layouts"]
        if layout.get("complete") is not True:
            missing.append("layouts_complete")
        elif layout.get("task") != task or layout.get("variants") != list(VARIANTS) or layout.get("states_per_variant") != 50:
            raise ValueError("OOD layout manifest differs from four predetermined 50-state conditions")
    return {"ready": not missing, "waiting_for": sorted(set(missing)),
            "gate_paths": {name: str(path.resolve()) for name, path in paths.items()}}


def make_env(*, protocol, shard_start, shard_count):
    """Retain the original OOD goal/control contract and add read-only telemetry."""
    from rl_vla.diagnostics.observation_bank import make_ood_env
    from rl_vla.envs.object_telemetry import ObjectTelemetryEnv

    if type(shard_start) is not int or type(shard_count) is not int or not 0 <= shard_start < 50 or not 1 <= shard_count <= 50:
        raise ValueError("Invalid OOD evaluation shard")
    record = json.loads(Path(protocol).read_text())
    if record["state_ids"] != list(range(50)) or record["max_steps"] != 280 or record["variant"] not in VARIANTS:
        raise ValueError("Expected frozen 50-state OOD protocol with unchanged 280-step budget")
    env = make_ood_env(protocol=protocol)
    try:
        base = env.unwrapped
        if shard_start + shard_count > len(base.state_ids):
            raise ValueError("OOD shard exceeds frozen state enumeration")
        base.state_ids = tuple(base.state_ids[shard_start:shard_start + shard_count])
        base.evaluation_index = 0
        return ObjectTelemetryEnv(env)
    except BaseException:
        env.close()
        raise


def merge_shards(reports, specs, *, config, checkpoint, variant, task):
    """Merge only the exact full fifty-case protocol; retain native artifact paths."""
    if len(reports) != len(specs) or not reports:
        raise ValueError("Missing OOD shards")
    episodes = []
    for report, spec in zip(reports, specs, strict=True):
        if report["checkpoint"] != checkpoint or report["config"] != spec["config"]:
            raise ValueError("OOD checkpoint or sampler configuration differs between shards")
        start, count = spec["start"], spec["count"]
        rows = report["episodes"]
        if len(rows) != count:
            raise ValueError("Incomplete OOD shard")
        if [row["seed"] for row in rows] != list(range(100000 + start, 100000 + start + count)):
            raise ValueError("OOD episode noise seeds differ")
        if [row["reset"]["initial_state_id"] for row in rows] != list(range(start, start + count)):
            raise ValueError("OOD state IDs differ")
        for row in rows:
            if row["reset"]["experiment_split"] != variant or row["reset"]["task_id"] != int(task[-1]):
                raise ValueError("OOD task/variant differs")
            if type(row["success"]) is not bool or not 0 < row["primitive_steps"] <= 280:
                raise ValueError("Invalid OOD success flag or controlled-step count")
            if "object_telemetry" not in row["reset"] or "trajectory" not in row or "video" not in row:
                raise ValueError("OOD episode is missing telemetry, trajectory, or video evidence")
        expected_rate = sum(row["success"] for row in rows) / count
        if abs(report["success_rate"] - expected_rate) > 1e-12:
            raise ValueError("OOD shard success summary differs from episode records")
        episodes.extend(rows)
    if [row["reset"]["initial_state_id"] for row in episodes] != list(range(50)) or [row["seed"] for row in episodes] != list(range(100000, 100050)):
        raise ValueError("OOD shards do not cover each predetermined state/noise pair once")
    return {"complete": True, "technical_only": False, "task": task, "variant": variant,
            "config": config, "checkpoint": checkpoint, "episodes": episodes,
            "successes": sum(row["success"] for row in episodes), "episode_count": 50,
            "success_rate": sum(row["success"] for row in episodes) / 50,
            "scope": "Local custom diagnostic relocations, not official LIBERO-PRO scores",
            "failure_labels": "Pending sustained-contact/lift and video review; failure is not automatically semantic"}
