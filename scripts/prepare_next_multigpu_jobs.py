"""Freeze a finite local queue without launching GPUs or altering prior contracts.

Missing LoRA and historical assets remain explicit dependencies. Current Object 1
swap is never presented as the unavailable historical Object 1-hard condition.
"""

import argparse
import copy
import hashlib
import json
import re
import tomllib
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCOPE = "Original pi05_libero; full action expert and projections trained; VLM/image encoder frozen; no LoRA"
E00 = "E00_object_2_seed0"
VARIANTS = ("reference", "quarter_lr_after25", "fixed_reference", "replay8", "combined")


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(8 * 1024**2):
            value.update(block)
    return value.hexdigest()


def encoded(value):
    return json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"


def immutable(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    content = value if isinstance(value, str) else encoded(value)
    if path.exists():
        if path.read_text() != content:
            raise ValueError(f"Refusing to rewrite immutable artifact: {path}")
    else:
        with path.open("x") as stream:
            stream.write(content)


def replace_scalar(text, section, key, value):
    pattern = rf"(?ms)(^\[{re.escape(section)}\]\s*\n)(.*?)(?=^\[|\Z)"
    match = re.search(pattern, text)
    if match is None:
        raise ValueError(f"Missing TOML section {section}")
    literal = json.dumps(value) if not isinstance(value, bool) else str(value).lower()
    body, count = re.subn(rf"(?m)^{re.escape(key)}\s*=.*$", f"{key} = {literal}", match[2])
    if count != 1:
        raise ValueError(f"Expected one TOML setting {section}.{key}")
    return text[:match.start()] + match[1] + body + text[match.end():]


def make_task(current, campaign, job_id, task, seed, variant, family, runtime_root=None):
    source = current / "tasks" / task
    destination = campaign / "configs" / job_id
    plan = json.loads((source / "protocol.json").read_text())
    old_plan = copy.deepcopy(plan)
    if runtime_root is not None:
        plan["pro_source"] = str(Path(runtime_root) / "third_party/libero_pro")
    iterations = [25, 50, 75, 100] if family == "E04" else [25, 50, 100]
    plan.update(purpose=f"Next campaign {family} {task} {variant}, RL seed {seed}",
                training_seed=seed, validation_iterations=iterations, online_validation=False,
                validation_execution="Evaluate immutable checkpoints after the continuous 100-iteration training process exits; training RNG/optimizer/replay unchanged",
                selection="Maximum binary validation success count; ties earliest; initialization eligible",
                selection_candidates=[0, *iterations],
                provenance={"source_protocol": str(source / "protocol.json"),
                            "source_protocol_sha256": digest(source / "protocol.json"),
                            "source_campaign": str(current),
                            "historical_comparison": "Destination reset families; no assertion of equality to missing October 28 training assets"})
    immutable(destination / "protocol.json", plan)
    for split in ("train", "validation", "test", "source"):
        content = (source / f"{split}.toml").read_text()
        content = replace_scalar(content, "env.kwargs", "protocol", str(destination / "protocol.json"))
        content = replace_scalar(content, "train", "seed", seed)
        content = replace_scalar(content, "train", "iterations", 100)
        content = replace_scalar(content, "train", "save_every", 25)
        if variant in ("fixed_reference", "combined"):
            content = replace_scalar(content, "rldt", "reference", "pretrained")
        if variant in ("replay8", "combined"):
            content = replace_scalar(content, "train", "critic_trajectories", 8)
        if variant in ("quarter_lr_after25", "combined"):
            content += '\n[campaign]\nactor_schedule = "quarter_after_iteration_25"\n'
        parsed = tomllib.loads(content)
        if parsed["policy"]["model_config"] != "pi05_libero" or parsed["policy"]["projection_only"]:
            raise ValueError("Destination native model scope unexpectedly changed")
        if (parsed["train"]["collect_decisions"], parsed["rldt"]["execute_steps"]) != (32, 5):
            raise ValueError("Destination interaction settings unexpectedly changed")
        immutable(destination / f"{split}.toml", content)
    hashes = {str(p): digest(p) for p in destination.iterdir() if p.is_file()}
    definition = old_plan["tasks"][task]
    for condition in ("pro", "source"):
        for record in definition[condition].values():
            if digest(record["path"]) != record["sha256"]:
                raise ValueError(f"Task asset has changed: {record['path']}")
            hashes[record["path"]] = record["sha256"]
    hashes[str(source / "protocol.json")] = digest(source / "protocol.json")
    return destination, iterations, hashes


def prepare(output, runtime_root, current, allow_independent_controls=False):
    output, runtime_root, current = (p.resolve() for p in (output, runtime_root, current))
    output.mkdir(parents=True, exist_ok=True)
    if (output / "jobs.jsonl").exists():
        raise FileExistsError("Queue already exists; never replace scheduler-owned state")
    verification_path = current / "base_checkpoint_verification.json"
    verification = json.loads(verification_path.read_text())["pi05_libero"]
    if not verification["verified"]:
        raise ValueError("Original pi05_libero checkpoint verification did not pass")
    initialization_hash = hashlib.sha256(encoded(verification).encode()).hexdigest()
    base_hashes = {item["path"]: item["sha256"] for item in verification["files"]}
    base_hashes[str(verification_path)] = digest(verification_path)
    jobs = []

    def add(job_id, family, task, variant, seed=0, mode="training", priority=10,
            dependencies=(), checkpoint=None, checkpoint_iteration=0, split=None):
        directory, validations, hashes = make_task(current, output, job_id, task, seed, variant, family, runtime_root)
        destination = output / "outputs" / job_id / "attempt_1"
        hashes.update(base_hashes)
        spec = {"job_id": job_id, "family_id": family, "task": task, "variant": variant,
                "training_seed": seed if mode != "evaluation" else None,
                "runtime_root": str(runtime_root), "mode": mode, "task_directory": str(directory),
                "config_path": str(directory / "train.toml"), "validation_iterations": validations,
                "trainable_scope": SCOPE if mode != "evaluation" else "Frozen evaluation; " + SCOPE,
                "initialization_hash": initialization_hash, "input_hashes": hashes,
                "checkpoint": str(checkpoint) if checkpoint else None,
                "checkpoint_iteration": checkpoint_iteration, "split": split,
                "source_lineage": "Current destination vla_trace_objects_20261031; not the historical October 28 checkpoints"}
        if mode == "evaluation":
            prior_output = current / "methods/rldt" / task
            candidates = []
            if checkpoint is None:
                candidates.append({"path": str(prior_output / f"baseline_{split}.json")})
            elif checkpoint.parent.parent == prior_output:
                candidates.extend([
                    {"path": str(prior_output / f"checkpoint_{checkpoint_iteration:06d}_{split}.json")},
                    {"path": str(prior_output / f"selected_{split}.json"),
                     "selection_path": str(prior_output / "selection.json"), "selected_iteration": checkpoint_iteration},
                ])
            spec.update(reuse_candidates=candidates, reuse_source_root=str(ROOT),
                        reuse_core_contract=str(prior_output / "input_contract.json"))
            if candidates:
                hashes[spec["reuse_core_contract"]] = digest(spec["reuse_core_contract"])
        spec_path = output / "specs" / f"{job_id}.json"
        immutable(spec_path, spec)
        required_artifacts = []
        if checkpoint:
            required_artifacts = [str(checkpoint / name) for name in ("expert.npz", "learner.npz", "rldt_pi05.json")]
        started_jobs = [E00] if mode == "training" and not allow_independent_controls else []
        jobs.append({"job_id": job_id, "family_id": family, "priority": priority, "task": task,
                     "variant": variant, "training_seed": spec["training_seed"], "sft_seed": None,
                     "initialization_hash": initialization_hash, "adapter_hash": None,
                     "trainable_scope": spec["trainable_scope"], "protocol_hash": digest(directory / "protocol.json"),
                     "config_path": spec["config_path"],
                     "command_argv": [str(ROOT / ".venv/bin/python"), str(runtime_root / "scripts/run_next_multigpu_job.py"),
                                      "--spec", str(spec_path), "--output", str(destination)],
                     "spec_path": str(spec_path), "spec_sha256": digest(spec_path), "output_directory": str(destination.parent),
                     "dependencies": list(dependencies), "required_started_jobs": started_jobs,
                     "required_artifacts": required_artifacts, "prerequisites": [], "blockers": [],
                     "gpu_count": 1, "estimated_memory_gb": 53 if mode == "training" else 34 if mode == "smoke" else 20,
                     "max_iterations": 100 if mode == "training" else 1 if mode == "smoke" else 0,
                     "max_steps": 16000 if mode == "training" else 10 if mode == "smoke" else 14000,
                     "teacher_budget": 0, "job_kind": {"training": "train", "evaluation": "evaluate", "smoke": "smoke"}[mode],
                     "state": "blocked_dependency", "scheduler_id": None, "started_at": None, "ended_at": None,
                     "exit_code": None, "attempt": 0,
                     "completion_artifact": "completion.json", "completion_key": "completed",
                     "estimated_runtime_seconds": None,
                     "runtime_estimate_note": "Measure from first completed local attempt; no speculative allocation request"})

    lora_blockers = ["No verified destination pi_0.5 LoRA checkpoint/configuration connected to the native RLDT loader",
                     "Native RLDT detaches the prefix and only trains the full action expert/projections; backbone LoRA RL unsupported",
                     "Require named adapter scope, effective-policy parity, real gradient/update, save/load and simulator smoke evidence"]
    for family, seed in (("E00", 0), ("E01", 1), ("E01", 2)):
        job_id = f"{family}_object_2_seed{seed}"
        jobs.append({"job_id": job_id, "family_id": family, "priority": 0 if family == "E00" else 10,
                     "task": "object_2", "variant": "existing_destination_lora_rldt", "training_seed": seed,
                     "sft_seed": None, "initialization_hash": None, "adapter_hash": None,
                     "trainable_scope": "unresolved; do not substitute native action-only scope",
                     "protocol_hash": None, "config_path": None, "command_argv": [], "output_directory": str(output / "outputs" / job_id),
                     "dependencies": [], "required_started_jobs": [E00] if seed else [], "required_artifacts": [],
                     "prerequisites": [], "blockers": lora_blockers, "gpu_count": 1, "estimated_memory_gb": None,
                     "max_iterations": 100, "max_steps": 16000, "teacher_budget": 0, "job_kind": "train",
                     "state": "blocked_dependency", "scheduler_id": None, "started_at": None, "ended_at": None,
                     "exit_code": None, "attempt": 0})

    add("smoke_native_object0", "E04", "object_0", "reference", mode="smoke", priority=1)
    add("smoke_schedule_object0", "E04", "object_0", "combined", mode="smoke", priority=1)
    add("smoke_native_object2", "E02", "object_2", "reference", mode="smoke", priority=1)
    for seed in range(3):
        add(f"E02_original_object_2_seed{seed}", "E02", "object_2", "original_action_expert", seed,
            dependencies=("smoke_native_object2",), priority=12)
    for variant in VARIANTS:
        for seed in range(3):
            deps = ["smoke_native_object0"]
            if variant in ("quarter_lr_after25", "combined"):
                deps.append("smoke_schedule_object0")
            add(f"E04_object_0_{variant}_seed{seed}", "E04", "object_0", variant, seed,
                dependencies=deps, priority=20 + VARIANTS.index(variant))

    sources = [("original", None, 0),
               ("current_object0_iter50", current / "methods/rldt/object_0/training/iteration_000050", 50),
               ("current_object1_swap_iter25", current / "methods/rldt/object_1/training/iteration_000025", 25)]
    for source_name, checkpoint, iteration in sources:
        for task in ("object_0", "object_1", "object_2"):
            for split in ("test", "source"):
                add(f"E03_{source_name}_{task}_{split}", "E03", task, source_name, mode="evaluation", priority=15,
                    checkpoint=checkpoint, checkpoint_iteration=iteration, split=split)

    families = [
        {"family_id": "E00", "prepared_jobs": 1, "blockers": lora_blockers},
        {"family_id": "E01", "prepared_jobs": 2, "blockers": lora_blockers},
        {"family_id": "E02", "prepared_jobs": 3, "blockers": ["LoRA-initialized frozen and trainable-backbone arms require E00 loader/scope readiness"], "prepared_scope": "Original pi05_libero full action-expert control, RL seeds 0/1/2"},
        {"family_id": "E03", "prepared_jobs": 18, "blockers": ["Historical Object0 iteration50 and Object1-hard iteration25 checkpoints absent", "Historical Object1-hard task assets absent", "E00-selected and matched initialization unavailable"],
         "prepared_scope": "Explicit local transfer extension: original/current Object0 iteration50/current Object1 published-swap iteration25 on current Object0/1/2 and original layouts; Object1 is not Object1-hard"},
        {"family_id": "E04", "prepared_jobs": 15, "blockers": ["Historical Object1-hard BDDL/state/protocol package absent; current Object1 swap is not interchangeable", "Fixed validation Q/action bank and policy-drift instrumentation require implementation before mechanistic claims"],
         "prepared_scope": "Object0 five variants, RL seeds0/1/2 on destination reset families; native reference/pretrained/replay8 and isolated quarter-LR subclass"},
        {"family_id": "E05", "prepared_jobs": 0, "blockers": ["Validated target-aware shaping, success/progress replay labels, compatible return clipping, verified demonstration importer/teacher absent"]},
        {"family_id": "E06", "prepared_jobs": 0, "blockers": ["Competent changed-layout teacher and validated executed/chunk labels absent"]},
        {"family_id": "E07", "prepared_jobs": 0, "blockers": ["Counterbalanced three-object successful demonstration bank and working VLM-LoRA SFT path absent"]},
        {"family_id": "E08", "prepared_jobs": 0, "blockers": ["E07 data and reliable grounding/localization labels required"]},
        {"family_id": "E09", "prepared_jobs": 0, "blockers": ["Validated recipe and nested verified 0/10/50/100 training demonstration subsets absent"]},
        {"family_id": "E10", "prepared_jobs": 0, "blockers": ["Validated curriculum states, reward scale and locked exploration temperature-floor pilot required"]},
        {"family_id": "E11", "prepared_jobs": 0, "blockers": ["Accessible sample package, labels, dataset agreement and validated embodiment/control adapter unresolved"]},
        {"family_id": "E12", "prepared_jobs": 0, "blockers": ["E03 results and instruction/goal-synchronized multitask collector required"]},
        {"family_id": "E13", "prepared_jobs": 0, "blockers": ["Validated RoboCasa mobile-base 12D adapter, target normalization, watchdog and measured evaluation budget required"]},
        {"family_id": "E14", "prepared_jobs": 0, "blockers": ["Useful-feedback recipe in both algorithms and JAX/Torch conversion parity required"]},
        {"family_id": "E15", "prepared_jobs": 0, "blockers": ["Current LIBERO diagnostics are already part of prerequisite campaign; reuse their audited results rather than duplicate jobs", "ManiSkill selected checkpoint/control normalization and fixed diagnostic rollout contract unresolved"]},
        {"family_id": "E16", "prepared_jobs": 0, "blockers": ["Validation-selected finalists and new locked confirmation reset families required"]},
        {"family_id": "E17", "prepared_jobs": 0, "blockers": ["Goal-conditioned replay and consistent reward/termination/feature recomputation absent"]},
    ]
    immutable(output / "families.json", {"families": families, "all_requested_families": [f"E{i:02d}" for i in range(18)]})
    immutable(output / "initialization_inventory.json", {"initialization_hash": initialization_hash, "original_verification": str(verification_path), "checkpoint": verification})
    immutable(output / "jobs.jsonl", "".join(json.dumps(j, sort_keys=True, allow_nan=False) + "\n" for j in jobs))
    immutable(output / "preparation_manifest.json", {
        "created_at": datetime.now(UTC).isoformat(), "runtime_root": str(runtime_root),
        "current_run": str(current), "job_count": len(jobs), "executable_jobs": sum(bool(j["command_argv"]) for j in jobs),
        "independent_controls_authorized": allow_independent_controls,
        "production_order": "E00 must start before native training unless the user authorizes independent controls",
        "global_gate": "Scheduler must wait for entire current experiment and analysis completion before every job",
        "selection": "Initialization included; count successes; earliest ties", "max_rl_iterations": 100,
        "evaluation_noise_seeds": [100000, 100049], "source_files": {str(Path(__file__).resolve()): digest(__file__)},
        "no_gpu_jobs_launched_by_preparation": True,
    })
    return {"output": str(output), "jobs": len(jobs), "executable_jobs": sum(bool(j["command_argv"]) for j in jobs),
            "training_jobs": sum(j["job_kind"] == "train" and bool(j["command_argv"]) for j in jobs),
            "evaluation_jobs": sum(j["job_kind"] == "evaluate" for j in jobs),
            "smoke_jobs": sum(j["job_kind"] == "smoke" for j in jobs)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--current-run", type=Path, required=True)
    parser.add_argument("--allow-independent-controls", action="store_true")
    args = parser.parse_args()
    print(json.dumps(prepare(args.output, args.runtime_root, args.current_run, args.allow_independent_controls), indent=2))
