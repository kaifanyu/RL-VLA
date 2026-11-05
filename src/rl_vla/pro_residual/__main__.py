"""CLI: resolve | check | train | evaluate. Run through scripts/run_flashsac_pro.sh."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

SMOKE = {"train": {"total_env_steps": 60, "learning_starts": 20, "batch_size": 16, "replay_capacity": 2000,
                   "log_every_steps": 5, "checkpoint_every_steps": 30, "validate_every_steps": 40,
                   "validate_at_steps": None, "validation_states": 1},
         "simulator": {"max_steps": 30},
         "record": {"train_every_episodes": 1, "max_train_recordings": 1},
         "flashsac": {"lr_decay_updates": 100}}


def _experiment(config: dict, run_dir: Path):
    from .experiment import Experiment

    return Experiment(config, run_dir)


def _sim_factory(exp):
    from .sim import SimClient

    def make(source: str):
        return SimClient(exp.config["simulator"], source, log_path=exp.run_dir / "logs" / "simulator_worker.log")

    return make


def _apply_overrides(config: dict, args) -> dict:
    from .spec import deep_merge, validate_config

    if getattr(args, "smoke", False):
        config = deep_merge(config, SMOKE)
        config["smoke"] = True
    if getattr(args, "total_env_steps", None):
        config["train"]["total_env_steps"] = args.total_env_steps
    if getattr(args, "max_episode_steps", None):
        config["simulator"]["max_steps"] = args.max_episode_steps
    if getattr(args, "device", None):
        config["train"]["device"] = args.device
    validate_config(config)
    return config


def cmd_resolve(args):
    from .experiment import resolve_run_dir
    from .spec import build_split, load_config, resolve_case, resolve_original_task

    config = load_config(args.config)
    case = resolve_case(config["task"])
    original = resolve_original_task(case)
    split = build_split(config, case, original)
    print(json.dumps({"label": config["task"].get("label"), "label_resolution": config["task"].get("label_resolution"),
                      "case": case, "retention_task": original,
                      "split": {k: split[k] for k in ("train", "validation", "test", "separates", "overlap_checks")},
                      "default_output_dir": str(resolve_run_dir(config["output_dir"]))}, indent=2))


def cmd_make_states(args):
    """Generate the config's state set in the simulator interpreter (CPU only, no policy)."""
    import copy
    import subprocess

    from .experiment import resolve_run_dir
    from .sim import simulator_environment
    from .spec import load_config, load_state_set, resolve_case, state_set_request

    config = load_config(args.config)
    generation = config.get("state_generation")
    target = args.output or config["task"].get("state_set")
    if not generation or not target:
        raise SystemExit("The config needs [state_generation] and task.state_set (or pass --output)")
    task = dict(config["task"], state_set=None)  # Resolve the published case first.
    case = resolve_case(task)
    request = state_set_request(case, generation["generation_seed"], generation["split_seed"], config["split"])
    output = resolve_run_dir(target)
    output.mkdir(parents=True, exist_ok=True)
    request_path = output / "request.json"
    if request_path.exists() and json.loads(request_path.read_text()) != request:
        raise SystemExit(f"{output} was started with a different request; choose a new directory")
    request_path.write_text(json.dumps(request, indent=2) + "\n")
    simulator = copy.deepcopy(config["simulator"])
    simulator["renderer"] = "osmesa"  # Same as the RLDT generator; states do not depend on it.
    python, env = simulator_environment(simulator, "pro")
    command = [python, "-m", "rl_vla.pro_residual.states", "--request", str(request_path),
               "--output", str(output), "--workers", str(args.workers)]
    print("Running:", " ".join(command), flush=True)
    subprocess.run(command, env=env, check=True)
    _, manifest = load_state_set(output)
    applied = resolve_case(config["task"]) if config["task"].get("state_set") else None
    print(json.dumps({"state_set": str(output), "states_sha256": manifest["states_sha256"],
                      "state_count": manifest["state_count"], "published_offset": manifest["published_offset"],
                      "rejected": len(manifest["rejected"]), "attempts_used": manifest["attempts_used"],
                      "resolves_with_config": bool(applied)}, indent=2))


def cmd_check(args):
    from .checks import run_checks
    from .experiment import prepare_new_run, resolve_run_dir
    from .spec import load_config

    config = _apply_overrides(load_config(args.config), args)
    run_dir = resolve_run_dir(args.output)
    prepare_new_run(run_dir)
    exp = _experiment(config, run_dir)
    exp.write_config()
    report = run_checks(exp, _sim_factory(exp), parity_steps=args.parity_steps,
                        reproduce_states=args.reproduce_states, output_name="verification")
    print(json.dumps({k: v["passed"] for k, v in report["checks"].items()}, indent=2))
    print(f"report: {run_dir / 'logs/verification.json'}  passed={report['passed']}")
    sys.exit(0 if report["passed"] else 1)


def cmd_train(args):
    from .experiment import prepare_new_run, resolve_run_dir
    from .spec import load_config
    from .train import train

    if args.resume:
        run_dir = resolve_run_dir(args.output)
        saved = json.loads((run_dir / "config/resolved_config.json").read_text())["config"]
        config = saved
        if args.config:
            from .spec import load_config as _load

            requested = _apply_overrides(_load(args.config), args)
            config = dict(saved, train=requested["train"], record=requested["record"])
        if args.total_env_steps:
            config["train"]["total_env_steps"] = args.total_env_steps
    else:
        if not args.config:
            raise SystemExit("--config is required for a new run")
        config = _apply_overrides(load_config(args.config), args)
        run_dir = resolve_run_dir(args.output or config["output_dir"])
        prepare_new_run(run_dir)
    exp = _experiment(config, run_dir)
    print(f"Run directory: {run_dir}", flush=True)
    summary = train(exp, _sim_factory(exp), resume=args.resume, smoke=bool(config.get("smoke")))
    print(json.dumps({k: summary[k] for k in ("env_steps", "episodes", "updates", "final_checkpoint", "base_unchanged")}, indent=2))


def cmd_evaluate(args):

    from .evaluate import evaluate, load_residual_checkpoint
    from .experiment import resolve_device, resolve_run_dir

    run_dir = resolve_run_dir(args.run)
    config = json.loads((run_dir / "config/resolved_config.json").read_text())["config"]
    overrides = {}
    # Evaluation always uses the suite horizon unless explicitly overridden; a training
    # smoke cap saved in the run's config must not leak into evaluation.
    config["simulator"]["max_steps"] = args.max_episode_steps
    if args.max_episode_steps:
        overrides["max_episode_steps"] = args.max_episode_steps
    if args.renderer and args.renderer != config["simulator"].get("renderer", "osmesa"):
        # EGL is not bit-repeatable between runs; osmesa is. Evaluating an EGL-trained run with
        # osmesa gives exact base/residual pairing and queue-comparable base numbers.
        overrides["renderer"] = {"trained_with": config["simulator"].get("renderer", "osmesa"),
                                 "evaluated_with": args.renderer}
        config["simulator"]["renderer"] = args.renderer
    exp = _experiment(config, run_dir)
    exp.ensure_split(create=False)
    if args.split == "test" and not args.heldout:
        raise SystemExit("The test split is held out. Pass --heldout to evaluate it (never for model selection).")
    states = args.initial_states if args.initial_states is not None else exp.split_states(args.split)
    if args.episodes:
        states = states[:args.episodes]
    allowed = set(exp.split_states(args.split))
    if not set(states) <= allowed:
        raise SystemExit(f"Initial states {sorted(set(states) - allowed)} are not in the {args.split} split")
    device = resolve_device(args.device or config["train"]["device"])
    exp.load_base()
    learner, payload, path = None, None, None
    if "residual" in args.variants:
        names = {"best": "best.pt", "latest": "latest.pt", "final": "final.pt"}
        path = Path(args.checkpoint) if args.checkpoint not in names else run_dir / "checkpoints" / names[args.checkpoint]
        learner = exp.make_learner(device)
        payload = load_residual_checkpoint(path, learner, device)
        if payload["case_id"] != exp.case["case_id"]:
            raise SystemExit("Checkpoint was trained on a different case")
        path = str(path.resolve())
    eval_id = args.eval_id or time.strftime("ev%Y%m%dT%H%M%S")
    metrics = evaluate(exp, _sim_factory(exp), split=args.split, initial_states=states, variants=tuple(args.variants),
                       learner=learner, checkpoint_payload=payload, checkpoint_path=path, eval_id=eval_id,
                       deterministic=not args.stochastic, record=not args.no_record, device=device,
                       overrides=overrides)
    print(json.dumps(metrics, indent=2))


def main():
    parser = argparse.ArgumentParser(prog="python -m rl_vla.pro_residual", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("resolve", help="Print the resolved task identity and split")
    p.add_argument("--config", required=True)
    p.set_defaults(func=cmd_resolve)
    p = sub.add_parser("make-states", help="Generate train/validation initial states (RLDT controlled-PRO protocol)")
    p.add_argument("--config", required=True)
    p.add_argument("--output", help="State set directory (default: task.state_set)")
    p.add_argument("--workers", type=int, default=2, help="Parallel simulator constructors (CPU)")
    p.set_defaults(func=cmd_make_states)
    p = sub.add_parser("check", help="Verification suite on the real checkpoint and simulator")
    p.add_argument("--config", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--parity-steps", type=int, default=15)
    p.add_argument("--reproduce-states", type=int, nargs="*", default=[])
    p.set_defaults(func=cmd_check)
    p = sub.add_parser("train", help="Train the FlashSAC residual (new run or --resume)")
    p.add_argument("--config")
    p.add_argument("--output", help="Run directory (default: config output_dir)")
    p.add_argument("--resume", action="store_true", help="Continue the run in --output from checkpoints/latest.pt")
    p.add_argument("--smoke", action="store_true", help="Tiny budget and 30-step episodes (pipeline check only)")
    p.add_argument("--total-env-steps", type=int)
    p.add_argument("--max-episode-steps", type=int, help="Non-protocol episode cap (smoke/debug only)")
    p.add_argument("--device")
    p.set_defaults(func=cmd_train)
    p = sub.add_parser("evaluate", help="Paired base-only vs base+residual evaluation")
    p.add_argument("--run", required=True)
    p.add_argument("--split", required=True, choices=["validation", "test", "retention", "train"])
    p.add_argument("--checkpoint", default="best", help="best | latest | final | path")
    p.add_argument("--variants", nargs="+", default=["base", "residual"], choices=["base", "residual"])
    p.add_argument("--episodes", type=int, help="Use only the first N states of the split")
    p.add_argument("--initial-states", type=int, nargs="+")
    p.add_argument("--eval-id")
    p.add_argument("--heldout", action="store_true", help="Required to evaluate the held-out test split")
    p.add_argument("--stochastic", action="store_true", help="Sample the residual instead of its mean")
    p.add_argument("--no-record", action="store_true")
    p.add_argument("--max-episode-steps", type=int, help="Non-protocol episode cap (smoke/debug only)")
    p.add_argument("--renderer", choices=["osmesa", "egl"],
                   help="Override the run's simulator renderer (recorded as a non-protocol override)")
    p.add_argument("--device")
    p.set_defaults(func=cmd_evaluate)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
