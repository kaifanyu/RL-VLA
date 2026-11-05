"""Run native evaluation in deterministic independent contiguous shards.

Each child uses the unchanged native evaluator, sampler and per-episode seed.
Only complete, correctly paired shards are published as a legacy-shaped report.
"""

import argparse
import hashlib
import json
import os
import re
import signal
import subprocess
import time
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FACTORY = "rl_vla.envs.libero_pro:make_env"
SHARD_FACTORY = "rl_vla.envs.libero_pro_shard:make_env"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def parse_cpuset(value):
    cpus = []
    for group in value.split(","):
        if not group or not re.fullmatch(r"\d+(?:-\d+)?", group):
            raise ValueError(f"Invalid CPU set: {value}")
        bounds = [int(x) for x in group.split("-")]
        start, stop = (bounds[0], bounds[-1])
        if stop < start:
            raise ValueError("Reversed CPU range")
        cpus.extend(range(start, stop + 1))
    if len(set(cpus)) != len(cpus):
        raise ValueError("CPU set contains duplicates")
    return cpus


def partition(count, shards):
    if type(count) is not int or type(shards) is not int or not 1 <= shards <= count:
        raise ValueError("Require 1 <= shards <= episode count")
    quotient, remainder = divmod(count, shards)
    result, offset = [], 0
    for i in range(shards):
        size = quotient + int(i < remainder)
        result.append((offset, size))
        offset += size
    return result


def shard_config(original, start, count, technical_max_steps=None):
    config = tomllib.loads(original)
    if config["env"]["factory"] != FACTORY:
        raise ValueError("Only the frozen official PRO evaluation factory is supported")
    if config["env"]["kwargs"]["split"] not in {"validation", "test", "source"}:
        raise ValueError("Never shard a training split")
    pattern = r'(?m)^factory[ \t]*=[ \t]*[\"\']' + re.escape(FACTORY) + r'[\"\'][ \t]*$'
    result, changed = re.subn(pattern, f'factory = "{SHARD_FACTORY}"', original)
    if changed != 1 or result.count("[env.kwargs]") != 1:
        raise ValueError("Expected one conventional TOML environment declaration")
    extra = f"\nshard_start = {start}\nshard_count = {count}\n"
    if technical_max_steps is not None:
        extra += f"technical_max_steps = {technical_max_steps}\n"
    result = result.replace("[env.kwargs]", "[env.kwargs]" + extra, 1)
    parsed = tomllib.loads(result)
    expected = dict(config["env"]["kwargs"], shard_start=start, shard_count=count)
    if technical_max_steps is not None:
        expected["technical_max_steps"] = technical_max_steps
    if parsed["env"]["kwargs"] != expected:
        raise ValueError("Unexpected shard configuration mutation")
    return result


def merge_reports(reports, *, original_config, expected_ids, seed, checkpoint, specifications,
                  technical_only=False):
    if len(reports) != len(specifications):
        raise ValueError("Missing evaluation shard")
    episodes = []
    for report, spec in zip(reports, specifications, strict=True):
        if report["checkpoint"] != checkpoint:
            raise ValueError("Evaluation checkpoint identity differs between shards")
        if report["config"] != spec["config"]:
            raise ValueError("Shard evaluator configuration differs from its frozen specification")
        rows = report["episodes"]
        start, count = spec["start"], spec["count"]
        if len(rows) != count:
            raise ValueError("Incomplete shard episode count")
        if [row["seed"] for row in rows] != list(range(seed + start, seed + start + count)):
            raise ValueError("Shard episode seeds differ")
        if [row["reset"]["initial_state_id"] for row in rows] != expected_ids[start:start + count]:
            raise ValueError("Shard episode state IDs differ")
        if any(row["reset"]["experiment_split"] != original_config["env"]["kwargs"]["split"] for row in rows):
            raise ValueError("Shard episode split differs")
        rate = sum(bool(row["success"]) for row in rows) / count
        if abs(report["success_rate"] - rate) > 1e-12:
            raise ValueError("Shard success summary differs from episode records")
        episodes.extend(rows)
    if [row["seed"] for row in episodes] != list(range(seed, seed + len(expected_ids))):
        raise ValueError("Shards leave a gap, overlap, or reorder the paired episode protocol")
    if [row["reset"]["initial_state_id"] for row in episodes] != expected_ids:
        raise ValueError("Merged state enumeration differs")
    return {"config": original_config, "checkpoint": checkpoint, "episodes": episodes,
            "success_rate": sum(bool(row["success"]) for row in episodes) / len(episodes),
            "technical_only": technical_only}


def run(args):
    output, config_path = args.output.resolve(), args.config.resolve()
    directory = output.with_suffix(".shards")
    if output.exists() or output.with_suffix(".episodes.jsonl").exists() or directory.exists():
        raise FileExistsError("Evaluation output or shard directory already exists; no implicit reruns")
    if not 0 <= args.record_episodes <= args.episodes:
        raise ValueError("Invalid video episode count")
    original_text = config_path.read_text()
    original_config = tomllib.loads(original_text)
    kwargs = original_config["env"]["kwargs"]
    protocol = Path(kwargs["protocol"]).resolve()
    plan = json.loads(protocol.read_text())
    definition = plan["tasks"][kwargs["task_key"]]
    ids = definition[f"{kwargs['split']}_state_ids"]
    if args.technical_subset:
        if kwargs["split"] != "validation" or not 1 <= args.episodes <= len(ids):
            raise ValueError("Technical subsets are restricted to validation states")
        ids = ids[:args.episodes]
    elif args.episodes != len(ids) or args.technical_max_steps is not None:
        raise ValueError("Production evaluation must enumerate the complete unchanged frozen split")
    shards = min(args.shards, args.episodes)
    sections = partition(args.episodes, shards)
    cpu_value = os.environ.get("RL_VLA_EVALUATION_CPUSET")
    if cpu_value is None:
        cpu_value = os.environ.get("RLDT_CPUSET")
    cpus = parse_cpuset(cpu_value) if cpu_value else sorted(os.sched_getaffinity(0))[:4 * shards]
    if len(cpus) < 4 * shards:
        raise ValueError("Allocate at least four CPUs per native evaluation subprocess")
    cpu_sections = partition(len(cpus), shards)
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not visible or "," in visible:
        raise ValueError("Assign one explicit evaluation GPU")
    directory.mkdir(parents=True)
    original_hashes = {str(config_path): digest(config_path), str(protocol): digest(protocol)}
    for condition in ("pro", "source"):
        for item in definition[condition].values():
            original_hashes[item["path"]] = item["sha256"]
    specifications, children, handles = [], [], []
    started = time.monotonic()
    status_path = directory / "status.json"
    try:
        for index, ((start, count), (cpu_start, cpu_count)) in enumerate(zip(sections, cpu_sections, strict=True)):
            shard = directory / f"shard_{index:02d}"
            shard.mkdir()
            child_config = shard / "config.toml"
            child_config.write_text(shard_config(original_text, start, count, args.technical_max_steps))
            destination = shard / "report.json"
            cpuset = ",".join(map(str, cpus[cpu_start:cpu_start + cpu_count]))
            command = ["bash", str(ROOT / "scripts/run_rldt.sh"), "evaluate", "--config", str(child_config),
                       "--output", str(destination), "--episodes", str(count), "--seed", str(args.seed + start),
                       "--record-episodes", str(max(0, min(count, args.record_episodes - start)))]
            if args.checkpoint is not None:
                command += ["--checkpoint", args.checkpoint]
            env = os.environ.copy()
            env.update(RLDT_CPUSET=cpuset, XLA_PYTHON_CLIENT_MEM_FRACTION="0.25",
                       XLA_PYTHON_CLIENT_PREALLOCATE="false", LP_NUM_THREADS="4")
            handle = (shard / "evaluate.log").open("x")
            handles.append(handle)
            child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=handle,
                                     stderr=subprocess.STDOUT, start_new_session=True)
            children.append(child)
            specifications.append({"index": index, "start": start, "count": count, "cpuset": cpuset,
                                   "seed": args.seed + start, "state_ids": ids[start:start + count],
                                   "pid": child.pid, "report": str(destination), "command": command,
                                   "config": tomllib.loads(child_config.read_text())})
        manifest = {"original_config": str(config_path), "original_hashes": original_hashes,
                    "checkpoint": args.checkpoint, "seed": args.seed, "episodes": args.episodes,
                    "state_ids": ids, "gpu": visible, "shards": specifications,
                    "technical_only": args.technical_subset, "technical_max_steps": args.technical_max_steps,
                    "sampler": "Unchanged rl_vla.rldt.evaluate; independent PRNGKey(seed+global_episode_index)"}
        write_json(directory / "manifest.json", manifest)
        write_json(status_path, {"state": "running", "shards": shards})
        while True:
            codes = [child.poll() for child in children]
            if any(code is not None and code != 0 for code in codes):
                raise RuntimeError(f"Native evaluation shard failed: exit codes {codes}")
            if all(code is not None for code in codes):
                break
            time.sleep(1)
        for path, expected in original_hashes.items():
            if digest(path) != expected:
                raise ValueError(f"Frozen evaluation input changed: {path}")
        reports = [json.loads(Path(spec["report"]).read_text()) for spec in specifications]
        merged = merge_reports(reports, original_config=original_config, expected_ids=ids, seed=args.seed,
                               checkpoint=args.checkpoint, specifications=specifications,
                               technical_only=args.technical_subset)
        merged["sharding"] = {"manifest": str(directory / "manifest.json"), "shards": shards,
                              "wall_seconds": time.monotonic() - started,
                              "pairing_verified": True, "original_inputs_unchanged": True,
                              "video_and_trajectory_paths": "Retained in each native shard; episode records point to them"}
        episode_path = output.with_suffix(".episodes.jsonl")
        temporary = episode_path.with_suffix(episode_path.suffix + ".tmp")
        with temporary.open("x") as stream:
            for episode in merged["episodes"]:
                stream.write(json.dumps(episode, allow_nan=False) + "\n")
        temporary.replace(episode_path)
        # Report is the commit marker: all shards and paired episode table exist.
        write_json(output, merged)
        write_json(status_path, {"state": "completed", "shards": shards,
                                "wall_seconds": time.monotonic() - started, "merged_report": str(output)})
        print(json.dumps({"completed": True, "episodes": len(merged["episodes"]),
                          "technical_only": args.technical_subset, "output": str(output)}), flush=True)
    except BaseException as exc:
        for child in children:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
        for child in children:
            if child.poll() is None:
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
        write_json(status_path, {"state": "failed", "error": repr(exc), "elapsed_seconds": time.monotonic() - started})
        raise
    finally:
        for handle in handles:
            handle.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=100000)
    parser.add_argument("--record-episodes", type=int, default=50)
    parser.add_argument("--shards", type=int, default=int(os.environ.get("RL_VLA_EVAL_SHARDS", "3")))
    parser.add_argument("--technical-subset", action="store_true", help="Validation-only technical smoke, not research accuracy")
    parser.add_argument("--technical-max-steps", type=int, help="Only with --technical-subset; shorten environment timeout for smoke")
    run(parser.parse_args())
