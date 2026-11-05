"""Generated LIBERO-PRO initial states, following the RLDT controlled-PRO protocol.

The procedure reproduces ``scripts/prepare_pro_suite_grasp.py`` from the RLDT campaign
(origin/main d0ef690) so FlashSAC can use the same train/validation/test roles:

* attempt ``k`` seeds NumPy and Python with ``generation_seed + k``, constructs a fresh
  ``OffScreenRenderEnv`` for the PRO BDDL and captures ``get_sim_state()`` before any
  explicit reset (the published PRO generator's sampling point);
* an attempt is rejected if the state is invalid, already satisfies the goal, or ends
  within ten settling actions; a qpos duplicate of a published or earlier state is rejected;
* attempts are consumed in seed order until ``train + validation`` states are accepted;
* those are permuted with ``default_rng(split_seed)`` and the published states are
  appended, so ids ``[0, train)`` are train, ``[train, train + validation)`` validation and
  the published states form the test split.

No policy is involved in generation or splitting.

``generate`` runs in the simulator interpreter (it imports LIBERO). The parent CLI
(``python -m rl_vla.pro_residual make-states``) launches it with the same environment as
the simulator worker.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

FORMAT = "rl-vla-pro-state-set-v1"
SETTLE_STEPS = 10
SETTLING_ACTION = [0.0] * 6 + [-1.0]  # Same as rl_vla.envs.libero.SETTLING_ACTION.


def sha256_file(path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def signature(state, nq: int) -> bytes:
    """Physical configuration (qpos), ignoring MuJoCo time; identical to the RLDT script."""
    return np.round(np.asarray(state)[1:1 + nq], 12).tobytes()


def generate_attempt(bddl: str, seed: int, shape: tuple, path: str) -> str:
    from libero.libero.envs import OffScreenRenderEnv

    np.random.seed(seed)
    random.seed(seed)
    env, state, nq, error = None, np.empty(0), 0, ""
    try:
        env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256)
        state = np.asarray(env.get_sim_state()).copy()
        nq = env.sim.model.nq
        if state.shape != tuple(shape) or not np.isfinite(state).all():
            raise ValueError("Generated state has incompatible shape or nonfinite values")
        env.seed(seed)
        env.reset()
        env.set_init_state(state)
        if env.check_success():
            raise ValueError("State already satisfies goal")
        for _ in range(SETTLE_STEPS):
            _, _, done, _ = env.step(SETTLING_ACTION)
            if done or env.check_success():
                raise ValueError("State ends during settling")
        if not np.isfinite(env.get_sim_state()).all():
            raise ValueError("State becomes nonfinite during settling")
    except Exception as exc:  # noqa: BLE001 - every failed attempt is kept for audit
        error = repr(exc)
    finally:
        if env is not None:
            env.close()
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as stream:
        np.savez(stream, state=state, nq=nq, seed=seed, error=error)
    temporary.replace(path)
    return str(path)


def collect_attempts(paths, published: np.ndarray, count: int):
    """Consume attempts in seed order (independent of worker completion order)."""
    generated, accepted, rejected = [], [], []
    physical, expected_nq = None, None
    for path in paths:
        with np.load(path, allow_pickle=False) as saved:
            state, nq, seed, error = saved["state"], int(saved["nq"]), int(saved["seed"]), str(saved["error"])
        if error:
            rejected.append({"generation_seed": seed, "error": error})
            continue
        if state.shape != published.shape[1:] or not np.isfinite(state).all() or nq <= 0:
            raise ValueError(f"Invalid cached generation result: {path}")
        if physical is None:
            expected_nq = nq
            physical = {signature(row, nq) for row in published}
            if len(physical) != len(published):
                raise ValueError("Published states have duplicate physical configurations")
        if nq != expected_nq:
            raise ValueError("State dimensions changed between simulator constructors")
        key = signature(state, nq)
        if key in physical:
            rejected.append({"generation_seed": seed, "error": "Duplicate physical configuration"})
            continue
        physical.add(key)
        generated.append(state)
        accepted.append({"generation_seed": seed, "qpos_sha256": hashlib.sha256(key).hexdigest()})
        if len(generated) == count:
            break
    return generated, accepted, rejected


def assemble(generated, accepted, published: np.ndarray, split_seed: int):
    """Permute generated states with the split seed and append the published states."""
    order = np.random.default_rng(split_seed).permutation(len(generated))
    combined = np.concatenate([np.asarray(generated)[order], published])
    return combined, [accepted[i] for i in order]


def splits(train: int, validation: int, test: int) -> dict:
    return {"train": list(range(train)), "validation": list(range(train, train + validation)),
            "test": list(range(train + validation, train + validation + test))}


def generate(request: dict, output: Path, workers: int, max_attempts: int = 1000) -> dict:
    """Build (or verify) one state set. Never overwrites a finished set."""
    from rl_vla.robustness import load_states

    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "state_set.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest["request"] != request:
            raise FileExistsError(f"{output} holds a different state set; choose a new directory")
        if sha256_file(output / "pro_states.npy") != manifest["states_sha256"]:
            raise ValueError("pro_states.npy no longer matches its manifest")
        return manifest
    published = load_states(request["published_states"])
    if sha256_file(request["published_states"]) != request["published_states_sha256"]:
        raise ValueError("Published state file differs from the resolved case")
    if sha256_file(request["bddl"]) != request["bddl_sha256"]:
        raise ValueError("BDDL differs from the resolved case")
    if len(published) != request["test"]:
        raise ValueError(f"Expected {request['test']} published states, found {len(published)}")
    count = request["train"] + request["validation"]
    attempts = output / "generation_attempts"
    attempts.mkdir(exist_ok=True)
    context_path = attempts / "context.json"
    context = {"request": request, "generator_sha256": sha256_file(__file__), "max_attempts": max_attempts}
    if context_path.exists() and json.loads(context_path.read_text()) != context:
        raise ValueError("Cannot resume generation after its inputs or generator changed")
    write_json(context_path, context)
    ordered, generated, accepted, rejected = [], [], [], []
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        for start in range(0, max_attempts, workers):
            paths = [attempts / f"{index:04d}.npz" for index in range(start, min(start + workers, max_attempts))]
            jobs = [pool.submit(generate_attempt, request["bddl"], request["generation_seed"] + index,
                                published.shape[1:], str(path))
                    for index, path in enumerate(paths, start=start) if not path.exists()]
            for future in as_completed(jobs):
                future.result()
            ordered.extend(paths)
            generated, accepted, rejected = collect_attempts(ordered, published, count)
            print(f"generated {len(generated)}/{count}, rejected {len(rejected)}, attempts {len(ordered)}", flush=True)
            if len(generated) == count:
                break
            if len(rejected) >= 20 and not generated:
                raise RuntimeError("Repeated state-generation failures; inspect generation_attempts/")
    if len(generated) != count:
        raise RuntimeError(f"Only {len(generated)} valid states after {max_attempts} attempts")
    combined, accepted = assemble(generated, accepted, published, request["split_seed"])
    if not np.array_equal(combined[count:], published):
        raise AssertionError("Published states must form the tail of the state array")
    states_path = output / "pro_states.npy"
    with states_path.with_suffix(".tmp").open("wb") as stream:
        np.save(stream, combined, allow_pickle=False)
    states_path.with_suffix(".tmp").replace(states_path)
    import importlib.metadata as metadata

    versions = {}
    for package in ("mujoco", "robosuite", "numpy", "bddl"):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    manifest = {
        "format": FORMAT, "request": request,
        "states_file": str(states_path), "states_sha256": sha256_file(states_path),
        "state_count": int(len(combined)), "state_vector_dim": int(combined.shape[1]),
        "published_offset": count, "splits": splits(request["train"], request["validation"], request["test"]),
        "accepted": accepted, "rejected": rejected, "attempts_used": len(ordered),
        "simulator_versions": versions, "generator_sha256": sha256_file(__file__),
        "method": ("RLDT controlled-PRO generation (scripts/prepare_pro_suite_grasp.py, origin/main d0ef690): "
                   "seed NumPy/Python with generation_seed + attempt; fresh OffScreenRenderEnv; capture before the "
                   "first explicit reset; reject invalid states, goals met before or within ten settling actions and "
                   "qpos duplicates; consume attempts in seed order; permute with default_rng(split_seed); append "
                   "the published states as the test split. No policy filtering."),
        "comparison_note": ("accepted[i].qpos_sha256 uses the same definition as RLDT's generation.json, so the two "
                            "state sets can be compared entry by entry without copying states."),
    }
    write_json(manifest_path, manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True, type=Path, help="JSON request written by the parent CLI")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    manifest = generate(json.loads(args.request.read_text()), args.output.resolve(), args.workers)
    print(json.dumps({k: manifest[k] for k in ("states_file", "states_sha256", "state_count", "published_offset",
                                                "attempts_used")}, indent=2))


if __name__ == "__main__":
    main()
