"""CPU-only immutable artifact exchange for separately compiled trace stages."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

STAGES = ("baseline", "trained", "baseline_projection")
POLICIES = ("pi05_libero", "rldt_000025", "rldt_000050", "rldt_000100")


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def save_context(folder, context):
    """Preserve tree structure and exact BF16 values without pickle/object arrays."""
    folder = Path(folder)
    arrays = {}

    def encode(value):
        if isinstance(value, dict):
            return {"kind": "dict", "items": [[key, encode(item)] for key, item in value.items()]}
        if isinstance(value, (tuple, list)):
            return {"kind": type(value).__name__, "items": [encode(item) for item in value]}
        if value is None:
            return {"kind": "none"}
        array = np.asarray(value)
        name = f"leaf_{len(arrays):03d}"
        dtype = str(array.dtype)
        if dtype == "bfloat16":
            array = array.astype(np.float32)
        if array.dtype.hasobject:
            raise TypeError("Context contains unsupported object dtype")
        arrays[name] = array
        return {"kind": "array", "key": name, "dtype": dtype, "shape": list(array.shape)}

    tree = encode(context)
    np.savez(folder / "context.npz", **arrays)
    atomic_json(folder / "context_schema.json", {"schema": 1, "tree": tree,
                "npz_sha256": sha(folder / "context.npz"),
                "encoding": "BF16 leaves stored as exactly representable float32; original dtype restored"})


def load_context(folder, array_factory=None):
    folder = Path(folder)
    schema = json.loads((folder / "context_schema.json").read_text())
    if schema.get("schema") != 1 or sha(folder / "context.npz") != schema["npz_sha256"]:
        raise ValueError("Saved baseline context hash/schema differs")
    if array_factory is None:
        array_factory = lambda array, dtype: np.asarray(array, dtype=dtype)
    with np.load(folder / "context.npz", allow_pickle=False) as archive:
        used = []

        def decode(node):
            kind = node["kind"]
            if kind == "dict":
                return {key: decode(value) for key, value in node["items"]}
            if kind in ("tuple", "list"):
                values = [decode(item) for item in node["items"]]
                return tuple(values) if kind == "tuple" else values
            if kind == "none":
                return None
            if kind != "array":
                raise ValueError("Unknown context node")
            array = archive[node["key"]]
            if list(array.shape) != node["shape"]:
                raise ValueError("Saved context leaf shape differs")
            used.append(node["key"])
            return array_factory(array, node["dtype"])

        context = decode(schema["tree"])
        if len(used) != len(set(used)) or set(used) != set(archive.files):
            raise ValueError("Saved context leaves are missing, repeated, or unused")
    return context


def artifact(path):
    path = Path(path).absolute()
    before = path.stat()
    digest = sha(path)
    after = path.stat()
    identity = [after.st_size, after.st_mtime_ns, after.st_ino]
    if identity != [before.st_size, before.st_mtime_ns, before.st_ino]:
        raise ValueError(f"Artifact changed while hashing: {path}")
    return {"path": str(path), "sha256": digest, "identity": identity}


def publish_stage(output, stage, *, plan, owned_paths, dependencies=()):
    if stage not in STAGES:
        raise ValueError("Unknown grid stage")
    output = Path(output)
    path = output / "stages" / stage / "summary.json"
    if path.exists():
        raise FileExistsError(f"Immutable stage already published: {path}")
    files = []
    for owned in owned_paths:
        owned = Path(owned)
        if not owned.exists():
            raise ValueError(f"Missing owned grid artifact: {owned}")
        files.extend(sorted(p for p in owned.rglob("*") if p.is_file()) if owned.is_dir() else [owned])
    launch_plan = output / "stages" / stage / "plan.json"
    if launch_plan.is_file():
        files.append(launch_plan)
    if len(set(files)) != len(files):
        raise ValueError("Stage artifact ownership overlaps")
    value = {"schema": 1, "complete": True, "stage": stage, **plan,
             "dependencies": [artifact(dep) for dep in dependencies],
             "artifacts": [artifact(file) for file in files]}
    atomic_json(path, value)
    return read_stage(output, stage)


def read_stage(output, stage):
    path = Path(output) / "stages" / stage / "summary.json"
    value = json.loads(path.read_text())
    if value.get("complete") is not True or value.get("stage") != stage:
        raise ValueError("Grid stage is incomplete or mismatched")
    return {"path": str(path), "sha256": sha(path), "summary": value}


def verify_stage(contract, *, full_hash=False):
    if sha(contract["path"]) != contract["sha256"]:
        raise ValueError("Grid stage completion artifact changed")
    for entry in contract["summary"]["dependencies"] + contract["summary"]["artifacts"]:
        path = Path(entry["path"])
        stat = path.stat()
        if [stat.st_size, stat.st_mtime_ns, stat.st_ino] != entry["identity"]:
            raise ValueError(f"Immutable stage artifact identity changed: {path}")
        if full_hash and sha(path) != entry["sha256"]:
            raise ValueError(f"Immutable stage artifact hash changed: {path}")


def aggregate_task(output):
    """Validate all policy/stage outputs; never publish a global campaign marker."""
    output = Path(output)
    contracts = [read_stage(output, stage) for stage in STAGES]
    for contract in contracts:
        verify_stage(contract, full_hash=True)
        launch = Path(contract["path"]).with_name("plan.json")
        if launch.exists():
            for source, expected in json.loads(launch.read_text())["source_sha256"].items():
                if sha(source) != expected:
                    raise ValueError("Stage source/profile contract changed before aggregation")
    common = ("task", "split", "state_ids", "steps", "selection_sha256", "checkpoint_audit_sha256", "manifest_sha256")
    baseline = contracts[0]["summary"]
    case_plan = json.loads((output / "case_plan.json").read_text())
    audit = case_plan["checkpoint_audit_contract"]
    if sha(audit["audit_path"]) != baseline["checkpoint_audit_sha256"]:
        raise ValueError("Native checkpoint audit changed before aggregation")
    if sha(Path(audit["audit_path"]).parent / "selection.json") != baseline["selection_sha256"]:
        raise ValueError("Locked selection changed before aggregation")
    for filename, expected in audit["files"].items():
        if sha(filename) != expected["sha256"]:
            raise ValueError("Native checkpoint changed before aggregation")
    if any(any(item["summary"][key] != baseline[key] for key in common) for item in contracts[1:]):
        raise ValueError("Grid stages do not share identical cases, schedule, and selection")
    for index, profile_id in ((0, "native_baseline_v1"), (1, "native_trained_v2"), (2, "native_baseline_v1")):
        if contracts[index]["summary"]["profile_id"] != profile_id:
            raise ValueError("Grid stage used the wrong native policy compiler profile")
    expected_dependencies = [[], [contracts[0]["path"]], [contracts[0]["path"], contracts[1]["path"]]]
    for contract, expected in zip(contracts, expected_dependencies, strict=True):
        if [entry["path"] for entry in contract["summary"]["dependencies"]] != expected:
            raise ValueError("Grid stage dependency chain is incomplete")
    shared_metadata = [json.loads((output / "shared" / f"state_{state:06d}" / "metadata.json").read_text())
                       for state in baseline["state_ids"]]
    expected_conditions = {"none", "no_image", "no_text"} | {
        f"{region}:{mode}" for region in ("target", "distractor", "random", "robot", "background", "old_target")
        for mode in ("background_fill", "mosaic")}
    backbone = None
    for policy in POLICIES:
        equality = json.loads((output / policy / "prefix_equality.json").read_text())
        if not equality.get("complete") or not equality.get("passed") or len(equality["observations"]) != len(baseline["state_ids"]):
            raise ValueError("Missing complete per-policy prefix identity audit")
        if backbone is None:
            backbone = equality["frozen_backbone_sha256"]
        if equality["frozen_backbone_sha256"] != backbone:
            raise ValueError("Grid policy frozen-backbone identities differ")
        for row, metadata in zip(equality["observations"], shared_metadata, strict=True):
            if (row.get("passed") is not True or row.get("prefix_max_abs_difference") != 0
                    or row.get("observation_sha256") != metadata["bank_record"]["observation_sha256"]
                    or row.get("baseline_feature_sha256") != metadata["prefix_features_sha256"]
                    or row.get("checkpoint_feature_sha256") != metadata["prefix_features_sha256"]):
                raise ValueError("Per-observation frozen-prefix equality audit does not pass")
        for index, state in enumerate(baseline["state_ids"]):
            folder = output / policy / f"state_{state:06d}"
            required = ["actions.npz", "own_trajectory_attention.npz", "metadata.json", "open_loop_summary.json", "open_loop_actions.npz"]
            if index < 5:
                required += ["layer_knockout_summary.json", "layer_knockout_arrays.npz"]
            for step in baseline["steps"]:
                required += [f"{stem}_step{step:02d}_{condition}.npz" for stem in ("expert", "views") for condition in ("none", "no_image", "no_text")]
                if index < 5:
                    required += [f"projection_ablation_step{step:02d}.{suffix}" for suffix in ("npz", "json")]
            if any(not (folder / name).is_file() for name in required):
                raise ValueError(f"Incomplete native policy trace: {folder}")
            open_loop = json.loads((folder / "open_loop_summary.json").read_text())
            if (open_loop.get("complete") is not True or len(open_loop["conditions"]) != 15
                    or {row["condition"] for row in open_loop["conditions"]} != expected_conditions):
                raise ValueError("Missing predeclared open-loop intervention condition")
            if index < 5:
                for step in baseline["steps"]:
                    projection = json.loads((folder / f"projection_ablation_step{step:02d}.json").read_text())
                    if projection.get("complete") is not True or projection.get("native_reconstruction_max_abs_difference") != 0:
                        raise ValueError("Projection comparison lacks exact native endpoints")
    for state in baseline["state_ids"]:
        folder = output / "pi05_base" / f"state_{state:06d}"
        if any(not (folder / name).is_file() for name in ("prefix.npz", "counterfactual_prefix.npz", "prefix_metadata.json", "counterfactual_prefix_metadata.json")):
            raise ValueError("Missing independent raw-base representation reference")
    controls = json.loads((output / "compiler_controls/summary.json").read_text())
    expected_controls = [(state, step) for state in baseline["state_ids"][:5] for step in baseline["steps"]]
    if (controls.get("complete") is not True or controls.get("states") != baseline["state_ids"][:5]
            or controls.get("steps") != baseline["steps"]
            or [(row["state_id"], row["step"]) for row in controls["cases"]] != expected_controls):
        raise ValueError("Missing predetermined compiler-only attention/hidden/velocity controls")
    for row in controls["cases"]:
        if sha(row["path"]) != row["sha256"]:
            raise ValueError("Compiler-only diagnostic control changed")
    value = {"complete": True, "completed": True, "task": baseline["task"], "states": len(baseline["state_ids"]),
             "policies": list(POLICIES), "raw_base": "VLM-only, action N/A",
             "selection_sha256": baseline["selection_sha256"],
             "checkpoint_audit_sha256": baseline["checkpoint_audit_sha256"],
             "checkpoint_hashes_verified_start_and_final": True,
             "stages": [{"stage": c["summary"]["stage"], "path": c["path"], "sha256": c["sha256"]} for c in contracts],
             "compiler_dispatch": "Separate baseline/trained processes; common baseline contexts and latent paths persisted exactly"}
    atomic_json(output / "summary.json", value)
    return value
