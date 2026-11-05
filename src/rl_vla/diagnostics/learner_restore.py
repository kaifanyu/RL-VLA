"""Read-only serialization checks shared by the native learner restore audit."""

import hashlib
from pathlib import Path

import numpy as np


def file_hash(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(8 * 1024**2):
            result.update(block)
    return result.hexdigest()


def array_record(value):
    array = np.asarray(value)
    flat = array.reshape(-1)
    checksum = hashlib.sha256()
    finite = True
    for offset in range(0, flat.size, 1024**2):
        block = np.ascontiguousarray(flat[offset:offset + 1024**2])
        finite = finite and bool(np.isfinite(block).all())
        checksum.update(memoryview(block.view(np.uint8)))
    return {"shape": list(array.shape), "dtype": str(array.dtype), "bytes": array.nbytes,
            "finite": finite, "sha256": checksum.hexdigest()}


def tree_records(tree):
    import jax
    return [{"path": jax.tree_util.keystr(path), **array_record(value)}
            for path, value in jax.tree_util.tree_flatten_with_path(tree)[0]]


def tree_schema(tree):
    import jax
    return [{"path": jax.tree_util.keystr(path), "shape": list(value.shape), "dtype": str(value.dtype)}
            for path, value in jax.tree_util.tree_flatten_with_path(tree)[0]]


def optimizer_counts(tree):
    import jax
    result = {}
    for path, value in jax.tree_util.tree_flatten_with_path(tree)[0]:
        if getattr(path[-1], "name", None) == "count":
            array = np.asarray(value)
            if array.shape != () or array.dtype != np.dtype("int32"):
                raise ValueError("Optimizer count must be scalar int32")
            result[jax.tree_util.keystr(path)] = int(array)
    if not result:
        raise ValueError("No native optimizer count leaves found")
    return result


def expected_updates(config, iteration):
    warmup = int(config["rldt"].get("critic_warmup_iterations", 2))
    return {"actor": max(iteration - warmup, 0) * int(config["train"]["actor_steps"]),
            "critic": iteration * int(config["train"]["critic_steps"])}


def check_restored_state(learner, checkpoint, template, original_params, expected):
    """Check actual restored leaves against both archives and the native template."""
    import jax
    records = tree_records(learner.state())
    if tree_schema(learner.state()) != template:
        raise ValueError("Restored state differs from the untouched template schema")
    if not all(row["finite"] for row in records):
        raise ValueError("Restored learner contains nonfinite leaves")
    with np.load(Path(checkpoint) / "learner.npz", allow_pickle=False) as archive:
        if set(archive.files) != {f"arr_{index}" for index in range(len(records))}:
            raise ValueError("Unexpected serialized learner leaf keys")
        for index, restored in enumerate(records):
            saved = array_record(archive[f"arr_{index}"])
            if saved != {key: value for key, value in restored.items() if key != "path"}:
                raise ValueError(f"Saved learner leaf {index} differs in dtype, shape or bytes from restored state")
    params = tree_records(learner.params)
    loaded_expert = learner.flow.load_params(checkpoint)
    if tree_records(loaded_expert) != params:
        raise ValueError("Restored learner.params differs from native expert archive load")
    del loaded_expert
    rule = learner.config["rldt"].get("reference", "moving")
    expected_reference = params if rule == "moving" else original_params
    if tree_records(learner.reference) != expected_reference:
        raise ValueError(f"Restored reference violates configured {rule} reference rule")
    counts = {"actor": optimizer_counts(learner.actor_opt), "critic": optimizer_counts(learner.q_opt)}
    if any(value != expected[kind] for kind, rows in counts.items() for value in rows.values()):
        raise ValueError(f"Restored optimizer counts disagree: {counts} versus {expected}")
    key = np.asarray(learner.key)
    if key.shape != (2,) or key.dtype != np.dtype("uint32"):
        raise ValueError("Restored PRNG key must have shape[2] and dtype uint32")
    return {"restored_leaves": records, "optimizer_counts": counts, "expected_optimizer_updates": expected,
            "prng_key": {**array_record(key), "value": key.tolist()},
            "reference_rule": rule, "reference_equals_checkpoint_params": rule == "moving",
            "reference_equals_original_params": tree_records(learner.reference) == original_params,
            "expert_params_exact": True, "serialized_leaf_bytes_exact": True,
            "template_shapes_and_dtypes_exact": True, "all_restored_leaves_finite": True}
