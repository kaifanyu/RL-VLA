"""Independent, read-only readiness checks for required analysis artifacts.

Producer metrics can precede rendering. A complete metrics JSON alone therefore
does not certify figures; manifests are published only after actual validation.
"""

import csv
import hashlib
import io
import json
import zipfile
from pathlib import Path

import numpy as np

TASKS = ("object_0", "object_1", "object_2")
MODELS = ("pi05_base", "pi05_libero")
POLICIES = ("pi05_libero", "rldt_25", "rldt_50", "rldt_100")
GROUPS = ("modality_routing", "layer_response", "action_vectors", "token_controls")


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def identity(path):
    stat = Path(path).stat()
    return [stat.st_size, stat.st_mtime_ns, stat.st_ino]


def read(path):
    return json.loads(Path(path).read_text(), parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def canonical(policy):
    return f"rldt_{int(policy.removeprefix('rldt_'))}" if policy.startswith("rldt_") else policy


def artifact(path, *, fields=(), expected_sha=None):
    """Decode actual content and bind stable bytes, not merely filename presence."""
    path = Path(path).resolve()
    before = identity(path)
    if before[0] == 0:
        raise ValueError("Empty artifact")
    suffix = path.suffix.lower()
    details = {}
    if suffix == ".png":
        from PIL import Image
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            image.load()
            if min(image.size) < 2:
                raise ValueError("Figure has no usable pixel dimensions")
            details = {"pixels": list(image.size), "mode": image.mode}
    elif suffix == ".pdf":
        from pypdf import PdfReader
        from pypdf.errors import PyPdfError
        try:
            reader = PdfReader(str(path), strict=True)
            if not reader.pages or any(float(page.mediabox.width) <= 0 or float(page.mediabox.height) <= 0 for page in reader.pages):
                raise ValueError("PDF has no usable pages")
            for page in reader.pages:
                if page.get_contents() is None:
                    raise ValueError("PDF page has no content stream")
                page.get_contents().get_data()
            details = {"pages": len(reader.pages)}
        except PyPdfError as error:
            raise ValueError(f"PDF could not be fully parsed: {error}") from error
    elif suffix == ".csv":
        with path.open(newline="") as stream:
            rows = csv.DictReader(stream)
            if not rows.fieldnames or not set(fields).issubset(rows.fieldnames):
                raise ValueError("CSV schema is missing required columns")
            count = 0
            for row in rows:
                if None in row or any(value is None for value in row.values()):
                    raise ValueError("CSV contains a truncated or malformed row")
                count += 1
            if count == 0:
                raise ValueError("Required table has no observations")
            details = {"rows": count, "columns": rows.fieldnames}
    elif suffix in (".npz", ".npy"):
        loaded = np.load(path, allow_pickle=False)
        arrays = {"array": loaded} if suffix == ".npy" else {name: loaded[name] for name in loaded.files}
        try:
            if not arrays or not set(fields).issubset(arrays):
                raise ValueError("Array artifact is missing required fields")
            validity = {}
            for name, value in arrays.items():
                if value.dtype.kind in "biufc" and not np.isfinite(value).all():
                    mask_name = name + "_valid" if name + "_valid" in arrays else ("valid" if name == "values" else None)
                    mask = arrays.get(mask_name) if mask_name else None
                    finite = np.isfinite(value)
                    if (mask is None or mask.dtype.kind != "b" or mask.shape != value.shape
                            or not np.array_equal(mask, finite) or not np.isnan(value[~finite]).all()):
                        raise ValueError("Array artifact contains nonfinite values without an exact Boolean validity mask")
                    validity[name] = {"mask": mask_name, "explicit_na_cells": int((~finite).sum())}
            details = {"arrays": {name: {"shape": list(value.shape), "dtype": str(value.dtype),
                                          **({"explicit_na": validity[name]} if name in validity else {})}
                                   for name, value in arrays.items()}}
        finally:
            if suffix == ".npz":
                loaded.close()
    elif suffix == ".json":
        value = read(path)
        if not isinstance(value, (dict, list)) or not value:
            raise ValueError("JSON artifact is empty or has no structured content")
        if fields and (not isinstance(value, dict) or not set(fields).issubset(value)):
            raise ValueError("JSON artifact is missing required fields")
        details = {"container": type(value).__name__}
    else:
        raise ValueError(f"Unsupported evidence artifact type: {suffix}")
    digest = sha(path)
    if identity(path) != before:
        raise ValueError("Artifact changed while it was being validated")
    if expected_sha is not None and digest != expected_sha:
        raise ValueError("Artifact differs from its producer's declared hash")
    return {"path": str(path), "sha256": digest, "identity": before, "kind": suffix[1:], "validation": details}


def manifest_path(root, domain, task, policy):
    return Path(root) / "figure_artifacts" / domain / task / canonical(policy) / "manifest.json"


def verify_manifest(path, *, full_hash=False):
    value = read(path)
    if value.get("complete") is not True or value.get("passed") is not True or not value.get("artifacts"):
        raise ValueError("Artifact manifest is incomplete")
    if value.get("validator_source", {}).get("sha256") != sha(__file__):
        raise ValueError("Artifact manifest used a different validator source")
    for row in value["artifacts"]:
        if identity(row["path"]) != row["identity"] or full_hash and sha(row["path"]) != row["sha256"]:
            raise ValueError(f"Bound analysis artifact changed: {row['path']}")
    return value


def cell_ready(root, domain, task, policy):
    path = manifest_path(root, domain, task, policy)
    if not path.exists():
        return False
    try:
        verify_manifest(path)
    except (OSError, ValueError, KeyError):
        return False
    return True


def cells():
    for task in TASKS:
        for model in MODELS:
            yield "probes", task, model
            yield "probes_counterfactual", task, model
        for policy in POLICIES:
            yield "causal_analysis", task, policy
        yield "supplemental_analysis", task, "all_policies"


def requirements(root, domain, task, policy):
    root = Path(root).resolve()
    folder = root / domain / task / policy
    if domain == "supplemental_analysis":
        folder = root / domain / task
    if domain == "causal_analysis" and policy.startswith("rldt_"):
        folder = root / domain / task / f"rldt_{int(policy.removeprefix('rldt_')):06d}"
    summary_path = folder / ("metrics.json" if domain.startswith("probes") else "summary.json")
    if not summary_path.exists():
        return None
    summary = read(summary_path)
    if summary.get("complete") is not True:
        return None
    if summary.get("task") != task:
        raise ValueError("Producer task identity differs")
    required = [(summary_path, (), None)]

    def add(name, fields=()):
        required.append((folder / name, fields, None))

    def figures(names):
        for name in names:
            for extension in ("png", "pdf"):
                add(f"{name}.{extension}")

    if domain == "probes":
        if summary.get("model") != policy or summary["specification"].get("fit_split") != "train" or summary["specification"].get("fit_states") != 100:
            raise ValueError("Probe identity or train-only specification differs")
        figures(("confusion",))
        add("specification.json", ("fit_split", "fit_states"))
        add("fixed_projection.npy")
        for kind in ("image_features", "prefix_features"):
            add(f"{kind}_probe.npz", ("center", "scale", "coefficients", "classes"))
            for split in ("validation", "test", "source"):
                if summary["measurements"][kind][split].get("status") != "complete":
                    raise ValueError("Probe producer omits a required split")
                add(f"{kind}_{split}_predictions.npz", ("scores", "labels", "state_ids", "cameras"))
                add(f"{kind}_{split}_localization.csv", ("state_id", "camera", "region", "iou"))
    elif domain == "probes_counterfactual":
        if summary.get("model") != policy or summary.get("states") != 50 or summary.get("refit") is not False:
            raise ValueError("Counterfactual probe scope/refit status differs")
        figures(("confusion", "roc", "prediction_maps"))
        add("input_manifest.json")
        for kind in ("image_features", "prefix_features"):
            if summary["measurements"][kind].get("original_probe_replay_bit_exact") is not True:
                raise ValueError("Counterfactual probe original-score replay is not exact")
            add(f"{kind}_prediction_maps.npz", ("original", "counterfactual", "labels", "valid", "state_ids", "cameras"))
            add(f"{kind}_localization.csv", ("state_id", "camera"))
            add(f"{kind}_paired_prompt_changes.csv", ("state_id", "camera"))
            add(f"{kind}_roc.json")
    elif domain == "causal_analysis":
        if (canonical(summary.get("policy", "")) != policy or summary.get("closed_loop_conditions") != 9
                or summary.get("paired_episodes_per_condition") != 10 or summary.get("open_loop_conditions") != 15
                or summary.get("open_loop_states") != 50):
            raise ValueError("Causal analysis does not cover the fixed cases/conditions")
        figures(("open_loop_sensitivity",))
        add("closed_loop_pairs.csv", ("condition", "episodes", "difference"))
        add("open_loop_actions.csv", ("state_id", "condition", "normalized_l2", "decoded_controller_l2"))
        add("grasp_choice_proxies.csv", ("state_id", "condition", "native_success"))
        add("paired_target_grasp_proxies.csv", ("condition", "episodes"))
        add("causal_sensitivity_ratios.csv", ("state_id", "replacement"))
    elif domain == "supplemental_analysis":
        if summary.get("states") != 50 or not summary.get("coverage"):
            raise ValueError("Supplemental analysis must declare full50-state scope and coverage")
        groups = summary.get("required_artifact_groups", {})
        if set(groups) != set(GROUPS):
            raise ValueError("Required supplemental figure/control/vector groups missing")
        declared = {}
        for row in summary.get("artifacts", []):
            path = Path(row["path"])
            path = path.resolve() if path.is_absolute() else (folder / path).resolve()
            if not path.is_relative_to(folder.resolve()) or path in declared:
                raise ValueError("Supplemental artifact path escapes its producer or duplicates another")
            declared[path] = row
        for name, paths in groups.items():
            group_paths = [Path(path).resolve() if Path(path).is_absolute() else (folder / path).resolve() for path in paths]
            if not group_paths or any(path not in declared for path in group_paths):
                raise ValueError(f"Supplemental group has undeclared artifacts: {name}")
            suffixes = {path.suffix.lower() for path in group_paths}
            if not {".png", ".pdf"}.issubset(suffixes) or not suffixes.intersection({".json", ".csv", ".npz"}):
                raise ValueError(f"Supplemental group lacks paired figures and raw data: {name}")
        for path, row in declared.items():
            required.append((path, (), row["sha256"]))
        amendment = summary["amendment"]
        amendment_path = Path(amendment["path"]).resolve()
        if not amendment_path.is_relative_to(root):
            raise ValueError("Supplemental amendment escaped run root")
        required.append((amendment_path, (), amendment["sha256"]))
    else:
        raise ValueError(domain)
    return required


def validate_cell(root, domain, task, policy):
    path = manifest_path(root, domain, task, policy)
    if path.exists():
        verify_manifest(path)
        return {"domain": domain, "task": task, "policy": policy, "status": "complete",
                "manifest": {"path": str(path.resolve()), "sha256": sha(path)}}
    try:
        expected = requirements(root, domain, task, policy)
        if expected is None:
            return {"domain": domain, "task": task, "policy": policy, "status": "pending", "reason": "Producer summary pending"}
        records = [artifact(file, fields=fields, expected_sha=digest) for file, fields, digest in expected]
    except (OSError, ValueError, KeyError, TypeError, EOFError, zipfile.BadZipFile) as error:
        # A producer can be between metrics publication and rendering. Missing
        # or partially written outputs must never produce premature completion.
        return {"domain": domain, "task": task, "policy": policy, "status": "pending",
                "reason": f"Required artifacts not yet validated: {type(error).__name__}: {error}"}
    manifest = {"schema": "analysis-figure-artifact-manifest-v1", "complete": True, "passed": True,
                "domain": domain, "task": task, "policy": policy, "artifacts": records,
                "validator_source": {"path": str(Path(__file__).resolve()), "sha256": sha(__file__)},
                "scope": "Actual required figures and compact source data decode successfully and their stable bytes are bound. No policy execution, refitting, new statistical settings or research score computation."}
    write(path, manifest)
    return {"domain": domain, "task": task, "policy": policy, "status": "complete",
            "manifest": {"path": str(path.resolve()), "sha256": sha(path)}}


def refresh(root):
    root = Path(root).resolve()
    results = []
    for domain, task, policy in cells():
        try:
            results.append(validate_cell(root, domain, task, policy))
        except (OSError, ValueError, KeyError, TypeError) as error:
            results.append({"domain": domain, "task": task, "policy": policy, "status": "failed", "reason": str(error)})
    failed = [row for row in results if row["status"] == "failed"]
    pending = [row for row in results if row["status"] == "pending"]
    complete = not failed and not pending
    summary = {"schema": "analysis-figure-artifact-coverage-v1", "complete": complete,
               "passed": False if failed else (True if complete else None),
               "expected_cells": 27, "completed_cells": sum(row["status"] == "complete" for row in results),
               "cells": results, "pending_cells": len(pending), "failed_cells": len(failed),
               "required_domains": {"probes": 6, "probes_counterfactual": 6, "causal_analysis": 12, "supplemental_analysis": 3},
               "validator_source": {"path": str(Path(__file__).resolve()), "sha256": sha(__file__)},
               "scope": "Required rendering/table/map coverage, independent of early producer metric completion markers; final scientific review remains separate."}
    if complete:
        for row in results:
            verify_manifest(row["manifest"]["path"], full_hash=True)
    write(root / "figure_artifacts/summary.json", summary)
    return summary


def verify_coverage(root, *, full_hash=False):
    """Consumer/final-gate check; a missing cell or stale binding fails closed."""
    root = Path(root)
    summary = read(root / "figure_artifacts/summary.json")
    expected = set(cells())
    actual = {(row["domain"], row["task"], row["policy"]) for row in summary.get("cells", [])}
    if (summary.get("complete") is not True or summary.get("passed") is not True
            or len(summary.get("cells", [])) != len(expected) or actual != expected
            or summary.get("validator_source", {}).get("sha256") != sha(__file__)):
        raise ValueError("Required figure/control/vector coverage is incomplete")
    for row in summary["cells"]:
        path = manifest_path(root, row["domain"], row["task"], row["policy"]).resolve()
        if row.get("status") != "complete" or Path(row["manifest"]["path"]).resolve() != path or sha(path) != row["manifest"]["sha256"]:
            raise ValueError("Coverage summary does not bind the expected complete manifest")
        verify_manifest(path, full_hash=full_hash)
    return summary
