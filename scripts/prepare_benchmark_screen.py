"""Freeze development-only PRO/Plus screening cells without consuming final tests."""

import argparse
import ast
import hashlib
import json
import multiprocessing
import os
import re
import shutil
import subprocess
import sys
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path, PurePosixPath

import numpy as np
import yaml
from prepare_pro_spatial_grasp import DATASET, PRO_COMMIT, REVISION, ROOT, download, record
from prepare_pro_suite_grasp import collect_attempts, generate_attempt, verify_record, write_json

from rl_vla.envs.libero import LiberoEnv
from rl_vla.robustness import load_states, plus_state_path

PLUS_COMMIT = "4976dc30028e805ff8094b55501d532c48fec182"
PLUS_DATASET = "Sylvest/LIBERO-plus"
PLUS_REVISION = "dd2bd61b7d9a6fef1abc52d606e983b41886a149"
PLUS_ARCHIVE_SHA256 = "96764a4bfbdaea98d4411598caeab235458318fe0f549611b93d1a323027b3cf"
PLUS_ARCHIVE_SIZE = 6395849578
DEVELOPMENT_COUNT = 5
HORIZONS = {"libero_spatial": 220, "libero_object": 280}
PLUS_CATEGORIES = ("Objects Layout", "Camera Viewpoints", "Robot Initial States")


def digest_file(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as file:
        while chunk := file.read(32 * 1024 * 1024):
            value.update(chunk)
    return value.hexdigest()


def source(benchmark):
    path = ROOT / "third_party" / f"libero_{benchmark}"
    expected = PRO_COMMIT if benchmark == "pro" else PLUS_COMMIT
    actual = subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"],
                                     text=True).strip()
    if actual != expected:
        raise ValueError(f"Unexpected {benchmark} source revision: {actual}")
    return path


def archive_relative_path(filename):
    """Discard the upstream build-machine prefix while preserving assets paths."""
    path = PurePosixPath(filename)
    if path.is_absolute() or ".." in path.parts or "assets" not in path.parts:
        raise ValueError(f"Unsafe or unexpected archive member: {filename}")
    return PurePosixPath(*path.parts[path.parts.index("assets"):])


def required_new_object_categories(repository):
    """Trace selected BDDL object types to their upstream XML directory categories."""
    wanted = set()
    for cell in plus_cells():
        text = Path(cell["real_bddl"]).read_text()
        objects = re.search(r"\(:objects\s+(.*?)\)\s*\(:", text, flags=re.DOTALL)
        if not objects:
            raise ValueError(f"Cannot read BDDL object definitions: {cell['real_bddl']}")
        wanted.update(re.findall(r"-\s+(\w+)", objects.group(1)))
    classes = ast.parse((repository / "libero/libero/envs/objects/custom_objects.py").read_text())
    categories = set()
    for cls in classes.body:
        if not isinstance(cls, ast.ClassDef):
            continue
        initializer = next((node for node in cls.body
                            if isinstance(node, ast.FunctionDef) and node.name == "__init__"), None)
        if initializer is None:
            continue
        args = initializer.args
        defaults = dict(zip([arg.arg for arg in args.args][-len(args.defaults):], args.defaults,
                            strict=True))
        obj = defaults.get("obj_name")
        if not isinstance(obj, ast.Constant) or obj.value not in wanted:
            continue
        for node in ast.walk(initializer):
            if (isinstance(node, ast.Constant) and isinstance(node.value, str)
                    and node.value.startswith("assets/new_objects/")):
                categories.add(node.value.split("/")[2])
    return sorted(categories)


def setup_plus(output):
    """Use a separate package overlay; never alter the working PRO environment."""
    repository = source("plus")
    assets = ROOT / "datasets/benchmark_assets/libero_plus"
    assets.mkdir(parents=True, exist_ok=True)
    archive, partial = assets / "assets.zip", assets / "assets.zip.part"
    if not archive.exists():
        if not partial.exists() or partial.stat().st_size != PLUS_ARCHIVE_SIZE:
            raise RuntimeError("Complete the pinned assets.zip download before setup-plus")
        if digest_file(partial) != PLUS_ARCHIVE_SHA256:
            raise ValueError("LIBERO-Plus asset archive SHA256 mismatch")
        partial.replace(archive)
    verification = assets / "verification.json"
    if not verification.exists():
        if archive.stat().st_size != PLUS_ARCHIVE_SIZE or digest_file(archive) != PLUS_ARCHIVE_SHA256:
            raise ValueError("LIBERO-Plus asset archive changed")
        with zipfile.ZipFile(archive) as package:
            categories = required_new_object_categories(repository)
            extracted = 0
            for member in package.infolist():
                relative = archive_relative_path(member.filename)
                parts = relative.parts
                # Other categories are optional until a screen cell uses them. Keep the
                # full verified archive so a later experiment can extend this subset.
                if len(parts) > 2 and parts[1] == "new_objects" and parts[2] not in categories:
                    continue
                target = assets.joinpath(*parts)
                if member.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    temporary = target.with_name(target.name + ".extracting")
                    # Reading the entire ZipExtFile checks the member CRC.
                    with package.open(member) as src, temporary.open("wb") as dst:
                        shutil.copyfileobj(src, dst, length=1024 * 1024)
                    temporary.replace(target)
                extracted += 1
                if extracted % 1000 == 0:
                    write_json(output / "plus_extraction_progress.json", {
                        "extracted_members": extracted, "new_object_categories": categories})
            members = len(package.infolist())
        write_json(verification, {"dataset": PLUS_DATASET, "revision": PLUS_REVISION,
                                 "archive_sha256": PLUS_ARCHIVE_SHA256,
                                 "archive_size": PLUS_ARCHIVE_SIZE, "total_archive_members": members,
                                 "crc_verified_extracted_members": extracted,
                                 "new_object_categories": categories,
                                 "scope": "All base assets plus traced new-object categories used by "
                                          "the six selected development cells; full archive retained."})
    unpacked = assets / "assets"
    if not unpacked.is_dir():
        raise FileNotFoundError("Unexpected Plus archive directory layout")
    link = repository / "libero/libero/assets"
    if not link.exists():
        link.symlink_to(unpacked, target_is_directory=True)
    if link.resolve() != unpacked.resolve():
        raise ValueError("Plus source asset path differs from verified assets")
    config = ROOT / ".cache/libero-plus-screen"
    config.mkdir(exist_ok=True)
    configuration = {"benchmark_root": str(repository / "libero/libero"),
                     "bddl_files": str(repository / "libero/libero/bddl_files"),
                     "init_states": str(repository / "libero/libero/init_files"),
                     "datasets": str(ROOT / "datasets"), "assets": str(unpacked)}
    path = config / "config.yaml"
    if path.exists() and yaml.safe_load(path.read_text()) != configuration:
        raise ValueError("Refusing to alter existing Plus configuration")
    path.write_text(yaml.safe_dump(configuration))
    runtime = ROOT / ".venv-plus"
    if not runtime.exists():
        subprocess.run([str(ROOT / ".venv-pro/bin/python"), "-m", "venv", "--without-pip",
                        str(runtime)], check=True)
    site = runtime / "lib/python3.11/site-packages"
    # The dependency overlay does not process PRO's editable-install .pth files.
    # Only Plus and this repo's src are added explicitly before those dependencies.
    overlay = "\n".join(map(str, [repository, ROOT / "src",
                                  ROOT / ".venv-pro/lib/python3.11/site-packages"])) + "\n"
    (site / "benchmark_dependencies.pth").write_text(overlay)
    uv = shutil.which("uv")
    if uv is None:
        raise FileNotFoundError("uv is not on PATH; source scripts/env.sh before setup-plus")
    subprocess.run([uv, "pip", "install", "--python",
                    str(runtime / "bin/python"), "--no-deps", "Wand==0.6.13",
                    "scikit-image==0.25.2", "tifffile==2025.6.11", "lazy-loader==0.4"], check=True)
    environment = os.environ | {"LIBERO_CONFIG_PATH": str(config)}
    smoke = subprocess.check_output([str(runtime / "bin/python"), "-c",
        "import libero.libero, wand.image, skimage; print(libero.libero.__file__)"],
        env=environment, text=True).strip()
    if not Path(smoke).resolve().is_relative_to(repository):
        raise ValueError("The Plus runtime imported a different LIBERO fork")
    write_json(output / "plus_runtime.json", {
        "python": str(runtime / "bin/python"), "config_path": str(config),
        "source": str(repository), "source_commit": PLUS_COMMIT,
        "dependency_overlay": str(ROOT / ".venv-pro/lib/python3.11/site-packages"),
        "asset_verification": record(verification), "imported_libero": smoke,
        "note": "Separate Plus imports and additional packages; shared read-only PRO dependencies."})


def mapping(repository):
    return ast.literal_eval(ast.parse(
        (repository / "libero/libero/benchmark/libero_suite_task_map.py").read_text()).body[0].value)


def pro_cells():
    repository = source("pro")
    tasks = mapping(repository)
    cells = []
    for suite in HORIZONS:
        for suffix, category in (("swap", "Position"), ("object", "Object")):
            condition = f"{suite}_{suffix}"
            for index, name in enumerate(tasks[condition]):
                bddl, bddl_record = download(f"bddl_files/{condition}/{name}.bddl")
                _, state_record = download(f"init_files/{condition}/{name}.pruned_init")
                key = f"pro_{suite}_{suffix}_{index:02d}"
                cells.append({"case_id": key, "benchmark": "pro", "suite": suite,
                              "task_id": index, "name": name, "category": category,
                              "condition_suite": condition, "difficulty": None,
                              "bddl": str(bddl), "real_bddl": str(bddl), "bddl_record": bddl_record,
                              "published_states": state_record, "source_commit": PRO_COMMIT,
                              "dataset": DATASET, "dataset_revision": REVISION,
                              "python": str(ROOT / ".venv-pro/bin/python"),
                              "config_path": str(ROOT / ".cache/libero-pro")})
    return cells


def plus_cells():
    repository = source("plus")
    tasks = mapping(repository)
    labels = json.loads((repository / "libero/libero/benchmark/task_classification.json").read_text())
    original = mapping(source("pro"))
    cells = []
    for suite in HORIZONS:
        base = original[suite][0]
        by_name = {row["name"]: row for row in labels[suite]}
        for category in PLUS_CATEGORIES:
            candidates = [(index, name, by_name[name]) for index, name in enumerate(tasks[suite])
                          if name.startswith(base + "_") and by_name[name]["category"] == category]
            # A predetermined moderate/high difficulty variant, not policy-based selection.
            candidates.sort(key=lambda row: (abs(row[2]["difficulty_level"] - 3), row[0]))
            index, name, label = candidates[0]
            bddl = repository / f"libero/libero/bddl_files/{suite}/{name}.bddl"
            real_bddl = bddl.with_name(name.split("_view_")[0] + ".bddl")
            states = plus_state_path(repository, suite, name)
            key = f"plus_{suite}_{category.lower().replace(' ', '_')}_{index:04d}"
            cells.append({"case_id": key, "benchmark": "plus", "suite": suite,
                          "task_id": index, "base_task_id": 0, "name": name,
                          "category": category, "difficulty": label["difficulty_level"],
                          "bddl": str(bddl), "real_bddl": str(real_bddl),
                          "bddl_record": record(real_bddl), "published_states": record(states),
                          "source_commit": PLUS_COMMIT, "dataset": PLUS_DATASET,
                          "dataset_revision": PLUS_REVISION,
                          "python": str(ROOT / ".venv-plus/bin/python"),
                          "config_path": str(ROOT / ".cache/libero-plus-screen")})
    return cells


def prepare_plan(output):
    cells = pro_cells() + plus_cells()
    for index, cell in enumerate(cells):
        cell.update(episodes=DEVELOPMENT_COUNT, split="development_only",
                    generation_seed=2027000000 + index * 10000,
                    state_origin="fresh_constructor_development_states_excluding_published_states",
                    states=str(output / "cases" / cell["case_id"] / "development_states.npy"))
    manifest = {
        "version": 1, "purpose": "Select challenging conditions using fresh development states only",
        "model_config": "pi05_libero", "checkpoint": str(ROOT / "checkpoints/openpi/pi05_libero"),
        "selection": "PRO all10 Spatial/Object tasks × Position/Object perturbations; Plus base task0 "
                     "per suite × Layout/Camera/Robot categories, closest difficulty3 then earliest ID.",
        "interpretation": "This is a development screen, not the official benchmark score. "
                          "Final adaptation tests must use independently frozen states/conditions.",
        "episodes_per_cell": DEVELOPMENT_COUNT, "horizons": HORIZONS,
        "camera_resolution": 256, "resize_size": 224, "settle_steps": 10,
        "action_horizon": 10, "execute_steps": 5, "cases": cells,
        "sources": ["https://github.com/Zxy-MLlab/LIBERO-PRO",
                    "https://github.com/sylvestf/LIBERO-plus",
                    "https://huggingface.co/datasets/Sylvest/LIBERO-plus"],
    }
    path = output / "screen_plan.json"
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise ValueError("Refusing to change a frozen screening plan")
    write_json(path, manifest)
    compatibility = {
        "libero_plus": {"priority": 1, "status": "setup_in_progress", "same_checkpoint": True,
                        "robot": "Panda; optional published Panda initial-joint variants",
                        "state_dim": 8, "action_dim": 7, "control_hz": 20,
                        "source": "https://github.com/sylvestf/LIBERO-plus"},
        "robocasa": {"priority": 2, "status": "requires_separate_embodiment_adaptation",
                     "same_checkpoint_drop_in": False, "official_rlinf_robot": "PandaOmron",
                     "official_rlinf_state_dim": 25, "official_rlinf_action_dim": 12,
                     "official_rlinf_example": "CloseDrawer",
                     "required_checkpoint": "RLinf/RLinf-Pi0-RoboCasa or new matched SFT",
                     "reason": "Mobile-base controls, state semantics, images and normalization differ "
                               "from the LIBERO pi05_libero policy; a separate comparison is needed.",
                     "sources": ["https://robocasa.ai/",
                                 "https://github.com/RLinf/RLinf/blob/main/docs/source-en/rst_source/examples/embodied/robocasa.rst"]},
    }
    write_json(output / "benchmark_compatibility.json", compatibility)
    print(f"Prepared {len(cells)} cells, {len(cells) * DEVELOPMENT_COUNT} development episodes", flush=True)


def exclusion_states(cell):
    """Exclude published tests and known adaptation states with the same BDDL."""
    paths = [Path(cell["published_states"]["path"])]
    if cell["benchmark"] == "pro":
        protocols = [ROOT / "runs/pro_spatial0_rldt_20261026/protocol.json"]
        protocols.extend((ROOT / "runs/pro_multimethod_20261027/tasks").glob("*/protocol.json"))
        for protocol in protocols:
            if not protocol.exists():
                continue
            for task in json.loads(protocol.read_text())["tasks"].values():
                if task["pro"]["bddl"]["sha256"] == cell["bddl_record"]["sha256"]:
                    paths.append(Path(task["pro"]["states"]["path"]))
    values = np.concatenate([load_states(path) for path in paths])
    # Published test rows can be present in an adaptation combined-state file.
    return np.unique(values, axis=0), [record(path) for path in paths]


def generate_cell(cell, directory):
    """One worker owns one fork/cell and validates all its saved reset observations."""
    os.environ["LIBERO_CONFIG_PATH"] = cell["config_path"]
    from libero.libero.envs import OffScreenRenderEnv

    directory.mkdir(parents=True, exist_ok=True)
    if (directory / "ready.json").exists():
        saved = json.loads((directory / "ready.json").read_text())
        verify_record(saved["states_record"])
        return saved
    verify_record(cell["bddl_record"])
    excluded, excluded_records = exclusion_states(cell)
    attempts = directory / "attempts"
    attempts.mkdir(exist_ok=True)
    context = {"case": cell, "excluded": excluded_records, "generator": record(Path(__file__))}
    context_path = attempts / "context.json"
    if context_path.exists() and json.loads(context_path.read_text()) != context:
        raise ValueError("Development generation inputs changed; use a new screen directory")
    write_json(context_path, context)
    paths = []
    for index in range(100):
        path = attempts / f"{index:03d}.npz"
        if not path.exists():
            generate_attempt(cell["bddl"], cell["generation_seed"] + index, excluded.shape[1:], path)
        paths.append(path)
        states, accepted, rejected = collect_attempts(paths, excluded, count=DEVELOPMENT_COUNT)
        if len(states) == DEVELOPMENT_COUNT:
            break
    if len(states) != DEVELOPMENT_COUNT:
        raise RuntimeError(f"No valid development states for {cell['case_id']}: {rejected}")
    target = Path(cell["states"])
    with target.with_suffix(".tmp").open("wb") as file:
        np.save(file, states, allow_pickle=False)
    target.with_suffix(".tmp").replace(target)
    backend = OffScreenRenderEnv(bddl_file_name=cell["bddl"], camera_heights=256, camera_widths=256,
                                 horizon=max(1000, HORIZONS[cell["suite"]] + 11))
    env = LiberoEnv(task_suite=cell["suite"], task_id=cell["task_id"], max_steps=HORIZONS[cell["suite"]],
                    backend=backend, initial_states=np.asarray(states),
                    task_description=backend.language_instruction)
    checks = []
    try:
        for index in range(DEVELOPMENT_COUNT):
            obs, info = env.reset(seed=cell["generation_seed"] + 5000 + index,
                                  options={"initial_state_id": index})
            if not np.isfinite(obs["state"]).all():
                raise ValueError("Nonfinite development observation")
            for name in ("observation/image", "observation/wrist_image"):
                image = obs["openpi"][name]
                if image.shape != (224, 224, 3) or image.dtype != np.uint8 or image.std() < 1:
                    raise ValueError("Invalid development camera")
            checks.append(info)
        import imageio.v3 as iio

        iio.imwrite(directory / "initial_front.png", obs["openpi"]["observation/image"])
    finally:
        env.close()
    result = cell | {"states_record": record(target), "states_sha256": digest_file(target),
                     "bddl_sha256": cell["bddl_record"]["sha256"], "accepted": accepted,
                     "rejected": rejected, "excluded_states": excluded_records,
                     "reset_checks": checks, "ready": True}
    write_json(directory / "ready.json", result)
    return result


def generate_screen(output, benchmark, workers):
    plan = json.loads((output / "screen_plan.json").read_text())
    cases = [cell for cell in plan["cases"] if cell["benchmark"] == benchmark]
    expected_runtime = Path(cases[0]["python"]).parent.parent
    if Path(sys.prefix).resolve() != expected_runtime.resolve():
        raise ValueError(f"Use the isolated runtime: {cases[0]['python']}")
    os.environ["LIBERO_CONFIG_PATH"] = cases[0]["config_path"]
    rows = []
    with ProcessPoolExecutor(max_workers=workers,
                             mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = [pool.submit(generate_cell, cell, output / "cases" / cell["case_id"])
                   for cell in cases]
        for future in as_completed(futures):
            rows.append(future.result())
            write_json(output / f"{benchmark}_preparation_progress.json",
                       {"ready": len(rows), "total": len(cases)})
            print(f"{benchmark}: {len(rows)}/{len(cases)} development cells ready", flush=True)
    rows.sort(key=lambda row: row["case_id"])
    write_json(output / f"{benchmark}_ready.json", plan | {"cases": rows})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("plan", "setup-plus", "generate-pro", "generate-plus"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if not 1 <= args.workers <= 8:
        parser.error("Use between one and eight workers")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if args.phase == "plan":
        prepare_plan(output)
    elif args.phase == "setup-plus":
        setup_plus(output)
    else:
        generate_screen(output, args.phase.split("-", 1)[1], args.workers)


if __name__ == "__main__":
    main()
