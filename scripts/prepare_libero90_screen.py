"""Four original LIBERO-90 development cells, separate from the PRO/Plus screen."""

import argparse
import json
import multiprocessing
import os
import subprocess
import urllib.request
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import prepare_benchmark_screen as shared
from prepare_pro_spatial_grasp import ROOT, record
from prepare_pro_suite_grasp import write_json

SOURCE_COMMIT = "f78abd68ee283de9f9be3c8f7e2a9ad60246e95c"
SFT_REVISION = "a4336d589d589045d1c56423ffdf3b88a0e19b1f"
TASK_IDS = (12, 13, 46, 47)
HORIZON = 400  # Pinned OpenPI examples/libero/main.py's LIBERO-90 horizon.


def prepare(output):
    source = ROOT / "third_party/openpi/third_party/libero"
    commit = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"],
                                     text=True).strip()
    if commit != SOURCE_COMMIT:
        raise ValueError("Original LIBERO source revision changed")
    tasks = shared.mapping(source)
    known_names = {name for suite in ("libero_spatial", "libero_object", "libero_goal", "libero_10")
                   for name in tasks[suite]}
    metadata = output / "training_data_metadata"
    metadata.mkdir(exist_ok=True)
    for filename in ("info.json", "tasks.jsonl"):
        destination = metadata / filename
        if not destination.exists():
            url = f"https://huggingface.co/datasets/physical-intelligence/libero/resolve/{SFT_REVISION}/meta/{filename}"
            with urllib.request.urlopen(url, timeout=60) as response:
                destination.write_bytes(response.read())
    info = json.loads((metadata / "info.json").read_text())
    task_rows = [json.loads(line) for line in (metadata / "tasks.jsonl").read_text().splitlines()]
    if info["total_tasks"] != 40 or len(task_rows) != 40:
        raise ValueError("Expected the documented 40-task LIBERO SFT dataset")
    cells = []
    for index, task_id in enumerate(TASK_IDS):
        name = tasks["libero_90"][task_id]
        if name in known_names:
            raise ValueError("Selected task definition overlaps the four documented SFT suites")
        directory = source / "libero/libero"
        bddl = directory / f"bddl_files/libero_90/{name}.bddl"
        states = directory / f"init_files/libero_90/{name}.pruned_init"
        key = f"libero90_{task_id:02d}"
        cells.append({
            "case_id": key, "benchmark": "libero90", "suite": "libero_90",
            "task_id": task_id, "name": name,
            "category": "Spatial relation" if task_id in (12, 13) else "Object choice",
            "category_note": "Research sample labels, not separate official LIBERO-90 subsuites",
            "difficulty": None, "max_steps": HORIZON, "bddl": str(bddl),
            "real_bddl": str(bddl), "bddl_record": record(bddl),
            "published_states": record(states), "source_repository": str(source),
            "source_commit": SOURCE_COMMIT, "dataset": "original LIBERO published initial states",
            "dataset_revision": SOURCE_COMMIT, "python": str(ROOT / ".venv/bin/python"),
            "config_path": str(ROOT / ".cache/libero"), "episodes": 5,
            "split": "development_only", "generation_seed": 2028000000 + index * 10000,
            "state_origin": "fresh_constructor_development_states_excluding_all_published_states",
            "states": str(output / "cases" / key / "development_states.npy"),
            "preparation_sources": [record(Path(__file__)), record(Path(shared.__file__))],
        })
    plan = {
        "version": 1, "purpose": "Separate original LIBERO-90 spatial/object manipulation pilot",
        "model_config": "pi05_libero", "checkpoint": str(ROOT / "checkpoints/openpi/pi05_libero"),
        "selection": "Fixed IDs12,13,46,47 chosen for spatial relation and object choice semantics; "
                     "no screening outcomes used in selection.",
        "interpretation": "Another original LIBERO task suite using the same Panda, not an "
                          "independent robot benchmark. These task definitions are outside the four "
                          "documented LIBERO SFT suites; pretraining novelty is not established.",
        "episodes_per_cell": 5, "horizons": {"libero_90": HORIZON},
        "camera_resolution": 256, "resize_size": 224, "settle_steps": 10,
        "action_horizon": 10, "execute_steps": 5, "cases": cells,
        "sft_data_evidence": {
            "dataset": "physical-intelligence/libero", "revision": SFT_REVISION,
            "total_tasks": 40, "metadata": [record(metadata / "info.json"),
                                             record(metadata / "tasks.jsonl")],
            "openpi_config": record(ROOT / "third_party/openpi/src/openpi/training/config.py"),
            "source_url": "https://huggingface.co/datasets/physical-intelligence/libero",
            "limitation": "Fine-tuning metadata does not enumerate all pi05_base pretraining data.",
        },
        "horizon_source": record(ROOT / "third_party/openpi/examples/libero/main.py"),
        "sources": ["https://github.com/Lifelong-Robot-Learning/LIBERO",
                    "https://github.com/Physical-Intelligence/openpi/tree/main/examples/libero"],
    }
    path = output / "screen_plan.json"
    if path.exists() and json.loads(path.read_text()) != plan:
        raise ValueError("Refusing to change the existing LIBERO-90 screen")
    write_json(path, plan)
    write_json(output / "future_dynamic_benchmark.json", {
        "benchmark": "LIBERO-MAX", "status": "official_code_available_not_launched",
        "code": "https://github.com/liberomax/LIBERO-MAX",
        "paper": "https://arxiv.org/abs/2609.36518", "website": "https://liberomax.github.io/",
        "full_pairs": 8000, "lite_pairs": 800, "event_types": 8,
        "protocol": "Matched Base/Dynamic rollouts with identical task, reset, policy seed and "
                    "executed prefix; one persistent change during execution.",
        "lineage": "5600 LIBERO-Plus and2400 LIBERO-PRO source cases; same LIBERO family.",
        "integration_needed": "Event timing, state/prefix replay, paired metrics and source assets; "
                              "existing static-reset screening does not implement this protocol.",
    })
    print(f"Frozen {len(cells)} original LIBERO-90 cells,20 fresh development episodes", flush=True)


def generate_case(case, directory):
    # Process-local extension; do not edit the shared/frozen 46-cell source files.
    shared.HORIZONS["libero_90"] = HORIZON
    return shared.generate_cell(case, directory)


def generate(output, workers):
    plan = json.loads((output / "screen_plan.json").read_text())
    os.environ["LIBERO_CONFIG_PATH"] = str(ROOT / ".cache/libero")
    rows = []
    with ProcessPoolExecutor(max_workers=workers,
                             mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = [pool.submit(generate_case, case, output / "cases" / case["case_id"])
                   for case in plan["cases"]]
        for future in as_completed(futures):
            rows.append(future.result())
            write_json(output / "preparation_progress.json", {"ready": len(rows), "total": 4})
            print(f"LIBERO-90: {len(rows)}/4 development cells ready", flush=True)
    write_json(output / "libero90_ready.json", plan | {"cases": sorted(rows, key=lambda x: x["case_id"])})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("plan", "generate"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.workers <= 4:
        parser.error("Use one to four workers for the four cells")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if args.phase == "plan":
        prepare(output)
    else:
        generate(output, args.workers)


if __name__ == "__main__":
    main()
