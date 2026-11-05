"""Measure the compiler-only attention drift floor on existing first-five controls."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from analyze_vla_trace_rank import read, sha

from rl_vla.diagnostics.attention_spectrum import attention_drift, key_groups, spectrum


def run(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    rows, sources = [], {}
    for task in ("object_0", "object_1", "object_2"):
        traces = root / "traces" / task
        controls = sorted((traces / "compiler_controls").glob("state_*/step*.json"))
        if len(controls) != 15:
            raise ValueError("Expected five states by three flow indices for compiler control")
        for path in controls:
            record = read(path)
            if (
                not record["complete"]
                or record["weights"] != "unchanged pi05_libero baseline"
                or record["same_profile_native_velocity_max_abs_difference"] != 0
            ):
                raise ValueError("Compiler control is not a verified unchanged baseline")
            arrays = Path(record["arrays"]["path"])
            if sha(arrays) != record["arrays"]["sha256"]:
                raise ValueError("Compiler control array hash mismatch")
            state, step = path.parent.name, record["step"]
            reference = traces / "pi05_libero" / state / f"expert_step{step:02d}_none.npz"
            metadata_path = reference.parent / "metadata.json"
            metadata = read(metadata_path)
            if record["observation_sha256"] != metadata["bank_record"]["observation_sha256"]:
                raise ValueError("Compiler control observation mismatch")
            for p in (path, arrays, reference, metadata_path):
                sources[str(p)] = sha(p)
            with (
                np.load(reference, allow_pickle=False) as a,
                np.load(arrays, allow_pickle=False) as b,
            ):
                if any(not np.array_equal(a[key], b[key]) for key in ("x_t", "t")):
                    raise ValueError("Compiler control latent/time mismatch")
                left, right = a["attention"], b["attention"]
            if left.shape != right.shape:
                raise ValueError("Compiler control attention axes differ")
            indices = key_groups(metadata, left.shape[-1])["visual"]
            for layer, batch, head in np.ndindex(left.shape[:3]):
                a = left[layer, batch, head][:, indices]
                b = right[layer, batch, head][:, indices]
                rows.append(
                    {
                        "task": task,
                        "state": state,
                        "step": step,
                        "layer": layer,
                        "head": head,
                        "effective_rank_delta": spectrum(b)["effective_rank"]
                        - spectrum(a)["effective_rank"],
                        **attention_drift(a, b),
                    }
                )
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["task"], row["step"]].append(row)
    summaries = []
    for (task, step), members in sorted(grouped.items()):
        summary = {"task": task, "step": step, "states": len({r["state"] for r in members})}
        for metric in (
            "effective_rank_delta",
            "relative_frobenius_delta",
            "conditional_js_nats",
            "mass_delta_all_keys",
        ):
            summary[metric] = float(np.mean([r[metric] for r in members]))
        summaries.append(summary)
    output.mkdir(parents=True, exist_ok=False)
    (output / "summary.json").write_text(
        json.dumps(
            {
                "complete": True,
                "summaries": summaries,
                "records": len(rows),
                "sources_sha256": sources,
                "scope": "Same baseline weights and saved context/latent, different original compiler profiles. Descriptive first-five-state floor; do not subtract these nonlinear distances from learned-checkpoint distances.",
            },
            indent=2,
        )
        + "\n"
    )
    with (output / "heads.jsonl").open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, allow_nan=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    run(args.root, args.output)
