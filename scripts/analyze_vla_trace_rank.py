"""Rank and paired checkpoint drift of native pi0.5 action-to-prefix attention."""

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from rl_vla.diagnostics.attention_spectrum import compare_attention, validate_pair


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def run(traces, output, *, policies, steps, rtol=1e-6):
    traces, output = Path(traces).resolve(), Path(output).resolve()
    baseline = traces / "pi05_libero"
    states = sorted(p.name for p in baseline.glob("state_*"))
    if not states or not policies or not steps:
        raise ValueError("Require baseline states, candidate policies and flow steps")
    sources = {}
    for policy in policies:
        if sorted(p.name for p in (traces / policy).glob("state_*")) != states:
            raise ValueError(f"State coverage differs for {policy}; refusing to drop pairs")
        for state in states:
            for name in ("pi05_libero", policy):
                for filename in ["metadata.json", *[f"expert_step{s:02d}_none.npz" for s in steps]]:
                    path = traces / name / state / filename
                    if not path.is_file():
                        raise FileNotFoundError(path)
    output.mkdir(parents=True, exist_ok=False)
    grouped = defaultdict(list)
    records = 0
    with (output / "heads.jsonl").open("w") as stream:
        for policy in policies:
            for state in states:
                left_dir, right_dir = baseline / state, traces / policy / state
                lm, rm = read(left_dir / "metadata.json"), read(right_dir / "metadata.json")
                for folder in (left_dir, right_dir):
                    path = folder / "metadata.json"
                    sources.setdefault(str(path), sha(path))
                for step in steps:
                    lp = left_dir / f"expert_step{step:02d}_none.npz"
                    rp = right_dir / lp.name
                    for path in (lp, rp):
                        if str(path) not in sources:
                            sources[str(path)] = sha(path)
                    with (
                        np.load(lp, allow_pickle=False) as left,
                        np.load(rp, allow_pickle=False) as right,
                    ):
                        validate_pair(left, right, lm, rm)
                        by_group = defaultdict(list)
                        for row in compare_attention(
                            left["attention"], right["attention"], lm, rtol=rtol
                        ):
                            row.update(policy=policy, state=state, step=step)
                            stream.write(json.dumps(row, allow_nan=False) + "\n")
                            by_group[row["key_group"]].append(row)
                            records += 1
                        # Observations are the independent resampling unit, not heads.
                        for group, rows in by_group.items():
                            fields = {
                                key: [r[key] for r in rows]
                                for key in (
                                    "effective_rank_delta",
                                    "relative_frobenius_delta",
                                    "mass_delta_all_keys",
                                    "conditional_js_nats",
                                )
                            }
                            fields.update(
                                {
                                    f"{side}_effective_rank": [
                                        r[f"{side}_spectrum"]["effective_rank"] for r in rows
                                    ]
                                    for side in ("reference", "candidate")
                                }
                            )
                            for metric, values in fields.items():
                                valid = [v for v in values if v is not None]
                                grouped[policy, step, group, metric].append(
                                    {
                                        "state": state,
                                        "value": float(np.mean(valid)) if valid else None,
                                        "defined_heads": len(valid),
                                        "total_heads": len(values),
                                    }
                                )
                print(f"{traces.name}/{policy}/{state}: rank and drift complete", flush=True)
    summaries = []
    for (policy, step, group, metric), rows in sorted(grouped.items()):
        values = np.asarray([r["value"] for r in rows if r["value"] is not None])
        interval = None
        if len(values) > 1:
            boot = np.random.default_rng(20261103).choice(values, (2000, len(values))).mean(-1)
            interval = np.quantile(boot, [0.025, 0.975]).tolist()
        summaries.append(
            {
                "policy": policy,
                "step": step,
                "key_group": group,
                "metric": metric,
                "defined_states": len(values),
                "total_states": len(rows),
                "mean": float(values.mean()) if len(values) else None,
                "state_bootstrap95": interval,
                "by_state": rows,
            }
        )
    result = {
        "complete": True,
        "task": traces.name,
        "states": len(states),
        "records": records,
        "policies": policies,
        "steps": steps,
        "numerical_rank_rtol": rtol,
        "sources_sha256": sources,
        "summaries": summaries,
        "interpretation": [
            "Exploratory reanalysis of fixed archived observations; no checkpoint selection.",
            "Cross attention is the action-query/prefix-key block of joint expert attention.",
            "Rank ceiling is min(queries, valid keys); low rank alone is not failure.",
            "Native all-key denominator retained; JS conditions on nonzero block mass.",
            "Raw rows retain each layer/head; intervals resample whole states, not heads.",
            "Compiler differences can cause drift; compare existing compiler_controls before attributing small changes to weights.",
            "Attention rank/drift does not establish semantic reasoning or causal policy use.",
        ],
    }
    (output / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--traces", required=True, type=Path, help="One task's existing traces directory"
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--policies", default="rldt_000025,rldt_000050,rldt_000100")
    parser.add_argument("--steps", default="0,5,9")
    parser.add_argument("--rtol", type=float, default=1e-6)
    args = parser.parse_args()
    policies = args.policies.split(",")
    steps = [int(s) for s in args.steps.split(",")]
    if len(set(steps)) != len(steps) or any(s not in range(10) for s in steps):
        parser.error("steps must be unique integers in 0..9")
    if len(set(policies)) != len(policies) or any(
        p not in ("rldt_000025", "rldt_000050", "rldt_000100") for p in policies
    ):
        parser.error("policies must be unique RLDT checkpoints 25/50/100")
    run(args.traces, args.output, policies=policies, steps=steps, rtol=args.rtol)
