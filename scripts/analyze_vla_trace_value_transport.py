"""Compare A@V interface contributions on the predeclared first-five trace states."""

import argparse
import json
from pathlib import Path

import numpy as np
from analyze_vla_trace_rank import read, sha

from rl_vla.diagnostics.attention_spectrum import validate_pair
from rl_vla.diagnostics.value_transport import compare_value_transport, stored_cache_hash


def run(traces, output):
    traces, output = Path(traces).resolve(), Path(output).resolve()
    states = sorted(p.name for p in (traces / "pi05_libero").glob("state_*"))[:5]
    if len(states) != 5:
        raise ValueError("Require the predeclared first-five states")
    output.mkdir(parents=True, exist_ok=False)
    sources, records = {}, 0
    with (output / "heads.jsonl").open("w") as stream:
        for state in states:
            shared = traces / "shared" / state
            schema_path = shared / "context_schema.json"
            schema = read(schema_path)
            context_path = shared / "context.npz"
            if sha(context_path) != schema["npz_sha256"]:
                raise ValueError("Context artifact hash differs from schema")
            fields = dict(schema["tree"]["items"])
            value_key = fields["kv_cache"]["items"][1]["key"]
            with np.load(context_path, allow_pickle=False) as data:
                values = data[value_key]
                cache_hashes = [
                    stored_cache_hash(data[item["key"]], item["dtype"])
                    for item in fields["kv_cache"]["items"]
                ]
            for p in (context_path, schema_path):
                sources[str(p)] = sha(p)
            left_dir = traces / "pi05_libero" / state
            lm = read(left_dir / "metadata.json")
            if cache_hashes != lm.get("kv_cache_sha256"):
                raise ValueError("Context values do not match captured native cache hashes")
            for policy in ("rldt_000025", "rldt_000050", "rldt_000100"):
                right_dir = traces / policy / state
                rm = read(right_dir / "metadata.json")
                if not lm.get("kv_cache_sha256") or lm["kv_cache_sha256"] != rm.get(
                    "kv_cache_sha256"
                ):
                    raise ValueError("Value comparison requires an identical frozen KV cache")
                for p in (left_dir / "metadata.json", right_dir / "metadata.json"):
                    sources[str(p)] = sha(p)
                for step in (0, 5, 9):
                    lp, rp = [p / f"expert_step{step:02d}_none.npz" for p in (left_dir, right_dir)]
                    for p in (lp, rp):
                        sources[str(p)] = sha(p)
                    with np.load(lp, allow_pickle=False) as a, np.load(rp, allow_pickle=False) as b:
                        validate_pair(a, b, lm, rm)
                        for row in compare_value_transport(
                            a["attention"], b["attention"], values, lm
                        ):
                            row.update(state=state, policy=policy, step=step)
                            stream.write(json.dumps(row, allow_nan=False) + "\n")
                            records += 1
            print(state, "value transport complete", flush=True)
    summary = {
        "complete": True,
        "task": traces.name,
        "states": states,
        "records": records,
        "sources_sha256": sources,
        "interpretation": "First-five-state secondary diagnostic; float64 A_block@V_block before output projection. Not native BF16 replay, semantic sufficiency, or causal isolation. No output projection, residual, suffix-value or MLP effects included. CKA compares action-query geometry, not independent examples.",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    run(args.traces, args.output)
