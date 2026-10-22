"""Freeze the complete plan after installing and verifying benchmark assets."""
import argparse
import json
import sys
from pathlib import Path

import yaml

from rl_vla.robustness import ASSETS, ROOT, SUITES, atomic_json, build_plan, task_map


def main(output):
    if (output / "queue_contract.json").exists():
        raise RuntimeError("Cannot replace a plan after the queue has started")
    pro = ROOT / "third_party/libero_pro"
    sys.path.insert(0, str(pro))
    from perturbation import BDDLParser, EnvironmentReplacePerturbator

    for suite in SUITES:
        for name in task_map(pro)[suite + "_env"]:
            original = pro / "libero/libero/bddl_files" / suite / (name + ".bddl")
            generator = EnvironmentReplacePerturbator(BDDLParser(original.read_text()),
                str(pro / "libero_ood/ood_environment.yaml"))
            target = ASSETS / "libero_pro_generated/bddl_files" / (suite + "_env") / (name + ".bddl")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(generator.perturb(suite, name, seed=20260929))
    for benchmark in ("plus", "pro"):
        root = ROOT / "third_party" / ("libero_" + benchmark) / "libero/libero"
        assets = ASSETS / "libero_plus/assets" if benchmark == "plus" else root / "assets"
        if not assets.is_dir():
            raise FileNotFoundError(f"Extract benchmark assets first: {assets}")
        # Some upstream arena helpers resolve relative to the Python source,
        # while others consult config.yaml. Both must reference the same assets.
        if benchmark == "plus":
            link = root / "assets"
            if not link.exists():
                link.symlink_to(assets, target_is_directory=True)
            if link.resolve() != assets.resolve():
                raise ValueError("Plus source assets and configured assets differ")
        config = ROOT.parent / ".cache" / ("rl-vla-libero-" + benchmark)
        config.mkdir(parents=True, exist_ok=True)
        (config / "config.yaml").write_text(yaml.safe_dump({
            "benchmark_root": str(root), "bddl_files": str(root / "bddl_files"),
            "init_states": str(root / "init_files"), "datasets": str(ROOT.parent / "datasets"),
            "assets": str(assets)}))
    plan = build_plan()
    plan["asset_verification"] = {name: json.loads((ASSETS / ("libero_" + name) / "verification.json").read_text())
                                   for name in ("plus", "pro")}
    atomic_json(output / "protocol.json", plan)
    for benchmark in ("pro", "plus"):
        categories = sorted({c["category"] for c in plan["cases"] if c["benchmark"] == benchmark})
        for category in categories:
            case = next(c for c in plan["cases"] if c["benchmark"] == benchmark and c["category"] == category)
            atomic_json(output / "preflight_cases" / f"{benchmark}_{category}.json", case)
    print(f"Validated {len(plan['cases'])} cases, {sum(c['episodes'] for c in plan['cases'])} episodes")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    main(parser.parse_args().output.resolve())
