"""Render readable, versioned causal figure derivatives after source validation."""
import argparse
import time
from pathlib import Path

from rl_vla.diagnostics import causal_figures as render
from rl_vla.diagnostics.figure_artifacts import sha, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--task", choices=render.TASKS)
    parser.add_argument("--policy", choices=render.POLICIES)
    parser.add_argument("--wait", action="store_true")
    args = parser.parse_args()
    sources = {str(Path(__file__).resolve()): sha(__file__), str(Path(render.__file__).resolve()): sha(render.__file__)}
    root = args.run.resolve()
    pending = [(task, policy) for task in render.TASKS for policy in render.POLICIES
               if (args.task is None or task == args.task) and (args.policy is None or policy == args.policy)]
    while pending:
        if any(sha(path) != expected for path, expected in sources.items()):
            raise ValueError("Causal presentation renderer changed after launch")
        for task, policy in list(pending):
            if render.render_cell(root, task, policy) is not None:
                pending.remove((task, policy))
                print(f"Validated causal derivative: {task}/{policy}", flush=True)
        coverage = render.verify_coverage(root)
        write(root / "figures/causal/summary.json", coverage)
        if coverage["failed_cells"]:
            raise ValueError("Causal derivative coverage includes an invalid completed artifact")
        if not args.wait or not pending:
            break
        print(f"Waiting for validated original causal figures: {len(pending)} cells", flush=True)
        time.sleep(30)


if __name__ == "__main__":
    main()
