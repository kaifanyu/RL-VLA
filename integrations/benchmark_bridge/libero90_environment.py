"""Original LIBERO-90 interface, separate from the frozen PRO/Plus screen."""

from benchmark_bridge.environment import SequentialCaseEnv, checked_case
from rl_vla import robustness


def make_env(*, ready_file, output):
    case = checked_case(ready_file)
    if case["benchmark"] != "libero90" or case["suite"] != "libero_90":
        raise ValueError("The LIBERO-90 wrapper only accepts original LIBERO-90 cells")
    if case.get("max_steps") != 400:
        raise ValueError("Use the pinned OpenPI LIBERO-90 horizon of400 controller steps")
    robustness.HORIZONS["libero_90"] = 400
    return SequentialCaseEnv(ready_file=ready_file, output=output)
