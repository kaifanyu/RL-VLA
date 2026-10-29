"""Load FlashSAC's network layers from an unmodified upstream checkout.

Upstream: https://github.com/Holiday-Robot/FlashSAC (MIT license), cloned to
``third_party/FlashSAC`` by ``scripts/fetch_flashsac.sh``. Only the pure-PyTorch
modules are imported (layer.py, network.py, utils/distribution.py). Importing the
``flash_rl`` package normally would execute ``flash_rl/__init__.py``, which pulls
in gymnasium, omegaconf and JAX through ``flash_rl.types``; none are needed here.
To avoid that, empty placeholder packages are registered for the parent names
before the three files are loaded, so their own absolute imports resolve to the
real upstream files without running any package ``__init__``.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import types
from pathlib import Path

PINNED_COMMIT = "87edc9061150ae9e962dd84e6544e27a1554b3ab"
_PACKAGES = ("flash_rl", "flash_rl.agents", "flash_rl.agents.utils", "flash_rl.agents.flashSAC")
_MODULES = (
    ("flash_rl.agents.utils.distribution", "agents/utils/distribution.py"),
    ("flash_rl.agents.flashSAC.layer", "agents/flashSAC/layer.py"),
    ("flash_rl.agents.flashSAC.network", "agents/flashSAC/network.py"),
)


def flashsac_root() -> Path:
    """FLASHSAC_ROOT overrides the default third_party/FlashSAC location."""
    override = os.environ.get("FLASHSAC_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    return Path(__file__).resolve().parents[2] / "third_party" / "FlashSAC"


def load_flashsac() -> types.SimpleNamespace:
    """Return the upstream ``layer`` and ``network`` modules."""
    loaded = sys.modules.get("flash_rl.agents.flashSAC.network")
    if loaded is not None and getattr(loaded, "__rl_vla_vendored__", False):
        return types.SimpleNamespace(layer=sys.modules["flash_rl.agents.flashSAC.layer"],
                                     network=loaded)
    if "flash_rl" in sys.modules and not getattr(sys.modules["flash_rl"], "__rl_vla_stub__", False):
        raise RuntimeError("A full flash_rl package is already imported; cannot load the "
                           "PyTorch-only FlashSAC layers alongside it")
    package_root = flashsac_root() / "flash_rl"
    missing = [rel for _, rel in _MODULES if not (package_root / rel).is_file()]
    if missing:
        raise FileNotFoundError(
            f"FlashSAC sources not found under {package_root} (missing {missing}). "
            "Run scripts/fetch_flashsac.sh or set FLASHSAC_ROOT to a FlashSAC checkout.")
    for name in _PACKAGES:
        if name not in sys.modules:
            stub = types.ModuleType(name)
            stub.__path__ = []  # Mark as a package without executing upstream __init__.
            stub.__rl_vla_stub__ = True
            sys.modules[name] = stub
    for name, rel in _MODULES:
        spec = importlib.util.spec_from_file_location(name, package_root / rel)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        module.__rl_vla_vendored__ = True
    return types.SimpleNamespace(layer=sys.modules["flash_rl.agents.flashSAC.layer"],
                                 network=sys.modules["flash_rl.agents.flashSAC.network"])
