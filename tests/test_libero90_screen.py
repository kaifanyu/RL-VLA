"""Keep the separate LIBERO-90 wrapper on its declared original-suite horizon."""

import importlib
from pathlib import Path

import pytest


@pytest.fixture
def wrapper(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "integrations"))
    module = importlib.import_module("benchmark_bridge.libero90_environment")
    monkeypatch.setattr(module.robustness, "HORIZONS", dict(module.robustness.HORIZONS))
    return module


def test_libero90_uses_official400step_cap(wrapper, monkeypatch):
    monkeypatch.setattr(wrapper, "checked_case", lambda _: {
        "benchmark": "libero90", "suite": "libero_90", "max_steps": 400})
    sentinel = object()
    monkeypatch.setattr(wrapper, "SequentialCaseEnv", lambda **_: sentinel)
    assert wrapper.make_env(ready_file="case", output="output") is sentinel
    assert wrapper.robustness.HORIZONS["libero_90"] == 400


@pytest.mark.parametrize("case", [
    {"benchmark": "pro", "suite": "libero_90", "max_steps": 400},
    {"benchmark": "libero90", "suite": "libero_spatial", "max_steps": 400},
    {"benchmark": "libero90", "suite": "libero_90", "max_steps": 520},
])
def test_other_benchmarks_or_horizons_rejected(wrapper, monkeypatch, case):
    monkeypatch.setattr(wrapper, "checked_case", lambda _: case)
    with pytest.raises(ValueError):
        wrapper.make_env(ready_file="case", output="output")
