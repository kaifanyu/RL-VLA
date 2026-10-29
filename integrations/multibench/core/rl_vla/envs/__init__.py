"""Environment factories. Real tasks must expose the documented observation/action contract."""

import importlib

import gymnasium as gym

from rl_vla.envs.toy import ToyReachEnv


def make_env(config: dict) -> gym.Env:
    """Construct one environment; the collector owns all resets.

    Example external config: ``{"kind": "external", "factory":
    "my_task.adapter:make_env", "kwargs": {"render_mode": None}}``.
    Only use trusted Python modules in the factory setting.
    """
    kind = config.get("kind", "toy")
    kwargs = dict(config.get("kwargs", {}))
    if kind == "toy":
        return ToyReachEnv(**kwargs)
    if kind != "external":
        raise ValueError(f"Unknown environment kind: {kind!r}")
    factory_path = config.get("factory", "")
    if factory_path.count(":") != 1:
        raise ValueError("An external factory must have the form 'module:function'")
    module_name, function_name = factory_path.split(":")
    factory = getattr(importlib.import_module(module_name), function_name)
    env = factory(**kwargs)
    if not isinstance(env, gym.Env):
        raise TypeError("The environment factory must return a Gymnasium Env")
    return env
