"""FlashSAC residual RL around a frozen OpenPI pi05_libero policy on LIBERO-PRO.

Entry point: ``python -m rl_vla.pro_residual`` (use ``scripts/run_flashsac_pro.sh``).
Pure modules (``spec``, ``residual``, ``rollout``, ``video``) import neither JAX
nor a simulator, so they are unit-testable in the root ``.venv``. ``base``
loads OpenPI in-process and must run in ``third_party/openpi/.venv``; ``sim``
runs the simulator in a separate child interpreter.
"""
