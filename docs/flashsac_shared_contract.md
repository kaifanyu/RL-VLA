# FlashSAC under the shared Object 0 contract

Config: `configs/flashsac_object_0.toml`. The earlier
`configs/flashsac_object_00.toml` and `runs/flashsac_object_00` are unchanged.

## What the contract sets

| Item | Setting |
|---|---|
| States | Canonical Oct31 array, `9f4d57d3...`: train 0-99, validation 100-149, test 150-199 |
| Evaluation reset | Seed `100000 + i` for the i-th state of the ordered split |
| Retention reset | Seed `100000 + i` for retention state i, as RLDT's source split |
| Retention environment | PRO fork in `.venv-pro`, BDDL instruction as prompt, as RLDT's source split |
| Base inference | `Pi05Flow`, as RLDT and DICE-RL (float32 action expert), 10 ODE steps |
| Evaluation noise | `PRNGKey(100000 + i)`, split once per base query, as `rl_vla.rldt.evaluate` |
| Training reset | Seed `seed + episode`, then a uniform train state, as `rl_vla.envs.libero_pro` |
| Budget | Stops after exactly `total_env_steps` commands; a pending episode ends as `budget_stop` |
| Milestones | Validation at exact command counts, on a second simulator, after that command's update |
| Validation RNG | Torch RNG is forked, so validation does not advance training RNG |
| Base weights | Training refuses any tensor hash other than `a7b7519f...` |

A run also stores `config/source_snapshot.tar.gz`, `config/python_packages.json` and
`config/launch.json`. Resuming from a mid-episode checkpoint drops the pending episode; its
commands stay counted.

## Commands

```bash
# Rebuild the state set only if it is missing.
third_party/openpi/.venv/bin/python scripts/make_flashsac_canonical_state_set.py

# Train, then evaluate the validation-selected checkpoint on test.
CUDA_VISIBLE_DEVICES=<gpu> bash scripts/run_flashsac_pro.sh train \
  --config configs/flashsac_object_0.toml
CUDA_VISIBLE_DEVICES=<gpu> bash scripts/run_flashsac_pro.sh evaluate \
  --run runs/flashsac_object_0 --split test --checkpoint best --heldout

# DICE-RL at K=1; omit --checkpoint for the frozen prior.
CUDA_VISIBLE_DEVICES=<gpu> third_party/openpi/.venv/bin/python scripts/evaluate_dice_k1.py \
  --split validation --checkpoint <dice checkpoint> --output runs/shared_contract_eval/<name>
```

`LD_LIBRARY_PATH` must include `.cache/native/usr/lib/x86_64-linux-gnu` for the DICE command.

## Verified

- Backend reset seeds match RLDT on validation states 100-103 (FlashSAC) and 100-101 (DICE K=1).
- The first two training states match RLDT (85, 47).
- With `Pi05Flow`, base-only actions on validation state 103 equal RLDT's recorded actions in all
  30 queries, and states 100-103 match RLDT's outcomes and step counts.
- Compare the full 50-state baseline with RLDT's 19/50 at the start of the real run.

## Not covered

RLDT stops by decisions (15,923 commands) and DICE-RL saves at episode ends, so neither has
checkpoints at exact 4K/8K/12K. Using exact milestones for them requires retraining.
