"""CPU-only construction checks for both official PPO policy variants."""

from __future__ import annotations

import argparse
import gc
import json
from collections import defaultdict
from pathlib import Path

from model_configs import METHODS, model_config
from model_contract import enforce_trainable_scope


def main():
    import torch
    from omegaconf import OmegaConf
    from rlinf.models.embodiment.openpi import get_model

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if torch.cuda.is_available():
        raise RuntimeError("This CPU construction check requires CUDA_VISIBLE_DEVICES=''")
    torch.set_num_threads(1)
    result = {}
    for method in METHODS:
        print(f"Constructing {method}", flush=True)
        model = get_model(OmegaConf.create(model_config(args.checkpoint, method)))
        counts = enforce_trainable_scope(model)
        dtypes = defaultdict(int)
        for parameter in model.parameters():
            if parameter.device.type != "cpu":
                raise RuntimeError("Unexpected non-CPU parameter")
            group = "trainable" if parameter.requires_grad else "frozen"
            dtypes[f"{group}/{parameter.dtype}"] += parameter.numel()
        result[method] = {"parameter_counts": counts, "dtypes": dict(dtypes)}
        print(json.dumps({method: result[method]}, indent=2), flush=True)
        del model
        gc.collect()
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
