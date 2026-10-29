"""Build controlled single-GPU experiments from the pinned RLinf PPO recipe."""

import argparse
import copy
import json
from pathlib import Path

import yaml
from pi_rl.model_configs import algorithm_overrides, model_config

ROOT = Path(__file__).resolve().parents[2]
UPSTREAM = ROOT / "third_party/rlinf/examples/embodiment/config"


def configuration(task, output, method, iterations=100, decisions=32):
    protocol = json.loads((task / "protocol.json").read_text())
    key, definition = next(iter(protocol["tasks"].items()))
    cfg = yaml.safe_load((UPSTREAM / "libero_spatial_ppo_openpi_pi05.yaml").read_text())
    cfg.pop("defaults")
    cfg["hydra"].pop("searchpath", None)
    cfg["cluster"] = {"num_nodes": 1, "component_placement": {"actor,env,rollout": "all"}}
    cfg["runner"].update({"max_epochs": iterations, "max_steps": iterations,
                           "save_interval": min(25, iterations),
                           # No Ray dashboard: capture worker stdout in the
                           # stage log instead of polling its disabled API.
                           "val_check_interval": -1, "per_worker_log": False})
    cfg["runner"]["logger"].update({"log_path": str(output), "project_name": "controlled-pro",
                                    "experiment_name": f"{key}_{method}",
                                    "logger_backends": ["tensorboard"]})
    model = model_config(ROOT / "checkpoints/openpi/pi05_libero_rlinf", method)
    model["enforce_native_scope"] = True
    model["scope_report_directory"] = str(output / "model_scope")
    cfg["actor"]["model"] = model
    # Keep the actor resident on the 96 GB GPU; this microbatch-one setup does
    # not need manual CPU offloading between rollout and optimization.
    cfg["actor"].update({"micro_batch_size": 1, "global_batch_size": min(8, decisions),
                          "seed": 0, "enable_offload": False})
    cfg["actor"]["fsdp_config"] = yaml.safe_load((UPSTREAM / "training_backend/fsdp.yaml").read_text())
    cfg["actor"]["fsdp_config"].update({"sharding_strategy": "no_shard", "use_orig_params": True,
                                       "save_full_model_weights": True})
    cfg["actor"]["optim"]["critic_warmup_steps"] = 2 if iterations >= 25 else 0
    cfg["algorithm"].update({"gamma": .999 ** 5, "update_epoch": 1,
                              **algorithm_overrides(method)})
    cfg["rollout"].update({"enable_offload": True, "pipeline_stage_num": 1})
    cfg["rollout"]["model"]["model_path"] = model["model_path"]
    env = {
        "env_type": "pro_controlled", "protocol": str(task / "protocol.json"),
        "task_key": key, "split": "train", "simulator_python": str(ROOT / ".venv-pro/bin/python"),
        "seed": 0, "total_num_envs": 1, "group_size": 1, "rollout_epoch": 1,
        "max_steps_per_rollout_epoch": decisions * 5,
        "max_episode_steps": definition["max_steps"], "primitive_gamma": .999,
        "auto_reset": True, "ignore_terminations": False, "is_eval": False,
        "recording_directory": str(output / "trajectories"),
        "video_cfg": {"save_video": False}, "init_params": {},
    }
    cfg["env"]["train"] = env
    cfg["env"]["eval"] = copy.deepcopy(env) | {
        "split": "validation", "is_eval": True, "seed": 100000,
        "recording_directory": str(output / "unused_online_validation"),
    }
    cfg["weight_syncer"] = yaml.safe_load((UPSTREAM / "weight_syncer/patch_syncer.yaml").read_text())
    return cfg


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--method", choices=["flow_sde", "flow_noise"], required=True)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--decisions", type=int, default=32)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    cfg = configuration(args.task.resolve(), output, args.method, args.iterations, args.decisions)
    (output / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    print(output / "config.yaml")
