"""CPU unit checks for paired evaluation and checkpoint scheduling; no policy downloads."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from multibench.evaluate import evaluate

HOOK = Path(__file__).parent / "core/rl_vla/checkpoint_validation.py"
spec = importlib.util.spec_from_file_location("multibench_validation_hook_test", HOOK)
hook = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hook)


class FakePolicy:
    execute_steps = 3
    metadata = {"unit_test": True}

    def __init__(self, config, checkpoint):
        self.seeds = []

    def reset(self, seed):
        self.seeds.append(seed)

    def predict(self, raw):
        motor = np.ones((3, 2), dtype=np.float32) * 2
        return np.zeros((50, 32)), motor, motor, {}


class FakeEnv:
    low, high = -np.ones(2), np.ones(2)

    def __init__(self):
        self.episode, self.closed, self.seeds = -1, False, []

    def reset(self, *, seed):
        self.episode += 1
        self.steps = 0
        self.seeds.append(seed)
        return {}, {"initial_state_id": [7, 12][self.episode]}

    def step(self, action):
        np.testing.assert_array_equal(action, self.high)
        self.steps += 1
        # Goal occurred before the boundary; success-once must not be lost.
        success = self.episode == 0 and self.steps == 1
        return {}, float(success), False, self.steps == 2, {"is_success": success}

    def close(self):
        self.closed = True


class EvaluationChecks(unittest.TestCase):
    def plan(self):
        return {"tasks": {"maniskill": {"max_steps": 2,
                "validation_state_ids": [7, 12], "validation_seeds": [91, 92]}}}

    def test_early_chunk_boundary_paired_states_and_success_once(self):
        with tempfile.TemporaryDirectory() as directory:
            env = FakeEnv()
            result = evaluate({}, self.plan(), "maniskill", "validation", Path(directory) / "report.json",
                              method="rldt", episodes=2, environment_factory=lambda: env,
                              policy_factory=FakePolicy)
            self.assertEqual(result["success_rate"], .5)
            self.assertEqual(result["controller_steps"], 4)
            self.assertEqual(result["policy_decisions"], 2)
            self.assertEqual(env.seeds, [100000, 100001])
            self.assertTrue(env.closed)
            self.assertEqual([e["environment_seed"] for e in result["episodes"]], [91, 92])
            self.assertTrue(all(e["truncated"] for e in result["episodes"]))

    def test_wrong_state_order_fails_and_closes(self):
        with tempfile.TemporaryDirectory() as directory:
            env = FakeEnv()
            plan = self.plan()
            plan["tasks"]["maniskill"]["validation_state_ids"] = [12, 7]
            with self.assertRaisesRegex(ValueError, "frozen evaluation"):
                evaluate({}, plan, "maniskill", "validation", Path(directory) / "report.json",
                         method="flow_sde", episodes=2, environment_factory=lambda: env,
                         policy_factory=FakePolicy)
            self.assertTrue(env.closed)
            self.assertFalse((Path(directory) / "report.json").exists())

    def test_checkpoint_schedule_and_earliest_tie(self):
        self.assertEqual(hook.scheduled_iterations({}), (25, 50, 100))
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            training = output / "training"
            training.mkdir()
            protocol = output / "protocol.json"
            protocol.write_text(json.dumps(self.plan()))
            experiment = {"method": "rldt", "benchmark": "maniskill", "task_directory": str(output),
                          "protocol_file": str(protocol), "protocol": {"online_validation": True,
                          "validation_iterations": [25, 50, 100]}}
            (output / "experiment.json").write_text(json.dumps(experiment))
            calls = []
            def run(command, **kwargs):
                calls.append(command)
                report_path = Path(command[command.index("--output") + 1])
                report_path.write_text(json.dumps({"success_rate": .5, "controller_steps": 4,
                    "episodes": [{"reset": {"initial_state_id": 7}, "success": True},
                                 {"reset": {"initial_state_id": 12}, "success": False}]}))
            with mock.patch.object(hook.subprocess, "run", run):
                for iteration in [25, 50, 75, 100]:
                    hook.run_checkpoint_validation(training, training / str(iteration), iteration,
                                                   {"controller_steps": iteration * 160}, "rldt")
            self.assertEqual(len(calls), 3)
            self.assertTrue(all("multibench.evaluate" in call for call in calls))
            self.assertTrue(all(call[2] == "native" for call in calls))
            self.assertEqual(hook.select_checkpoint(output)["iteration"], 25)
            self.assertFalse((output / "validation_000075.json").exists())


if __name__ == "__main__":
    unittest.main()
