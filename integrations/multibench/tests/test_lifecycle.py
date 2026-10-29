"""CPU campaign lifecycle checks; external learners and Slurm are not launched."""
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

from multibench import run_experiment, submit


class LifecycleTests(unittest.TestCase):
    def prepare(self, root, method):
        campaign = root / 'campaign'
        task = campaign / 'tasks' / 'robocasa'
        output = campaign / 'experiments' / f'robocasa_{method}'
        task.mkdir(parents=True)
        output.mkdir(parents=True)
        source = root / 'source.py'
        source.write_text('# frozen source\n')
        contract = campaign / 'input_contract.json'
        contract.write_text(json.dumps({'ready': True, 'sha256': {
            'source.py': hashlib.sha256(source.read_bytes()).hexdigest()}}))
        experiment = {'benchmark': 'robocasa', 'method': method,
                      'task_directory': str(task), 'protocol': {'validation_iterations': [25, 50, 100]}}
        (output / 'experiment.json').write_text(json.dumps(experiment))
        (output / 'training_config.yaml').write_text('test: true\n')
        (campaign / 'plan.json').write_text(json.dumps({'experiments': [str(output)]}))
        return campaign, task, output, contract

    def run_fake(self, method, fail_training=False):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            campaign, task, output, contract = self.prepare(root, method)
            calls = []

            def execute(command, **kwargs):
                calls.append(command)
                if 'multibench.evaluate' in command:
                    if method == 'flow_sde':
                        # Real Torch baseline loading writes its trainable-scope
                        # provenance under training before the learner starts.
                        scope = output / 'training' / 'model_scope'
                        scope.mkdir(parents=True, exist_ok=True)
                        (scope / 'baseline.json').write_text('{}\n')
                    get = lambda flag: command[command.index(flag) + 1]
                    report = {'split': get('--split'), 'successes': 0, 'success_rate': 0.,
                              'episodes': [{}] * 50, 'controller_steps': 100, 'policy_decisions': 20,
                              'elapsed_seconds': 2., 'model_load_seconds': 1.}
                    Path(get('--output')).write_text(json.dumps(report))
                else:
                    if fail_training:
                        raise subprocess.CalledProcessError(7, command)
                    (output / 'training').mkdir(exist_ok=True)
                    rows = [{'iteration': i, 'checkpoint': str(output / 'training' / str(i)),
                             'successes': 0, 'episodes': 50, 'report': str(output / f'validation_{i:06d}.json'),
                             'training_counters': {'primitive_steps': i * 160, 'decisions': i * 32,
                                                   'elapsed_seconds': i * 2.}}
                            for i in (25, 50, 100)]
                    (output / 'checkpoint_validations.jsonl').write_text(
                        ''.join(json.dumps(row) + '\n' for row in rows))

            with mock.patch.object(run_experiment, 'ROOT', root), \
                 mock.patch.object(run_experiment.subprocess, 'run', execute):
                if fail_training:
                    with self.assertRaises(subprocess.CalledProcessError):
                        run_experiment.run(output, contract)
                else:
                    result = run_experiment.run(output, contract)
                    self.assertEqual(result['state'], 'completed')
                    self.assertEqual(result['selected_checkpoint']['iteration'], 25)
            status = json.loads((output / 'status.json').read_text())
            evaluations = [call for call in calls if 'multibench.evaluate' in call]
            self.assertEqual(len(evaluations), 2 if fail_training else 3)
            expected_backend = 'native' if method == 'rldt' else 'torch'
            for command in calls:
                self.assertEqual(command[2], expected_backend)
            self.assertEqual([c[c.index('--split') + 1] for c in evaluations[:2]], ['validation', 'test'])
            self.assertTrue(all('--checkpoint' not in c for c in evaluations[:2]))
            self.assertTrue(all(c[c.index('--episodes') + 1] == '50' for c in evaluations))
            self.assertEqual(status['state'], 'failed' if fail_training else 'completed')
            self.assertTrue((campaign / 'results.json').is_file())
            if not fail_training:
                self.assertEqual(evaluations[-1][evaluations[-1].index('--checkpoint') + 1],
                                 str(output / 'training' / '25'))
                self.assertEqual(evaluations[0][evaluations[0].index('--seed') + 1], '100000')
                self.assertEqual(evaluations[-1][evaluations[-1].index('--seed') + 1], '200000')
                expected_config = task / 'validation.toml' if method == 'rldt' else output / 'training_config.yaml'
                self.assertTrue(all(c[c.index('--config') + 1] == str(expected_config) for c in evaluations))

    def test_native_paired_baselines_continuous_training_and_selected_test(self):
        self.run_fake('rldt')

    def test_torch_paired_baselines_continuous_training_and_selected_test(self):
        self.run_fake('flow_sde')

    def test_failed_training_preserves_partial_results_and_skips_selected_test(self):
        self.run_fake('flow_sde', fail_training=True)

    def test_changed_frozen_source_fails_before_any_external_command(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, _, output, contract = self.prepare(root, 'rldt')
            (root / 'source.py').write_text('# modified\n')
            with mock.patch.object(run_experiment, 'ROOT', root), \
                 mock.patch.object(run_experiment.subprocess, 'run') as execute:
                with self.assertRaisesRegex(RuntimeError, 'Frozen experiment'):
                    run_experiment.run(output, contract)
                execute.assert_not_called()

    def test_slurm_submissions_are_sequential_with_durable_receipts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            campaign, _, output, contract = self.prepare(root, 'rldt')
            second = output.parent / 'robocasa_flow_sde'
            second.mkdir()
            (campaign / 'plan.json').write_text(json.dumps({'experiments': [str(output), str(second)]}))
            with mock.patch.object(run_experiment, 'ROOT', root), mock.patch.object(submit, 'ROOT', root), \
                 mock.patch.object(submit.subprocess, 'check_output', side_effect=['123;cluster\n', '456\n']) as send:
                jobs = submit.submit(campaign, contract, dependency='99')
            self.assertEqual([job['job_id'] for job in jobs], ['123', '456'])
            self.assertEqual([job['dependency'] for job in jobs], ['99', '123'])
            for call, dependency in zip(send.call_args_list, ['afterany:99', 'afterany:123']):
                command = call.args[0]
                self.assertEqual(command[command.index('--dependency') + 1], dependency)
            self.assertEqual(json.loads((campaign / 'submission.json').read_text())['state'], 'submitted')


if __name__ == '__main__':
    unittest.main()
