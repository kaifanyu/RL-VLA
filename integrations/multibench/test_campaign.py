"""Check real generated protocols and isolation invariants without loading policies."""
import json
from pathlib import Path
import tomllib
import unittest
import yaml
from multibench.environment import ROOT, process_config, state_id

CAMPAIGN=ROOT/'runs/multibench_20261028/campaign'

class CampaignTests(unittest.TestCase):
    def test_frozen_splits_are_disjoint_and_native(self):
        for benchmark in ('robotwin','robocasa','maniskill'):
            protocol=json.loads((CAMPAIGN/'tasks'/benchmark/'protocol.json').read_text())
            self.assertEqual(protocol['validation_iterations'],[25,50,100])
            definition=protocol['tasks'][benchmark]
            all_ids=[]
            for split, n in [('train',100),('validation',50),('test',50)]:
                self.assertEqual(len(definition[f'{split}_seeds']),n)
                self.assertEqual([state_id(benchmark,s) for s in definition[f'{split}_seeds']],definition[f'{split}_state_ids'])
                all_ids.extend(definition[f'{split}_state_ids'])
            self.assertEqual(len(set(all_ids)),200)

    def test_all_methods_start_raw_base_and_keep_native_shape(self):
        for benchmark, motor, images in [('robotwin',14,3),('robocasa',12,3),('maniskill',7,1)]:
            task=CAMPAIGN/'tasks'/benchmark
            config=tomllib.loads((task/'train.toml').read_text())
            self.assertEqual(config['policy']['benchmark'],benchmark)
            self.assertEqual(Path(config['policy']['checkpoint']).name,'pi05_base')
            self.assertEqual(config['train']['iterations'],100)
            self.assertEqual(config['rldt']['execute_steps'],5)
            self.assertFalse(config['recording']['enabled'])
            torch=yaml.safe_load((CAMPAIGN/'experiments'/f'{benchmark}_flow_sde'/'training_config.yaml').read_text())
            model=torch['actor']['model']
            self.assertEqual(Path(model['model_path']).name,'pi05_base_rlinf')
            self.assertEqual(model['action_dim'],motor)
            self.assertEqual(model['openpi']['action_env_dim'],motor)
            self.assertEqual(model['openpi']['num_images_in_input'],images)
            self.assertEqual(model['openpi']['action_horizon'],50)
            self.assertTrue(model['openpi']['discrete_state_input'])
            self.assertEqual(torch['runner']['max_steps'],100)
            self.assertEqual(torch['env']['train']['max_steps_per_rollout_epoch'],160)
            self.assertEqual(torch['actor']['global_batch_size'],8)

    def test_separate_simulator_processes_and_render_visibility(self):
        for benchmark in ('robotwin','robocasa','maniskill'):
            cfg=process_config(benchmark,CAMPAIGN/'tasks'/benchmark/'protocol.json','validation')
            self.assertTrue(Path(cfg['python']).is_file())
            self.assertEqual(cfg['gpu_rendering'],benchmark!='robocasa')
            self.assertIn('multibench/core',cfg['process_env']['PYTHONPATH'])
            self.assertNotIn(str(ROOT/'src'),cfg['process_env']['PYTHONPATH'].split(':'))

if __name__=='__main__':
    unittest.main()
