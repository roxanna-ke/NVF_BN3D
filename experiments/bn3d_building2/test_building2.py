"""Offline checks: python3 -B -m unittest experiments.bn3d_building2.test_building2."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from experiments.bn3d_building2 import run

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


class Building2Tests(unittest.TestCase):
    def test_freeze_config_consistency(self):
        _, train, evaluation = run.configs(HERE / 'config.json')
        self.assertTrue(train['mesh_path'].endswith('/building2/building2.obj'))
        self.assertEqual(train['camera'], dict(width=512, height=512, fx=256, fy=256, cx=256, cy=256))
        self.assertEqual(train['audit'], dict(enabled=True, map_observations=[3, 10, 20, 35]))
        self.assertEqual(evaluation['modes'], ['train', 'eval'])
        self.assertEqual(evaluation['inference_seeds'], [42000, 43000, 44000])
        self.assertEqual([g['checkpoint_observations'] for g in evaluation['comparisons']], [3, 10, 20, 30, 50])
        self.assertEqual(train['frozen_inputs'], train['output_root'] + '/prepared/frozen_inputs.json')
        for group in evaluation['comparisons']:
            for policy in ['nvf', 'random']:
                self.assertEqual(group[f'{policy}_run'], train['output_root'] + f'/{policy}/50/seed42')

    def test_prepare_rewrite_and_freeze_identity(self):
        _, train, evaluation = run.configs(HERE / 'config.json')
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            mesh = root / 'building2.obj'
            mesh.write_text('v 0 0 0\nv 1 1 1\n')
            (root / 'meta_data.json').write_text(json.dumps({'frames': [
                {'camtoworld': np.eye(4).tolist()} for _ in range(200)]}))
            train.update(mesh_path=str(mesh), data_root=str(root), output_root=str(root / 'output'))
            manifest = run.tools.prepare(train)
            path = Path(train['output_root']) / 'prepared/manifest.json'
            before = path.read_bytes()
            frozen = {'normalization': {'world_to_canonical': np.eye(4).tolist(),
                                         'canonical_to_world': np.eye(4).tolist()}}
            item = run.frozen_item(train, frozen)
            first_directory, first_spec = run.protocol.dataset_directory(item, evaluation)
            training = copy.deepcopy(train)
            training.update(policy='nvf', seed=42, total_images=50, git_revision='example',
                            intrinsics=manifest['intrinsics'])
            self.assertEqual(run.tools.prepare(training), manifest)
            self.assertEqual(path.read_bytes(), before, 'Training prepare must preserve frozen manifest bytes')
            later = dict(item, config=training, run=str(root / 'output/nvf/50/seed42'))
            self.assertEqual(run.protocol.dataset_directory(later, evaluation), (first_directory, first_spec))
            self.assertEqual(len(item['split']['initial']), 3)
            self.assertEqual(len(item['split']['test']), 40)

    def test_submit_arbitrary_cwd_and_config_mapping(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            log = root / 'argv.json'
            stub = root / 'runai'
            stub.write_text(f'#!{sys.executable}\nimport os,json,sys\n'
                            'from pathlib import Path\n'
                            'Path(os.environ["BUILDING2_TEST_LOG"]).write_text(json.dumps(sys.argv[1:]))\n')
            stub.chmod(0o755)
            env = dict(os.environ, PATH=str(root) + os.pathsep + os.environ['PATH'],
                       BUILDING2_TEST_LOG=str(log), CONTAINER_REPO='/mnt/scratch/test/repo',
                       SCRATCH_PVC='test-scratch', HOME_PVC='test-home')
            for task in ['freeze', 'nvf', 'random', 'evaluate']:
                for config in [[], ['experiments/bn3d_building2/config.json'], [str(HERE / 'config.json')]]:
                    with self.subTest(task=task, config=config):
                        subprocess.run(['bash', str(HERE / 'job.sh'), 'submit', task, *config],
                                       cwd=root, env=env, check=True, capture_output=True)
                        args = json.loads(log.read_text())
                        self.assertEqual(args[args.index('--working-dir') + 1], '/tmp')
                        self.assertEqual(args[-3:], ['worker', task,
                            '/mnt/scratch/test/repo/experiments/bn3d_building2/config.json'])
                        self.assertIn('claimname=test-scratch,path=/mnt/scratch', args)
                        self.assertIn('claimname=test-home,path=/mnt/home', args)

    def test_worker_enters_repository(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            stub = root / 'python'
            log = root / 'worker.json'
            stub.write_text(f'#!{sys.executable}\nimport os,json,sys\n'
                            'from pathlib import Path\n'
                            'Path(os.environ["BUILDING2_TEST_LOG"]).write_text(json.dumps(dict('
                            'cwd=os.getcwd(), args=sys.argv[1:], pythonpath=os.environ["PYTHONPATH"])))\n')
            stub.chmod(0o755)
            env = dict(os.environ, PYTHON_BIN=str(stub), CONTAINER_REPO=str(ROOT),
                       BUILDING2_TEST_LOG=str(log))
            subprocess.run(['bash', str(HERE / 'job.sh'), 'worker', 'freeze'],
                           cwd=root, env=env, check=True, capture_output=True)
            result = json.loads(log.read_text())
            self.assertEqual(Path(result['cwd']).resolve(), ROOT)
            self.assertEqual(result['args'][:4], ['-B', '-m', 'experiments.bn3d_building2.run', 'freeze'])
            self.assertEqual(result['pythonpath'].split(os.pathsep)[0], str(ROOT))


if __name__ == '__main__':
    unittest.main()
