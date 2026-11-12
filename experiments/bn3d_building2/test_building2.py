"""Offline checks: python3 -B -m unittest experiments.bn3d_building2.test_building2."""
import copy
import ast
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from experiments.bn3d_building2 import run
from experiments.yawpitch_audit import load_frozen_inputs
from experiments.bn3d_eval_protocol.sampling import projected_inside
from nvf.bn3d_data import preserve_mesh_materials
from PIL import Image

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
        self.assertEqual(train['test_set'], 'new20')
        self.assertEqual(evaluation['test_sets'], ['new'])
        self.assertEqual(evaluation['sampling']['ring_counts'], [4, 6, 5, 4])
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
            self.assertEqual(item['split']['test'], [])
            self.assertEqual(item['split']['initial'], [131, 18, 154])
            self.assertNotIn('old_poses', first_spec)
            self.assertNotIn('old_frame_ids', first_spec)

    def test_four_height_rings_and_top_with_both_principal_points(self):
        _, _, evaluation = run.configs(HERE / 'config.json')
        vertices = np.array(list(itertools.product([-1., 1.], [-.7, .7], [-.8, .8])))
        image = np.zeros((512, 512, 4), np.uint8)
        image[100:400, 100:400, 3] = 255
        for center in [255.5, 256.]:
            k = np.array([[256., 0, center], [0, 256., center], [0, 0, 1.]])
            views = run.protocol.generate_views(vertices, k, evaluation['sampling'], render=lambda pose: image)
            self.assertEqual(len(views), 20)
            self.assertEqual([sum(v['ring'] == f'ring_{i}' for v in views) for i in range(4)], [4, 6, 5, 4])
            self.assertEqual(views[-1]['ring'], 'top')
            for i, fraction in enumerate([.95, .65, .35, .05]):
                for view in views:
                    if view['ring'] == f'ring_{i}':
                        self.assertAlmostEqual(view['pose'][2][3], -.8 + fraction * 1.6)
            for view in views:
                self.assertTrue(projected_inside(vertices, np.asarray(view['pose']), k, 12))

    def test_principal_point_is_independent_of_explicit_image_size(self):
        import torch
        # Run the repository's actual size initialization method without loading
        # the CUDA-heavy camera module and its transitive dependencies.
        tree = ast.parse((ROOT / 'nerfstudio/cameras/cameras.py').read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Cameras')
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef)
                      and n.name == '_init_get_height_width')
        method.returns = None
        for arg in method.args.args:
            arg.annotation = None
        namespace = {'torch': torch}
        exec(compile(ast.Module(body=[method], type_ignores=[]), '<camera-size>', 'exec'), namespace)
        size = namespace['_init_get_height_width']
        camera = SimpleNamespace(device='cpu')
        for center in [255.5, 256.]:
            c = torch.tensor([[center]])
            self.assertEqual(size(camera, 512, c).item(), 512)
            self.assertEqual(size(camera, None, c).item(), int(center * 2))

    def test_material_assignments_survive_annotation_and_exceptions(self):
        class Polygons:
            def __init__(self):
                self.indices = np.array([0, 1, 2, 1], np.int32)
            def __len__(self):
                return len(self.indices)
            def foreach_get(self, name, output):
                output[:] = self.indices
            def foreach_set(self, name, values):
                self.indices[:] = values

        polygons = Polygons()
        class Slots(list):
            def clear(self):
                super().clear()
                polygons.indices[:] = 0  # Blender's destructive slot-clear behavior.

        slots = Slots(['wood', 'plaster', 'stone'])
        mesh = SimpleNamespace(materials=slots, polygons=polygons, update=lambda: None)
        objects = [SimpleNamespace(data=mesh), SimpleNamespace(data=mesh)]
        for failure in [False, True, False]:
            try:
                with preserve_mesh_materials(objects):
                    slots.clear()
                    slots.append('annotation')
                    if failure:
                        raise RuntimeError('annotation failed')
                    slots.clear()
                    slots.extend(['wood', 'plaster', 'stone'])
            except RuntimeError:
                pass
            self.assertEqual(slots, ['wood', 'plaster', 'stone'])
            np.testing.assert_array_equal(polygons.indices, [0, 1, 2, 1])

    def test_new20_freeze_and_training_loader_roundtrip(self):
        _, train, evaluation = run.configs(HERE / 'config.json')
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            train.update(output_root=str(root / 'training'), data_root=str(root),
                         mesh_path=str(root / 'building2.obj'),
                         frozen_inputs=str(root / 'training/prepared/frozen_inputs.json'))
            evaluation['output_root'] = str(root / 'evaluation')
            (root / 'building2.obj').write_text('v 0 0 0\nv 1 1 1\n')
            (root / 'meta_data.json').write_text(json.dumps({'frames': [
                {'camtoworld': np.eye(4).tolist()} for _ in range(200)]}))
            image = np.zeros((512, 512, 4), np.uint8)
            image[100:400, 100:400] = [100, 150, 200, 255]
            class Scene:
                world_to_canonical = canonical_to_world = np.eye(4)
                mesh = SimpleNamespace(vertices=np.array(list(itertools.product(
                    [-1., 1.], [-.7, .7], [-.8, .8]))))
                renders = 0
                def __init__(self, config):
                    pass
                def render_pose(self, pose):
                    Scene.renders += 1
                    return image.copy()

            with patch.dict(sys.modules, {'nvf.env.Scene': SimpleNamespace(BN3DScene=Scene),
                                         'nvf.env.utils': SimpleNamespace(set_seed=lambda seed: None),
                                         'bpy': SimpleNamespace(app=SimpleNamespace(version_string='fixture'))}), \
                    patch.object(run.protocol, 'runtime_info', return_value={}):
                run.freeze(train, evaluation)
            self.assertEqual(Scene.renders, 43)  # initial3 + two renders per new view; no metadata test images.
            frozen, item = run.verify(train, evaluation)
            self.assertNotIn('old', frozen)
            test, initial = load_frozen_inputs(train, frozen['normalization'],
                item['split']['initial'], [], [np.eye(4) for _ in range(200)])
            self.assertEqual((len(test), len(initial)), (20, 3))
            np.testing.assert_array_equal(test[0], image)
            self.assertFalse(list(root.rglob('old')))
            # Reuse accepts only the exact completed new20 freeze.
            run.freeze(train, evaluation)
            frozen['test'][0]['pose'][0][3] += 1
            run.write(train['frozen_inputs'], frozen)
            with self.assertRaisesRegex(ValueError, 'pose/path'):
                load_frozen_inputs(train, frozen['normalization'], item['split']['initial'],
                                   [], [np.eye(4) for _ in range(200)])

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
                            'if "-c" in sys.argv:\n'
                            '    os.execv(sys.executable, [sys.executable, *sys.argv[1:]])\n'
                            'Path(os.environ["BUILDING2_TEST_LOG"]).write_text(json.dumps(dict('
                            'cwd=os.getcwd(), args=sys.argv[1:], pythonpath=os.environ["PYTHONPATH"])))\n'
                            'print("training stdout", flush=True)\n'
                            'print("original training failure", file=sys.stderr, flush=True)\n'
                            'sys.exit(int(os.environ.get("BUILDING2_TEST_EXIT", "0")))\n')
            stub.chmod(0o755)
            (root / 'config.json').write_text(json.dumps({'train_config': 'train.json'}))
            (root / 'train.json').write_text(json.dumps({'output_root': str(root / 'output')}))
            env = dict(os.environ, PYTHON_BIN=str(stub), CONTAINER_REPO=str(ROOT),
                       BUILDING2_TEST_LOG=str(log))
            subprocess.run(['bash', str(HERE / 'job.sh'), 'worker', 'freeze', str(root / 'config.json')],
                           cwd=root, env=env, check=True, capture_output=True)
            result = json.loads(log.read_text())
            self.assertEqual(Path(result['cwd']).resolve(), ROOT)
            self.assertEqual(result['args'][:4], ['-B', '-m', 'experiments.bn3d_building2.run', 'freeze'])
            self.assertEqual(result['pythonpath'].split(os.pathsep)[0], str(ROOT))
            first_log, = (root / 'output/_logs').glob('freeze.log.*')
            first_contents = first_log.read_text()
            self.assertIn('training stdout', first_contents)
            self.assertIn('original training failure', first_contents)
            # A replacement Pod must fail too, and must preserve the first log.
            failed = subprocess.run(
                ['bash', str(HERE / 'job.sh'), 'worker', 'freeze', str(root / 'config.json')],
                cwd=root, env=dict(env, BUILDING2_TEST_EXIT='17'), capture_output=True)
            self.assertEqual(failed.returncode, 17)
            self.assertEqual(first_log.read_text(), first_contents)
            self.assertEqual(len(list((root / 'output/_logs').glob('freeze.log.*'))), 2)


if __name__ == '__main__':
    unittest.main()
