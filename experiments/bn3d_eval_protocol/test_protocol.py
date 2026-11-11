"""Meaningful CPU checks for geometry, occupancy guards, and aggregation.

Run from the repository root: python3 -m unittest experiments.bn3d_eval_protocol.test_protocol
"""
import copy
import itertools
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image

from experiments.bn3d_eval_protocol.sampling import generate_views, projected_inside
from experiments.bn3d_eval_protocol.run import aggregate, summarize, verify_files, sha256, write


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.settings = json.loads((Path(__file__).parent / 'config.json').read_text())['sampling']
        self.vertices = np.array(list(itertools.product([-1, 1], [-.7, .7], [-.8, .8])), float)

    def test_package_import_ignores_same_named_top_level_modules(self):
        # Reproduce a notebook/plugin process which already imported unrelated
        # modules named sampling and run. Neither may affect our imports.
        script = '''
import sys, types
sys.modules['sampling'] = types.ModuleType('sampling')
sys.modules['run'] = types.ModuleType('run')
before = list(sys.path)
from experiments.bn3d_eval_protocol import run
assert run.generate_views.__module__ == 'experiments.bn3d_eval_protocol.sampling'
assert sys.path == before, 'Import mutated the interpreter search path'
assert not hasattr(sys.modules['sampling'], 'generate_views')
'''
        subprocess.run([sys.executable, '-B', '-c', script],
                       cwd=Path(__file__).resolve().parents[2], check=True)

    def test_ground_and_complete_framing_with_both_intrinsics(self):
        for focal in [256, 711.111274]:
            k = np.array([[focal, 0, 255.5], [0, focal, 255.5], [0, 0, 1.]])
            views = generate_views(self.vertices, k, self.settings)
            self.assertEqual(len(views), 20)
            poses = np.asarray([v['pose'] for v in views])
            self.assertTrue((poses[:, 2, 3] >= -.8).all())
            self.assertTrue((poses[:, 2, 3] < 0).any())
            self.assertEqual([sum(v['ring'] == f'ring_{i}' for v in views) for i in range(3)], [4, 6, 9])
            for pose in poses:
                self.assertTrue(projected_inside(self.vertices, pose, k, 12))
                np.testing.assert_allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-10)
                self.assertAlmostEqual(np.linalg.det(pose[:3, :3]), 1.)
            again = generate_views(self.vertices, k, self.settings)
            self.assertEqual([v['pose'] for v in views], [v['pose'] for v in again])

    def test_actual_empty_render_cannot_be_frozen(self):
        k = np.array([[711., 0, 255.5], [0, 711., 255.5], [0, 0, 1.]])
        with self.assertRaisesRegex(ValueError, 'foreground'):
            generate_views(self.vertices, k, self.settings,
                           render=lambda pose: np.zeros((512, 512, 4), dtype=np.uint8))

    def test_valid_alpha_is_recorded_and_ground_override_checked(self):
        k = np.array([[711., 0, 255.5], [0, 711., 255.5], [0, 0, 1.]])
        image = np.zeros((512, 512, 4), dtype=np.uint8)
        image[100:400, 100:400, 3] = 255
        views = generate_views(self.vertices, k, self.settings, render=lambda pose: image)
        self.assertTrue(all(.2 <= v['foreground_fraction'] <= .37 for v in views))
        settings = copy.deepcopy(self.settings)
        settings['ground_z'] = 0
        with self.assertRaisesRegex(ValueError, 'ground_z'):
            generate_views(self.vertices, k, settings)

    def test_foreground_metrics_pool_pixels_before_psnr(self):
        rows = []
        for pixels, mse in [(10, .01), (30, .09)]:
            rows.append({'foreground_pixels': pixels, 'fg_mse': mse, 'full_mse': mse,
                         'fg_ssim': .5, 'fg_lpips_spatial': .2, 'full_ssim': .7, 'full_lpips': .1})
        value = aggregate(rows)
        self.assertAlmostEqual(value['fg_mse'], .07)
        self.assertAlmostEqual(value['full_mse'], .05)
        self.assertAlmostEqual(value['fg_psnr'], -10 * np.log10(.07))

    def test_modified_frozen_gt_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = root / 'view.png'
            Image.new('RGBA', (8, 8), 'red').save(path)
            manifest = {'files': {'view.png': sha256(path)}}
            verify_files(root, manifest)
            Image.new('RGBA', (8, 8), 'blue').save(path)
            with self.assertRaisesRegex(ValueError, 'modified'):
                verify_files(root, manifest)

    def test_summary_pairs_by_frame_without_gt_and_reports_contrasts(self):
        # Exercise the real summary pipeline with 40/20 tiny synthetic images.
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            cfg = {'output_root': temp, 'comparisons': [{'id': 'fixture'}],
                   'modes': ['train', 'eval'], 'inference_seeds': [42]}
            for which, count in [('old', 40), ('new', 20)]:
                for mode in cfg['modes']:
                    for policy in ['nvf', 'random']:
                        dest = root / 'evaluations/fixture' / which / mode / policy / 'seed42'
                        (dest / 'predictions').mkdir(parents=True)
                        for i in range(count):
                            Image.new('RGB', (512, 512), 'red' if policy == 'nvf' else 'blue').save(dest / 'predictions' / f'view_{i:02d}.png')
                        metrics = {k: 1. if policy == 'nvf' else 0. for k in [
                            'full_psnr', 'fg_psnr', 'full_ssim', 'fg_ssim', 'full_lpips', 'fg_lpips_spatial']}
                        write(dest / 'metrics.json', metrics)
                        write(dest / 'complete.json', {'files': {'metrics.json': sha256(dest / 'metrics.json')}})
            summarize(cfg)
            paired = root / 'comparisons/fixture/new/train/view_00.png'
            with Image.open(paired) as im:
                self.assertEqual(im.size, (1024, 538))
                self.assertEqual(im.getpixel((10, 40)), (255, 0, 0))
                self.assertEqual(im.getpixel((522, 40)), (0, 0, 255))
            self.assertTrue((root / 'policy_differences.csv').exists())
            self.assertFalse(list((root / 'comparisons').rglob('*gt*')))


if __name__ == '__main__':
    unittest.main()
