"""CPU regression tests for passive recording; no Blender/CUDA smoke test."""
import ast
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from experiments.yawpitch_audit import SelectionAudit, map_indices

ROOT = Path(__file__).resolve().parents[1]


def isolated_class(file, name, namespace):
    # Execute the actual class body without importing CUDA-only transitive deps.
    tree = ast.parse((ROOT / file).read_text())
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name)
    code = ast.Module(body=[node], type_ignores=[])
    exec(compile(code, file, 'exec'), namespace)
    return namespace[name]


class AuditTests(unittest.TestCase):
    def test_density_audit_does_not_change_mask_or_random_state(self):
        cls = isolated_class('nvf/active_mapping/agents/Sampler.py', 'BaseSampler', {
            'torch': torch, 'np': np,
            'get_density': lambda positions, model: positions[:, :1]})
        cfg = SimpleNamespace(camera_aabb=torch.tensor([[-2., -2., -2.], [2., 2., 2.]]),
                              target_aabb=torch.tensor([[-2., -2., -2.], [2., 2., 2.]]),
                              density_threshold=1e-4)
        sampler = cls(cfg)
        sampler.pipeline = SimpleNamespace(trainer=SimpleNamespace(pipeline=SimpleNamespace(
            model=SimpleNamespace(device='cpu'))))
        poses = torch.tensor([[0., 0., 0., 1., -1., 0., 0.],
                              [0., 0., 0., 1., 1., 0., 0.],
                              [0., 0., 0., 0., -1., 0., 0.],
                              [0., 0., 0., 1., 0., 0., 0.]])
        expected = sampler.filter_poses(poses)
        state = torch.get_rng_state().clone()
        sampler.audit_enabled = True
        actual = sampler.filter_poses(poses)
        self.assertTrue(torch.equal(expected, actual))
        self.assertTrue(torch.equal(state, torch.get_rng_state()))
        audit = sampler.last_filter_audit
        ids = np.flatnonzero(audit['quaternion_valid'])[audit['density_valid']]
        np.testing.assert_array_equal(ids, [0, 3])
        np.testing.assert_array_equal(audit['raw_poses'][ids], expected.numpy())

    def test_real_get_reward_observer_keeps_same_forward_and_return(self):
        cls = isolated_class('nvf/active_mapping/agents/BaseAgent.py', 'BaseAgent',
                             {'torch': torch, 'np': np})
        agent = cls.__new__(cls)
        maps = torch.arange(24, dtype=torch.float32).reshape(3, 1, 2, 4)
        calls = []
        def get_cost(**kwargs):
            calls.append(kwargs)
            return maps
        agent.pipeline = SimpleNamespace(get_cost=get_cost, trainer=SimpleNamespace(
            pipeline=SimpleNamespace(model=SimpleNamespace(renderer_entropy=object()))))
        agent.plan_hist = []
        poses = torch.zeros(3, 7)
        baseline = agent.get_reward(poses)
        observed = []
        agent.reward_observer = lambda p, m, s: observed.append((p, m, s))
        state = torch.get_rng_state().clone()
        result = agent.get_reward(poses)
        self.assertTrue(torch.equal(baseline, result))
        self.assertIs(observed[0][1], maps)
        self.assertIs(observed[0][2], result)
        self.assertEqual(len(calls), 2)  # Exactly once per call, no visualization rerender.
        self.assertTrue(torch.equal(state, torch.get_rng_state()))

    def test_map_selection_preserves_global_rng_and_first_tie(self):
        np.random.seed(8)
        state = np.random.get_state()
        ids, groups = map_indices([3, 3, 1], np.eye(3), np.zeros(3), seed=99)
        self.assertEqual(groups['selected'], 0)
        self.assertIn(0, ids)
        after = np.random.get_state()
        np.testing.assert_array_equal(state[1], after[1])
        self.assertEqual(state[2:], after[2:])

    def test_archive_indices_means_and_scaled_intrinsics(self):
        poses = torch.tensor([[0., 0., 0., 1., -1., 0., 0.],
                              [0., 0., 0., 1., 1., 0., 0.],
                              [0., 0., 0., 1., 0., 0., 0.]])
        raw = {'raw_poses': poses.numpy(), 'quaternion_valid': np.ones(3, bool),
               'density': np.array([0., 1., 0.]), 'density_valid': np.array([True, False, True])}
        model = torch.nn.Linear(1, 1)
        model.renderer_entropy = object()
        agent = SimpleNamespace(sampler=SimpleNamespace(last_filter_audit=raw,
            center=np.zeros(3), density_threshold=1e-4), config=SimpleNamespace(
            object_aabb=torch.tensor([[-1., -1., -1.], [1., 1., 1.]]), mu=.95),
            pipeline=SimpleNamespace(trainer=SimpleNamespace(pipeline=SimpleNamespace(model=model))))
        config = {'audit': {'map_observations': [3]}, 'camera': {
            'width': 512, 'height': 512, 'fx': 256, 'fy': 256, 'cx': 256, 'cy': 256},
            'min_foreground_fraction': .01}
        with tempfile.TemporaryDirectory() as temp:
            (Path(temp) / 'checkpoint_03.ckpt').write_bytes(b'fixture')
            audit = SelectionAudit(agent, temp, config, 3, 0)
            audit.sampling(poses[[0, 2]])
            maps = torch.tensor([1., 2.])[:, None, None, None].expand(2, 1, 300, 300)
            audit.reward(poses[[0, 2]], maps, maps.mean((-1, -2)))
            audit.selected(1, .4)
            directory = Path(temp) / 'audit/03'
            metadata = json.loads((directory / 'batch_00_entropy.json').read_text())
            self.assertEqual(metadata['intrinsics'][0], [150., 0., 150.])
            with np.load(directory / 'batch_00_entropy.npz') as saved:
                np.testing.assert_array_equal(saved['raw_candidate_ids'], [0, 2])
                np.testing.assert_allclose(saved['entropy'].mean((-1, -2)), saved['scores'])
            selected = json.loads((directory / 'batch_00_selected.json').read_text())
            self.assertEqual(selected['raw_candidate_id'], 2)


if __name__ == '__main__':
    unittest.main()
