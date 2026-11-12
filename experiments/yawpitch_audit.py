"""Optional passive recording for the existing yawpitch training loop.

No torch import at module scope; selection of diagnostic views uses an isolated
NumPy generator, never the training or candidate proposal random stream.
"""
from pathlib import Path
import json

import numpy as np

from nvf.bn3d_data import sha256


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def map_indices(scores, positions, center, seed):
    scores = np.asarray(scores).reshape(-1)
    if not len(scores) or not np.isfinite(scores).all():
        raise ValueError('Expected nonempty finite candidate scores')
    best = int(scores.argmax())  # same first-occurrence tie rule as torch.argmax
    top = np.argsort(-scores, kind='stable')[:10].tolist()
    random = np.random.default_rng(seed).choice(len(scores), min(10, len(scores)), replace=False).tolist()
    distances = np.linalg.norm(np.asarray(positions) - center, axis=1)
    matches = np.flatnonzero(np.abs(distances - distances[best]) <= .05 * max(distances[best], 1e-12))
    matches = matches[matches != best]
    if not len(matches) and len(scores) > 1:
        order = np.argsort(np.abs(distances - distances[best]), kind='stable')
        matches = order[order != best][:1]
    matched = matches[:10].tolist()
    return sorted(set([best] + top + random + matched)), {
        'selected': best, 'top10': top, 'random10': random, 'distance_matched': matched}


class SelectionAudit:
    def __init__(self, agent, run_dir, config, observation, batch):
        self.agent, self.config = agent, config
        self.n, self.batch = observation, batch
        self.root = Path(run_dir) / 'audit' / f'{observation:02d}'
        self.root.mkdir(parents=True, exist_ok=True)
        self.stem = f'batch_{batch:02d}'

    def sampling(self, candidates):
        audit = self.agent.sampler.last_filter_audit
        raw = audit['raw_poses']
        valid_ids = np.flatnonzero(audit['quaternion_valid'])
        filtered_ids = valid_ids[np.asarray(audit['density_valid']).reshape(-1)]
        returned_ids = filtered_ids[:len(candidates)]
        returned = candidates.detach().cpu().numpy()
        np.testing.assert_array_equal(raw[returned_ids], returned)
        center = self.agent.sampler.center
        distances = np.linalg.norm(raw[:, -3:] - center, axis=1)
        bounds = np.asarray(self.agent.config.object_aabb.cpu())
        scale = float(np.linalg.norm(bounds[1] - bounds[0]))
        arrays = dict(raw_poses=raw, quaternion_valid=audit['quaternion_valid'],
                      density_valid=audit['density_valid'], filtered_ids=filtered_ids,
                      returned_ids=returned_ids, center_distance=distances,
                      normalized_center_distance=distances / scale)
        if audit['density'] is not None:
            arrays['density_after_quaternion_filter'] = audit['density']
        path = self.root / f'{self.stem}_sampling.npz'
        np.savez_compressed(path, **arrays)
        self.returned_ids = returned_ids
        self.sampling_record = {'file': str(path.relative_to(self.root.parent.parent)),
            'sha256': sha256(path), 'raw_count': len(raw), 'filtered_count': len(filtered_ids),
            'returned_count': len(returned), 'object_diagonal': scale,
            'center': center.tolist(), 'density_threshold': self.agent.sampler.density_threshold,
            'raw_distance': distances.tolist(), 'filtered_distance': distances[filtered_ids].tolist(),
            'returned_distance': distances[returned_ids].tolist()}
        return self.sampling_record

    def reward(self, poses, maps, scores):
        if self.n not in self.config['audit'].get('map_observations', [3, 10, 20, 35]):
            return
        values = scores.detach().cpu().numpy().reshape(-1)
        pose_array = poses.detach().cpu().numpy()
        ids, groups = map_indices(values, pose_array[:, -3:], self.agent.sampler.center,
                                  910000 + self.n * 100 + self.batch)
        # get_cost returns [candidate, trajectory=1, H, W]. Do not recompute it.
        selected = maps[ids].detach().cpu().numpy()
        selected = selected.reshape(len(ids), *selected.shape[-2:])
        np.testing.assert_allclose(selected.mean(axis=(-1, -2)), values[ids], rtol=2e-5, atol=2e-5)
        path = self.root / f'{self.stem}_entropy.npz'
        np.savez_compressed(path, candidate_ids=ids, raw_candidate_ids=self.returned_ids[ids],
                            poses=pose_array[ids], entropy=selected, scores=values[ids])
        model = self.agent.pipeline.trainer.pipeline.model
        height, width = selected.shape[-2:]
        camera = self.config['camera']
        k = [[camera['fx'] * width / camera['width'], 0., camera['cx'] * width / camera['width']],
             [0., camera['fy'] * height / camera['height'], camera['cy'] * height / camera['height']],
             [0., 0., 1.]]
        checkpoint = self.root.parent.parent / f'checkpoint_{self.n:02d}.ckpt'
        renderer = model.renderer_entropy
        write_json(self.root / f'{self.stem}_entropy.json', {
            'observation_count': self.n, 'batch': self.batch, 'groups': groups,
            'source': 'actual_selection_forward', 'file': path.name, 'sha256': sha256(path),
            'checkpoint': str(checkpoint), 'checkpoint_sha256': sha256(checkpoint),
            'height': height, 'width': width, 'intrinsics': k,
            'module_training': {name: bool(module.training) for name, module in model.named_modules()},
            'renderer': type(renderer).__name__, 'renderer_iteration': self.n - 3,
            'configured_d0': self.config.get('d0', 1.), 'configured_mu': self.agent.config.mu,
            'camera_index': 0, 'aggregation': 'mean over all pixels', 'all_scores': values.tolist()})

    def selected(self, candidate_id, fraction):
        write_json(self.root / f'{self.stem}_selected.json', {
            'candidate_id': candidate_id, 'raw_candidate_id': int(self.returned_ids[candidate_id]),
            'foreground_fraction': fraction,
            'passes_random_threshold': fraction >= self.config['min_foreground_fraction'],
            'center_distance': self.sampling_record['returned_distance'][candidate_id],
            'sampling': self.sampling_record})


def load_frozen_inputs(config, normalization, initial_ids, test_ids, poses):
    """Read and verify pretraining shared images without touching global RNG."""
    from PIL import Image
    from experiments.bn3d_eval_protocol.run import verify_files
    path = Path(config['frozen_inputs'])
    inputs = json.loads(path.read_text())
    if inputs['normalization'] != normalization:
        raise ValueError('Frozen input normalization differs from training')
    dataset = Path(inputs['dataset'])
    if sha256(dataset / 'manifest.json') != inputs['manifest_sha256']:
        raise ValueError('Frozen test manifest changed')
    manifest = json.loads((dataset / 'manifest.json').read_text())
    verify_files(dataset, manifest)
    expected_k = np.array([[config['camera']['fx'], 0, config['camera']['cx']],
                           [0, config['camera']['fy'], config['camera']['cy']], [0, 0, 1.]])
    np.testing.assert_array_equal(manifest['spec']['intrinsics'], expected_k)
    if manifest['spec']['cycles_samples'] != config['cycles_samples']:
        raise ValueError('Frozen rendering configuration differs from training')

    def images(which, ids):
        rows = inputs[which]
        key = 'id' if which == 'test' else 'frame_id'
        if [r[key] for r in rows] != ids:
            raise ValueError(f'Frozen {which} frame IDs differ')
        result = []
        for row in rows:
            if sha256(Path(row['file'])) != row['sha256']:
                raise ValueError(f'Frozen image modified: {row["file"]}')
            if which != 'test':
                np.testing.assert_allclose(row['pose'], poses[row['frame_id']], atol=1e-7)
            image = np.asarray(Image.open(row['file']).convert('RGBA'))
            if image.shape != (512, 512, 4):
                raise ValueError('Frozen image must be 512x512 RGBA')
            result.append(image)
        return result
    if config.get('test_set') == 'new20':
        views = manifest['views']
        if set(views) != {'new'} or len(views['new']) != 20 or test_ids:
            raise ValueError('new20 must be the only test set; no metadata test split')
        rows = inputs['test']
        if len(rows) != 20:
            raise ValueError('Expected 20 frozen test records')
        for row, view in zip(rows, views['new']):
            if row['pose'] != view['pose'] or Path(row['file']) != dataset / view['file']:
                raise ValueError('Frozen new20 pose/path differs from manifest')
        test_images = images('test', [v['id'] for v in views['new']])
    else:
        test_images = images('old', test_ids)
    return test_images, images('initial', initial_ids)
