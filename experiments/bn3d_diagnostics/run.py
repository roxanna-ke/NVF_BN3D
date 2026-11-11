"""Building1 diagnostics. Imports of CUDA/Blender dependencies are lazy."""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def csv_write(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def load_config(path):
    cfg = read(path)
    if len(cfg['score_seeds']) != 10 or len(set(cfg['score_seeds'])) != 10:
        raise ValueError('Exactly ten distinct scoring seeds required')
    if len(cfg['proposal_seeds']) != 5 or cfg['raw_proposals_per_batch'] != 614:
        raise ValueError('Counterfactual requires 5 x 614 raw poses')
    if cfg['reset_iterations'] != 5000:
        raise ValueError('Branches require reset5000')
    for run in set(cfg['runs'] + cfg['main_pair'] + cfg['branch_runs']):
        if run not in ('nvf/20', 'random/20', 'nvf/50', 'random/50'):
            raise ValueError(f'Unsupported historical run {run}')
    if cfg['branch_runs'] != ['nvf/50']:
        raise ValueError('Branches are restricted to NVF50 (30 fits maximum)')
    if cfg['main_pair'] != ['nvf/50', 'random/50']:
        raise ValueError('Main counterfactual pair must be NVF50/Random50')
    if any(type(n) is not int or n < 3 for n in cfg['states'] + cfg['branch_states']):
        raise ValueError('Invalid observation state')
    for key in ('score_seeds', 'proposal_seeds', 'branch_seeds', 'inference_seeds'):
        if not cfg[key] or any(type(s) is not int or not 0 <= s < 2**32 - 1000 for s in cfg[key]):
            raise ValueError(f'Invalid {key}')
    if sorted(cfg['distance_edges']) != cfg['distance_edges'] or len(set(cfg['distance_edges'])) < 2:
        raise ValueError('Distance edges must increase')
    output = Path(cfg['output_root']).resolve()
    for key in ('history_root', 'frozen_eval_root'):
        root = Path(cfg[key]).resolve()
        if output == root or root in output.parents:
            raise ValueError('Output must be outside historical/frozen roots')
    return cfg


def inventory(cfg):
    """Report every requested artifact; never substitute latest or another budget."""
    rows = []
    for run in cfg['runs']:
        directory = Path(cfg['history_root']) / run / f"seed{cfg['history_seed']}"
        total = int(run.split('/')[1])
        for n in sorted(set(cfg['states'] + cfg['branch_states'])):
            base = ['effective_config.json', 'normalization.json', 'mapper/config/config.yml',
                    'trajectory.json', f'checkpoint_{n:02d}.ckpt']
            missing = [s for s in base if not (directory / s).is_file()]
            candidate = directory / f'candidates_{n:02d}.json'
            obs_missing = [f'observations/{i:02d}.png' for i in range(n)
                           if not (directory / f'observations/{i:02d}.png').is_file()]
            rows.append({'run': run, 'state': n, 'directory': str(directory),
                'status': 'outside_budget' if n > total else ('missing' if missing else 'available'),
                'missing': missing, 'missing_observations': obs_missing,
                'candidates_status': 'available' if candidate.is_file() else (
                    'terminal_no_selection_expected' if n == total else 'missing'),
                'metrics_present': (directory / f'metrics_{n:02d}.json').is_file()})
    return rows


def item_for(row, cfg):
    directory = Path(row['directory'])
    config = read(directory / 'effective_config.json')
    manifest = next((read(parent / 'prepared/manifest.json') for parent in directory.parents
                     if (parent / 'prepared/manifest.json').is_file()), None)
    if manifest is None:
        raise FileNotFoundError(f'No prepared manifest above {directory}')
    k = np.asarray(config.get('intrinsics', manifest['intrinsics']), dtype=float)
    if not np.array_equal(k, np.asarray(manifest['intrinsics'])):
        raise ValueError('Run/manifest intrinsics mismatch')
    camera = config.get('camera', {'width': manifest['width'], 'height': manifest['height'],
        'fx': float(k[0,0]), 'fy': float(k[1,1]), 'cx': float(k[0,2]), 'cy': float(k[1,2])})
    if not np.array_equal(k, [[camera['fx'],0,camera['cx']], [0,camera['fy'],camera['cy']], [0,0,1]]):
        raise ValueError('Camera/intrinsics mismatch')
    config = dict(config, camera=camera)
    if not (camera['width'] == camera['height'] == 512 and
            np.isclose(camera['fx'], 256) and np.isclose(camera['fy'], 256)):
        raise ValueError(f'Not historical FOV90: {directory}')
    trajectory = read(directory / 'trajectory.json')
    n = row['state']
    if len(trajectory) < n:
        raise ValueError(f'Trajectory shorter than state {n}: {directory}')
    split = manifest['splits'][str(cfg['history_seed'])]
    if [v['frame_id'] for v in trajectory[:3]] != split['initial']:
        raise ValueError('Initial frame provenance mismatch')
    return {'run': str(directory), 'config': config, 'manifest': manifest,
            'normalization': read(directory / 'normalization.json'), 'split': split,
            'checkpoint': str(directory / f'checkpoint_{n:02d}.ckpt'),
            'trajectory': trajectory[:n], 'state': n, 'label': row['run']}


def parameters(item):
    c = item['config']
    return SimpleNamespace(method='NVF', d0=c.get('d0', 1.), use_huber=True,
        use_var=True, use_vis=True, mu=.95, use_uniform=False, train_use_tensorboard=False)


def load_reward(item):
    """Strict checkpoint restore, then restore non-state-dict planning settings."""
    import torch
    import yaml
    from eval import set_params
    from nerfstudio.data.scene_box import SceneBox
    from nvf.active_mapping.active_mapping import ActiveMapper
    path = Path(item['checkpoint'])
    before = sha(path)
    state = torch.load(path, map_location='cpu', weights_only=False)
    weights = {k.removeprefix('_model.'): v for k, v in state['pipeline'].items()
               if k.startswith('_model.')}
    yaml_path = Path(item['run']) / 'mapper/config/config.yml'
    model_cfg = yaml.load(yaml_path.read_text(), Loader=yaml.Loader)
    count = weights['field.embedding_appearance.embedding.weight'].shape[0]
    model = model_cfg.pipeline.model.setup(scene_box=SceneBox(aabb=weights['field.aabb']),
        num_train_data=count, metadata={}, device='cuda:0')
    model.load_state_dict(weights, strict=True)
    model.to('cuda:0')
    mapper = ActiveMapper.__new__(ActiveMapper)
    mapper.trainer = SimpleNamespace(pipeline=SimpleNamespace(model=model, device='cuda:0'))
    c = item['config']['camera']
    mapper.train_img_size = (c['height'], c['width'])
    mapper.camera_intrinsics = tuple(c[k] for k in ('fx', 'fy', 'cx', 'cy'))
    mapper.fov = 90.
    set_params(parameters(item), mapper)
    model.renderer_entropy.set_iteration(item['state'] - 3)
    model.train_nvf = False
    model.train()  # train_nvf() leaves the historical model in training mode.
    if sha(path) != before:
        raise ValueError('Checkpoint changed while loading')
    metadata = {'checkpoint': str(path), 'sha256': before, 'bytes': path.stat().st_size,
        'stored_step': state.get('step'), 'observation_state': item['state'],
        'embedding_capacity': count, 'model_config_sha256': sha(yaml_path),
        'effective_config_sha256': sha(Path(item['run']) / 'effective_config.json'),
        'strict_restore': True, 'training_mode': model.training,
        'reward': vars(parameters(item)), 'renderer_iteration': item['state'] - 3,
        'resolution': [300, 300], 'step_note': 'Stored step is not observation count; historical save defaults to zero.'}
    return mapper, metadata


def scene_and_config(item, workspace):
    import torch
    from nvf.env.Scene import BN3DScene
    from nvf.bn3d_data import transform_pose
    from tools.bn3d_yawpitch import environment_config, read_metadata
    c = item['config']
    meta = read_metadata(c['data_root'], c['camera'])
    env = environment_config(c, meta, np.asarray(item['manifest']['intrinsics']))
    scene = BN3DScene(env)
    if not np.allclose(scene.world_to_canonical, item['normalization']['world_to_canonical'], atol=1e-7):
        raise ValueError('Scene normalization mismatch')
    centers = np.array([transform_pose(np.asarray(f['camtoworld']), scene.world_to_canonical)[:3, 3]
        for i, f in enumerate(meta['frames']) if i not in item['split']['test']])
    bounds = torch.tensor(np.stack([centers.min(0), centers.max(0)]), dtype=torch.float32)
    cfg = SimpleNamespace(env=env, model='instant-ngp', sampler='BaseSampler', n_sample=512,
        camera_aabb=bounds, target_aabb=bounds, object_aabb=torch.tensor(scene.get_aabb(), dtype=torch.float32),
        check_density=True, density_threshold=1e-4, horizon=100,
        world_up=c['world_up'], target_height_fraction=c['target_height_fraction'], train_iter=5000,
        config_home=str(workspace / 'mapper'), dataset_path=str(workspace / 'dataset'),
        exp_folder=str(workspace), **{k: v for k, v in vars(parameters(item)).items() if k != 'method'})
    return scene, cfg


def tensors(item):
    import torch
    from nvf.env.utils import pose2tensor
    return torch.stack([torch.as_tensor(pose2tensor(np.asarray(v['pose'])), dtype=torch.float32)
                        for v in item['trajectory']])


class Visibility:
    """Repo mesh/camera convention, with background excluded BEFORE face indexing."""
    def __init__(self, scene, poses):
        self.scene = scene
        import torch
        from nvf.metric.mesh_metrics import VisibilityMeshMetric
        self.metric = VisibilityMeshMetric(scene)
        self.seen = np.zeros(len(scene.mesh.faces), dtype=bool)
        self.areas = np.asarray(scene.mesh.area_faces)
        for pose in poses:
            faces, _ = self.raster(pose)
            self.seen[np.unique(faces[faces >= 0])] = True

    def raster(self, pose):
        import torch
        from nvf.active_mapping.mapping_utils import to_transform
        from pytorch3d.renderer import PerspectiveCameras, MeshRasterizer, RasterizationSettings
        m = self.metric
        transform = to_transform(pose).to(m.device).float()
        rotation = transform[:3, :3] @ torch.diag(torch.tensor([-1., 1., -1.], device=m.device))
        translation = -rotation.T @ transform[:3, 3]
        scale = 300 / m.W
        camera = PerspectiveCameras(R=rotation[None], T=translation[None],
            focal_length=((m.fx * scale, m.fy * scale),),
            principal_point=((m.cx * scale, m.cy * scale),), image_size=((300, 300),),
            in_ndc=False, device=m.device)
        fragments = MeshRasterizer(camera, RasterizationSettings(image_size=(300, 300),
            blur_radius=0., faces_per_pixel=1))(m.mesh)
        return (fragments.pix_to_face[0, ..., 0].cpu().numpy(),
                fragments.zbuf[0, ..., 0].cpu().numpy())

    def maps(self, pose):
        faces, depth = self.raster(pose)
        fg = faces >= 0
        unseen = np.zeros_like(fg)
        unseen[fg] = ~self.seen[faces[fg]]
        new_faces = np.unique(faces[unseen])
        import trimesh
        position = pose.detach().cpu().numpy()[-3:][None]
        try:
            _, surface_distance, _ = trimesh.proximity.closest_point(self.scene.mesh, position)
        except (ImportError, ModuleNotFoundError):
            _, surface_distance, _ = trimesh.proximity.closest_point_naive(self.scene.mesh, position)
        return {'faces': faces, 'foreground': fg, 'unseen': unseen, 'depth': depth}, {
            'surface_distance': float(surface_distance[0]),
            'center_distance': float(np.linalg.norm(position[0] - np.asarray(self.scene.get_aabb()).mean(0))),
            'foreground_fraction': float(fg.mean()), 'new_faces': len(new_faces),
            'new_area': float(self.areas[new_faces].sum())}


def decomposition(entropy, maps, edges):
    result = {'score': float(entropy.mean())}
    masks = {'fg': maps['foreground'], 'bg': ~maps['foreground'], 'unseen': maps['unseen']}
    for lo, hi in zip(edges, edges[1:] + [float('inf')]):
        masks[f'depth_{lo}_{hi}'] = maps['foreground'] & (maps['depth'] >= lo) & (maps['depth'] < hi)
    for name, mask in masks.items():
        result[name + '_pixels'] = int(mask.sum())
        result[name + '_mean'] = float(entropy[mask].mean()) if mask.any() else None
        result[name + '_contribution'] = float(entropy[mask].sum() / entropy.size)
    return result


def score_fixed(mapper, poses, visibility, cfg, dest):
    import torch
    from nvf.env.utils import set_seed
    dest.mkdir(parents=True, exist_ok=False)
    np.save(dest / 'poses.npy', poses.cpu().numpy())
    rows, geometry = [], []
    # Disk-backed maps avoid holding candidates x seeds x 300 x 300 in RAM.
    maps_array = np.lib.format.open_memmap(dest / 'entropy_repeat0.npy', mode='w+', dtype='float32',
        shape=(len(poses), 300, 300))
    for i, pose in enumerate(poses):
        maps, stats = visibility.maps(pose)
        geometry.append(stats)
        np.savez_compressed(dest / f'gt_{i:04d}.npz', **maps)
    scores = np.empty((len(cfg['score_seeds']), len(poses)))
    for repeat, seed in enumerate(cfg['score_seeds']):
        set_seed(seed)
        for i, pose in enumerate(poses):
            with torch.no_grad():
                entropy = mapper.get_cost(pose[None, None], return_image=True).reshape(300, 300).cpu().numpy()
            if not np.isfinite(entropy).all():
                raise ValueError('Nonfinite NVF entropy')
            if repeat == 0:
                maps_array[i] = entropy
            with np.load(dest / f'gt_{i:04d}.npz') as gt:
                stats = decomposition(entropy, gt, cfg['distance_edges'])
                if repeat == 0:
                    stats.update(binary_metrics(entropy[gt['foreground']], gt['unseen'][gt['foreground']]))
            scores[repeat, i] = stats['score']
            rows.append({'repeat': repeat, 'seed': seed, 'candidate': i, **geometry[i], **stats})
        maps_array.flush()
        print(f'{dest}: scoring seed {seed} complete', flush=True)
    csv_write(dest / 'scores.csv', rows)
    means = scores.mean(0)
    summary = [{'candidate': i, 'mean': float(means[i]), 'std': float(scores[:, i].std()),
        'wins': int((scores.argmax(1) == i).sum()), **geometry[i]} for i in range(len(poses))]
    csv_write(dest / 'summary.csv', summary)
    np.save(dest / 'scores.npy', scores)
    write(dest / 'geometry.json', geometry)
    write(dest / 'stability.json', rank_stability(scores))
    analysis_summaries(rows, summary, cfg, dest)
    return scores, geometry


def historical_candidates(item):
    import torch
    path = Path(item['run']) / f"candidates_{item['state']:02d}.json"
    if not path.exists():
        return None
    record = read(path)
    poses = [p for batch in record['batches'] for p in batch['poses']]
    if not poses:
        raise ValueError(f'Empty recorded candidate set: {path}')
    return torch.tensor(poses, dtype=torch.float32)


def counterfactual(items, cfg, dest):
    import torch
    from nvf.active_mapping.agents.bn3d_yawpitch_sampler import BN3DYawPitchSampler
    from nvf.env.utils import set_seed
    from nvf.visibility.visibility import get_density
    for key in ('normalization', 'split'):
        if items[0][key] != items[1][key]:
            raise ValueError(f'Counterfactual mismatch: {key}')
    for key in ('intrinsics', 'asset_sha256', 'metadata_sha256'):
        if items[0]['manifest'][key] != items[1]['manifest'][key]:
            raise ValueError(f'Counterfactual mismatch: {key}')
    if any(items[0]['config'][k] != items[1]['config'][k]
           for k in ('world_up', 'target_height_fraction')):
        raise ValueError('Counterfactual proposal laws differ')
    scene, agent_cfg = scene_and_config(items[0], dest)
    sampler = BN3DYawPitchSampler(agent_cfg)
    raw = []
    for seed in cfg['proposal_seeds']:
        set_seed(seed)
        raw.append(sampler._sample_poses(614))  # deliberately bypass learned density filtering
    poses = torch.cat(raw)
    dest.mkdir(parents=True, exist_ok=False)
    np.save(dest / 'raw_poses.npy', poses.numpy())
    rows = []
    masks = []
    for item in items:
        mapper, metadata = load_reward(item)
        write(dest / (item['label'].replace('/', '_') + '_checkpoint.json'), metadata)
        with torch.no_grad():
            density = get_density(poses[:, -3:].cuda(), mapper.trainer.pipeline.model).reshape(-1).cpu().numpy()
        valid = np.isfinite(density) & (density < 1e-4)
        masks.append(valid)
        for i, d in enumerate(density):
            rows.append({'model': item['label'], 'candidate': i, 'batch': i // 614,
                'seed': cfg['proposal_seeds'][i // 614], 'density': float(d) if np.isfinite(d) else None,
                'accepted': bool(valid[i])})
        del mapper
        gc.collect()
        torch.cuda.empty_cache()
    csv_write(dest / 'density.csv', rows)
    write(dest / 'overlap.json', {'both': int((masks[0] & masks[1]).sum()),
        'nvf_only': int((masks[0] & ~masks[1]).sum()), 'random_only': int((~masks[0] & masks[1]).sum()),
        'neither': int((~masks[0] & ~masks[1]).sum()),
        'per_batch_retained_first512': [[int(min(512, m[j:j+614].sum())) for j in range(0,3070,614)] for m in masks]})


def analyze(cfg, inventory_rows, root):
    import torch
    available = {(r['run'], r['state']): r for r in inventory_rows if r['status'] == 'available'}
    for n in cfg['states']:
        pair = [available.get((run, n)) for run in cfg['main_pair']]
        if all(pair):
            counterfactual([item_for(r, cfg) for r in pair], cfg, root / 'density' / f'state{n:02d}')
        for run in cfg['runs']:
            row = available.get((run, n))
            if row is None:
                continue
            item = item_for(row, cfg)
            dest = root / 'history' / run / f'state{n:02d}'
            dest.mkdir(parents=True, exist_ok=False)
            mapper, metadata = load_reward(item)
            write(dest / 'checkpoint.json', metadata)
            poses = historical_candidates(item)
            if poses is None:
                write(dest / 'missing_candidates.json', {'reason': row['candidates_status'],
                    'note': 'No substitute candidate set used for historical repeated scoring.'})
            else:
                scene, _ = scene_and_config(item, dest)
                vis = Visibility(scene, tensors(item))
                scores, geometry = score_fixed(mapper, poses, vis, cfg, dest / 'fixed_candidates')
                candidate_previews(mapper, poses, scores, geometry, vis, item, cfg, dest / 'previews')
                fixed = dest / 'fixed_candidates'
                write(fixed / 'complete.json', {'files': {p.name: sha(p) for p in fixed.iterdir() if p.is_file()},
                    'trajectory_sha256': sha(Path(item['run']) / 'trajectory.json'),
                    'candidates_sha256': sha(Path(item['run']) / f'candidates_{n:02d}.json')})
                del vis, scene
            del mapper
            gc.collect()
            torch.cuda.empty_cache()



def candidate_previews(mapper, poses, scores, geometry, visibility, item, cfg, dest):
    """Selected views plus a radial sweep; d0 ablation uses paired ray RNG seeds."""
    import torch
    from PIL import Image, ImageDraw
    from nvf.active_mapping.mapping_utils import to_transform
    from nvf.env.utils import set_seed
    dest.mkdir(parents=True, exist_ok=False)
    center = np.asarray(visibility.scene.get_aabb()).mean(0)
    choices, _ = branch_choices(scores, geometry, poses.numpy(), center,
        cfg['history_seed'] + item['state'], cfg['distance_match_tolerance'])
    views = [(name, poses[index]) for name,index in choices.items() if index is not None]
    record = read(Path(item['run']) / f"candidates_{item['state']:02d}.json")
    offset = sum(len(b['poses']) for b in record['batches'][:record['chosen_batch']])
    views.append(('historical_selected', poses[offset + record['chosen']]))
    top = poses[choices['top']]
    for scale in (.75, 1., 1.25, 1.5):
        radial = top.clone()
        radial[-3:] = torch.tensor(center + scale * (top[-3:].numpy() - center), dtype=radial.dtype)
        views.append((f'radial_{scale}', radial))  # orientation held fixed, no density rejection
    model = mapper.trainer.pipeline.model
    renderer = model.renderer_entropy
    rows = []
    for name, pose in views:
        maps, stats = visibility.maps(pose)
        camera = mapper.camera_from_transform(to_transform(pose), 300, 300, 'cuda:0')
        seed = cfg['score_seeds'][0]
        with torch.no_grad():
            set_seed(seed)
            entropy = mapper.get_cost(pose[None,None], return_image=True).reshape(300,300).cpu().numpy()
            d0 = renderer.d0
            try:
                renderer.d0 = 0.
                set_seed(seed)
                unweighted = mapper.get_cost(pose[None,None], return_image=True).reshape(300,300).cpu().numpy()
            finally:
                renderer.d0 = d0
            nvf, entropy_flag = model.use_nvf, model.calculate_entropy
            try:
                model.use_nvf = False
                model.calculate_entropy = False
                set_seed(seed)
                outputs = model.get_outputs_for_camera_ray_bundle(camera.generate_rays(camera_indices=0, aabb_box=None))
                rgb = outputs['rgb'].clamp(0,1).cpu().numpy()
                depth = outputs['depth'].squeeze().cpu().numpy()
                opacity = outputs['accumulation'].squeeze().cpu().numpy()
            finally:
                model.use_nvf, model.calculate_entropy = nvf, entropy_flag
        if not all(np.isfinite(a).all() for a in (entropy, unweighted, rgb, depth, opacity)):
            raise ValueError('Nonfinite preview output')
        np.savez_compressed(dest / f'{name}.npz', pose=pose.numpy(), entropy=entropy,
            entropy_d0_zero=unweighted, distance_effect=entropy-unweighted,
            rgb=rgb, predicted_depth=depth, opacity=opacity, **maps)
        rows.append({'view': name, **stats, **decomposition(entropy, maps, cfg['distance_edges']),
            'd0_zero_score': float(unweighted.mean()), 'distance_effect': float((entropy-unweighted).mean())})

        def heat(a, mask=None):
            mask = np.isfinite(a) if mask is None else mask & np.isfinite(a)
            values = a[mask]
            lo, hi = np.percentile(values, [1,99]) if values.size else (0.,1.)
            x = np.clip((a-lo)/max(float(hi-lo), 1e-12), 0,1)
            color = np.stack([x, 1-np.abs(2*x-1), 1-x], axis=-1)
            color[~mask] = 0
            return (color*255).astype(np.uint8)
        panels = [('RGB', (rgb*255).astype(np.uint8)), ('predicted depth', heat(depth)),
            ('opacity', heat(opacity)), ('NVF entropy', heat(entropy)),
            ('GT depth (foreground)', heat(maps['depth'], maps['foreground'])),
            ('unseen GT faces', np.repeat(maps['unseen'][...,None],3,axis=-1).astype(np.uint8)*255),
            ('NVF d0=0', heat(unweighted)), ('distance effect', heat(entropy-unweighted))]
        canvas = Image.new('RGB', (1200,640), 'white')
        draw = ImageDraw.Draw(canvas)
        for i,(label,array) in enumerate(panels):
            x,y = (i%4)*300, (i//4)*320
            draw.text((x+4,y+3), label, fill='black')
            canvas.paste(Image.fromarray(array), (x,y+20))
        canvas.save(dest / f'{name}.png')
    csv_write(dest / 'radial_and_selected.csv', rows)
    write(dest / 'display.json', {'heatmap_scaling': 'per-panel foreground/finite 1st–99th percentile; compare NPZ numeric values',
        'radial': 'scale translation about canonical AABB center; hold orientation fixed; unfiltered diagnostic poses',
        'distance_ablation': 'd0=0 paired torch seed; all other renderer parameters unchanged'})

def frozen_dataset(item, cfg):
    from experiments.bn3d_eval_protocol.run import verify_files
    paths = ([Path(cfg['frozen_directory']) / 'manifest.json'] if cfg.get('frozen_directory') else
             sorted((Path(cfg['frozen_eval_root']) / '_frozen').glob('*/manifest.json')))
    matches = []
    for path in paths:
        manifest = read(path)
        spec = manifest['spec']
        if (spec['intrinsics'] == item['manifest']['intrinsics'] and
            spec['normalization'] == item['normalization'] and
            spec['assets'] == item['manifest']['asset_sha256'] and
            spec['metadata_sha256'] == item['manifest']['metadata_sha256'] and
            spec['old_frame_ids'] == item['split']['test'] and
            spec['cycles_samples'] == item['config']['cycles_samples']):
            matches.append((path.parent, manifest))
    if len(matches) != 1:
        raise ValueError(f'Expected one matching existing frozen dataset, found {len(matches)}; '
                         'set frozen_directory explicitly if ambiguous. Never regenerates GT.')
    directory, manifest = matches[0]
    verify_files(directory, manifest)
    if {k: len(v) for k, v in manifest['views'].items()} != {'old': 40, 'new': 20}:
        raise ValueError('Expected frozen old40/new20 dataset')
    return directory, manifest


def evaluate_branch(model, frozen, manifest, cfg, dest):
    """Use existing protocol metric helpers and identical camera/frame-seed semantics."""
    import torch
    from PIL import Image
    from nvf.env.utils import set_seed
    from nerfstudio.cameras.cameras import Cameras, CameraType
    from experiments.bn3d_eval_protocol.run import score_images, aggregate
    dest.mkdir(parents=True, exist_ok=False)
    k = np.asarray(manifest['spec']['intrinsics'])
    model.use_nvf = False
    model.calculate_entropy = False
    summaries = []
    for mode in cfg['eval_modes']:
        model.train(mode == 'train')
        for which, views in manifest['views'].items():
            for seed in cfg['inference_seeds']:
                rows = []
                for index, view in enumerate(views):
                    camera = Cameras(fx=float(k[0,0]), fy=float(k[1,1]), cx=float(k[0,2]), cy=float(k[1,2]),
                        width=512, height=512, camera_type=CameraType.PERSPECTIVE,
                        camera_to_worlds=torch.tensor(np.asarray(view['pose'])[:3,:4], dtype=torch.float32)[None]).to('cuda:0')
                    set_seed(seed + view.get('original_frame_id', index))
                    with torch.no_grad():
                        rgb = model.get_outputs_for_camera_ray_bundle(camera.generate_rays(camera_indices=0, aabb_box=None))['rgb']
                        if not torch.isfinite(rgb).all():
                            raise ValueError('Nonfinite branch RGB')
                        rgba = np.asarray(Image.open(frozen / view['file']).convert('RGBA'))
                        stats = score_images(model, rgb.clamp(0,1), rgba)
                    rows.append({'view': view['id'], **stats})
                csv_write(dest / f'{which}_{mode}_{seed}.csv', rows)
                summaries.append({'test_set': which, 'mode': mode, 'seed': seed, **aggregate(rows)})
    csv_write(dest / 'aggregate.csv', summaries)
    write(dest / 'dataset.json', {'directory': str(frozen), 'manifest_sha256': sha(frozen / 'manifest.json'),
        'helper': 'experiments.bn3d_eval_protocol.run.score_images/aggregate',
        'inference_seeds': cfg['inference_seeds'], 'modes': cfg['eval_modes']})
    return summaries


def branch_choices(scores, geometry, poses, center, seed, tolerance):
    top = int(scores.mean(0).argmax())
    rng = np.random.default_rng(seed)
    distances = np.array([g['surface_distance'] for g in geometry])
    alternatives = np.flatnonzero(np.arange(len(poses)) != top)
    if not len(alternatives):
        raise ValueError('At least two branch candidates required')
    delta = np.abs(distances - distances[top])
    absolute_tolerance = tolerance * distances[top]
    matched = alternatives[delta[alternatives] <= absolute_tolerance]
    fallback = not len(matched)
    if fallback:
        matched = alternatives[np.isclose(delta[alternatives], delta[alternatives].min())]
    choices = {'D': None, 'top': top, 'random': int(rng.integers(len(poses))),
        'distance_matched': int(rng.choice(matched)),
        'oracle': int(np.argmax([g['new_area'] for g in geometry]))}
    return choices, {'distance_definition': 'minimum Euclidean camera-to-GT-triangle-surface',
        'relative_tolerance': tolerance, 'absolute_tolerance': float(absolute_tolerance), 'nearest_fallback': fallback,
        'distances': {k: None if v is None else float(distances[v]) for k,v in choices.items()},
        'oracle_definition': 'maximum newly visible GT triangle area within fixed candidate set'}


def branches(cfg, inventory_rows, root):
    import torch
    import yaml
    from PIL import Image
    from eval import set_params
    from nvf.active_mapping.agents.BaseAgent import BaseAgent
    from nvf.active_mapping.mapping_utils import to_transform
    from nvf.env.utils import set_seed
    analysis = Path(cfg['output_root']) / 'analyze'
    if not (analysis / 'complete.json').is_file():
        raise FileNotFoundError('branches requires a completed analyze command')
    if read(analysis / 'effective_config.json') != cfg:
        raise ValueError('Analysis/config mismatch; use the same config for analyze and branches')
    requested = [r for r in inventory_rows if r['run'] in cfg['branch_runs'] and r['state'] in cfg['branch_states']]
    unavailable = [r for r in requested if r['status'] != 'available' or r['missing_observations'] or r['candidates_status'] != 'available']
    if unavailable or len(requested) != len(cfg['branch_states']):
        write(root / 'unavailable_branches.json', unavailable)
        raise FileNotFoundError('Required branch history unavailable; no partial branch success')
    all_results = []
    for row in requested:
        if row['run'] not in cfg['branch_runs'] or row['state'] not in cfg['branch_states']:
            continue
        if row['status'] != 'available' or row['missing_observations']:
            continue
        item = item_for(row, cfg)
        frozen, manifest = frozen_dataset(item, cfg)
        dest = root / row['run'] / f"state{row['state']:02d}"
        dest.mkdir(parents=True, exist_ok=False)
        poses = historical_candidates(item)
        if poses is None:
            write(dest / 'unavailable.json', {'reason': 'No recorded candidate set for branches'})
            continue
        scene, agent_cfg = scene_and_config(item, dest)
        history_poses = tensors(item)
        source = analysis / 'history' / row['run'] / f"state{row['state']:02d}"
        metadata = read(source / 'checkpoint.json')
        if metadata['sha256'] != sha(item['checkpoint']):
            raise ValueError('Historical checkpoint changed since analysis')
        selection = source / 'fixed_candidates'
        integrity = read(selection / 'complete.json')
        for name, expected in integrity['files'].items():
            if sha(selection / name) != expected:
                raise ValueError(f'Analysis selection artifact changed: {name}')
        if integrity['trajectory_sha256'] != sha(Path(item['run']) / 'trajectory.json') or integrity['candidates_sha256'] != sha(Path(item['run']) / f"candidates_{item['state']:02d}.json"):
            raise ValueError('Historical trajectory/candidates changed since analyze')
        if not np.array_equal(np.load(selection / 'poses.npy'), poses.numpy()):
            raise ValueError('Analysis/historical candidates differ')
        scores = np.load(selection / 'scores.npy')
        geometry = read(selection / 'geometry.json')
        write(dest / 'source_checkpoint.json', metadata)
        write(dest / 'analysis_source.json', {'directory': str(selection), 'complete_sha256': sha(selection / 'complete.json')})
        images = [np.asarray(Image.open(Path(item['run']) / f'observations/{i:02d}.png').convert('RGBA'))
                  for i in range(item['state'])]
        source_hashes = {f'observations/{i:02d}.png': sha(Path(item['run']) / f'observations/{i:02d}.png')
                         for i in range(item['state'])}
        source_hashes['trajectory.json'] = sha(Path(item['run']) / 'trajectory.json')
        write(dest / 'training_sources.json', source_hashes)
        center = np.asarray(scene.get_aabb()).mean(0)
        rendered = {}
        choices, selection_meta = branch_choices(scores, geometry, poses.numpy(), center,
            cfg['history_seed'] + item['state'], cfg['distance_match_tolerance'])
        write(dest / 'selection.json', {'choices': choices, 'selection_seed': cfg['history_seed'] + item['state'], **selection_meta})
        for seed in cfg['branch_seeds']:
            fitted = {}
            for name, selected in choices.items():
                branch = dest / f'seed{seed}' / name
                branch.mkdir(parents=True, exist_ok=False)
                if selected in fitted:
                    original, results = fitted[selected]
                    write(branch / 'alias.json', {'candidate': selected, 'alias_of': str(original), 'reason': 'Identical D+pose with same training seed'})
                    all_results.extend({'run': row['run'], 'state': row['state'], 'training_seed': seed, 'branch': name, **r} for r in results)
                    csv_write(root / 'branch_metrics.csv', all_results)
                    continue
                branch_images, branch_poses = list(images), list(history_poses)
                if selected is not None:
                    if selected not in rendered:
                        set_seed(cfg['history_seed'] * 10000 + selected)
                        rendered[selected] = scene.render_pose(to_transform(poses[selected]).numpy())
                    branch_images.append(rendered[selected])
                    branch_poses.append(poses[selected])
                    Image.fromarray(rendered[selected]).save(branch / 'added_observation.png')
                local_cfg = SimpleNamespace(**vars(agent_cfg))
                local_cfg.config_home = str(branch / 'mapper')
                local_cfg.dataset_path = str(branch / 'dataset')
                local_cfg.exp_folder = str(branch)
                (branch / 'mapper').mkdir()
                set_seed(seed)
                agent = BaseAgent(local_cfg)
                agent.pipeline.reset()
                agent.pipeline.config.max_num_iterations = 5000
                agent.pipeline.trainer.config.max_num_iterations = 5000
                agent.pipeline.trainer.config.nvf_num_iterations = item['config'].get('nvf_iterations', 500)
                agent.pipeline.trainer.config.nvf_batch_size = item['config'].get('nvf_batch_size', 65536)
                set_params(parameters(item), agent.pipeline)
                agent.pipeline.trainer.pipeline.model.renderer_entropy.set_iteration(len(branch_images) - 3)
                # Save actual effective trainer config (BaseAgent writes before these overrides).
                Path(agent.pipeline.config_pth).write_text(yaml.dump(agent.pipeline.config))
                agent.step = 0
                agent.process_obs(branch_images, branch_poses)
                checkpoint = branch / 'checkpoint.ckpt'
                agent.pipeline.save_ckpt(str(checkpoint), step=4999)
                identity = {'source': metadata, 'branch': name, 'seed': seed, 'candidate': selected,
                    'observations': len(branch_images), 'reset_iterations': 5000,
                    'checkpoint_sha256': sha(checkpoint), 'dataset_sha256': sha(frozen / 'manifest.json'),
                    'training': 'fresh BaseAgent + reset, full D or D+one, RGB5000 then configured NVF training'}
                write(branch / 'checkpoint_metadata.json', identity)
                results = evaluate_branch(agent.pipeline.trainer.pipeline.model, frozen, manifest, cfg, branch / 'evaluation')
                fitted[selected] = (branch, results)
                all_results.extend({'run': row['run'], 'state': row['state'], 'training_seed': seed,
                    'branch': name, **r} for r in results)
                csv_write(root / 'branch_metrics.csv', all_results)
                write(branch / 'complete.json', identity)
                del agent
                gc.collect()
                torch.cuda.empty_cache()
                print(f'Finished {branch}', flush=True)
        del scene
    baseline = {(r['run'],r['state'],r['training_seed'],r['test_set'],r['mode'],r['seed']):r
                for r in all_results if r['branch'] == 'D'}
    deltas = []
    for r in all_results:
        b = baseline[(r['run'],r['state'],r['training_seed'],r['test_set'],r['mode'],r['seed'])]
        deltas.append({**r, **{f'delta_{k}': r[k]-b[k] for k in
            ('full_mse','fg_mse','full_psnr','fg_psnr','full_ssim','fg_ssim','full_lpips','fg_lpips_spatial')}})
    for r in deltas:
        r['fg_mse_decrease'] = -r['delta_fg_mse']
        r['full_mse_decrease'] = -r['delta_full_mse']
    csv_write(root / 'branch_deltas.csv', deltas)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('analyze', 'branches'))
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    root = Path(cfg['output_root']) / args.command
    root.mkdir(parents=True, exist_ok=False)  # never mix partial or incompatible executions
    write(root / 'effective_config.json', cfg)
    rows = inventory(cfg)
    write(root / 'history_inventory.json', rows)
    write(root / 'code.json', {p.name: sha(p) for p in Path(__file__).parent.glob('*.py')})
    try:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError('GPU CLI requires CUDA; no CPU training/rendering fallback')
        from experiments.bn3d_eval_protocol.run import runtime_info
        write(root / 'runtime.json', runtime_info())
        if not any(r['status'] == 'available' for r in rows):
            raise FileNotFoundError('No requested checkpoints available; see history_inventory.json')
        (analyze if args.command == 'analyze' else branches)(cfg, rows, root)
        write(root / 'complete.json', {'status': 'finished_available_history',
            'missing_history': any(r['status'] == 'missing' for r in rows)})
    except Exception as error:
        write(root / 'failure.json', {'type': type(error).__name__, 'message': str(error)})
        raise


if __name__ == '__main__':
    main()
