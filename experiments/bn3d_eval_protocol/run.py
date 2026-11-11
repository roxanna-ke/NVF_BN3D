"""Post-hoc 2x2 protocol comparison. Heavy dependencies are imported on demand."""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace

import numpy as np
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if not __package__:
    raise SystemExit('Run from the repository root: python -m experiments.bn3d_eval_protocol.run <command>')

from .sampling import generate_views
from nvf.bn3d_data import sha256, transform_pose


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def code_digest():
    files = [HERE / 'run.py', HERE / 'sampling.py',
             ROOT / 'nerfstudio/models/instant_ngp.py',
             ROOT / 'nerfstudio/fields/nerfacto_field.py',
             ROOT / 'nerfstudio/model_components/ray_samplers.py',
             ROOT / 'nerfstudio/model_components/renderers.py',
             ROOT / 'nvf/env/BlenderRenderer.py', ROOT / 'nvf/env/Scene.py',
             ROOT / 'nvf/bn3d_data.py']
    return {str(p.relative_to(ROOT)): sha256(p) for p in files}


def runtime_info():
    import torch
    versions = {}
    for package in ['numpy', 'Pillow', 'torchmetrics', 'nerfacc', 'PyYAML']:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = 'unknown'
    return {'python': sys.version, 'torch': str(torch.__version__),
            'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(0),
            'packages': versions}


def load_config(path):
    cfg = read(path)
    if not cfg['comparisons'] or len(set(g['id'] for g in cfg['comparisons'])) != len(cfg['comparisons']):
        raise ValueError('Comparison IDs must be unique')
    for group in cfg['comparisons']:
        if not group['id'].replace('_', '').isalnum() or group['checkpoint_observations'] < 3:
            raise ValueError('Invalid comparison ID or observation count')
    if set(cfg['modes']) != {'train', 'eval'} or len(cfg['modes']) != 2:
        raise ValueError('This experiment requires both train and eval modes')
    if not cfg['inference_seeds'] or len(set(cfg['inference_seeds'])) != len(cfg['inference_seeds']):
        raise ValueError('Require unique inference seeds')
    if any(type(s) is not int or not 0 <= s < 2 ** 32 - 200 for s in cfg['inference_seeds']):
        raise ValueError('Inference seed must allow base + frame ID within NumPy seed range')
    return cfg


def prepared(run):
    for parent in Path(run).parents:
        path = parent / 'prepared/manifest.json'
        if path.is_file():
            return read(path)
    raise FileNotFoundError(f'No prepared manifest above {run}')


def pair_inputs(group, cfg):
    pair = []
    for policy in ['nvf', 'random']:
        run = Path(group[f'{policy}_run'])
        config = read(run / 'effective_config.json')
        manifest = prepared(run)
        normalization = read(run / 'normalization.json')
        n = group['checkpoint_observations']
        ckpt = run / f'checkpoint_{n:02d}.ckpt'
        # Never guess "latest" or accidentally compare Random24 to NVF50.
        for required in [ckpt, run / f'metrics_{n:02d}.json', run / 'mapper/config/config.yml']:
            if not required.is_file():
                raise FileNotFoundError(required)
        split = manifest['splits'][str(cfg['seed'])]
        item = {'run': str(run), 'config': config, 'manifest': manifest,
                'normalization': normalization, 'split': split, 'checkpoint': str(ckpt)}
        pair.append(item)
    a, b = pair
    for key in ['mesh_path', 'data_root', 'cycles_samples']:
        if a['config'][key] != b['config'][key]:
            raise ValueError(f'Policy inputs differ: {key}')
    for key in ['intrinsics', 'mesh_sha256', 'asset_sha256', 'metadata_sha256']:
        if a['manifest'][key] != b['manifest'][key]:
            raise ValueError(f'Policy manifests differ: {key}')
    if a['normalization'] != b['normalization'] or a['split'] != b['split']:
        raise ValueError('Policy normalization/splits differ')
    if len(a['split']['test']) != 40:
        raise ValueError('Expected 40 old test views')
    for item in pair:
        trajectory = read(Path(item['run']) / 'trajectory.json')
        ids = [r['frame_id'] for r in trajectory[:3]]
        if ids != item['split']['initial']:
            raise ValueError('Initial pose provenance mismatch')
    return pair


def freeze_spec(item, cfg):
    meta_path = Path(item['config']['data_root']) / 'meta_data.json'
    if sha256(meta_path) != item['manifest']['metadata_sha256']:
        raise ValueError('Metadata hash differs from training')
    mesh = Path(item['config']['mesh_path'])
    for name, value in item['manifest']['asset_sha256'].items():
        if sha256(mesh.parent / name) != value:
            raise ValueError(f'Asset changed: {name}')
    meta = read(meta_path)
    k = item['manifest']['intrinsics']
    matrix = np.asarray(item['normalization']['world_to_canonical'])
    old_poses = [transform_pose(np.asarray(meta['frames'][i]['camtoworld']), matrix).tolist()
                 for i in item['split']['test']]
    return {'version': 1, 'width': 512, 'height': 512, 'intrinsics': k,
            'normalization': item['normalization'], 'old_frame_ids': item['split']['test'],
            'old_poses': old_poses, 'assets': item['manifest']['asset_sha256'],
            'metadata_sha256': item['manifest']['metadata_sha256'],
            'cycles_samples': item['config']['cycles_samples'], 'seed': cfg['seed'],
            'sampling': cfg['sampling'], 'code': code_digest()}


def dataset_directory(item, cfg):
    spec = freeze_spec(item, cfg)
    return Path(cfg['output_root']) / '_frozen' / digest(spec), spec


def verify_files(directory, manifest):
    for name, expected in manifest['files'].items():
        if sha256(directory / name) != expected:
            raise ValueError(f'Frozen/completed artifact modified: {directory / name}')


def freeze(group, cfg):
    item = pair_inputs(group, cfg)[0]
    freeze_item(item, cfg)


def freeze_item(item, cfg):
    dest, spec = dataset_directory(item, cfg)
    if dest.exists():
        manifest = read(dest / 'manifest.json')
        if manifest['spec'] != spec:
            raise ValueError('Frozen spec mismatch')
        verify_files(dest, manifest)
        print(f'Reusing verified frozen GT: {dest}', flush=True)
        return
    from nvf.env.Scene import BN3DScene
    from nvf.env.utils import set_seed
    import bpy
    k = np.asarray(spec['intrinsics'])
    env = SimpleNamespace(resolution=(512, 512),
        fov=float(np.degrees(2 * np.arctan(512 / (2 * k[0, 0])))),
        fx=k[0, 0], fy=k[1, 1], cx=k[0, 2], cy=k[1, 2],
        mesh_path=item['config']['mesh_path'], cycles=True,
        cycles_samples=spec['cycles_samples'], resolution_percentage=100, gpu=True,
        scene='BN3DScene')
    set_seed(cfg['seed'])
    scene = BN3DScene(env)
    if not np.allclose(scene.world_to_canonical, spec['normalization']['world_to_canonical'], atol=1e-7):
        raise ValueError('Blender normalization differs from training')
    vertices = np.asarray(scene.mesh.vertices)
    # A canonical Z-up ground plane; by default mesh AABB minimum Z.
    ground = float(vertices[:, 2].min()) if cfg['sampling']['ground_z'] is None else cfg['sampling']['ground_z']
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.freeze-', dir=dest.parent) as temp:
        stage = Path(temp) / 'dataset'
        stage.mkdir()
        records = {'old': [], 'new': []}

        def save(which, record, image):
            if image.shape != (512, 512, 4) or image.dtype != np.uint8:
                raise ValueError('Expected Blender uint8 RGBA 512x512')
            path = stage / which / f'{record["id"]}.png'
            path.parent.mkdir(exist_ok=True)
            Image.fromarray(image).save(path)
            record['file'] = str(path.relative_to(stage))
            record['foreground_fraction'] = float((image[..., 3] > 0).mean())
            if record['foreground_fraction'] <= 0:
                raise ValueError('Empty GT')
            records[which].append(record)

        # Reuse archived old GT when available, otherwise regenerate once and
        # freeze. RGB was not saved during original runs in every experiment.
        old_cache = Path(item['config']['output_root']) / 'foreground_eval/reference'
        for frame_id, pose in zip(spec['old_frame_ids'], spec['old_poses']):
            cache = old_cache / f'frame_{frame_id:03d}.png'
            if cache.exists():
                image = np.asarray(Image.open(cache).convert('RGBA'))
                origin = str(cache)
            else:
                set_seed(cfg['seed'] * 10000 + frame_id)
                image = scene.render_pose(np.asarray(pose))
                origin = 'regenerated_once_from_original_assets_pose_and_K'
            save('old', {'id': f'frame_{frame_id:03d}', 'pose': pose,
                         'original_frame_id': frame_id, 'origin': origin}, image)
        old_fractions = Path(item['run']) / 'reference_foreground.json'
        if old_fractions.exists() and not np.array_equal(
                [v['foreground_fraction'] for v in records['old']], read(old_fractions)):
            raise ValueError('Old GT silhouettes differ from original records')

        def render(pose):
            set_seed(cfg['seed'])
            return scene.render_pose(pose)
        for view in generate_views(vertices, k, cfg['sampling'], render=render):
            image = view.pop('rgba')
            save('new', view, image)
        assert len(records['new']) == 20
        # Do not accidentally reuse an initial training camera as a test view.
        initial = read(Path(item['run']) / 'trajectory.json')[:3]
        for view in records['new']:
            if any(np.allclose(view['pose'], row['pose'], atol=1e-5) for row in initial):
                raise ValueError('New test pose equals initial training pose')
        manifest = {'spec': spec, 'ground_z': ground, 'views': records,
                    'runtime': {**runtime_info(), 'blender': bpy.app.version_string},
                    'files': {str(f.relative_to(stage)): sha256(f) for f in sorted(stage.rglob('*.png'))}}
        write(stage / 'manifest.json', manifest)
        stage.rename(dest)
    print(f'Frozen old40/new20 GT: {dest}', flush=True)


def load_model(item):
    import torch
    import yaml
    from nerfstudio.data.scene_box import SceneBox
    path = Path(item['checkpoint'])
    expected_hash = sha256(path)
    # Trusted local training artifact, using the same YAML object config as the
    # existing evaluator. No checkpoint/config is written back.
    try:
        state = torch.load(path, map_location='cpu', weights_only=False)
    except TypeError:
        state = torch.load(path, map_location='cpu')
    if sha256(path) != expected_hash:
        raise ValueError('Checkpoint changed while loading')
    weights = {k[len('_model.'):]: v for k, v in state['pipeline'].items() if k.startswith('_model.')}
    count = weights['field.embedding_appearance.embedding.weight'].shape[0]
    model_config = yaml.load((Path(item['run']) / 'mapper/config/config.yml').read_text(), Loader=yaml.Loader)
    model = model_config.pipeline.model.setup(
        scene_box=SceneBox(aabb=weights['field.aabb']), num_train_data=count,
        metadata={}, device='cuda:0')
    model.load_state_dict(weights, strict=True)
    model.to('cuda:0')
    model.use_nvf = False
    model.calculate_entropy = False
    # RGB path only; these flags disable unused auxiliary outputs, not RGB.
    del state, weights
    return model, expected_hash


def score_images(model, prediction, rgba):
    import torch
    from torchmetrics.functional.image.ssim import structural_similarity_index_measure
    pred = prediction.permute(2, 0, 1)[None]
    gt = torch.as_tensor(rgba[..., :3].copy(), device=pred.device, dtype=pred.dtype).permute(2, 0, 1)[None] / 255
    mask = torch.as_tensor(rgba[..., 3] > 0, device=pred.device)
    count = int(mask.sum())
    if count == 0:
        raise ValueError('Empty foreground')
    error = (pred - gt).square()
    full_ssim, smap = structural_similarity_index_measure(pred, gt, data_range=1., return_full_image=True)
    model.lpips.net.spatial = False
    full_lpips = float(model.lpips.net(pred, gt, normalize=True).mean())
    model.lpips.net.spatial = True
    lmap = model.lpips.net(pred, gt, normalize=True)
    if smap.shape != pred.shape or lmap.shape[-2:] != mask.shape:
        raise ValueError('Unexpected SSIM/LPIPS map shape for this torchmetrics version')
    return {'foreground_pixels': count, 'full_mse': float(error.mean()),
            'fg_mse': float(error[0, :, mask].mean()), 'full_ssim': float(full_ssim),
            'fg_ssim': float(smap[0, :, mask].mean()), 'full_lpips': full_lpips,
            'fg_lpips_spatial': float(lmap[0, 0][mask].mean())}


def aggregate(rows):
    count = sum(r['foreground_pixels'] for r in rows)
    result = {'views': len(rows), 'foreground_pixels': count}
    for name in ['full_mse', 'full_ssim', 'full_lpips']:
        result[name] = float(np.mean([r[name] for r in rows]))
    for name in ['fg_mse', 'fg_ssim', 'fg_lpips_spatial']:
        result[name] = sum(r[name] * r['foreground_pixels'] for r in rows) / count
    for prefix in ['full', 'fg']:
        result[f'{prefix}_psnr'] = float(-10 * np.log10(max(result[f'{prefix}_mse'], 1e-30)))
    return result


def evaluate(group, cfg):
    import torch
    from nvf.env.utils import set_seed
    from nerfstudio.cameras.cameras import Cameras, CameraType
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required; submit the job on the cluster')
    items = pair_inputs(group, cfg)
    frozen, spec = dataset_directory(items[0], cfg)
    manifest = read(frozen / 'manifest.json')
    if manifest['spec'] != spec:
        raise ValueError('Frozen dataset spec mismatch')
    verify_files(frozen, manifest)
    dataset_hash = sha256(frozen / 'manifest.json')
    k = np.asarray(spec['intrinsics'])
    runtime = runtime_info()
    for policy, item in zip(['nvf', 'random'], items):
        set_seed(cfg['seed'])
        model, checkpoint_hash = load_model(item)
        model_config_hash = sha256(Path(item['run']) / 'mapper/config/config.yml')
        for mode in cfg['modes']:
            model.train(mode == 'train')
            assert model.training == model.field.training == model.sampler.training == (mode == 'train')
            for which, views in manifest['views'].items():
                for seed in cfg['inference_seeds']:
                    dest = Path(cfg['output_root']) / 'evaluations' / group['id'] / which / mode / policy / f'seed{seed}'
                    save_predictions = seed == cfg['inference_seeds'][0]
                    identity = {'dataset_sha256': dataset_hash, 'checkpoint_sha256': checkpoint_hash,
                                'model_config_sha256': model_config_hash, 'code': code_digest(),
                                'policy': policy, 'test_set': which, 'mode': mode,
                                'inference_seed': seed, 'save_predictions': save_predictions,
                                'runtime': runtime}
                    if dest.exists():
                        done = read(dest / 'complete.json')
                        if done['identity'] != identity:
                            raise ValueError(f'Completed result identity mismatch: {dest}')
                        verify_files(dest, done)
                        print(f'Reusing verified evaluation: {dest}', flush=True)
                        continue
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    with tempfile.TemporaryDirectory(prefix='.eval-', dir=dest.parent) as temp:
                        stage = Path(temp) / 'result'
                        stage.mkdir()
                        if save_predictions:
                            (stage / 'predictions').mkdir()
                            array = np.lib.format.open_memmap(stage / 'prediction_float32.npy', mode='w+',
                                dtype=np.float32, shape=(len(views), 512, 512, 3))
                        rows = []
                        for index, view in enumerate(views):
                            camera = Cameras(fx=float(k[0, 0]), fy=float(k[1, 1]),
                                cx=float(k[0, 2]), cy=float(k[1, 2]), width=512, height=512,
                                camera_type=CameraType.PERSPECTIVE,
                                camera_to_worlds=torch.tensor(np.asarray(view['pose'])[:3, :4],
                                    dtype=torch.float32)[None]).to('cuda:0')
                            # Old views retain seed_base + original frame ID,
                            # same for policies and modes. No RNG is consumed by
                            # metrics before the next frame seed is reset.
                            frame_seed = seed + view.get('original_frame_id', index)
                            set_seed(frame_seed)
                            with torch.no_grad():
                                rgb = model.get_outputs_for_camera_ray_bundle(
                                    camera.generate_rays(camera_indices=0, aabb_box=None))['rgb']
                                if not torch.isfinite(rgb).all():
                                    raise ValueError('Non-finite prediction')
                                rgb = rgb.clamp(0, 1)
                                rgba = np.asarray(Image.open(frozen / view['file']).convert('RGBA'))
                                row = score_images(model, rgb, rgba)
                            rows.append({'view': view['id'], 'seed': frame_seed, **row,
                                'full_psnr': float(-10 * np.log10(max(row['full_mse'], 1e-30))),
                                'fg_psnr': float(-10 * np.log10(max(row['fg_mse'], 1e-30)))})
                            if save_predictions:
                                output = rgb.detach().cpu().numpy()
                                array[index] = output
                                Image.fromarray((output * 255).astype(np.uint8)).save(stage / 'predictions' / f'{view["id"]}.png')
                            print(f'{group["id"]} {which} {mode} {policy} seed{seed} '
                                  f'{index + 1}/{len(views)}', flush=True)
                        if save_predictions:
                            array.flush()
                            del array
                        with (stage / 'per_view.csv').open('w', newline='') as stream:
                            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                            writer.writeheader()
                            writer.writerows(rows)
                        metrics = {**identity, **aggregate(rows),
                            'appearance': 'camera_index_0' if mode == 'train' else (
                                'mean_embedding' if model.field.use_average_appearance_embedding else 'zero_embedding'),
                            'num_train_embeddings': model.field.embedding_appearance.in_dim,
                            'stratified': bool(model.sampler.training),
                            'note': 'Mode contrast changes both sampler and appearance; not a pure noise ablation.'}
                        if which == 'old':
                            n = group['checkpoint_observations']
                            metrics['original_metrics'] = read(Path(item['run']) / f'metrics_{n:02d}.json')
                            metrics['full_psnr_minus_original'] = metrics['full_psnr'] - metrics['original_metrics']['psnr']
                        write(stage / 'metrics.json', metrics)
                        files = {str(f.relative_to(stage)): sha256(f) for f in sorted(stage.rglob('*')) if f.is_file()}
                        write(stage / 'complete.json', {'identity': identity, 'files': files})
                        stage.rename(dest)
        del model
        gc.collect()
        torch.cuda.empty_cache()


def summarize(cfg):
    out = Path(cfg['output_root'])
    rows, contrasts = [], []
    names = ['full_psnr', 'fg_psnr', 'full_ssim', 'fg_ssim', 'full_lpips', 'fg_lpips_spatial']
    for group in cfg['comparisons']:
        values = {}
        for which in ['old', 'new']:
            for mode in cfg['modes']:
                for policy in ['nvf', 'random']:
                    directory = out / 'evaluations' / group['id'] / which / mode / policy
                    samples = []
                    for seed in cfg['inference_seeds']:
                        done = read(directory / f'seed{seed}/complete.json')
                        verify_files(directory / f'seed{seed}', done)
                        metrics = read(directory / f'seed{seed}/metrics.json')
                        samples.append(metrics)
                        rows.append({'comparison': group['id'], 'test_set': which, 'mode': mode,
                                     'policy': policy, 'seed': seed, **{k: metrics[k] for k in names}})
                    values[which, mode, policy] = {k: float(np.mean([s[k] for s in samples])) for k in names}
                # Prediction-only side-by-side images, identical framing. No GT.
                destination = out / 'comparisons' / group['id'] / which / mode
                destination.mkdir(parents=True, exist_ok=True)
                seed = cfg['inference_seeds'][0]
                paths = [out / 'evaluations' / group['id'] / which / mode / p / f'seed{seed}/predictions'
                         for p in ['nvf', 'random']]
                files = sorted(paths[0].glob('*.png'))
                expected = 40 if which == 'old' else 20
                if len(files) != expected or {p.name for p in files} != {p.name for p in paths[1].glob('*.png')}:
                    raise ValueError('Missing/mismatched prediction frames')
                pages = []
                for index, path in enumerate(files):
                    canvas = Image.new('RGB', (1024, 538), '#222222')
                    draw = ImageDraw.Draw(canvas)
                    draw.text((8, 6), f'NVF | {path.stem} | {mode} | seed {seed}', fill='white')
                    draw.text((520, 6), f'Random | {path.stem} | {mode} | seed {seed}', fill='white')
                    for column, base in enumerate(paths):
                        canvas.paste(Image.open(base / path.name).convert('RGB'), (column * 512, 26))
                    canvas.save(destination / path.name)
                    if index % 5 == 0:
                        page = Image.new('RGB', (1024, 538 * 5), '#222222')
                        pages.append(page)
                    page.paste(canvas, (0, index % 5 * 538))
                for index, page in enumerate(pages):
                    page.save(destination / f'contact_sheet_{index:02d}.jpg', quality=95)
                for name in names:
                    diffs = [next(r[name] for r in rows if r['comparison'] == group['id'] and r['test_set'] == which and r['mode'] == mode and r['policy'] == 'nvf' and r['seed'] == seed)
                             - next(r[name] for r in rows if r['comparison'] == group['id'] and r['test_set'] == which and r['mode'] == mode and r['policy'] == 'random' and r['seed'] == seed)
                             for seed in cfg['inference_seeds']]
                    contrasts.append({'comparison': group['id'], 'test_set': which, 'mode': mode,
                        'metric': name, 'nvf_minus_random_mean': float(np.mean(diffs)),
                        'inference_seed_std': float(np.std(diffs, ddof=1)) if len(diffs) > 1 else 0.0})
        effects = {}
        for name in names:
            gap = lambda which, mode: values[which, mode, 'nvf'][name] - values[which, mode, 'random'][name]
            effects[name] = {'new_minus_old_gap_at_train': gap('new', 'train') - gap('old', 'train'),
                'eval_minus_train_gap_at_old': gap('old', 'eval') - gap('old', 'train'),
                'eval_minus_train_gap_at_new': gap('new', 'eval') - gap('new', 'train')}
        write(out / 'comparisons' / group['id'] / 'protocol_effects.json', effects)
    for filename, entries in [('summary.csv', rows), ('policy_differences.csv', contrasts)]:
        with (out / filename).open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(entries[0]))
            writer.writeheader()
            writer.writerows(entries)
    write(out / 'experiment_config.json', cfg)
    print(f'Complete: {out}/comparisons (predictions only; GT stays in _frozen)', flush=True)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('command', choices=['plan', 'freeze', 'evaluate', 'summarize', 'all'])
    parser.add_argument('--config', type=Path, default=HERE / 'config.json')
    parser.add_argument('--comparison', help='Run one comparison ID only')
    parser.add_argument('--check-paths', action='store_true', help='Validate cluster input files in plan')
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.comparison:
        cfg['comparisons'] = [g for g in cfg['comparisons'] if g['id'] == args.comparison]
        if not cfg['comparisons']:
            parser.error('Unknown comparison')
    if args.command == 'plan':
        print(json.dumps({'output_root': cfg['output_root'], 'conditions': ['old/train', 'new/train', 'old/eval', 'new/eval'],
            'views': {'old': 40, 'new': 20}, 'resolution': [512, 512],
            'comparisons': [g['id'] for g in cfg['comparisons']], 'inference_seeds': cfg['inference_seeds'],
            'new_sampling': cfg['sampling']}, indent=2))
        if args.check_paths:
            for group in cfg['comparisons']:
                pair_inputs(group, cfg)
        return
    if args.command == 'all':
        # Isolate Blender, CUDA model loading, and CPU summaries in separate
        # processes. All GT freezes finish before any inference begins.
        commands = [('freeze', g['id']) for g in cfg['comparisons']]
        commands += [('evaluate', g['id']) for g in cfg['comparisons']]
        commands.append(('summarize', args.comparison))
        for group in cfg['comparisons']:
            pair_inputs(group, cfg)
        # Freeze a configuration snapshot for subprocesses so editing the
        # repository config during a long job cannot change later conditions.
        with tempfile.TemporaryDirectory(prefix='bn3d-eval-config-') as temp:
            snapshot = Path(temp) / 'config.json'
            write(snapshot, cfg)
            for command, group_id in commands:
                argv = [sys.executable, '-B', '-m', 'experiments.bn3d_eval_protocol.run',
                        command, '--config', str(snapshot)]
                if group_id:
                    argv += ['--comparison', group_id]
                subprocess.run(argv, check=True, cwd=ROOT)
        return
    if args.command == 'summarize':
        summarize(cfg)
    else:
        for group in cfg['comparisons']:
            (freeze if args.command == 'freeze' else evaluate)(group, cfg)


if __name__ == '__main__':
    main()
