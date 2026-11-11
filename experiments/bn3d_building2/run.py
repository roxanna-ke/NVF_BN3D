"""Freeze once, then run the existing yaw/pitch trainer and evaluation protocol."""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

import numpy as np

from experiments.bn3d_eval_protocol import run as protocol
from nvf.bn3d_data import sha256, transform_pose
from tools import bn3d_yawpitch as tools

HERE = Path(__file__).resolve().parent
read, write = protocol.read, protocol.write


def configs(path):
    path = path.resolve()
    bundle = read(path)
    train = tools.load_config(path.parent / bundle['train_config'])
    evaluation = protocol.load_config(path.parent / bundle['eval_config'])
    if train['seeds'] != [42] or evaluation['seed'] != 42:
        raise ValueError('Building2 requires seed 42')
    if train['camera'] != dict(width=512, height=512, fx=256., fy=256., cx=256., cy=256.):
        raise ValueError('Building2 requires K256/FOV90')
    if evaluation['inference_seeds'] != [42000, 43000, 44000]:
        raise ValueError('Unexpected inference seeds')
    if [g['checkpoint_observations'] for g in evaluation['comparisons']] != [3, 10, 20, 30, 50]:
        raise ValueError('Unexpected checkpoint counts')
    root = Path(train['output_root'])
    if Path(train['frozen_inputs']) != root / 'prepared/frozen_inputs.json':
        raise ValueError('frozen_inputs must be output_root/prepared/frozen_inputs.json')
    for group in evaluation['comparisons']:
        for policy in ['nvf', 'random']:
            if Path(group[f'{policy}_run']) != root / policy / '50/seed42':
                raise ValueError('Evaluation must use the matching 50-view training runs')
    return bundle, train, evaluation


def frozen_item(train, frozen):
    manifest = read(Path(train['output_root']) / 'prepared/manifest.json')
    return dict(run=str(Path(train['output_root']) / 'prepared/freeze_input'),
                config=train, manifest=manifest, normalization=frozen['normalization'],
                split=manifest['splits']['42'])


def verify(train, evaluation):
    root = Path(train['output_root']) / 'prepared'
    done = read(root / 'freeze_complete.json')
    if done['train_config'] != train or done['eval_config'] != evaluation:
        raise ValueError('Configuration differs from frozen snapshot')
    protocol.verify_files(root, done)
    frozen = read(train['frozen_inputs'])
    item = frozen_item(train, frozen)
    directory, spec = protocol.dataset_directory(item, evaluation)
    if str(directory) != frozen['dataset'] or sha256(directory / 'manifest.json') != frozen['manifest_sha256']:
        raise ValueError('Frozen dataset identity/hash mismatch')
    manifest = read(directory / 'manifest.json')
    if manifest['spec'] != spec:
        raise ValueError('Frozen spec differs from evaluation spec')
    protocol.verify_files(directory, manifest)
    for which, count in [('initial', 3), ('old', 40)]:
        rows = frozen[which]
        ids = item['split']['initial' if which == 'initial' else 'test']
        if len(rows) != count or [r['frame_id'] for r in rows] != ids:
            raise ValueError(f'Invalid frozen {which} provenance')
        for row in rows:
            if not Path(row['file']).is_absolute() or sha256(row['file']) != row['sha256']:
                raise ValueError(f'Frozen PNG changed: {row["file"]}')
    return frozen, item


def freeze(train, evaluation):
    root = Path(train['output_root']) / 'prepared'
    if root.exists():
        verify(train, evaluation)  # Partial directories fail; never render over them.
        print(f'Reusing verified freeze: {root}', flush=True)
        return
    root.mkdir(parents=True, exist_ok=False)
    write(root / 'train_config.json', train)
    write(root / 'eval_config.json', evaluation)
    manifest = tools.prepare(train)
    from PIL import Image
    from nvf.env.Scene import BN3DScene
    from nvf.env.utils import set_seed
    set_seed(42)
    meta = tools.read_metadata(Path(train['data_root']), train['camera'])
    scene = BN3DScene(tools.environment_config(train, meta, np.asarray(manifest['intrinsics'])))
    normalization = dict(world_to_canonical=scene.world_to_canonical.tolist(),
                         canonical_to_world=scene.canonical_to_world.tolist())
    initial = []
    initial_dir = root / 'initial'
    initial_dir.mkdir()
    for frame_id in manifest['splits']['42']['initial']:
        pose = transform_pose(np.asarray(meta['frames'][frame_id]['camtoworld']), scene.world_to_canonical)
        set_seed(420000 + frame_id)
        rgba = scene.render_pose(pose)
        if rgba.shape != (512, 512, 4) or rgba.dtype != np.uint8 or not rgba[..., 3].any():
            raise ValueError('Invalid initial RGBA rendering')
        path = initial_dir / f'frame_{frame_id:03d}.png'
        Image.fromarray(rgba).save(path)
        initial.append(dict(pose=pose.tolist(), file=str(path), sha256=sha256(path), frame_id=frame_id))
    frozen = dict(normalization=normalization, initial=initial)
    item = frozen_item(train, frozen)
    synthetic = Path(item['run'])
    synthetic.mkdir()
    write(synthetic / 'trajectory.json', initial)
    # Keep item.config.output_root identical to training: it controls the old GT cache.
    protocol.freeze_item(item, evaluation)
    directory, _ = protocol.dataset_directory(item, evaluation)
    dataset = read(directory / 'manifest.json')
    protocol.verify_files(directory, dataset)
    frozen.update(dataset=str(directory), manifest_sha256=sha256(directory / 'manifest.json'),
                  old=[dict(pose=v['pose'], file=str(directory / v['file']),
                            sha256=sha256(directory / v['file']), frame_id=v['original_frame_id'])
                       for v in dataset['views']['old']])
    write(train['frozen_inputs'], frozen)
    files = {str(p.relative_to(root)): sha256(p) for p in sorted(root.rglob('*')) if p.is_file()}
    write(root / 'freeze_complete.json', dict(train_config=train, eval_config=evaluation, files=files))
    verify(train, evaluation)
    print(f'Frozen initial3/old40/new20: {train["frozen_inputs"]}', flush=True)


def execute(task, bundle, train, evaluation):
    frozen, item = verify(train, evaluation)
    out = Path(evaluation['output_root'] if task == 'evaluate' else train['output_root'])
    snapshot = out / '_jobs' / task
    # Exclusive creation also prevents two workers from claiming the same task.
    if task in ['nvf', 'random']:
        run = out / task / '50/seed42'
        if run.exists():
            raise FileExistsError(f'Run already exists; inspect/move it before retrying: {run}')
    elif (out / 'evaluations').exists():
        raise FileExistsError('Evaluation output already exists; inspect/move it before retrying')
    snapshot.mkdir(parents=True, exist_ok=False)
    write(snapshot / 'config.json', bundle)
    write(snapshot / 'train.json', train)
    write(snapshot / 'eval.json', evaluation)
    write(snapshot / 'frozen_inputs.json', frozen)
    inputs = {'frozen_inputs': sha256(train['frozen_inputs']),
              'prepared_manifest': item['manifest'], 'dataset_manifest': read(Path(frozen['dataset']) / 'manifest.json')}
    if task == 'evaluate':
        for policy in ['nvf', 'random']:
            completed = Path(train['output_root']) / policy / '50/seed42/complete.json'
            if not completed.is_file():
                raise FileNotFoundError(f'Training must finish before evaluation: {completed}')
        pairs = [protocol.pair_inputs(g, evaluation) for g in evaluation['comparisons']]
        checkpoints = {}
        for pair in pairs:
            for entry in pair:
                directory, _ = protocol.dataset_directory(entry, evaluation)
                if str(directory) != frozen['dataset'] or entry['split'] != item['split']:
                    raise ValueError('Training inputs do not match pretraining freeze')
                for path in [Path(entry['checkpoint']), *[Path(entry['run']) / name for name in
                             ['complete.json', 'effective_config.json', 'normalization.json', 'trajectory.json',
                              'mapper/config/config.yml', Path(entry['checkpoint']).name.replace('checkpoint_', 'metrics_').replace('.ckpt', '.json')]]]:
                    checkpoints[str(path)] = sha256(path)
        inputs['training_files'] = checkpoints
    write(snapshot / 'inputs.json', inputs)
    if task == 'evaluate':
        for group in evaluation['comparisons']:
            subprocess.run([sys.executable, '-B', '-m', 'experiments.bn3d_eval_protocol.run',
                            'evaluate', '--config', str(snapshot / 'eval.json'), '--comparison', group['id']], check=True)
        subprocess.run([sys.executable, '-B', '-m', 'experiments.bn3d_eval_protocol.run',
                        'summarize', '--config', str(snapshot / 'eval.json')], check=True)
        for path, expected in inputs['training_files'].items():
            if sha256(path) != expected:
                raise ValueError(f'Training input changed during evaluation: {path}')
    else:
        tools.run(train, task, seed=42, total=50)
    verify(train, evaluation)
    write(snapshot / 'complete.json', {'task': task, 'inputs_sha256': sha256(snapshot / 'inputs.json')})


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('task', choices=['freeze', 'nvf', 'random', 'evaluate'])
    parser.add_argument('--config', type=Path, default=HERE / 'config.json')
    args = parser.parse_args()
    bundle, train, evaluation = configs(args.config)
    if args.task == 'freeze':
        freeze(train, evaluation)
    else:
        execute(args.task, bundle, train, evaluation)


if __name__ == '__main__':
    main()
