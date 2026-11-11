# Building2: frozen seed42 yaw/pitch experiment

Uses the existing `tools.bn3d_yawpitch.run` for NVF50 and Random50 with seed 42, 512 candidates, 512×512 rendering, `fx=fy=256` (90° FOV), and **`cx=cy=256`**. This corrects the prior plan's principal point of 255.5 by half a pixel to match the actual archived Building1 yawpitch `effective_config.json` intrinsics; that older config has no `camera` field. Audit is enabled at mapping observations 3, 10, 20, and 35.

The mesh filename is `building2.obj`, verified in the local Building2 asset directory (not Building1's `House.obj`). The cluster paths mirror the local data layout:

- `/mnt/scratch/imos-students/ke/data/buildnet3d/building_models/building2/building2.obj`
- `/mnt/scratch/imos-students/ke/data/REAL/building2/meta_data.json`

Training output: `/mnt/scratch/imos-students/ke/outputs/bn3d_yawpitch/building2`.
Evaluation output: `/mnt/scratch/imos-students/ke/outputs/bn3d_building2_eval_v1/building2`.

## Submission

Deploy these files and the trainer's `frozen_inputs`/audit support to the shared repository first. On the login node, these commands work from any cwd:

```bash
JOB=/mnt/scratch/imos-students/ke/repos/NVF_BN3D/experiments/bn3d_building2/job.sh
bash "$JOB" submit freeze
# Wait for freeze to finish successfully before submitting either training job.
bash "$JOB" submit nvf
bash "$JOB" submit random
# Wait for BOTH training jobs to finish successfully.
bash "$JOB" submit evaluate
```

There is no automatic dependency scheduling. No task invokes preflight, smoke, or an environment probe. Run:ai containers start safely in `/tmp`; the worker then changes to `CONTAINER_REPO` so historical trainer resources resolve relative to the repository. `PYTHONPATH` also includes that repository. Both shared PVCs are mounted: scratch at `/mnt/scratch`, home at `/mnt/home`. The wrapper follows the existing project/image/resource defaults and accepts `CONTAINER_REPO`, `PYTHON_BIN`, `RUNAI_PROJECT`, `MY_IMAGE`, `MY_GPU_AMOUNT`, `NODE`, `SCRATCH_PVC`, `HOME_PVC`, and cluster identity environment overrides.

An optional third argument selects the bundle config:

```bash
bash "$JOB" submit freeze experiments/bn3d_building2/config.json
```

Relative bundle paths are relative to the repository, independent of caller cwd. Absolute bundle paths must be inside the submitting repository and are mapped to the same relative location under `CONTAINER_REPO`. `train_config` and `eval_config` inside the bundle are relative to the bundle's directory. Keep both files in the deployed repository. The default bundle uses `train.json` and `eval.json`.

## Frozen inputs and integrity

`freeze` calls the trainer's `prepare`, creates the same Blender scene/normalization, transforms metadata poses, and renders initial3 once. It saves a synthetic `prepared/freeze_input/trajectory.json` so the existing evaluation helper can check new-test/initial-pose overlap without training checkpoints. The only shared evaluator change extracts `freeze_item(item, cfg)` from `freeze`; the existing paired-checkpoint wrapper remains intact.

The freeze item uses the actual training config, including its unchanged `output_root`. `dataset_directory(item, eval_config)` therefore computes exactly the same frozen dataset directory before training and during subsequent evaluation. The evaluator generates old40/new20 once under its content-derived `_frozen` directory. Freeze verifies PNG hashes and writes `prepared/frozen_inputs.json`:

```json
{
  "normalization": {
    "world_to_canonical": [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],
    "canonical_to_world": [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
  },
  "initial": [{"pose": "actual 4x4 matrix", "file": "/absolute/initial.png", "sha256": "hash", "frame_id": 0}],
  "old": [{"pose": "actual 4x4 matrix", "file": "/absolute/old.png", "sha256": "hash", "frame_id": 1}],
  "dataset": "/absolute/evaluation/_frozen/spec-digest",
  "manifest_sha256": "hash of dataset/manifest.json"
}
```

This is a schema illustration: generated normalization contains the full actual normalization matrices, each pose is a numeric 4×4 matrix, and the lists have 3 and 40 entries in split order. The trainer validates this file, normalization, intrinsics, poses, IDs, and hashes, then loads the frozen initial and old images instead of rerendering them.

Freeze snapshots both configs and hashes its prepared artifacts in `freeze_complete.json`. A repeated freeze only reuses a fully verified completed freeze with identical configs. A partial prepared directory fails. Training/evaluation snapshot configs, frozen inputs, and input provenance under their output root's `_jobs/<task>/`; evaluation also records checkpoint, metrics, trajectory, normalization, and mapper-config hashes. Task directories are claimed exclusively; existing training/evaluation output or task directories fail rather than being overwritten, including completed ones. Inspect and move aside affected paths explicitly before retrying a failed task. Do not edit configs, source, assets, or frozen artifacts between freeze and evaluation; source hashes participate in the frozen dataset identity.

Evaluation invokes `experiments.bn3d_eval_protocol.run evaluate` for counts 3, 10, 20, 30, and 50, always from each policy's 50-view run. It reuses the freeze, runs both `train` and `eval` inference modes with seeds 42000, 43000, and 44000 on old40/new20, then invokes the existing summary command. Frozen dataset identity and files are checked before and after tasks; training input hashes are checked again after evaluation.

Run the offline tests with:

```bash
python3 -B -m unittest experiments.bn3d_building2.test_building2
```

The four tests cover freeze/training/evaluation config consistency, byte-identical `prepare` manifest rewrites and pre/post-training dataset identity, fake Run:ai submission from an arbitrary cwd (default, relative, and absolute config paths), and worker repository cwd. The fake executables never contact the cluster or launch training. Invoke `job.sh` through `bash`; executable permission is not required.

Local checks passed for Python/shell syntax and configs. CPU preparation on the actual Building2 OBJ and metadata found 200 poses and the expected initial3/old40 split, with byte-identical manifest rewrites. A CPU projection check using the OBJ import axis conversion `(x, y, z) -> (x, -z, y)` found all 12,760 vertices inside each initial view (frame IDs 131, 18, 154). NumPy emitted matrix-operation warnings, but explicit checks confirmed finite projections and positive depths for all vertices. This checks geometry only; no Blender render or cluster job was executed.
