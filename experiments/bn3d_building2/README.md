# Building2: frozen new20 seed42 yaw/pitch experiment

Runs the existing `tools.bn3d_yawpitch.run` for NVF50 and Random50 with seed42,
512 candidates, 512×512 rendering, `fx=fy=256` (90° FOV), and `cx=cy=256`.
Audit is enabled at mapping observations 3, 10, 20, and 35.

The principal point is not an NVF compatibility requirement. Original dataset
metadata uses 255.5. Nerfstudio infers dimensions as twice the principal point
when width/height are omitted, producing 511 for 255.5; commit `167770e` fixed the
metric camera by explicitly passing dimensions. Current rendering, training,
reward, and evaluation use explicit 512×512 dimensions. Building2 retains the
historical FOV90 principal point of 256. It rerenders the assets and does not use
original dataset RGB as training or test images.

Cluster assets:

- `/mnt/scratch/imos-students/ke/data/buildnet3d/building_models/building2/building2.obj`
- `/mnt/scratch/imos-students/ke/data/REAL/building2/meta_data.json`

Training output: `/mnt/scratch/imos-students/ke/outputs/bn3d_yawpitch/building2`.
Evaluation output: `/mnt/scratch/imos-students/ke/outputs/bn3d_building2_eval_v1/building2`.

## Migration and materials

Archive both previous Building2 output roots before using this revision, after
confirming the previous jobs have no running or pending workers. Earlier frozen
images contain material corruption and use a different test protocol. Recreate
freeze and train both policies from the beginning; do not reuse previous images
or checkpoints. This change does not delete or move any existing outputs.

`bpycv.render_data()` first renders RGB, then temporarily replaces mesh material
slots for instance annotation. Clearing the slots resets per-face material
indices; restoring only the material list leaves all faces using slot zero.
`BN3DScene.render()` now preserves both slots and polygon material indices in a
`finally` block, also on failed renders. This keeps original OBJ/MTL textures
assigned to their original faces. Neutral world lighting and the existing
Standard color transform remain part of the rendering protocol; an exact color
match to the original dataset's lighting is not assumed.

## Submission

Deploy the updated repository (including the renderer, trainer and frozen-input
loader) to the shared cluster repository first. Run from any cwd:

```bash
JOB=/mnt/scratch/imos-students/ke/repos/NVF_BN3D/experiments/bn3d_building2/job.sh
bash "$JOB" submit freeze
# Wait for successful freeze and inspect the rendered images.
bash "$JOB" submit nvf
bash "$JOB" submit random
# Wait for BOTH 50-view training jobs to finish successfully.
bash "$JOB" submit evaluate
```

There is no automatic dependency scheduling or smoke test. Containers start in
`/tmp`; the worker enters `CONTAINER_REPO` and adds it to `PYTHONPATH`. Both scratch
and home PVCs are mounted. The wrapper accepts `CONTAINER_REPO`, `PYTHON_BIN`,
`RUNAI_PROJECT`, `MY_IMAGE`, `MY_GPU_AMOUNT`, `NODE`, `SCRATCH_PVC`, `HOME_PVC`, and
cluster identity environment overrides. Invoke `job.sh` through `bash`.

An optional third argument selects the bundle config:

```bash
bash "$JOB" submit freeze experiments/bn3d_building2/config.json
```

Relative bundle paths are repository-relative. Absolute bundle paths must be
inside the submitting repository and are mapped under `CONTAINER_REPO`.
`train_config` and `eval_config` are relative to the bundle directory.

## Frozen inputs and evaluation

Building2 uses exactly one test set, `new20`. Four rings at fractions
0.95/0.65/0.35/0.05 of the building height, ordered highest to lowest, contain
4/6/5/4 views respectively, plus one overhead
view. Azimuth offset is 15°. Cameras remain above ground, the entire building fits
with a 12-pixel margin, alpha foreground occupancy must be at least 20%, and its
target is approximately 35%. Failure to meet these constraints stops freeze.
Test construction never reads model predictions or evaluation scores.

Initial metadata frame IDs remain fixed at `[131, 18, 154]`. No metadata frames
are reserved for testing; all 200 metadata positions define the candidate AABB.
The frozen manifest records an empty metadata test split. Both policies use
identical initial3 images and the same new20 images/poses for online metrics and
final evaluation. Test images do not enter gradient updates or selection reward.
Online geometry and visibility metrics remain enabled.

`freeze` prepares metadata/assets, renders initial3, creates a synthetic
`prepared/freeze_input/trajectory.json` for initial/test pose-overlap checks, and
renders new20 into the evaluation output's `_frozen/<spec-digest>/new/` directory.
It writes `prepared/frozen_inputs.json` containing:

- `normalization`: both coordinate transformation matrices.
- `initial`: three rows with metadata `frame_id`, normalized `pose`, absolute
  image `file`, and `sha256`.
- `test`: twenty rows with view `id`, normalized `pose`, absolute image `file`,
  and `sha256`.
- `dataset` and `manifest_sha256`: frozen dataset path and manifest identity.

The trainer verifies hashes, poses, intrinsics, and rendering settings before
loading these images. Freeze snapshots both configs and hashes prepared artifacts
in `freeze_complete.json`. Repeated freeze accepts only a fully verified,
identical completed freeze. Training preparation preserves the manifest bytes.
Do not edit configs, source, assets, or frozen files between freeze and evaluation;
rendering/evaluation source hashes participate in dataset identity.

Final evaluation reads checkpoints at 3, 10, 20, 30, and 50 observations from each
50-view training run. It evaluates new20 in both `train` and `eval` inference modes
with seeds 42000/43000/44000. Reports include foreground and full-image PSNR,
SSIM/LPIPS, policy differences, and mode contrasts. The primary comparison is
new20/eval foreground PSNR, pooling foreground MSE before conversion to PSNR.
Mode contrasts change both sampler and appearance behavior. Prediction comparison
images show NVF and Random; GT remains archived separately.

## Failed Pods and retries

The worker writes combined stdout/stderr, including native subprocess output, to
`<output_root>/_logs/<task>.log.<unique suffix>` and prints the path at startup.
Each Pod gets a new file and training failures return a failure exit code.
Configuration/filesystem errors before log creation remain in Pod logs only.

`_jobs/<task>` is an exclusive claim created before training; `<policy>/50/seed42`
is created by the trainer. A first attempt can fail after creating either, so
replacement Pods can report directory-exists errors even when submission began
with no output. Inspect the earliest Pod or persistent attempt log for the first
failure. After stopping the old job, inspect completion markers and move aside
both affected paths before retrying. Completed runs are not overwritten or
resumed. Preserve logs for diagnosis.

## Offline checks and limits

```bash
python3 -B -m unittest experiments.bn3d_building2.test_building2 \
  experiments.bn3d_eval_protocol.test_protocol experiments.test_yawpitch_audit
```

Tests cover the new20 freeze/verification/training-loader path with a fake scene,
four-height geometry and framing, tamper rejection, material restoration after
slot clearing (including exceptions), principal point versus explicit dimensions,
new-only summaries, passive selection auditing, submission path mapping, and
persistent logs/failure exit codes. Fake scenes/executables never launch Blender,
training, or cluster jobs. Real Blender/GPU rendering remains to be validated on
the cluster; local tests do not establish rendered visual correctness.
