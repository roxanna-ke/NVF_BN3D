"""BN3D building1 NVF/random active mapping; run inside the NVF CUDA environment."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import subprocess
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nvf.bn3d_data import sha256, split_frames, transform_pose, project_opengl

EXPERIMENTS = (("nvf", 20), ("nvf", 30), ("nvf", 50), ("random", 20), ("random", 50))

def read_metadata(data_root):
    """Read poses only; dataset images and camera intrinsics are not inputs."""
    meta = json.loads((Path(data_root) / "meta_data.json").read_text())
    for frame in meta["frames"]:
        pose = np.asarray(frame["camtoworld"], dtype=float)
        if pose.shape != (4, 4) or not np.isfinite(pose).all():
            raise ValueError("Invalid camera pose")
        if not np.allclose(pose[3], [0, 0, 0, 1]) or not np.allclose(pose[:3,:3].T @ pose[:3,:3], np.eye(3), atol=1e-4) or not np.isclose(np.linalg.det(pose[:3,:3]), 1, atol=1e-4):
            raise ValueError("Invalid camera rotation")
    meta["height"] = meta["width"] = 512
    return meta

def validate_experiment(policy, total):
    if (policy, total) not in EXPERIMENTS:
        raise ValueError(f"Unsupported experiment: {policy}/{total}")



def load_config(path):
    config = json.loads(Path(path).read_text())
    for key in ("mesh_path", "data_root", "output_root"):
        config[key] = str(Path(config[key]).expanduser().resolve())
    if not 0 < config["min_foreground_fraction"] <= 1:
        raise ValueError("Invalid foreground fraction")
    for key in ("random_max_attempts", "random_max_batches", "candidates"):
        if not isinstance(config[key], int) or config[key] < 1:
            raise ValueError(f"Invalid {key}")
    return config


def prepare(config):
    """CPU-only metadata checks; Blender scene checks happen in run()."""
    meta = read_metadata(Path(config["data_root"]))
    mesh = Path(config["mesh_path"])
    if not mesh.is_file():
        raise FileNotFoundError(mesh)
    k = np.array([[256., 0., 256.], [0., 256., 256.], [0., 0., 1.]])
    output = Path(config["output_root"]) / "prepared"
    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "mesh_sha256": sha256(mesh),
        "asset_sha256": {path.name: sha256(path) for path in sorted(mesh.parent.iterdir())
                         if path.is_file() and path.suffix.lower() in (".obj", ".mtl", ".png", ".jpg", ".jpeg")},
        "metadata_sha256": sha256(Path(config["data_root"]) / "meta_data.json"),
        "intrinsics": k.tolist(),
        "width": meta["width"], "height": meta["height"],
        "splits": {str(seed): dict(zip(("initial", "test"), split_frames(meta, seed)))
                   for seed in config["seeds"]},
    }
    # NVF and Random jobs can prepare the same inputs concurrently.
    temporary = output / f".manifest.{os.getpid()}.{uuid4().hex}.json"
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    temporary.replace(output / "manifest.json")
    return manifest


def environment_config(config, meta, k, *, gpu=True, samples=None):
    return SimpleNamespace(
        resolution=(meta["height"], meta["width"]), fov=90,
        fx=float(k[0, 0]), fy=float(k[1, 1]),
        cx=float(k[0, 2]), cy=float(k[1, 2]),
        mesh_path=config["mesh_path"], cycles=True,
        cycles_samples=samples if samples is not None else config.get("cycles_samples", 100),
        resolution_percentage=100, gpu=gpu, scene="BN3DScene",
    )


def preflight(config, policy):
    """Import the actual OBJ and render one known metadata camera without training."""
    from PIL import Image
    from nvf.env.Scene import BN3DScene

    if policy not in ("nvf", "random"):
        raise ValueError(policy)
    manifest = prepare(config)
    meta = read_metadata(Path(config["data_root"]))
    k = np.asarray(manifest["intrinsics"])
    scene = BN3DScene(environment_config(config, meta, k, gpu=False, samples=1))
    pose = transform_pose(np.asarray(meta["frames"][0]["camtoworld"], dtype=float),
                          scene.world_to_canonical)
    uv, depth = project_opengl(np.asarray(scene.mesh.vertices), pose, k)
    visible = (depth > 0) & np.isfinite(uv).all(axis=1)
    visible &= (uv[:, 0] >= 0) & (uv[:, 0] < meta["width"])
    visible &= (uv[:, 1] >= 0) & (uv[:, 1] < meta["height"])
    if int(visible.sum()) < 100:
        raise ValueError("BN3D OBJ/camera projection has fewer than 100 visible vertices")
    rendered = scene.render_pose(pose)
    mask = rendered[..., 3] > 0
    if not mask.any():
        raise ValueError("BN3D Blender render has an empty foreground mask")
    output = Path(config["output_root"]) / "prepared"
    Image.fromarray(rendered).save(output / f"preflight_{policy}.png")
    result = {"visible_vertices": int(visible.sum()),
              "rendered_foreground_fraction": float(mask.mean()),
              "intrinsics": k.tolist(),
              "original_silhouette_check": False,
              "preview": str(output / f"preflight_{policy}.png")}
    (output / f"preflight_{policy}.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def run(config, policy, seed, total):
    validate_experiment(policy, total)
    import torch
    from PIL import Image
    from nvf.active_mapping.agents.BaseAgent import BaseAgent
    from nvf.env.Scene import BN3DScene
    from nvf.env.utils import pose2tensor, set_seed
    from nvf.metric.MetricTracker import RefMetricTracker
    from eval import set_params

    if policy not in ("nvf", "random"):
        raise ValueError(policy)
    if seed not in config["seeds"]:
        raise ValueError("Seed is absent from experiment config")
    manifest = prepare(config)
    meta = read_metadata(Path(config["data_root"]))
    init_ids = manifest["splits"][str(seed)]["initial"]
    test_ids = manifest["splits"][str(seed)]["test"]
    run_dir = Path(config["output_root"]) / policy / str(total) / f"seed{seed}"
    if (run_dir / "complete.json").exists():
        raise FileExistsError(f"Already complete: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=False)
    config = dict(config, policy=policy, total_images=total, seed=seed)
    revision = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
    config["git_revision"] = revision.stdout.strip() or "unknown"
    config["intrinsics"] = manifest["intrinsics"]
    (run_dir / "effective_config.json").write_text(json.dumps(config, indent=2) + "\n")
    set_seed(seed)
    k = np.asarray(manifest["intrinsics"])
    env_cfg = environment_config(config, meta, k)
    scene = BN3DScene(env_cfg)
    matrix = scene.world_to_canonical
    raw_poses = [np.asarray(f["camtoworld"], dtype=float) for f in meta["frames"]]
    poses = [transform_pose(p, matrix) for p in raw_poses]
    mesh_vertices = np.asarray(scene.mesh.vertices)
    for index in init_ids:
        uv, depth = project_opengl(mesh_vertices, poses[index], k)
        visible = (depth > 0) & np.isfinite(uv).all(axis=1)
        visible &= (uv[:, 0] >= 0) & (uv[:, 0] < meta["width"])
        visible &= (uv[:, 1] >= 0) & (uv[:, 1] < meta["height"])
        if visible.sum() < 100:
            raise ValueError(f"OBJ/camera alignment failed at initial frame {index}")
    (run_dir / "normalization.json").write_text(json.dumps({
        "world_to_canonical": matrix.tolist(),
        "canonical_to_world": scene.canonical_to_world.tolist(),
    }, indent=2) + "\n")

    # Re-render both initial observations and held-out GT with the same scene.
    test_images = scene.render_poses([poses[i] for i in test_ids])
    init_images = scene.render_poses([poses[i] for i in init_ids])
    if any(image[..., 3].max() == 0 for image in init_images):
        raise ValueError("Blender produced an empty foreground mask")
    obs_dir = run_dir / "observations"
    obs_dir.mkdir(exist_ok=True)
    for n, image in enumerate(init_images):
        Image.fromarray(image).save(obs_dir / f"{n:02d}.png")
    observations = list(init_images)
    obs_poses = [torch.as_tensor(pose2tensor(p), dtype=torch.float32) for p in (poses[i] for i in init_ids)]
    trajectory = [{"source": "initial", "frame_id": i, "pose": poses[i].tolist()} for i in init_ids]

    centers = np.array([poses[i][:3, 3] for i in range(len(poses)) if i not in test_ids])
    camera_bounds = np.stack([centers.min(0), centers.max(0)])
    object_bounds = np.asarray(scene.get_aabb())
    cfg = SimpleNamespace(
        env=env_cfg, model="instant-ngp", sampler="BaseSampler",
        n_sample=config.get("candidates", 512), camera_aabb=torch.tensor(camera_bounds, dtype=torch.float32),
        target_aabb=torch.tensor(camera_bounds, dtype=torch.float32),
        object_aabb=torch.tensor(object_bounds, dtype=torch.float32),
        check_density=True, density_threshold=1e-4,
        horizon=total - 2,
        world_up=config["world_up"],
        target_height_fraction=config["target_height_fraction"],
        train_iter=config.get("train_iter", 5000), d0=config.get("d0", 1.0),
        use_uniform=False, use_huber=True, use_var=True, use_vis=True, mu=0.95,
        train_use_tensorboard=False, config_home=str(run_dir / "mapper"),
        dataset_path=str(run_dir / "dataset"), exp_folder=str(run_dir),
    )
    (run_dir / "mapper").mkdir(exist_ok=True)
    agent = BaseAgent(cfg)
    from nvf.active_mapping.agents.bn3d_yawpitch_sampler import BN3DYawPitchSampler
    agent.sampler = BN3DYawPitchSampler(cfg)
    agent.sampler.pipeline = agent.pipeline
    agent.pipeline.trainer.config.nvf_num_iterations = config.get("nvf_iterations", 500)
    agent.pipeline.trainer.config.nvf_batch_size = config.get("nvf_batch_size", 65536)

    class EvaluationEnvironment:
        def __init__(self):
            self.scene = scene

        def get_images(self, mode, return_quat=False):
            if mode != "eval" or return_quat:
                raise ValueError(mode)
            return test_images, [torch.tensor(poses[i], dtype=torch.float32) for i in test_ids]

    tracker = RefMetricTracker(cfg, EvaluationEnvironment())
    tracker.init_trajectory(obs_poses)
    (run_dir / "trajectory.json").write_text(json.dumps(trajectory, indent=2) + "\n")
    (run_dir / "reference_foreground.json").write_text(json.dumps([float((im[...,3] > 0).mean()) for im in test_images]))
    elapsed_steps = []

    def record_timing(n, started):
        elapsed_steps.append(time.perf_counter() - started)
        remaining = float(np.mean(elapsed_steps) * (total - n))
        (run_dir / "timing.json").write_text(json.dumps({
            "completed_observation_counts": list(range(3, n + 1)),
            "elapsed_seconds": elapsed_steps,
            "estimated_remaining_seconds": remaining,
        }, indent=2) + "\n")
        print(f"{policy} seed {seed}: {n}/{total} observations trained; "
              f"step {elapsed_steps[-1]/3600:.2f} h; "
              f"estimated remaining {remaining/3600:.1f} h", flush=True)

    for n in range(3, total + 1):
        step_start = time.perf_counter()
        agent.pipeline.reset()
        agent.pipeline.config.max_num_iterations = cfg.train_iter
        set_params(SimpleNamespace(method="NVF", **vars(cfg)), agent.pipeline)
        agent.pipeline.trainer.pipeline.model.renderer_entropy.set_iteration(n - 3)
        new_images = observations if n == 3 else observations[-1:]
        new_poses = obs_poses if n == 3 else obs_poses[-1:]
        agent.step = n - 3
        agent.process_obs(new_images, new_poses)
        agent.pipeline.save_ckpt(str(run_dir / f"checkpoint_{n:02d}.ckpt"))
        with torch.no_grad():
            metrics, predicted = tracker.get_metric(agent.pipeline)
            if n in (3, 10, 20, 30, total):
                visuals = run_dir / "visualizations"
                visuals.mkdir(exist_ok=True)
                Image.fromarray((predicted[0].permute(1, 2, 0).clip(0, 1)
                                 .cpu().numpy() * 255).astype(np.uint8)).save(
                                     visuals / f"rgb_{n:02d}.png")
                entropy = agent.pipeline.visualize(
                    obs_poses[0][None], render_option="entropy", height=256, width=256)
                np.save(visuals / f"entropy_{n:02d}.npy", entropy.detach().cpu().numpy())
        (run_dir / f"metrics_{n:02d}.json").write_text(json.dumps(
            {name: float(value) for name, value in metrics.items()}, indent=2) + "\n")
        if n == total:
            record_timing(n, step_start)
            break
        next_pose, image, selection = select_observation(agent, scene, cfg, config, policy, n)
        matrix_pose = selection["matrix_pose"]
        chosen = selection["chosen"]
        Image.fromarray(image).save(obs_dir / f"{n:02d}.png")
        observations.append(image)
        obs_poses.append(next_pose)
        tracker.visibility_metric.update(next_pose[None])
        trajectory.append({"source": policy, "pose": matrix_pose,
                           "candidate_index": chosen})
        (run_dir / f"candidates_{n:02d}.json").write_text(json.dumps(selection) + "\n")
        (run_dir / "trajectory.json").write_text(json.dumps(trajectory, indent=2) + "\n")
        record_timing(n, step_start)
    (run_dir / "complete.json").write_text(json.dumps({"policy": policy, "seed": seed,
                                                        "observations": len(observations), "training_rounds": total - 2}) + "\n")


def select_observation(agent, scene, cfg, config, policy, observation):
    import torch
    from nvf.active_mapping.mapping_utils import to_transform
    batches, attempts = [], 0
    max_batches = 1 if policy == "nvf" else config["random_max_batches"]
    for batch in range(max_batches):
        candidates = agent.sampler(cfg.n_sample, pose=agent.current_pose)
        if len(candidates) == 0:
            raise RuntimeError("No valid camera candidates")
        scores = agent.get_reward(candidates) if policy == "nvf" else None
        order = [int(scores.reshape(-1).argmax().item())] if policy == "nvf" else torch.randperm(len(candidates)).tolist()
        record = {"poses": candidates.detach().cpu().tolist(),
                  "scores": None if scores is None else scores.detach().cpu().reshape(-1).tolist(), "attempts": []}
        batches.append(record)
        for chosen in order:
            pose = candidates[chosen].detach().cpu()
            matrix = to_transform(pose).numpy()
            image = scene.render_pose(matrix)
            fraction = float((image[..., 3] > 0).mean())
            attempts += 1
            accepted = policy == "nvf" or fraction >= config["min_foreground_fraction"]
            record["attempts"].append({"index": chosen, "foreground_fraction": fraction, "accepted": accepted})
            if accepted:
                return pose, image, {"batches": batches, "chosen_batch": batch, "chosen": chosen,
                                     "matrix_pose": matrix.tolist(), "render_attempts": attempts}
            if attempts >= config["random_max_attempts"]:
                break
        if attempts >= config["random_max_attempts"]:
            break
    failure = Path(cfg.exp_folder) / f"rejected_{observation:02d}.json"
    failure.write_text(json.dumps(batches))
    raise RuntimeError(f"Random observation {observation}: no foreground >= {config['min_foreground_fraction']} after {attempts} renders; see {failure}")


def summarize(config):
    root = Path(config["output_root"])
    result = {}
    for policy, total in EXPERIMENTS:
        for seed in config["seeds"]:
            directory = root / policy / str(total) / f"seed{seed}"
            if not (directory / "complete.json").exists():
                raise FileNotFoundError(f"Incomplete run: {directory}")
            result[f"{policy}/{total}/seed{seed}"] = json.loads((directory / f"metrics_{total:02d}.json").read_text())
    for total in (20, 50):
        for seed in config["seeds"]:
            nvf, random = (result[f"{p}/{total}/seed{seed}"] for p in ("nvf", "random"))
            result[f"difference/{total}/seed{seed}"] = {k: nvf[k] - random[k] for k in nvf}
    output = root / "summary"
    output.mkdir(exist_ok=True)
    (output / "final.json").write_text(json.dumps(result, indent=2) + "\n")
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "preflight", "run", "summarize"))
    parser.add_argument("--config", required=True)
    parser.add_argument("--policy", choices=("nvf", "random"))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--total-images", type=int)
    args = parser.parse_args()
    config = load_config(args.config)
    if args.command == "prepare":
        print(json.dumps(prepare(config), indent=2))
    elif args.command == "preflight":
        if args.policy is None:
            parser.error("preflight requires --policy")
        print(json.dumps(preflight(config, args.policy), indent=2))
    elif args.command == "summarize":
        print(summarize(config))
    else:
        if args.policy is None or args.seed is None or args.total_images is None:
            parser.error("run requires --policy, --seed and --total-images")
        run(config, args.policy, args.seed, args.total_images)


if __name__ == "__main__":
    main()
