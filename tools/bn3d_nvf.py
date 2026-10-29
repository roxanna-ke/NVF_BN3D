"""BN3D building1 NVF/random active mapping; run inside the NVF CUDA environment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from nvf.bn3d_data import read_metadata, sha256, split_frames, transform_pose, project_opengl


def load_config(path):
    config = json.loads(Path(path).read_text())
    for key in ("mesh_path", "data_root", "output_root"):
        config[key] = str(Path(config[key]).expanduser().resolve())
    return config


def prepare(config):
    """CPU-only metadata checks; Blender scene checks happen in run()."""
    meta = read_metadata(Path(config["data_root"]))
    mesh = Path(config["mesh_path"])
    if not mesh.is_file():
        raise FileNotFoundError(mesh)
    k = np.asarray(meta["frames"][0]["intrinsics"])
    for frame in meta["frames"]:
        if not np.allclose(frame["intrinsics"], k):
            raise ValueError("BN3D run requires one common camera intrinsics matrix")
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
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def run(config, policy, seed):
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
    run_dir = Path(config["output_root"]) / policy / f"seed{seed}"
    if (run_dir / "complete.json").exists():
        raise FileExistsError(f"Already complete: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "effective_config.json").write_text(json.dumps(config, indent=2) + "\n")
    set_seed(seed)
    k = np.asarray(manifest["intrinsics"])
    env_cfg = SimpleNamespace(
        resolution=(meta["height"], meta["width"]), fov=90,
        fx=float(k[0, 0]), fy=float(k[1, 1]),
        cx=float(k[0, 2]), cy=float(k[1, 2]),
        mesh_path=config["mesh_path"], cycles=True,
        cycles_samples=config.get("cycles_samples", 100),
        resolution_percentage=100, gpu=True, scene="BN3DScene",
    )
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
        horizon=18,
        train_iter=config.get("train_iter", 5000), d0=config.get("d0", 1.0),
        use_uniform=False, use_huber=True, use_var=True, use_vis=True, mu=0.95,
        train_use_tensorboard=False, config_home=str(run_dir / "mapper"),
        dataset_path=str(run_dir / "dataset"), exp_folder=str(run_dir),
    )
    (run_dir / "mapper").mkdir(exist_ok=True)
    agent = BaseAgent(cfg)
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
    for n in range(3, 21):
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
            if n in (3, 10, 20):
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
        if n == 20:
            break
        candidates = agent.sampler(cfg.n_sample, pose=agent.current_pose)
        if len(candidates) == 0:
            raise RuntimeError("No valid camera candidates")
        if policy == "nvf":
            scores = agent.get_reward(candidates)
            chosen = int(scores.reshape(-1).argmax().item())
        else:
            scores = None
            chosen = int(torch.randint(len(candidates), (1,)).item())
        next_pose = candidates[chosen].detach().cpu()
        from nvf.active_mapping.mapping_utils import to_transform
        matrix_pose = to_transform(next_pose).numpy()
        image = scene.render_pose(matrix_pose)
        Image.fromarray(image).save(obs_dir / f"{n:02d}.png")
        observations.append(image)
        obs_poses.append(next_pose)
        tracker.visibility_metric.update(next_pose[None])
        trajectory.append({"source": policy, "pose": matrix_pose.tolist(),
                           "candidate_index": chosen})
        (run_dir / f"candidates_{n:02d}.json").write_text(json.dumps({
            "poses": candidates.detach().cpu().tolist(),
            "scores": None if scores is None else scores.detach().cpu().reshape(-1).tolist(),
            "chosen": chosen,
        }) + "\n")
        (run_dir / "trajectory.json").write_text(json.dumps(trajectory, indent=2) + "\n")
    (run_dir / "complete.json").write_text(json.dumps({"policy": policy, "seed": seed,
                                                        "observations": len(observations)}) + "\n")


def summarize(config):
    import csv

    root = Path(config["output_root"])
    rows = []
    for policy in ("nvf", "random"):
        for seed in config["seeds"]:
            run_dir = root / policy / f"seed{seed}"
            if not (run_dir / "complete.json").is_file():
                raise FileNotFoundError(f"Incomplete run: {run_dir}")
            for n in range(3, 21):
                metrics = json.loads((run_dir / f"metrics_{n:02d}.json").read_text())
                rows.append({"policy": policy, "seed": seed, "observations": n, **metrics})
    output = root / "summary"
    output.mkdir(exist_ok=True)
    columns = list(rows[0])
    with (output / "metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    result = {}
    for key in columns[3:]:
        values = {policy: np.array([r[key] for r in rows
                                    if r["policy"] == policy and r["observations"] == 20])
                  for policy in ("nvf", "random")}
        result[key] = {
            policy: {"mean": float(a.mean()), "std": float(a.std(ddof=1))}
            for policy, a in values.items()
        }
        result[key]["paired_difference"] = (values["nvf"] - values["random"]).tolist()
    (output / "final.json").write_text(json.dumps(result, indent=2) + "\n")
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "run", "summarize"))
    parser.add_argument("--config", required=True)
    parser.add_argument("--policy", choices=("nvf", "random"))
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    config = load_config(args.config)
    if args.command == "prepare":
        print(json.dumps(prepare(config), indent=2))
    elif args.command == "summarize":
        print(summarize(config))
    else:
        if args.policy is None or args.seed is None:
            parser.error("run requires --policy and --seed")
        run(config, args.policy, args.seed)


if __name__ == "__main__":
    main()
