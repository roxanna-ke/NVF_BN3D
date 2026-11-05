"""BN3D look-at-axis rotations with the existing BaseSampler position rules."""
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from nvf.active_mapping.agents.Sampler import BaseSampler


class BN3DYawPitchSampler(BaseSampler):
    def __init__(self, cfg):
        super().__init__(cfg)
        bounds = np.asarray(cfg.object_aabb.cpu(), dtype=float)
        self.center = (bounds[0] + bounds[1]) / 2
        self.up = np.asarray(cfg.world_up, dtype=float)
        if self.up.shape != (3,) or not np.isfinite(self.up).all() or np.linalg.norm(self.up) < 1e-9:
            raise ValueError("world_up must be a finite nonzero 3-vector")
        self.up /= np.linalg.norm(self.up)
        fractions = np.asarray(cfg.target_height_fraction, dtype=float)
        if fractions.shape != (2,) or not np.isfinite(fractions).all() or not 0 <= fractions[0] <= fractions[1] <= 1:
            raise ValueError("target_height_fraction must be ordered within [0,1]")
        # Project the AABB onto the configured building axis.
        extent = np.dot((bounds[1] - bounds[0]) / 2, np.abs(self.up))
        self.heights = (fractions * 2 - 1) * extent

    def _sample_poses(self, count, retry=0):
        # Same independent uniform XYZ sampling as native sample_poses.
        positions = torch.stack([
            torch.FloatTensor(count).uniform_(self.camera_aabb[i, 0], self.camera_aabb[i, 1])
            for i in range(3)
        ], dim=-1)
        heights = torch.FloatTensor(count).uniform_(*self.heights).numpy()
        targets = self.center + heights[:, None] * self.up
        forward = targets - positions.numpy()
        norms = np.linalg.norm(forward, axis=1)
        if np.any(norms < 1e-9):
            if retry >= 3:
                raise ValueError("Camera coincides with axis target after 4 attempts")
            return self._sample_poses(count, retry + 1)
        forward /= norms[:, None]
        right = np.cross(forward, self.up)
        lengths = np.linalg.norm(right, axis=1)
        if np.any(lengths < 1e-7):
            if retry >= 3:
                raise ValueError("Vertical camera-axis degeneracy after 4 attempts")
            return self._sample_poses(count, retry + 1)
        right /= lengths[:, None]
        camera_up = np.cross(right, forward)
        matrices = np.stack([right, camera_up, -forward], axis=-1)
        quaternions = torch.tensor(Rotation.from_matrix(matrices).as_quat(), dtype=positions.dtype)
        return torch.cat([quaternions, positions], dim=-1)

    def sample(self, n_sample, *args, **kwargs):
        if self.pipeline is not None and self.cfg.check_density:
            poses = self._sample_poses(int(n_sample * 1.2))
            return self.filter_poses(poses)[:n_sample]
        return self._sample_poses(n_sample)
