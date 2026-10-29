import unittest

import numpy as np

from nvf.bn3d_data import normalization, project_opengl, split_frames, transform_pose


class BN3DDataTests(unittest.TestCase):
    def test_world_normalization_preserves_camera_rotation(self):
        vertices = np.array([[-4., -2., 0.], [6., 3., 5.]])
        world_to_canonical, inverse = normalization(vertices)
        self.assertTrue(np.allclose(world_to_canonical @ inverse, np.eye(4)))
        normalized = vertices @ world_to_canonical[:3, :3].T + world_to_canonical[:3, 3]
        self.assertAlmostEqual(float(np.max(normalized.max(0) - normalized.min(0))), 2.0)
        pose = np.eye(4)
        pose[:3, 3] = [2., 1., 3.]
        converted = transform_pose(pose, world_to_canonical)
        self.assertTrue(np.allclose(converted[:3, :3], pose[:3, :3]))
        self.assertTrue(np.allclose(converted[:3, 3],
                                    world_to_canonical[:3, :3] @ pose[:3, 3]
                                    + world_to_canonical[:3, 3]))

    def test_opengl_projection_and_split(self):
        k = np.array([[100., 0., 50.], [0., 100., 50.], [0., 0., 1.]])
        points = np.array([[0., 0., -2.], [1., 1., -2.]])
        uv, depth = project_opengl(points, np.eye(4), k)
        self.assertTrue(np.allclose(uv, [[50., 50.], [100., 0.]]))
        self.assertTrue(np.all(depth > 0))
        meta = {"frames": [None] * 200}
        initial, test = split_frames(meta, 42)
        self.assertEqual(len(initial), 3)
        self.assertEqual(len(test), 40)
        self.assertFalse(set(initial) & set(test))
        self.assertEqual(initial, split_frames(meta, 42)[0])


if __name__ == "__main__":
    unittest.main()
