"""BN3D data preparation. Mesh import follows AVS/tools/bn3d_common.py.

This module deliberately has no dependency on Blender until import_obj is called.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


def read_metadata(data_root: Path) -> dict:
    data_root = Path(data_root)
    meta = json.loads((data_root / "meta_data.json").read_text())
    if meta["width"] != 512 or meta["height"] != 512:
        raise ValueError("Expected 512x512 BN3D images")
    if len(meta["frames"]) != 200:
        raise ValueError("Expected 200 BN3D camera frames")
    for frame in meta["frames"]:
        pose = np.asarray(frame["camtoworld"], dtype=float)
        k = np.asarray(frame["intrinsics"], dtype=float)
        if pose.shape != (4, 4) or k.shape != (3, 3):
            raise ValueError("Invalid camera matrix shape")
        if not np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-4):
            raise ValueError("Camera rotation is not orthonormal")
        if not (data_root / "images" / frame["rgb_path"]).is_file():
            raise FileNotFoundError(frame["rgb_path"])
    return meta


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalization(vertices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Map the mesh bounding-box center to zero and longest side to two."""
    vertices = np.asarray(vertices, dtype=float)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) == 0:
        raise ValueError("Expected nonempty Nx3 vertices")
    low, high = vertices.min(0), vertices.max(0)
    longest = np.max(high - low)
    if not np.isfinite(longest) or longest <= 0:
        raise ValueError("Degenerate mesh")
    scale = 2.0 / longest
    matrix = np.eye(4)
    matrix[:3, :3] *= scale
    matrix[:3, 3] = -(low + high) * scale / 2
    return matrix, np.linalg.inv(matrix)


def transform_pose(pose: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Apply a uniform world normalization without scaling camera rotation."""
    result = np.array(pose, dtype=float, copy=True)
    result[:3, 3] = matrix[:3, :3] @ result[:3, 3] + matrix[:3, 3]
    return result


def split_frames(meta: dict, seed: int) -> tuple[list[int], list[int]]:
    test = list(range(0, len(meta["frames"]), 5))
    train = sorted(set(range(len(meta["frames"]))) - set(test))
    init = np.random.default_rng(seed).choice(train, size=3, replace=False).tolist()
    return init, test


def project_opengl(vertices: np.ndarray, pose: np.ndarray, intrinsics: np.ndarray):
    """Projection in image-edge coordinates for OpenGL camera axes."""
    camera = (np.asarray(vertices) - pose[:3, 3]) @ pose[:3, :3]
    depth = -camera[:, 2]
    uv = np.column_stack((intrinsics[0, 0] * camera[:, 0] / depth + intrinsics[0, 2],
                          -intrinsics[1, 1] * camera[:, 1] / depth + intrinsics[1, 2]))
    return uv, depth


def import_obj(path: Path):
    """Import all OBJ meshes; call before creating the camera and lights."""
    import bpy

    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    if bpy.app.version[0] < 4:
        bpy.ops.import_scene.obj(filepath=str(path), split_mode="OFF", axis_forward="-Z", axis_up="Y")
    else:
        bpy.ops.wm.obj_import(filepath=str(path), forward_axis="NEGATIVE_Z", up_axis="Y")
    objects = [obj for obj in bpy.context.selected_objects if obj.type == "MESH"]
    if not objects:
        raise ValueError("OBJ import produced no meshes")
    bpy.context.view_layer.update()
    vertices = np.concatenate([
        np.array([tuple(obj.matrix_world @ vertex.co) for vertex in obj.data.vertices])
        for obj in objects
    ])
    return objects, vertices
