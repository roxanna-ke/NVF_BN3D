"""Dependency-light, Z-up, ground-aware adaptation of native hotdog_eval.

Native ring counts/order (4 high, 6 middle, 9 low) + one overhead = 20.
Each view's radius is fitted to the actual mesh and intrinsics, independently
of any checkpoint. Ring heights are measured from the ground, not the center.
"""
import numpy as np


def look_at(position, target):
    back = np.asarray(position, float) - target
    back /= np.linalg.norm(back)
    up = np.array([0., 0., 1.])
    right = np.cross(up, back)
    if np.linalg.norm(right) < 1e-7:
        # Same vertical degeneracy convention as native pose_point_to.
        right = np.cross(np.array([.1, 0., 1.]), back)
    right /= np.linalg.norm(right)
    pose = np.eye(4)
    pose[:3, :3] = np.stack([right, np.cross(back, right), back], axis=1)
    pose[:3, 3] = position
    return pose


def projected_inside(vertices, pose, k, margin=12, size=512):
    camera = np.einsum('ni,ij->nj', vertices - pose[:3, 3], pose[:3, :3])
    depth = -camera[:, 2]
    if np.any(depth <= 1e-6):
        return False
    uv = np.column_stack([k[0, 0] * camera[:, 0] / depth + k[0, 2],
                          -k[1, 1] * camera[:, 1] / depth + k[1, 2]])
    return bool(np.isfinite(uv).all() and (uv >= margin).all() and (uv <= size - margin).all())


def fit_view(vertices, k, target, position_at, margin):
    """Find a conservative external camera with all mesh vertices in frame."""
    span = float(np.linalg.norm(np.ptp(vertices, axis=0)))
    # Stay outside the entire mesh AABB horizontally (or above it for top).
    low, high = span, span
    while not projected_inside(vertices, look_at(position_at(high), target), k, margin):
        high *= 1.5
        if high > 100 * span:
            raise ValueError('Cannot fit building in camera')
    # Search downwards from an external, fitting pose. The lower bound is
    # supplied geometrically, so no model-density collision heuristic is used.
    low = span * .05
    for _ in range(36):
        mid = (low + high) / 2
        position = position_at(mid)
        external = np.any((position < vertices.min(0) - .01 * span) |
                          (position > vertices.max(0) + .01 * span))
        if external and projected_inside(vertices, look_at(position, target), k, margin):
            high = mid
        else:
            low = mid
    return high * 1.002


def generate_views(vertices, k, settings, render=None):
    """Return exactly 20 views. With render, enforce actual alpha occupancy.

    GT occupancy is a synthetic evaluation-set construction criterion only.
    It never uses NVF/Random predictions or evaluation scores.
    """
    vertices, k = np.asarray(vertices, float), np.asarray(k, float)
    low, high = vertices.min(0), vertices.max(0)
    target = (low + high) / 2
    ground = low[2] if settings['ground_z'] is None else float(settings['ground_z'])
    if ground > low[2] + 1e-5:
        raise ValueError('ground_z must not cut through the building mesh')
    counts = settings['ring_counts']
    fractions = settings['ring_height_fractions']
    if counts != [4, 6, 9] or len(fractions) != 3 or any(not 0 <= f <= 1 for f in fractions):
        raise ValueError('Require native 4/6/9 rings and three height fractions in [0,1]')
    if high[2] <= ground:
        raise ValueError('Building height must be positive')
    minimum = settings['min_foreground_fraction']
    desired = settings['target_foreground_fraction']
    if not 0 < minimum <= desired < 1:
        raise ValueError('Invalid foreground thresholds')
    margin = settings['margin_pixels']
    if not 0 <= margin < 128:
        raise ValueError('Invalid projection margin')
    views = []

    def add(position_at, ring, angle=None):
        distance = fit_view(vertices, k, target, position_at, margin)
        closest = distance
        image = None
        for attempt in range(9):
            pose = look_at(position_at(distance), target)
            if render is None:
                break
            image = render(pose)
            fraction = float((image[..., 3] > 0).mean())
            if fraction < minimum:
                # First attempt is the closest complete-building pose. If it
                # fails, do not silently freeze an undersized test image.
                if attempt == 0:
                    raise ValueError(f'{ring}: foreground {fraction:.3f} < {minimum}; '
                                     'review mesh/ground/margin or explicitly revise config')
                distance = (closest + distance) / 2
                continue
            if fraction <= desired * 1.05:
                break
            closest = distance
            distance *= min(1.25, np.sqrt(fraction / desired))
        if render is not None:
            # Always render the final pose, including when the loop exhausted.
            pose = look_at(position_at(distance), target)
            image = render(pose)
            fraction = float((image[..., 3] > 0).mean())
            if fraction < minimum:
                raise ValueError(f'{ring}: final foreground below threshold')
        else:
            fraction = None
        assert pose[2, 3] >= ground
        assert projected_inside(vertices, pose, k, margin)
        views.append({'id': f'view_{len(views):02d}', 'ring': ring,
                      'azimuth_degrees': angle, 'pose': pose.tolist(),
                      'foreground_fraction': fraction, 'rgba': image})

    for ring, (count, fraction) in enumerate(zip(counts, fractions)):
        z = ground + fraction * (high[2] - ground)
        for index in range(count):
            degrees = settings['azimuth_offset_degrees'] + index * 360 / count
            angle = np.deg2rad(degrees)
            direction = np.array([np.cos(angle), np.sin(angle)])
            def position_at(radius, direction=direction, z=z):
                return np.array([*(target[:2] + radius * direction), z])
            add(position_at, f'ring_{ring}', degrees)
    add(lambda distance: target + np.array([0., 0., distance]), 'top')
    assert len(views) == 20
    return views
