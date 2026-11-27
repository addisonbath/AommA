"""World frame: find the floor in the point map and stand the scene upright on it.

The world frame is Z-up with the floor at z = 0. The camera sits on the Z axis
at its height above the floor, and +Y is the direction it faces along the floor.
"""

from typing import NamedTuple

import numpy as np

# Points within this distance of the fitted plane count as floor, in meters.
_INLIER_M = 0.03
_MIN_FLOOR_POINTS = 500
# A depth sensor's floor points needed to set the floor's height from them (the LiDAR's map is 256 x 192).
_MIN_MEASURED_FLOOR = 200
# Without a floor mask to start from, the floor is the plane, among the background points, that the most
# of them lie on, of those that face up within this angle of the camera's up and lie at least this far
# below it. The bedroom's mask was the rug in the next room, seen through a doorway, and a level camera,
# the old guess, put the floor 0.25 m too low and slid every floor-standing model back through the wall.
# A floor mask's plane tilted further than this from the camera's up is not the floor either: KitchenScan 011's
# mask was its tiled backsplash, a wall, which with the pose's up put the camera 0.25 m off the floor.
_MAX_TILT_DEG = 50
_MIN_DROP_M = 0.3
_TRIALS = 600
_SAMPLE = 20000
# A point is on the floor only if its own surface faces the way the floor does, within this angle. A plane
# can pass through many points it is not the surface of: KitchenScan 007, its floor out of view, had one
# slicing across the backsplash, the cabinet fronts and the oven door, 45 degrees off level, that more points
# lay within 3 cm of than of any level surface the built-ins' masks left.
_FACING_DEG = 20


class FloorFit(NamedTuple):
    """The floor plane in the camera frame (`normal` @ p = `offset`), and whether it was seen: found on floor
    the photo or the depth sensor shows, not a guess (the lowest level surface, which may be a counter's top,
    or a level camera)."""

    normal: np.ndarray
    offset: float
    seen: bool

    @property
    def camera_to_world(self) -> np.ndarray:
        # Orient the normal towards the camera, which sits at the origin.
        normal, offset = (-self.normal, -self.offset) if self.offset > 0 else (self.normal, self.offset)
        return camera_over_floor(normal, -offset)


def fit_camera_to_world(
    points: np.ndarray, floor_mask: np.ndarray, object_mask: np.ndarray,
    up: np.ndarray | None = None, measured: tuple[np.ndarray, np.ndarray] | None = None,
) -> np.ndarray:
    """Return the 4x4 camera-to-world matrix implied by the floor plane (see `fit_floor`)."""
    return fit_floor(points, floor_mask, object_mask, up, measured).camera_to_world


def fit_floor(
    points: np.ndarray, floor_mask: np.ndarray, object_mask: np.ndarray,
    up: np.ndarray | None = None, measured: tuple[np.ndarray, np.ndarray] | None = None,
) -> FloorFit:
    """The floor plane, from the camera-frame point map.

    `points` is the (H, W, 3) camera-frame point map, `floor_mask` a rough
    segmentation of the floor, and `object_mask` every pixel owned by an object.
    `up`, the world's up in the camera frame when it is known (a capture's pose),
    fixes the floor's tilt, leaving only its height to find (`_floor_offset`), from
    `measured` if given: a depth sensor's camera-frame points on the floor mask, and all of them.
    Only the mask's points whose surfaces face up are taken as floor (see `_FACING_DEG`).
    """
    valid = np.isfinite(points).all(axis=-1)
    normals = surface_normals(points)
    usable = valid & ~object_mask
    background, background_normals = points[usable], normals[usable]
    on_mask = usable & floor_mask

    if up is not None:
        normal = up / np.linalg.norm(up)
        offset, seen = _floor_offset(normal, points[on_mask & facing(normals, normal)], background, measured)
        return FloorFit(normal, offset, seen)

    if on_mask.sum() >= _MIN_FLOOR_POINTS:
        normal, offset = _fit_plane(points[on_mask])
        if _faces_up(normal):
            # The segmenter may catch only a patch of floor; grow to every
            # background point on the same plane and refit for a steadier estimate.
            for _ in range(3):
                near = np.abs(background @ normal - offset) < _INLIER_M
                normal, offset = _fit_plane(background[near])
            return FloorFit(normal, offset, True)
    if (found := _find_floor(background, background_normals)) is not None:
        return FloorFit(*found, False)
    # No floor to be found: assume the camera is level and the lowest
    # background points are at floor height.
    normal = np.array([0.0, 1.0, 0.0])
    return FloorFit(normal, float(np.percentile(background @ normal, 1)), False)


def surface_normals(points: np.ndarray) -> np.ndarray:
    """Each point's unit surface normal (H, W, 3), from its neighbours across and down; NaN where they are missing.
    Its sign is arbitrary."""
    across, down = np.full_like(points, np.nan), np.full_like(points, np.nan)
    across[:, 1:-1] = points[:, 2:] - points[:, :-2]
    down[1:-1] = points[2:] - points[:-2]
    normals = np.cross(across, down)
    with np.errstate(invalid="ignore", divide="ignore"):
        return normals / np.linalg.norm(normals, axis=-1, keepdims=True)


def facing(normals: np.ndarray, normal: np.ndarray, within_deg: float = _FACING_DEG) -> np.ndarray:
    """Which surface `normals` face along `normal`, either way, within `within_deg`; False where unknown."""
    with np.errstate(invalid="ignore"):
        return np.abs(normals @ normal) > np.cos(np.radians(within_deg))


def _faces_up(normal: np.ndarray) -> bool:
    """Whether a plane's `normal` is within `_MAX_TILT_DEG` of the camera's up, either way."""
    return abs(float(normal[1])) / np.linalg.norm(normal) >= np.cos(np.radians(_MAX_TILT_DEG))


def camera_over_floor(normal: np.ndarray, height: float) -> np.ndarray:
    """The 4x4 camera-to-world matrix of a camera `height` above the floor, the world's up being `normal` in the
    camera frame: Z up, the floor at 0, the camera over the origin, the world's x the camera's right laid flat."""
    normal = normal / np.linalg.norm(normal)
    right = np.array([1.0, 0.0, 0.0])
    x_axis = right - (right @ normal) * normal
    x_axis /= np.linalg.norm(x_axis)
    matrix = np.eye(4)
    matrix[:3, :3] = np.stack([x_axis, np.cross(normal, x_axis), normal])
    matrix[2, 3] = height
    return matrix


def _floor_offset(
    normal: np.ndarray, seed: np.ndarray, background: np.ndarray, measured: tuple[np.ndarray, np.ndarray] | None,
) -> tuple[float, bool]:
    """The floor's offset along the known up `normal`, and whether the floor was seen: where the measured floor
    points lie, if there are enough of them, else the floor mask's; else, unseen, the lowest level of the
    background that many points share. Then settled on every point near it, as the plane fit grows from its seed.

    With the up from the pose, the LiDAR's floor points put the camera within 3 cm of RoomPlan's height on 14 of
    the 15 test shots that show floor (MoGe's, fitted to the LiDAR, within 4 cm).
    """
    seen = True
    if measured is not None and len(measured[0]) >= _MIN_MEASURED_FLOOR:
        on, every = measured[0] @ normal, measured[1] @ normal
    elif len(seed) >= _MIN_FLOOR_POINTS:
        on, every = seed @ normal, background @ normal
    else:
        seen = False
        every = np.sort(background @ normal)
        low = every[every < -_MIN_DROP_M]
        # The first band of _INLIER_M either side holding _MIN_FLOOR_POINTS points.
        span = low[_MIN_FLOOR_POINTS - 1:] - low[: len(low) - _MIN_FLOOR_POINTS + 1]
        dense = np.flatnonzero(span < 2 * _INLIER_M)
        if not len(dense):
            return float(np.percentile(every, 1)), False
        on = low[dense[0]:dense[0] + _MIN_FLOOR_POINTS]
    offset = float(np.median(on))
    for _ in range(3):
        near = every[np.abs(every - offset) < _INLIER_M]
        if not len(near):
            break
        offset = float(np.median(near))
    return offset, seen


def _find_floor(background: np.ndarray, normals: np.ndarray) -> tuple[np.ndarray, float] | None:
    """The floor among the background points, as (unit normal, offset), or None if no plane qualifies.

    Tries planes through three random points whose surfaces face up, keeps the one most points lie on and face
    the way of (see `_FACING_DEG`), of those facing up (see `_MAX_TILT_DEG`, `_MIN_DROP_M`), and refits it to them.
    """
    rng = np.random.default_rng(0)
    chosen = rng.choice(len(background), min(len(background), _SAMPLE), replace=False)
    points, point_normals = background[chosen], normals[chosen]
    up = np.array([0.0, 1.0, 0.0])
    # Drawn from all points, the three would seldom all be on a floor that is a small share of the photo.
    candidates = points[facing(point_normals, up, _MAX_TILT_DEG)]
    if len(candidates) < _MIN_FLOOR_POINTS:
        return None
    best, best_count = None, _MIN_FLOOR_POINTS - 1
    for _ in range(_TRIALS):
        a, b, c = candidates[rng.choice(len(candidates), 3, replace=False)]
        normal = np.cross(b - a, c - a)
        if np.linalg.norm(normal) < 1e-6:
            continue
        normal /= np.linalg.norm(normal)
        if normal @ up < 0:
            normal = -normal
        offset = float(normal @ a)
        # Facing up, and below the camera (which sits at the origin) by enough to be a floor.
        if normal @ up < np.cos(np.radians(_MAX_TILT_DEG)) or offset > -_MIN_DROP_M:
            continue
        count = int(((np.abs(points @ normal - offset) < _INLIER_M) & facing(point_normals, normal)).sum())
        if count > best_count:
            best, best_count = (normal, offset), count
    if best is None:
        return None
    normal, offset = best
    for _ in range(3):
        normal, offset = _fit_plane(background[(np.abs(background @ normal - offset) < _INLIER_M) & facing(normals, normal)])
    return normal, offset


def _fit_plane(points: np.ndarray) -> tuple[np.ndarray, float]:
    """Least-squares plane through `points` as (unit normal, offset): normal @ p = offset."""
    centre = points.mean(axis=0)
    normal = np.linalg.svd(points - centre, full_matrices=False)[2][2]
    return normal, float(normal @ centre)


def to_world(points: np.ndarray, camera_to_world: np.ndarray) -> np.ndarray:
    """Transform (..., 3) camera-frame points into the world frame."""
    return points @ camera_to_world[:3, :3].T + camera_to_world[:3, 3]
