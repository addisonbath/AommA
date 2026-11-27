"""Pose stage: find the rotation, scale and position that make an asset match the photo.

A generated model arrives in its own frame with no relation to the photo. We
try many candidate rotations; for each, scale and position follow in closed
form from the object's mask, and the candidate is scored by how well a quick
point-splat render matches the mask's shape and the photo's colours.
"""

from collections.abc import Callable
from typing import Literal

import numpy as np
from PIL import Image
from pydantic import BaseModel
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

from aomma.layout import Camera, Vec3

# Side of the small window, in pixels, in which candidates are rendered and compared.
_GRID = 96
_COARSE_YAW_DEG = 15
_FINE_YAW_DEG = 3
# How much colour disagreement (0-1) counts against silhouette overlap (0-1).
_COLOR_WEIGHT = 0.5
# Settling may tip an object by at most this much, and may cost at most this much overlap.
_MAX_SETTLE_DEG = 30
_SETTLE_IOU_COST = 0.05
# Levelling: a flat surface says which way is up if it holds this share of the model's
# area, counting triangles within this angle of one another as one surface.
_MIN_LEVEL_SHARE = 0.08
_LEVEL_SPREAD_DEG = 10
# Levelling is kept unless it costs more overlap than this: then the surface was not really a level one.
_LEVEL_IOU_COST = 0.15
# Fitting a model to its measured box may stretch any one dimension by at most this factor either way.
_MAX_STRETCH = 1.6
# A point counts as hidden by a wall only if it is this far beyond the wall's face, in meters,
# and a model is clear of the walls when no more than this share of it is hidden.
_WALL_SLACK_M = 0.03
_HIDDEN_OK = 0.02
# Pulling a model clear of a wall may bring it at most this share of the way to the camera.
_MAX_PULL = 0.5
# A floor-standing model whose lowest point is higher than this, in meters, floats and may be re-seated.
_RESEAT_MIN_M = 0.02
# With `contact="trim"`, this share of a model's surface, its lowest, does not count when it is
# stood on a surface: a generated mesh can have a stray spike hanging below it.
_TRIM_SHARE = 0.01

class Wall(tuple):
    """A wall, as the solver needs it: the bottom-left corner of the face towards the room,
    the vector along that face, the unit normal into the room, and the wall's height.

    It unpacks as those four. Its `openings` are the doors and doorways through it, each as
    (from, to, top): shares of the way along the face, and the height of its top. Lines of sight
    pass through them (`behind_walls`); the room still ends at the wall (`large.share_past_walls`).
    A plain 4-tuple is a wall with none.
    """

    openings: tuple[tuple[float, float, float], ...] = ()

    def __new__(cls, corner, along, normal, height, openings=()):
        wall = super().__new__(cls, (corner, along, normal, height))
        wall.openings = tuple(openings)
        return wall


def through_openings(wall: Wall | tuple, across: np.ndarray, up: np.ndarray) -> np.ndarray:
    """Which crossings of the wall's face, at these shares of the way `across` it and heights `up`, pass through an opening."""
    passes = np.zeros(len(across), dtype=bool)
    for low, high, top in getattr(wall, "openings", ()):
        passes |= (across >= low) & (across <= high) & (up < top)
    return passes


class PoseSettings(BaseModel):
    """Choices in how poses are solved. The defaults are what runs use."""

    # How much the share of a partly hidden object's model that a wall hides counts
    # against its silhouette overlap while its pose is searched for. 0 ignores walls.
    wall_weight: float = 0.0
    # Whether a solved model that a wall still hides is then pulled clear of it.
    pull_forward: bool = False
    # What is done with a floor-standing model left floating (see `reseat_on_floor`).
    reseat: Literal["none", "drop", "stretch"] = "none"
    # What a model stands on: its lowest surface less the lowest `_TRIM_SHARE` of it ("trim"), so
    # that a stray spike does not hold it up, or its single lowest point ("lowest"). Trim fixed a bed
    # standing on a spike and a bookshelf posed on its back; it cost one nightstand 0.05 of fit.
    contact: Literal["lowest", "trim"] = "trim"
    # How much the pose search of a partly hidden object counts against a candidate, as the
    # typical |log| of the ratio, how far its visible surface is from where the depth map puts it.
    depth_weight: float = 0.0
    # How much the same search counts against a candidate, as the |log| of the ratio, how far
    # its size is from the size the VLM estimated for the object.
    size_weight: float = 0.0
    # Degrees about the vertical by which the search's starting heading, and the limit on how far
    # the front may turn from the camera, are turned. A redraw drawn from another angle than the
    # photo gives a model whose "front" is not the object's front: the garage's dryer was posed
    # side-on, its plain side to the camera, and turned 90 degrees it faced the camera.
    turn_deg: float = 0.0
    # Whether a redrawn object's model is then stretched onto the box the depth map measured for it,
    # where that fits its outline better than its own proportions (see `fit_to_box`).
    fit_box: bool = True
    # Whether, in a staged run, a model standing on the floor or a built-in is held to the room's shell:
    # stood on its support, squared to the walls, kept out of them and backed against one it nearly
    # touches (see `stages.large.hold_to_shell`).
    hold_to_shell: bool = True


class Pose(BaseModel):
    id: str
    translate: Vec3
    rotate_xyz_deg: Vec3
    scale: float
    # Silhouette overlap with the object's mask and mean colour difference, both 0-1, as the camera
    # sees the model: without the share of it, `hidden_by_wall`, that a wall stands in front of.
    iou: float
    color_error: float
    hidden_by_wall: float = 0.0
    # In a staged run, what holding the model to the shell moved (see `stages.large.ShellMoves`).
    shell_moves: dict | None = None


def rotation_z(degrees: float) -> np.ndarray:
    a = np.radians(degrees)
    return np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])


def _rotation_x(degrees: float) -> np.ndarray:
    a = np.radians(degrees)
    return np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])


def _rotation_y(degrees: float) -> np.ndarray:
    a = np.radians(degrees)
    return np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])


# The floor as a contact surface: the plane z = 0, facing up.
FLOOR = (np.array([0.0, 0.0, 1.0]), 0.0)

# The six ways to stand an asset up: each of its axes, either way, pointing at the sky.
# Objects rest on surfaces, so one of these plus a turn about the vertical covers most poses.
UPRIGHTS = [np.eye(3), _rotation_x(90), _rotation_x(-90), _rotation_x(180), _rotation_y(90), _rotation_y(-90)]


def euler_xyz_deg(rotation: np.ndarray) -> Vec3:
    """Angles for USD's rotateXYZ (turn about X, then Y, then Z) that reproduce `rotation`."""
    if abs(rotation[2, 0]) > 1 - 1e-9:
        # Looking straight along the Y turn: X and Z turns coincide, so put it all in X.
        y = -np.sign(rotation[2, 0]) * 90.0
        x = np.degrees(np.arctan2(-rotation[1, 2], rotation[1, 1]))
        return (float(x), float(y), 0.0)
    y = -np.degrees(np.arcsin(rotation[2, 0]))
    x = np.degrees(np.arctan2(rotation[2, 1], rotation[2, 2]))
    z = np.degrees(np.arctan2(rotation[1, 0], rotation[0, 0]))
    return (float(x), float(y), float(z))


def matrix_from_euler_xyz_deg(angles: Vec3) -> np.ndarray:
    return rotation_z(angles[2]) @ _rotation_y(angles[1]) @ _rotation_x(angles[0])


# glTF models are Y-up; converting to our Z-up assets turns their axes like this.
_GLTF_TO_ASSET = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])


def view_rotation(mask: np.ndarray, camera: Camera) -> np.ndarray:
    """The rotation an asset has if it was generated facing the camera as photographed.

    View-aligned generators (Pixal3D) build the model in the frame of the crop
    they were shown: its front faces the viewer along the line of sight through
    the crop's centre. This is that orientation, as an asset-to-world rotation.
    """
    rows, cols = np.nonzero(mask)
    u, v = (cols.min() + cols.max() + 1) / 2, (rows.min() + rows.max() + 1) / 2
    forward = np.array([(u - camera.width / 2) / camera.focal_px, -(v - camera.height / 2) / camera.focal_px, -1.0])
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, (0.0, 1.0, 0.0))
    right /= np.linalg.norm(right)
    crop_axes = np.stack([right, np.cross(right, forward), -forward], axis=1)
    return camera.matrix[:3, :3] @ crop_axes @ _GLTF_TO_ASSET.T


# The asset axis a view-aligned generator points the shown side of a model along: towards the viewer.
_ASSET_FRONT = _GLTF_TO_ASSET @ np.array([0.0, 0.0, 1.0])
# A model generated from the photo is never posed with its front turned further than this from the
# camera. The outline alone barely tells a box's front from its back, and a search free to turn it
# all the way round put a media console's closed back, and a row of books' page edges, to the camera.
_MAX_TURN_DEG = 90


def turned_away(rotation: np.ndarray, facing: np.ndarray) -> bool:
    """Whether `rotation` turns a model's front more than `_MAX_TURN_DEG` from where `facing` points it.

    Only the turn about the vertical counts; a front that points nearly
    straight up or down (a book lying on its back) is never turned away.
    """
    front, wanted = rotation @ _ASSET_FRONT, facing @ _ASSET_FRONT
    front[2] = wanted[2] = 0.0
    if np.linalg.norm(front) < 0.3 or np.linalg.norm(wanted) < 0.3:
        return False
    return front @ wanted / np.linalg.norm(front) / np.linalg.norm(wanted) < np.cos(np.radians(_MAX_TURN_DEG))


def hidden_by_walls(world: np.ndarray, walls: list[Wall], eye: np.ndarray) -> float:
    """Share of the points that a wall stands in front of, seen from `eye` (see `behind_walls`)."""
    return float(behind_walls(world, walls, eye).mean())


def behind_walls(world: np.ndarray, walls: list[Wall], eye: np.ndarray) -> np.ndarray:
    """Which of the points a wall stands in front of, seen from `eye`.

    A wall hides a point if the point is beyond its face and the line of sight
    to the point passes through the face. Sight lines that pass a wall's end
    do not count: the photo shows what is there, and a "wall" may be a door
    standing open into the room.
    """
    hidden = np.zeros(len(world), dtype=bool)
    for wall in walls:
        corner, along, normal, height = wall
        beyond = -((world - corner) @ normal)
        eye_side = (eye - corner) @ normal
        # Where each line of sight crosses the face's plane, as a share of the way to the point.
        crossing = eye_side / np.maximum(eye_side + beyond, 1e-9)
        hit = eye + crossing[:, None] * (world - eye)
        across = (hit - corner) @ along / (along @ along)
        on_face = (across >= 0) & (across <= 1) & (hit[:, 2] >= 0) & (hit[:, 2] <= height)
        hidden |= (beyond > _WALL_SLACK_M) & on_face & ~through_openings(wall, across, hit[:, 2])
    return hidden


def pull_forward(pose: "Pose", points: np.ndarray, walls: list[Wall], eye: np.ndarray) -> "Pose":
    """Bring a model that a wall hides towards the camera until it stands clear, shrinking it in proportion.

    Moved along its lines of sight and shrunk by the same share, a model keeps
    exactly the outline it had in the photo, so the fit found for it stands.
    Its height above the floor changes, though: it is no longer where the
    solver stood it. A model that cannot be cleared within `_MAX_PULL` is left alone.
    """
    world = pose.scale * points @ matrix_from_euler_xyz_deg(pose.rotate_xyz_deg).T + np.array(pose.translate)

    def hidden(share: float) -> float:
        return hidden_by_walls(eye + share * (world - eye), walls, eye)

    if hidden(1.0) <= _HIDDEN_OK or hidden(1 - _MAX_PULL) > _HIDDEN_OK:
        return pose
    near, far = 1 - _MAX_PULL, 1.0
    for _ in range(12):
        middle = (near + far) / 2
        near, far = (middle, far) if hidden(middle) <= _HIDDEN_OK else (near, middle)
    translate = eye + near * (np.array(pose.translate) - eye)
    return pose.model_copy(update={"scale": pose.scale * near, "translate": tuple(float(v) for v in translate)})


def lowest(heights: np.ndarray, trim: bool = False) -> float:
    """The height a model stands on: its lowest point, or with `trim` its lowest less a stray `_TRIM_SHARE` of it."""
    return float(np.percentile(heights, 100 * _TRIM_SHARE)) if trim else float(heights.min())


def placed_points(pose: "Pose", points: np.ndarray) -> np.ndarray:
    """The asset's points where `pose` puts them in the world."""
    return pose.scale * points @ matrix_from_euler_xyz_deg(pose.rotate_xyz_deg).T + np.array(pose.translate)


def reseat_on_floor(
    pose: "Pose", points: np.ndarray, mode: Literal["drop", "stretch"]
) -> tuple["Pose", np.ndarray | None] | None:
    """Bring a floor-standing model that floats back down onto the floor; None if it already stands on it.

    "drop" lowers it whole, which moves its outline down in the photo.
    "stretch" keeps its top where it is and lengthens it downwards, by at most
    `_MAX_STRETCH`, lowering it the rest of the way if that is not enough. For a
    model whose hidden lower part came out too short (a desk's legs behind a
    chair), that keeps the outline of what the photo shows.

    Returns the new pose, and for a stretch a reshaping matrix like
    `fit_to_box`'s: 4x4, acting on column vectors, taking the asset's points to
    a reshaped asset with its longest side 1 m and its pivot bottom-centre,
    which the returned pose places. The pose's scores are left as they were.
    """
    rotation = matrix_from_euler_xyz_deg(pose.rotate_xyz_deg)
    turned = pose.scale * points @ rotation.T
    world = turned + np.array(pose.translate)
    low, top = float(world[:, 2].min()), float(world[:, 2].max())
    if low <= _RESEAT_MIN_M:
        return None
    if mode == "drop":
        translate = np.array(pose.translate) - (0.0, 0.0, low)
        return pose.model_copy(update={"translate": tuple(float(v) for v in translate)}), None
    # Stretch about the top: z -> top - k (top - z), which takes `low` to 0 when k is not capped.
    k = min(top / (top - low), _MAX_STRETCH)
    stretch = np.diag([1.0, 1.0, k])
    rise = top * (1 - k)
    rise -= top - k * (top - low)  # what a capped stretch leaves above the floor is dropped
    shaped = turned @ stretch.T
    lowest, highest = shaped.min(axis=0), shaped.max(axis=0)
    pivot = np.array([(lowest[0] + highest[0]) / 2, (lowest[1] + highest[1]) / 2, lowest[2]])
    longest = float((highest - lowest).max())
    reshape = np.eye(4)
    reshape[:3, :3] = stretch @ (pose.scale * rotation) / longest
    reshape[:3, 3] = -pivot / longest
    translate = stretch @ np.array(pose.translate) + (0.0, 0.0, rise) + pivot
    return pose.model_copy(
        update={"translate": tuple(float(v) for v in translate), "rotate_xyz_deg": (0.0, 0.0, 0.0), "scale": longest}
    ), reshape


def outline_fit(
    world: np.ndarray,
    colors: np.ndarray,
    mask: np.ndarray,
    image: Image.Image,
    camera: Camera,
    frame: np.ndarray | None = None,
    occluders: np.ndarray | None = None,
) -> tuple[float, float]:
    """Silhouette IoU and colour error of points already placed in the world, as the solvers score them."""
    pixels, depth = _project((world - camera.matrix[:3, 3]) @ camera.matrix[:3, :3], camera)
    return _Target(mask, image, frame=frame, ignore=occluders).compare(pixels, depth, colors)


def solve_pose(
    object_id: str,
    points: np.ndarray,
    colors: np.ndarray,
    mask: np.ndarray,
    image: Image.Image,
    camera: Camera,
    depth_m: float,
    contact: tuple[np.ndarray, float] | None,
    guesses: tuple[np.ndarray, ...] = (),
    settle: bool = True,
    facets: tuple[np.ndarray, ...] | None = None,
    trim: bool = False,
    facing: np.ndarray | None = None,
) -> Pose:
    """Find the pose of an asset (given as coloured surface samples) that best matches the photo.

    `mask` is the object's mask in the photo and `depth_m` its distance from the
    camera. `contact` is a surface the object must touch, as a plane
    (unit normal pointing into the room, offset) with normal @ p = offset: the
    floor for things standing on it, a wall for things hanging on it.
    `guesses` are extra rotations to try besides the upright search. `settle`
    says the object rests on a level surface and should sit flat on it;
    `facets`, the model's triangle normals and areas, let its own flat
    surfaces say which way is up. With `trim` the model touches `contact`
    with its lowest surface, not a stray spike below it (see `lowest`). With
    `facing`, the rotation a view-aligned generator built the model in (see
    `view_rotation`), no rotation turns its front away from the camera.
    """
    target = _Target(mask, image)
    camera_rotation = camera.matrix[:3, :3]
    camera_position = camera.matrix[:3, 3]

    def evaluate(rotation: np.ndarray) -> tuple[float, float, float, float, np.ndarray]:
        """Score one candidate; returns (score, iou, color_error, scale, world translation)."""
        local = points @ (camera_rotation.T @ rotation).T
        low, high = local.min(axis=0), local.max(axis=0)
        offset = (low + high) / 2
        centred = local - offset

        # Start on the ray through the mask's centre at the measured depth, sized
        # so the projected box matches the mask's box, then correct for perspective.
        f = camera.focal_px
        centre = np.array(
            [(target.centre[0] - camera.width / 2) / f, -(target.centre[1] - camera.height / 2) / f, -1.0]
        ) * depth_m
        scale = depth_m / f * np.sqrt(target.size[0] * target.size[1] / ((high[0] - low[0]) * (high[1] - low[1])))
        for _ in range(3):
            pixels, depth = _project(scale * centred + centre, camera)
            factor, shift = target.box_match(pixels.min(axis=0), pixels.max(axis=0))
            scale *= factor
            shift *= depth_m / f
            centre += np.array([shift[0], -shift[1], 0.0])

        pixels, depth = _project(scale * centred + centre, camera)
        iou, color_error = target.compare(pixels, depth, colors)
        translation = camera_rotation @ (centre - scale * offset) + camera_position
        return iou - _COLOR_WEIGHT * color_error, iou, color_error, scale, translation

    best = None

    def consider(rotation: np.ndarray) -> None:
        nonlocal best
        if facing is not None and turned_away(rotation, facing):
            return
        result = evaluate(rotation)
        if best is None or result[0] > best[1][0]:
            best = (rotation, result)

    for rotation in guesses:
        consider(rotation)
    for upright in UPRIGHTS:
        for yaw in range(0, 360, _COARSE_YAW_DEG):
            consider(rotation_z(yaw) @ upright)
    coarse = best[0]
    for delta in range(-_COARSE_YAW_DEG + _FINE_YAW_DEG, _COARSE_YAW_DEG, _FINE_YAW_DEG):
        consider(rotation_z(delta) @ coarse)

    if settle:
        # View-aligned generators get the direction they are looked at from
        # right but the height wrong, building models that lean. A shelf's top or
        # a book's cover is level in life, so level it; that is worth more than
        # some of the outline, which a model built for the wrong height cannot match exactly anyway.
        levelled = level_rotation(facets, best[0]) if facets is not None else None
        result = evaluate(levelled) if levelled is not None else None
        if result is not None and result[1] >= best[1][1] - _LEVEL_IOU_COST:
            best = (levelled, result)
        else:
            # Otherwise tip the object flat onto its support, unless that visibly worsens the fit.
            settled = settle_rotation(points, best[0])
            if settled is not None:
                result = evaluate(settled)
                if result[1] >= best[1][1] - _SETTLE_IOU_COST:
                    best = (settled, result)

    rotation, (_, iou, color_error, scale, translation) = best
    if contact is not None:
        # Distance and size trade off exactly in a single view. Slide the asset
        # along the camera's line of sight, resizing to keep its image fixed,
        # until its nearest point touches the surface.
        normal, offset = contact
        nearest = lowest(scale * (points @ rotation.T) @ normal, trim) + translation @ normal - offset
        camera_gap = camera_position @ normal - offset
        factor = camera_gap / (camera_gap - nearest)
        scale *= factor
        translation = camera_position + factor * (translation - camera_position)

    return Pose(
        id=object_id,
        translate=tuple(float(v) for v in translation),
        rotate_xyz_deg=euler_xyz_deg(rotation),
        scale=float(scale),
        iou=round(float(iou), 3),
        color_error=round(float(color_error), 3),
    )


def level_rotation(facets: tuple[np.ndarray, ...], rotation: np.ndarray) -> np.ndarray | None:
    """Turn a model the small amount that makes its main near-level surface level; None if it has none.

    `facets` are the model's triangle normals and areas in its own frame. The
    surface may face up or down (a shelf's top and its underside are the same
    evidence), and must be within the settling limit of level already.
    """
    normals, areas = facets[:2]
    normals = normals @ rotation.T
    near = np.abs(normals[:, 2]) > np.cos(np.radians(_MAX_SETTLE_DEG + _LEVEL_SPREAD_DEG))
    if not near.any():
        return None
    upward = normals[near] * np.sign(normals[near, 2])[:, None]
    weights = areas[near]
    together = np.cos(np.radians(_LEVEL_SPREAD_DEG))

    # Start from a spread of the candidates and let each drift to the middle of the surface around it.
    rng = np.random.default_rng(0)
    best_share, best_normal = 0.0, None
    for normal in upward[rng.choice(len(upward), min(len(upward), 24), replace=False, p=weights / weights.sum())]:
        for _ in range(5):
            members = upward @ normal > together
            normal = (upward[members] * weights[members, None]).sum(axis=0)
            normal /= np.linalg.norm(normal)
        share = weights[upward @ normal > together].sum() / areas.sum()
        if share > best_share:
            best_share, best_normal = share, normal
    if best_share < _MIN_LEVEL_SHARE or best_normal[2] < np.cos(np.radians(_MAX_SETTLE_DEG)):
        return None
    turn = _turn_onto(best_normal, np.array([0.0, 0.0, 1.0]))
    return None if turn is None else turn @ rotation


def _turn_onto(normal: np.ndarray, target: np.ndarray) -> np.ndarray | None:
    """The smallest rotation taking unit vector `normal` to `target` (Rodrigues' formula); None under a degree."""
    axis = np.cross(normal, target)
    sine, cosine = np.linalg.norm(axis), normal @ target
    if sine < np.sin(np.radians(1)):
        return None
    k = axis / sine
    cross = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + sine * cross + (1 - cosine) * cross @ cross


def settle_rotation(points: np.ndarray, rotation: np.ndarray) -> np.ndarray | None:
    """Tip a nearly-resting object so that it sits flat; None if it already does.

    A rigid object on a level surface rests on a face of its convex hull. Find
    the largest hull face that points roughly downwards and turn the object the
    small amount needed to make that face level.
    """
    rng = np.random.default_rng(0)
    sample = points[rng.choice(len(points), min(len(points), 4000), replace=False)] @ rotation.T
    hull = ConvexHull(sample)
    normals = hull.equations[:, :3]
    corners = sample[hull.simplices]
    areas = np.linalg.norm(np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]), axis=1) / 2

    downward = normals[:, 2] < -np.cos(np.radians(_MAX_SETTLE_DEG))
    if not downward.any():
        return None
    normals, areas = normals[downward], areas[downward]
    # A flat face is many small triangles with nearly the same normal: total their areas.
    same_face = normals @ normals.T > np.cos(np.radians(5))
    face = same_face[np.argmax(same_face @ areas)]
    normal = (normals[face] * areas[face, None]).sum(axis=0)
    normal /= np.linalg.norm(normal)

    turn = _turn_onto(normal, np.array([0.0, 0.0, -1.0]))
    return None if turn is None else turn @ rotation


def solve_occluded_pose(
    object_id: str,
    points: np.ndarray,
    colors: np.ndarray,
    visible: np.ndarray,
    occluders: np.ndarray,
    frame: np.ndarray,
    image: Image.Image,
    camera: Camera,
    start_scale: float,
    start_base: Vec3,
    guesses: tuple[np.ndarray, ...] = (),
    facets: tuple[np.ndarray, ...] | None = None,
    fixed_scale: bool = False,
    walls: list[Wall] | None = None,
    wall_weight: float = 0.0,
    trim: bool = False,
    distances: np.ndarray | None = None,
    depth_weight: float = 0.0,
    expected_size: float | None = None,
    size_weight: float = 0.0,
    facing: np.ndarray | None = None,
    turn_within_deg: float | None = None,
) -> Pose:
    """Place a complete model of an object that the photo shows only partly.

    The visible mask is too small to size the model from, so start from where
    the proxy box put the object (`start_scale` for its longest side, and
    `start_base`, the bottom-centre of its box) and adjust size and position on
    its support. A candidate is scored on the part of its outline that is not
    behind `occluders`, against the `visible` mask. `frame` is a mask covering
    roughly where the whole object may project, to size the comparison window.
    `facets`, the model's triangle normals and areas, let its flat surfaces
    level each candidate rotation before it is tried. `fixed_scale` keeps
    `start_scale` as it is, for a model whose true size is already known.
    With a `wall_weight`, the share of a candidate that `walls` hide from the
    camera counts against it: what shows of the object fits as well bigger and
    further back, and nothing else stops a candidate backing through a wall.

    The outline alone does not pin down a hidden object's size or distance,
    so two more terms may count against a candidate: with a `depth_weight`,
    how far its visible surface is from where `distances` (the depth map,
    per pixel, along the camera's axis) puts the object; with a `size_weight`,
    how far its size is from `expected_size`. `trim` stands it on its lowest
    surface, not on a stray spike below it. With `facing`, as in `solve_pose`,
    no rotation turns the model's front away from the camera. With
    `turn_within_deg`, the only rotations tried are the `guesses` turned about
    the vertical by up to that much: a model known to stand upright is not
    tipped onto a side (the office desk, searched over every upright, was laid
    on its back), nor turned side-on (the kitchen's fridge, whose outline from
    the side fits as well).
    """
    target = _Target(visible, image, frame=frame, ignore=occluders, distances=distances if depth_weight else None)
    camera_rotation = camera.matrix[:3, :3]
    camera_position = camera.matrix[:3, 3]

    def place(rotation: np.ndarray, scale: float, x: float, y: float) -> np.ndarray:
        """World translation that stands the turned, scaled model with its box's bottom-centre at (x, y)."""
        turned = scale * points @ rotation.T
        low, high = turned.min(axis=0), turned.max(axis=0)
        return np.array([x - (low[0] + high[0]) / 2, y - (low[1] + high[1]) / 2, start_base[2] - lowest(turned[:, 2], trim)])

    def evaluate(rotation: np.ndarray, scale: float, x: float, y: float) -> tuple[float, float, float]:
        world = scale * points @ rotation.T + place(rotation, scale, x, y)
        pixels, depth = _project((world - camera_position) @ camera_rotation, camera)
        if (depth <= 0.05).any():
            return -1.0, 0.0, 1.0
        iou, color_error = target.compare(pixels, depth, colors)
        score = iou - _COLOR_WEIGHT * color_error
        if wall_weight and walls:
            score -= wall_weight * hidden_by_walls(world, walls, camera_position)
        if depth_weight and (off := target.depth_disagreement(pixels, depth)) is not None:
            score -= depth_weight * off
        if size_weight and expected_size:
            score -= size_weight * abs(np.log(scale / expected_size))
        return score, iou, color_error

    start = (start_scale, start_base[0], start_base[1])
    if turn_within_deg is None:
        grid = [rotation_z(yaw) @ upright for upright in UPRIGHTS for yaw in range(0, 360, _COARSE_YAW_DEG)]
    else:
        reach = int(turn_within_deg) // _COARSE_YAW_DEG * _COARSE_YAW_DEG
        grid = [rotation_z(yaw) @ guess for guess in guesses for yaw in range(-reach, reach + 1, _COARSE_YAW_DEG)]
    if facing is not None:
        grid = [rotation for rotation in grid if not turned_away(rotation, facing)]
    # Refining every rotation is too slow, so refine the guesses and the few
    # grid rotations that look best at the starting size and position.
    ranked = sorted(grid, key=lambda rotation: -evaluate(rotation, *start)[0])

    candidates = [*guesses, *ranked[:3]]
    if facets is not None:
        # Generated models lean (see `solve_pose`), whichever way round they are tried.
        candidates = [levelled if (levelled := level_rotation(facets, r)) is not None else r for r in candidates]
    best = None
    for rotation in candidates:
        # Search size as a ratio so it cannot go negative.
        found = minimize(
            lambda v: -evaluate(rotation, start_scale * np.exp(0.0 if fixed_scale else v[0]), v[1], v[2])[0],
            x0=(0.0, start_base[0], start_base[1]),
            method="Nelder-Mead",
            options={"maxiter": 120, "xatol": 0.005, "fatol": 0.002, "initial_simplex": [
                (0.0, start_base[0], start_base[1]),
                (0.15, start_base[0], start_base[1]),
                (0.0, start_base[0] + 0.15 * start_scale, start_base[1]),
                (0.0, start_base[0], start_base[1] + 0.15 * start_scale),
            ]},
        )
        scale = start_scale if fixed_scale else start_scale * float(np.exp(found.x[0]))
        result = evaluate(rotation, scale, found.x[1], found.x[2])
        if best is None or result[0] > best[1][0]:
            best = ((rotation, scale, found.x[1], found.x[2]), result)

    (rotation, scale, x, y), (_, iou, color_error) = best
    return Pose(
        id=object_id,
        translate=tuple(float(v) for v in place(rotation, scale, x, y)),
        rotate_xyz_deg=euler_xyz_deg(rotation),
        scale=float(scale),
        iou=round(float(iou), 3),
        color_error=round(float(color_error), 3),
    )


def fit_to_box(
    solved: Pose,
    points: np.ndarray,
    colors: np.ndarray,
    size: Vec3,
    base: Vec3,
    yaw_deg: float,
    visible: np.ndarray,
    occluders: np.ndarray,
    frame: np.ndarray,
    image: Image.Image,
    camera: Camera,
    facets: tuple[np.ndarray, ...] | None = None,
    walls: list[Wall] = (),
    eye: np.ndarray | None = None,
    max_stretch: float = _MAX_STRETCH,
    turns: tuple[float, ...] = (0.0,),
    facing: np.ndarray | None = None,
    guide: bool = False,
    shift_within: float = 0.0,
) -> tuple[Pose, np.ndarray | None] | None:
    """Stretch a posed model onto the box measured for its object; returns the new pose and a reshaping matrix.

    A model generated from a redrawn picture has the right look but not the
    right proportions: the generator misjudges the height it is seen from and
    builds a shelf too deep and too tall. The depth map measured the real
    object's box (`size`, bottom-centre `base`, turned `yaw_deg`), so each of
    the model's dimensions is stretched to it, by at most `max_stretch` either way.

    The matrix (4x4, acting on column vectors) takes the asset's points to a
    reshaped asset that stands upright in the box's frame with its longest side
    1 m and its pivot bottom-centre; the returned pose places that asset.
    `facets`, the model's triangle normals, areas and centres, let the height
    be matched at the model's top surface.

    Boxes are measured from what shows of an object, and for one that is
    mostly hidden the box is poor. So the model is posed both ways and the
    better outline kept: returns None, keeping its own proportions, unless the
    fitted model matches the photo better than `solved` does. Both are judged
    as the camera at `eye` sees them, without what `walls` hide: a model too
    long in its own proportions can reach into a wall. Kept regardless,
    the stretch cost 12 of 15 redrawn models fit (0.06 on average, a box measured
    too short squashing a tray); it saved a refrigerator whose own model was a
    slab 2.65 m long.

    `turns` (degrees about the vertical) are the model's turns within the box to try after squaring it, the best
    outline kept: a box from elsewhere (RoomPlan's) holds the object whichever side is its front. With `facing`, a
    turn that faces the model's front away from the camera (`turned_away`) is not tried. With `guide`, the box is
    RoomPlan's, not measured from what shows, and only guides the model: it is turned onto the box's heading and stood
    at its place whether or not the outline fits better, the outline choosing only among `turns`, but keeps its own
    shape and size; no reshaping matrix is returned (None). Stretched onto RoomPlan's boxes, models stood up to 1.6
    times their box's height and filled it only to the stretch's limit, so RoomPlan's boxes are guides, not for
    reshaping the assets. With `shift_within` (metres), the outline
    also chooses where in plan the box stands, within that of `base`: a box from elsewhere is only so exact.
    """
    to_box = rotation_z(-yaw_deg) @ matrix_from_euler_xyz_deg(solved.rotate_xyz_deg)
    # Stretching a model that sits askew in the box would shear it, so first
    # square it up: turn it until its own footprint lines up with the box's sides.
    askew = _footprint_angle_deg(points @ to_box.T) if facets is None else _upright_faces_angle_deg(facets, to_box)
    squared = rotation_z(-askew) @ to_box

    def seen(world: np.ndarray) -> tuple[float, float]:
        hidden = behind_walls(world, walls, eye) if walls else np.zeros(len(world), dtype=bool)
        if hidden.all():
            return 0.0, 1.0
        return outline_fit(world[~hidden], colors[~hidden], visible, image, camera, frame, occluders)

    def stretched(to_box: np.ndarray) -> tuple[np.ndarray, float, tuple[float, float]]:
        """The reshaping matrix, the longest side and the outline fit, for the model turned by `to_box` into the box."""
        turned = solved.scale * points @ to_box.T
        extent = turned.max(axis=0) - turned.min(axis=0)
        if facets is not None and not guide:
            # The box's height was measured to the object's top surface, which on a
            # model can sit below its highest point (a raised rim, a warped corner).
            normals, areas, centres = facets
            level = (normals @ to_box.T)[:, 2] > np.cos(np.radians(_LEVEL_SPREAD_DEG))
            if level.any():
                heights = solved.scale * (centres[level] @ to_box.T)[:, 2] - turned[:, 2].min()
                bins = np.floor(heights / 0.01).astype(int)
                totals = np.bincount(bins - bins.min(), weights=areas[level])
                top = (np.nonzero(totals >= 0.25 * totals.max())[0][-1] + bins.min() + 0.5) * 0.01
                if top > 0.6 * extent[2]:
                    extent[2] = top
        stretch = np.ones(3) if guide else np.clip(np.array(size) / extent, 1 / max_stretch, max_stretch)
        shaped = turned * stretch
        low, high = shaped.min(axis=0), shaped.max(axis=0)
        pivot = np.array([(low[0] + high[0]) / 2, (low[1] + high[1]) / 2, low[2]])
        longest = float((high - low).max())
        reshape = np.eye(4)
        reshape[:3, :3] = np.diag(stretch * solved.scale / longest) @ to_box
        reshape[:3, 3] = -pivot / longest
        world = (shaped - pivot) @ rotation_z(yaw_deg).T + np.array(base)
        # Coarse then fine, over offsets in plan within `shift_within` of the box's place.
        best = (seen(world), np.zeros(3))
        for step, around in ((shift_within / 2, np.zeros(3)), (shift_within / 8, None)):
            if step <= 0:
                break
            centre = best[1] if around is None else around
            for dx in (-2, -1, 0, 1, 2):
                for dy in (-2, -1, 0, 1, 2):
                    offset = centre + (dx * step, dy * step, 0.0)
                    if np.hypot(offset[0], offset[1]) > shift_within + 1e-9 or not (dx or dy):
                        continue
                    found = seen(world + offset)
                    if found[0] > best[0][0]:
                        best = (found, offset)
        return reshape, longest, best[0], best[1], to_box, pivot

    tried = [rotation_z(turn) @ squared for turn in turns]
    tried = [t for t in tried if facing is None or not turned_away(rotation_z(yaw_deg) @ t, facing)] or [squared]
    reshape, longest, (iou, color_error), offset, turn, pivot = max(
        (stretched(t) for t in tried), key=lambda found: found[2][0]
    )
    if guide:
        # Its own shape and size, turned and stood in the box: the asset is placed, not reshaped.
        rotation = rotation_z(yaw_deg) @ turn
        pose = Pose(
            id=solved.id,
            translate=tuple(float(v) for v in np.array(base) + offset - rotation_z(yaw_deg) @ pivot),
            rotate_xyz_deg=euler_xyz_deg(rotation),
            scale=solved.scale,
            iou=round(float(iou), 3),
            color_error=round(float(color_error), 3),
        )
        return pose, None
    if iou <= seen(placed_points(solved, points))[0]:
        return None
    pose = Pose(
        id=solved.id,
        translate=tuple(float(v) for v in np.array(base) + offset),
        rotate_xyz_deg=(0.0, 0.0, float(yaw_deg)),
        scale=longest,
        iou=round(float(iou), 3),
        color_error=round(float(color_error), 3),
    )
    return pose, reshape


def _square_turnings() -> list[np.ndarray]:
    """The 24 rotations that take a box's axes onto its axes: every side up, turned four ways."""
    found = []
    for order in ((0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)):
        for signs in np.ndindex(2, 2, 2):
            turn = np.zeros((3, 3))
            turn[range(3), order] = np.where(np.array(signs) == 0, 1.0, -1.0)
            if np.linalg.det(turn) > 0:
                found.append(turn)
    return found


SQUARE_TURNINGS = _square_turnings()


def in_box(
    object_id: str,
    points: np.ndarray,
    colors: np.ndarray,
    size: Vec3,
    base: Vec3,
    yaw_deg: float,
    seen: Callable[[np.ndarray, np.ndarray], tuple[float, float]],
    up: np.ndarray | None = None,
) -> tuple[Pose, np.ndarray]:
    """A model stretched onto its item's gate 4 box (`size` along the box's x, y and up, bottom-centre `base`, turned
    `yaw_deg`): the box gives the item's place and proportions, the photo only which way the model sits in it. Each of
    the 24 square turnings of the asset (`SQUARE_TURNINGS`: any side up, turned four ways; a generated asset's own axes
    follow its object's sides) is stretched onto the box and judged by `seen` (world points and their colours to
    outline fit and colour error); the best outline kept. A model generated from the photo cannot see a thin item's
    depth: OfficeScan 007's notebook came out 13 cm deep for its 3 cm, and the outline alone stood it on edge.

    With `up`, the asset's direction the photo's pose stood upright (in the asset's own frame), only the four turnings
    that keep the asset's side nearest it up are tried: the outline can't tell which way up a roundish thing stands
    (OfficeScan 007's potted plant, laid on its side, won 0.74 to 0.71).

    Returns the pose and the reshaping matrix (4x4, on column vectors), as `fit_to_box`'s: the reshaped asset stands
    upright in the box's frame, its longest side 1 m, its pivot bottom-centre."""
    turnings = SQUARE_TURNINGS
    if up is not None:
        side = np.zeros(3)
        side[np.argmax(np.abs(up))] = np.sign(up[np.argmax(np.abs(up))])
        turnings = [turn for turn in SQUARE_TURNINGS if np.allclose(turn[2], side)]
    best = None
    for turn in turnings:
        turned = points @ turn.T
        low, high = turned.min(axis=0), turned.max(axis=0)
        stretch = np.asarray(size) / np.maximum(high - low, 1e-6)
        shaped = turned * stretch
        pivot = np.array([(low[0] + high[0]) / 2 * stretch[0], (low[1] + high[1]) / 2 * stretch[1], low[2] * stretch[2]])
        world = (shaped - pivot) @ rotation_z(yaw_deg).T + np.asarray(base)
        fit = seen(world, colors)
        if best is None or fit[0] > best[0][0]:
            longest = float(max(size))
            reshape = np.eye(4)
            reshape[:3, :3] = np.diag(stretch / longest) @ turn
            reshape[:3, 3] = -pivot / longest
            best = (fit, reshape, longest)
    (iou, color_error), reshape, longest = best
    pose = Pose(id=object_id, translate=tuple(float(v) for v in base), rotate_xyz_deg=(0.0, 0.0, float(yaw_deg)),
                scale=longest, iou=round(float(iou), 3), color_error=round(float(color_error), 3))
    return pose, reshape



def _upright_faces_angle_deg(facets: tuple[np.ndarray, ...], rotation: np.ndarray) -> float:
    """How far, within 45 degrees either way, a turned model's upright faces are from square to the axes.

    Furniture is mostly panels at right angles, so its fronts, backs and sides
    all agree on one direction once angles are counted modulo a quarter turn.
    That agreement is steadier than the footprint's outline, which legs and
    overhangs pull about.
    """
    normals, areas = facets[0] @ rotation.T, facets[1]
    upright = np.abs(normals[:, 2]) < 0.3
    if not upright.any():
        return 0.0
    # Multiplying an angle by four folds every quarter turn onto the same direction.
    angles = 4 * np.arctan2(normals[upright, 1], normals[upright, 0])
    mean = np.arctan2((areas[upright] * np.sin(angles)).sum(), (areas[upright] * np.cos(angles)).sum())
    return float(np.degrees(mean / 4))


def _footprint_angle_deg(points: np.ndarray) -> float:
    """How far, within 45 degrees either way, the points' footprint is turned from square to the axes.

    The footprint's direction is that of the smallest rectangle that holds it,
    which always lies along one edge of its convex hull.
    """
    rng = np.random.default_rng(0)
    plan = points[rng.choice(len(points), min(len(points), 4000), replace=False), :2]
    hull = plan[ConvexHull(plan).vertices]
    best_area, best_angle = np.inf, 0.0
    for start, end in zip(hull, np.roll(hull, -1, axis=0)):
        angle = np.arctan2(*(end - start)[::-1])
        along = np.array([np.cos(angle), np.sin(angle)])
        across = np.array([-along[1], along[0]])
        area = np.ptp(hull @ along) * np.ptp(hull @ across)
        if area < best_area:
            best_area, best_angle = area, np.degrees(angle)
    return float((best_angle + 45) % 90 - 45)


def _project(points_camera: np.ndarray, camera: Camera) -> tuple[np.ndarray, np.ndarray]:
    """Pixel coordinates and depth of points already in the camera's frame."""
    depth = -points_camera[:, 2]
    pixels = np.stack(
        [
            camera.width / 2 + camera.focal_px * points_camera[:, 0] / depth,
            camera.height / 2 - camera.focal_px * points_camera[:, 1] / depth,
        ],
        axis=1,
    )
    return pixels, depth


class _Target:
    """The object's mask and pixels, resampled into a small window for comparison."""

    def __init__(
        self,
        mask: np.ndarray,
        image: Image.Image,
        frame: np.ndarray | None = None,
        ignore: np.ndarray | None = None,
        distances: np.ndarray | None = None,
    ):
        """`frame` widens the window beyond the mask; `ignore` marks pixels that say nothing
        about this object (other objects in front of it). The part of the window beyond
        the photo's edge says nothing either: an object cut off by the frame carries on there.
        `distances`, the depth map along the camera's axis (NaN where unknown), is kept
        for `depth_disagreement`."""
        rows, cols = np.nonzero(mask)
        self.centre = np.array([cols.min() + cols.max() + 1, rows.min() + rows.max() + 1]) / 2
        self.size = np.array([cols.max() + 1 - cols.min(), rows.max() + 1 - rows.min()], dtype=float)
        # Which ends of the mask's box the photo's edge cuts, low (left, top) then high (right, bottom), by axis.
        self.cut = np.array([
            [cols.min() == 0, rows.min() == 0],
            [cols.max() == mask.shape[1] - 1, rows.max() == mask.shape[0] - 1],
        ])

        rows, cols = np.nonzero(mask if frame is None else mask | frame)
        low = np.array([cols.min(), rows.min()], dtype=float)
        span = np.array([cols.max() + 1, rows.max() + 1], dtype=float) - low
        # A margin around the mask, so a candidate that spills outside it is penalised.
        self.origin = low - 0.25 * span
        self.pixels_per_cell = 1.5 * span.max() / _GRID
        extent = np.ceil(1.5 * span / self.pixels_per_cell).astype(int)
        self.shape = (int(extent[1]), int(extent[0]))
        box = (*self.origin, *(self.origin + extent * self.pixels_per_cell))
        size = (self.shape[1], self.shape[0])

        # The window can hang over the photo's edge; `transform` fills that part
        # with black where `resize` would refuse.
        def window(picture: Image.Image) -> np.ndarray:
            return np.asarray(picture.transform(size, Image.Transform.EXTENT, box, Image.Resampling.BILINEAR))

        def window_mask(binary: np.ndarray) -> np.ndarray:
            return window(Image.fromarray(binary.astype(np.uint8) * 255)) > 127

        self.mask = window_mask(mask)
        self.rgb = window(image.convert("RGB")).astype(np.float32) / 255
        self.ignore = ~window_mask(np.ones(mask.shape, dtype=bool))
        if ignore is not None:
            self.ignore |= window_mask(ignore)
        self.distances = None
        if distances is not None:
            known = np.isfinite(distances) & (distances > 0)
            sampled = window(Image.fromarray(np.where(known, distances, 0).astype(np.float32)))
            self.distances = np.where(window_mask(known), sampled, np.nan)

    def box_match(self, lo: np.ndarray, hi: np.ndarray) -> tuple[float, np.ndarray]:
        """The size factor and pixel shift that put a projected box (`lo`, `hi`) onto the mask's box.

        Where the photo's edge cuts the mask, the object carries on past it, so that end
        says nothing: the size comes from the axes cut at neither end, and an axis cut at
        one end is lined up by its other. A chair whose base runs off the bottom of the
        photo, fitted whole inside the cut box, stood too small and, put on the floor,
        too far back. It still reaches at least to the edge:
        a squatter model sized by its width alone stopped short of it, and stood too far
        back the other way.
        """
        whole = ~self.cut.any(axis=0)
        if whole.all():
            factor = float(np.sqrt(np.prod(self.size) / np.prod(hi - lo)))
        elif whole.any():
            factor = float((self.size / (hi - lo))[whole][0])
        else:
            factor = 1.0
        factor = max([factor, *(self.size / (hi - lo))[~whole]])
        middle = (lo + hi) / 2
        lo, hi = middle + factor * (lo - middle), middle + factor * (hi - middle)
        shift = self.centre - middle
        for axis in (0, 1):
            low_cut, high_cut = self.cut[:, axis]
            if high_cut and not low_cut:
                shift[axis] = self.centre[axis] - self.size[axis] / 2 - lo[axis]
            elif low_cut and not high_cut:
                shift[axis] = self.centre[axis] + self.size[axis] / 2 - hi[axis]
        return factor, shift

    def depth_disagreement(self, pixels: np.ndarray, depth: np.ndarray) -> float | None:
        """How far the points' visible surface is from the measured distances, as the median |log| ratio
        over the object's visible pixels that they cover; None without distances, or if they cover too few."""
        if self.distances is None:
            return None
        cells = np.floor((pixels - self.origin) / self.pixels_per_cell).astype(int)
        inside = (cells >= 0).all(axis=1) & (cells[:, 0] < self.shape[1]) & (cells[:, 1] < self.shape[0])
        # Write far points first so that the nearest point in each cell wins.
        order = np.argsort(-depth[inside])
        cells = cells[inside][order]
        nearest = np.full(self.shape, np.nan)
        nearest[cells[:, 1], cells[:, 0]] = depth[inside][order]
        judged = self.mask & ~self.ignore & np.isfinite(nearest) & np.isfinite(self.distances) & (nearest > 0)
        if judged.sum() < 10:
            return None
        return float(np.median(np.abs(np.log(nearest[judged] / self.distances[judged]))))

    def compare(self, pixels: np.ndarray, depth: np.ndarray, colors: np.ndarray) -> tuple[float, float]:
        """Splat the points into the window; return (silhouette IoU, colour error)."""
        cells = np.floor((pixels - self.origin) / self.pixels_per_cell).astype(int)
        inside = (cells >= 0).all(axis=1) & (cells[:, 0] < self.shape[1]) & (cells[:, 1] < self.shape[0])
        # Write far points first so that the nearest point in each cell wins.
        order = np.argsort(-depth[inside])
        cells, shades = cells[inside][order], colors[inside][order]
        silhouette = np.zeros(self.shape, dtype=bool)
        rendered = np.zeros((*self.shape, 3), dtype=np.float32)
        silhouette[cells[:, 1], cells[:, 0]] = True
        rendered[cells[:, 1], cells[:, 0]] = shades

        silhouette &= ~self.ignore
        union = (silhouette | self.mask).sum()
        both = silhouette & self.mask
        iou = both.sum() / union if union else 0.0
        if both.sum() < 10:
            return float(iou), 1.0
        # Match overall brightness first: the photo's lighting is not the asset's.
        gain = self.rgb[both].mean() / max(rendered[both].mean(), 1e-6)
        return float(iou), float(np.abs(rendered[both] * gain - self.rgb[both]).mean())
