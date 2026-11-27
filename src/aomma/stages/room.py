"""Room shell: find the walls in the point map and stand a slab in for each."""

import numpy as np
from PIL import Image
from scipy import ndimage

from pydantic import BaseModel

from aomma.layout import Camera, Vec3
from aomma.stages.pose import Wall
from aomma.stages.proxy import Opening, Proxy
from aomma.stages.score import project

_MAX_WALLS = 4
_SAMPLE = 20000
_TRIALS = 400
# Points within this distance of a wall's plane belong to it, in meters.
_INLIER_M = 0.04
# The depth map smears a wall into a layer thicker than that (15 cm on the bedroom's blue wall); every
# point this close to a found wall's plane is part of it, so the layer is not fitted again as more walls.
_LAYER_M = 0.25
# A wall's face stands behind its layer, and is moved further back, by at most this much, until the
# objects' own depth points within its span are in front of it: the photo shows them, so no wall hides them.
# Points further behind are seen through an opening (a doorway or window; see `cut_openings`).
_MAX_PUSH_M = 0.3
_IN_FRONT_SLACK_M = 0.03
# How far in front of a wall's face its own points can be: its whole layer, and how far it was set back.
WALL_REACH_M = _LAYER_M + _MAX_PUSH_M
# How far past the photo's edge a wall it shows running off the frame is carried (see `reach_frame`).
_PAST_FRAME_M = 0.5
# An opening's wall is the one the wall pixels within this share of the photo's width of it stand on: the
# office's door frame keeps its door 11 pixels (1%) from the wall's.
_AROUND_OPENING = 0.05
# A wall must hold at least this share of the background points and be this tall.
_MIN_SHARE = 0.06
_MIN_HEIGHT_M = 0.5
_THICKNESS_M = 0.05
# Walls are drawn at least this tall, since photos rarely show the ceiling.
_ROOM_HEIGHT_M = 2.4
# Walls within this angle of square to the largest wall are made exactly square.
_SQUARE_UP_DEG = 8
# Two walls this near parallel, each end of the shorter this close to the longer's line, are one wall (see
# `merge_walls`). Less than a chimney breast stands out (30 cm and more). Starting values.
_SAME_LINE_DEG = 5
_SAME_LINE_M = 0.15
# Two walls whose ends are both within this distance of where their lines cross share a corner there.
_CORNER_REACH_M = 1.0
# A window's frame is looked for in 1 cm steps across each side of its cut, from this far inside it to this far out,
# against the wall's colour this far out. The frame's outer edge is where the wall starts: the first step from which on
# the photo is within `_FRAME_CONTRAST` of the wall's colour (8-bit RGB), and it must stand within `_FRAME_EDGE_M` of
# the cut, since the window's mask takes in some of its frame (OfficeScan 005, the bedroom: the edge 0-2 cm out); a side
# that reads otherwise (shading, a lamp beside it: the office) is not counted. Only the photo pixels whose points stand
# within `_FRAME_NEAR_M` of the wall's face show it.
_FRAME_INSIDE_M = 0.12
_FRAME_OUTSIDE_M = 0.15
_FRAME_STEP_M = 0.01
_FRAME_WALL_M = (0.15, 0.25)
_FRAME_CONTRAST = 40.0
_FRAME_EDGE_M = 0.05
_FRAME_NEAR_M = 0.3
# Photo textures: texels per meter, and the largest texture side.
_TEXELS_PER_M = 160
_MAX_TEXTURE = 2048
# A photo pixel shows the surface if its 3D point is within this distance of it, in meters.
_ON_SURFACE_M = 0.08
# A baseboard is looked for in the band this high above the floor, in 1 cm steps, among points this close to
# the wall's surface (it may stand up to this far behind the slab's face). It is this thick, and between these heights.
_BASEBOARD_BAND_M = 0.30
_BASEBOARD_BEHIND_M = 0.05
_BASEBOARD_STEP_M = 0.01
_BASEBOARD_NEAR_M = 0.04
BASEBOARD_THICKNESS_M = 0.015
_BASEBOARD_HEIGHT_M = (0.04, 0.25)
# A step of the band is baseboard if its median colour is this far (sRGB, 0-255) from the wall's, and the
# photo tells whether there is one only from this many points spread over this share of the wall's length.
_BASEBOARD_CONTRAST = 30.0
# Up its height a baseboard changes colour by less than this from one step to the next (the bookshelf's: 66,
# where it meets the floor); from a white baseboard to the wall above, the colour jumps by more (140 there).
_BASEBOARD_STEP_CHANGE = 80.0
_BASEBOARD_MIN_POINTS = 60
_BASEBOARD_SPREAD = 0.2
# The wall's surface a baseboard stands on is measured in runs this long along the wall, from at least this many of
# the wall's own points up to twice the band's height.
_SURFACE_RUN_M = 0.2
_SURFACE_RUN_MIN_POINTS = 20
# A wall has no baseboard only if the photo shows its bottom over this share of its length, this close (sRGB,
# 0-255) to its colour just above.
_PLAIN_SPREAD = 0.4
_PLAIN_CONTRAST = 15.0


class Baseboard(BaseModel):
    height: float
    # Linear RGB 0-1.
    color: Vec3


class BaseboardSighting(BaseModel):
    """What the photo shows of one wall's baseboard."""

    # The baseboard the photo shows, if it shows one.
    baseboard: Baseboard | None = None
    # The photo shows the wall's bottom well and in the wall's own colour: it has no baseboard.
    plain: bool = False
    # How many pixels show the baseboard: the more, the more its height and colour are to be trusted.
    points: int = 0


def fit_walls(
    image: Image.Image,
    points_world: np.ndarray,
    background: np.ndarray,
    camera_position: np.ndarray,
    in_front: np.ndarray | None = None,
    max_walls: int = _MAX_WALLS,
    tidy: bool = True,
    min_height: float = _MIN_HEIGHT_M,
) -> list[Proxy]:
    """Fit upright slabs to the walls among the `background` pixels (not floor, not objects).

    Walls are vertical, so seen from above each is a straight line: find the
    line most background points sit on, take it out with the layer around it,
    and repeat. `in_front` are world points the photo shows of the objects;
    a wall is set back behind those near it (see `_MAX_PUSH_M`). With
    `tidy`, the walls are made to agree with each other (see `tidy_walls`).
    A wall must span `min_height` from its lowest point to its highest: lower
    it for pixels already known to be one wall, where a backsplash between
    counter and wall cabinets may be all that shows.
    """
    rng = np.random.default_rng(0)
    points = points_world[background]
    pixels = np.asarray(image)[background].astype(np.float32) / 255
    if len(points) > _SAMPLE:
        keep = rng.choice(len(points), _SAMPLE, replace=False)
        points, pixels = points[keep], pixels[keep]
    total = len(points)

    walls = []
    while len(walls) < max_walls and len(points) >= 2:
        plan = points[:, :2]
        best = np.zeros(len(plan), dtype=bool)
        for _ in range(_TRIALS):
            a, b = plan[rng.choice(len(plan), 2, replace=False)]
            along = b - a
            length = np.linalg.norm(along)
            if length < 0.2:
                continue
            normal = np.array([-along[1], along[0]]) / length
            inliers = np.abs((plan - a) @ normal) < _INLIER_M
            if inliers.sum() > best.sum():
                best = inliers
        if best.sum() < _MIN_SHARE * total:
            break

        # Refit the line to all of its points, then collect its points again.
        centre = plan[best].mean(axis=0)
        along = np.linalg.svd(plan[best] - centre, full_matrices=False)[2][0]
        normal = np.array([-along[1], along[0]])
        if normal @ (camera_position[:2] - centre) < 0:
            normal = -normal  # face the room
        ahead = (plan - centre) @ normal
        on_wall = np.abs(ahead) < _INLIER_M
        layer = np.abs(ahead) < _LAYER_M
        heights = points[on_wall, 2]

        if np.percentile(heights, 99) - np.percentile(heights, 1) >= min_height:
            span_of = (plan - centre) @ along
            start, end = np.percentile(span_of[on_wall], [1, 99])
            # The slab's front face sits at the back of the wall's layer, and behind the objects against it. Only
            # the layer along the wall's own length: past its end a wall turning away behind it (OfficeScan 005's
            # notch, beside the poster wall) set the poster wall 18 cm back.
            face = min(0.0, float(np.percentile(ahead[layer & (span_of > start) & (span_of < end)], 2)))
            if in_front is not None and len(in_front):
                offset = (in_front[:, :2] - centre) @ normal
                span = (in_front[:, :2] - centre) @ along
                near = (span > start) & (span < end) & (offset < face) & (offset > face - _MAX_PUSH_M)
                if near.any():
                    face = max(face - _MAX_PUSH_M, min(face, float(np.percentile(offset[near], 2)) - _IN_FRONT_SLACK_M))
            middle = centre + along * (start + end) / 2 + normal * (face - _THICKNESS_M / 2)
            walls.append(
                Proxy(
                    id=f"wall_{len(walls) + 1}",
                    size=(float(end - start), _THICKNESS_M, float(max(np.percentile(heights, 99), _ROOM_HEIGHT_M))),
                    translate=(float(middle[0]), float(middle[1]), 0.0),
                    yaw_deg=float(np.degrees(np.arctan2(along[1], along[0]))),
                    # The median ignores pictures and shadows on the wall.
                    color=tuple(float(c) for c in np.median(pixels[on_wall], axis=0) ** 2.2),
                )
            )
        points, pixels = points[~layer], pixels[~layer]
    return tidy_walls(walls) if tidy else walls


# A survey wall and the wall it meets are square to each other within this (as `_SQUARE_UP_DEG` squares walls up).
_END_SQUARE_DEG = _SQUARE_UP_DEG
# Points a wall's or a built-in's own pixels are read from stand at least this high (the floor's edge bleeds into masks).
_END_ABOVE_FLOOR_M = 0.15
# Fewer of its own points than this say nothing of where a mask lies.
_END_MIN_POINTS = 200
# Of a mask's points, these lowest and highest shares are its spread's noise, not its extent.
_END_SPREAD = (5, 95)


def built_in_ends(
    wall_masks: dict[str, np.ndarray], built_masks: dict[str, np.ndarray], points_world: np.ndarray, same_m: float,
) -> dict[str, str]:
    """Survey walls that are a built-in's end, each with that built-in.

    A built-in stands out from the wall behind it as far as its front. A surface square to that wall, beside the
    built-in, that lies wholly between the wall and the built-in's front (never past it, within `same_m`) is the
    built-in's end panel, not a wall: KitchenScan 010's tall cabinet column, whose beige end the survey named
    `back_wall`. A wall the room carries on past, a step or an archway's side, runs on past
    any built-in's front. Read in plan, from the run's own points; the wall behind is another survey wall's mask.
    """
    valid = np.isfinite(points_world).all(axis=-1)
    with np.errstate(invalid="ignore"):
        high = valid & (points_world[..., 2] > _END_ABOVE_FLOOR_M)
    built_area = np.any(list(built_masks.values()), axis=0) if built_masks else np.zeros(high.shape, bool)

    def plan(mask: np.ndarray) -> np.ndarray | None:
        found = points_world[mask & high][:, :2]
        return found if len(found) >= _END_MIN_POINTS else None

    def line(found: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        centre = found.mean(axis=0)
        _, _, vt = np.linalg.svd(found - centre, full_matrices=False)
        return centre, vt[1]

    own = {oid: plan(mask & ~built_area) for oid, mask in wall_masks.items()}
    lines = {oid: line(found) for oid, found in own.items() if found is not None}
    near = ndimage.generate_binary_structure(2, 2)
    ends = {}
    square = np.sin(np.radians(_END_SQUARE_DEG))
    for oid, found in own.items():
        if found is None:
            continue
        _, normal = lines[oid]
        beside = ndimage.binary_dilation(wall_masks[oid], near, iterations=3)
        for built_id, built_mask in sorted(built_masks.items(), key=lambda kv: -int((beside & kv[1]).sum())):
            if not (beside & built_mask).any() or (built := plan(built_mask)) is None:
                continue
            for other, (centre, across) in lines.items():
                if other == oid or abs(float(normal @ across)) > square:
                    continue
                # Toward the room: the side of the wall behind that the built-in stands on.
                standing = (built - centre) @ across
                if np.median(standing) < 0:
                    across, standing = -across, -standing
                back, front = np.percentile(standing, _END_SPREAD)
                # The built-in stands against that wall: it runs along it further than it stands out from it (a run
                # of cabinets along the surface itself measures its length here, not its depth).
                along = np.ptp(np.percentile(built @ np.array([across[1], -across[0]]), _END_SPREAD))
                lo, hi = np.percentile((found - centre) @ across, _END_SPREAD)
                if (front - back < along and back >= -same_m and front > same_m and lo >= -same_m
                        and hi <= front + same_m):
                    ends[oid] = built_id
                    break
            if oid in ends:
                break
    return ends


def wall_behind(built_in: Proxy, camera_position: np.ndarray) -> Proxy:
    """A wall standing against the back of `built_in`, for a wall the photo shows only covered by it.

    The kitchen's wall over the fridge shows nothing but its cabinets. The box's
    back is its long side further from the camera; the wall spans the box's width.
    """
    yaw = np.radians(built_in.yaw_deg)
    along, across = np.array([np.cos(yaw), np.sin(yaw)]), np.array([-np.sin(yaw), np.cos(yaw)])
    length, depth = built_in.size[0], built_in.size[1]
    if length < depth:
        along, across, length, depth = across, along, depth, length
    centre = np.array(built_in.translate[:2])
    if across @ (centre - camera_position[:2]) < 0:
        across = -across  # away from the camera
    middle = centre + across * (depth + _THICKNESS_M) / 2
    return Proxy(
        id=f"behind_{built_in.id}",
        size=(float(length), _THICKNESS_M, _ROOM_HEIGHT_M),
        translate=(float(middle[0]), float(middle[1]), 0.0),
        yaw_deg=float(np.degrees(np.arctan2(along[1], along[0]))),
        color=built_in.color,
    )


def merge_walls(walls: list[Proxy]) -> list[Proxy]:
    """Walls lying on one line made one: the longer's line, carried over both's ends.

    The survey can name one wall twice: OfficeScan 005's desk wall came as a wall and, at its window, a
    "back wall", fitted 4 cm apart and built as two slabs with a step between them.
    """
    merged: list[Proxy] = []
    for wall in sorted(walls, key=lambda w: -w.size[0]):
        yaw = np.radians(wall.yaw_deg)
        ends = np.array(wall.translate[:2]) + np.outer([-0.5, 0.5], [np.cos(yaw), np.sin(yaw)]) * wall.size[0]
        for index, kept in enumerate(merged):
            turn = (wall.yaw_deg - kept.yaw_deg) % 180
            along = np.array([np.cos(np.radians(kept.yaw_deg)), np.sin(np.radians(kept.yaw_deg))])
            normal = np.array([-along[1], along[0]])
            centre = np.array(kept.translate[:2])
            if min(turn, 180 - turn) < _SAME_LINE_DEG and np.abs((ends - centre) @ normal).max() < _SAME_LINE_M:
                span = np.concatenate([[-kept.size[0] / 2, kept.size[0] / 2], (ends - centre) @ along])
                middle = centre + along * (span.min() + span.max()) / 2
                merged[index] = kept.model_copy(update={
                    "size": (float(span.max() - span.min()), kept.size[1], max(kept.size[2], wall.size[2])),
                    "translate": (float(middle[0]), float(middle[1]), 0.0),
                })
                break
        else:
            merged.append(wall)
    return merged


def tidy_walls(walls: list[Proxy]) -> list[Proxy]:
    """Make separately fitted walls agree: square corners, shared corner lines, one height.

    Each wall is fitted to its own points, so neighbours come out a degree or
    two off square, stop short of the corner, and differ slightly in height.
    """
    if not walls:
        return walls
    height = max(wall.size[2] for wall in walls)
    ends = []
    for index, wall in enumerate(walls):
        yaw = wall.yaw_deg
        # Rooms are nearly always rectangular: snap to the first (largest) wall's grid.
        turn = (yaw - walls[0].yaw_deg) % 90
        if index and min(turn, 90 - turn) < _SQUARE_UP_DEG:
            yaw -= turn if turn < 45 else turn - 90
        along = np.array([np.cos(np.radians(yaw)), np.sin(np.radians(yaw))])
        centre = np.array(wall.translate[:2])
        ends.append([centre - along * wall.size[0] / 2, centre + along * wall.size[0] / 2])

    # Where two walls' lines cross near an end of each, that crossing is the corner.
    for a in range(len(ends)):
        for b in range(a + 1, len(ends)):
            direction_a, direction_b = ends[a][1] - ends[a][0], ends[b][1] - ends[b][0]
            cross = direction_a[0] * direction_b[1] - direction_a[1] * direction_b[0]
            if abs(cross) < 1e-6:
                continue  # parallel
            gap = ends[b][0] - ends[a][0]
            corner = ends[a][0] + direction_a * (gap[0] * direction_b[1] - gap[1] * direction_b[0]) / cross
            nearest = [min((0, 1), key=lambda end: np.linalg.norm(ends[w][end] - corner)) for w in (a, b)]
            if all(np.linalg.norm(ends[w][end] - corner) < _CORNER_REACH_M for w, end in zip((a, b), nearest)):
                ends[a][nearest[0]], ends[b][nearest[1]] = corner, corner

    tidied = []
    for wall, (start, end) in zip(walls, ends):
        along = end - start
        middle = (start + end) / 2
        tidied.append(
            wall.model_copy(
                update={
                    "size": (float(np.linalg.norm(along)), wall.size[1], height),
                    "translate": (float(middle[0]), float(middle[1]), 0.0),
                    "yaw_deg": float(np.degrees(np.arctan2(along[1], along[0]))),
                }
            )
        )
    return tidied


def wall_face(wall: Proxy, camera_position: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A wall slab's room-facing face as (bottom-left corner, vector along it, unit normal into the room)."""
    yaw = np.radians(wall.yaw_deg)
    along = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    normal = np.array([-np.sin(yaw), np.cos(yaw), 0.0])
    centre = np.array(wall.translate)
    if normal @ (camera_position - centre) < 0:
        normal = -normal
    # Seen from the room with the normal towards the viewer, "left" is this way.
    left_to_right = np.cross(normal, (0.0, 0.0, 1.0))
    origin = centre + normal * wall.size[1] / 2 - left_to_right * wall.size[0] / 2
    return origin, left_to_right * wall.size[0], normal


def reach_frame(walls: list[Proxy], wall_pixels: np.ndarray, points_world: np.ndarray, camera: Camera) -> list[Proxy]:
    """Carry each wall the photo shows running off its left or right edge on past the frame.

    A wall is fitted from its own depth points, which thin out near the photo's
    edge, so it ends inside the frame and the render shows nothing beyond it.
    For each `wall_pixels` pixel in the edge
    column, the wall its depth point stands on is found, and that wall's end
    is moved to where the pixel's line of sight meets it, and `_PAST_FRAME_M`
    beyond. A wall already reaching that far is left as it is.
    """
    if not walls:
        return walls
    height, width = wall_pixels.shape
    eye = camera.matrix[:3, 3]
    faces = [wall_face(wall, eye) for wall in walls]
    reach = [[0.0, wall.size[0]] for wall in walls]
    for column in (0, width - 1):
        for row in np.nonzero(wall_pixels[:, column])[0]:
            point = points_world[row, column]
            if not np.isfinite(point).all():
                continue
            off = [abs((point - origin) @ normal) for origin, _, normal in faces]
            nearest = int(np.argmin(off))
            if off[nearest] > WALL_REACH_M:
                continue
            origin, along, normal = faces[nearest]
            # The line of sight through the pixel's centre, in the camera's full-resolution frame.
            u, v = (column + 0.5) * camera.width / width, (row + 0.5) * camera.height / height
            sight = camera.matrix[:3, :3] @ np.array([u - camera.width / 2, camera.height / 2 - v, -camera.focal_px])
            facing = sight @ normal
            if abs(facing) < 1e-9:
                continue
            hit = eye + sight * ((origin - eye) @ normal) / facing
            spot = (hit - origin) @ along / (along @ along) * np.linalg.norm(along)
            reach[nearest][0] = min(reach[nearest][0], spot - _PAST_FRAME_M)
            reach[nearest][1] = max(reach[nearest][1], spot + _PAST_FRAME_M)
    reached = []
    for wall, (origin, along, _), (start, end) in zip(walls, faces, reach):
        if start == 0.0 and end == wall.size[0]:
            reached.append(wall)
            continue
        unit = along / np.linalg.norm(along)
        # The face's centre moves along it; the slab's centre is behind the face by the same amount.
        shift = unit * ((start + end) / 2 - wall.size[0] / 2)
        reached.append(wall.model_copy(update={
            "size": (float(end - start), wall.size[1], wall.size[2]),
            "translate": (float(wall.translate[0] + shift[0]), float(wall.translate[1] + shift[1]), wall.translate[2]),
        }))
    return reached


def _sight(camera: Camera, rows: np.ndarray, columns: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """The lines of sight through the centres of these pixels of a picture of `shape`, in the world frame."""
    height, width = shape
    u, v = (columns + 0.5) * camera.width / width, (rows + 0.5) * camera.height / height
    return np.stack([u - camera.width / 2, camera.height / 2 - v, np.full(len(u), -camera.focal_px)], axis=-1) @ camera.matrix[:3, :3].T


def cut_openings(
    walls: list[Proxy],
    openings: dict[str, tuple[np.ndarray, float, bool]],
    wall_pixels: np.ndarray,
    points_world: np.ndarray,
    camera: Camera,
) -> list[Proxy]:
    """Cut each door, doorway and window through the wall it stands in, carrying that wall on past it if need be.

    `openings` holds each one's mask, listed width, and whether it is a window.
    Its wall is the one the `wall_pixels` around its mask stand on; its sides are
    where its pixels' lines of sight meet that wall's face, between the 2nd and
    98th percentiles, and its top the 98th percentile of their heights. A
    window's bottom, its sill, is their 2nd percentile; a door's is the floor. One near the photo's side
    edge is made at least its listed width, towards the edge. The wall
    is carried `_PAST_FRAME_M` past a door's or doorway's sides: what the photo shows
    through a doorway at its edge (the office's hall, the bedroom's bathroom)
    once left the wall short of the frame. A window's
    wall is carried only as far as the window.
    """
    if not walls:
        return walls
    shape = wall_pixels.shape
    eye = camera.matrix[:3, 3]
    faces = [wall_face(wall, eye) for wall in walls]
    reach = [[0.0, wall.size[0]] for wall in walls]
    cuts: list[list[tuple[str, float, float, float, float, bool]]] = [[] for _ in walls]
    for oid, (mask, listed_width, window) in openings.items():
        if not mask.any():
            continue
        ring = (ndimage.distance_transform_edt(~mask) <= _AROUND_OPENING * shape[1]) & ~mask & wall_pixels
        around = points_world[ring]
        around = around[np.isfinite(around).all(axis=1)]
        if not len(around):
            continue
        off = np.stack([np.abs((around - origin) @ normal) for origin, _, normal in faces])
        nearest = off.argmin(axis=0)[off.min(axis=0) <= WALL_REACH_M]
        if not len(nearest):
            continue
        index = int(np.bincount(nearest).argmax())
        origin, along, normal = faces[index]
        length = float(np.linalg.norm(along))

        def spots(rows: np.ndarray, columns: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            """Where these pixels' lines of sight meet the wall's face: metres along it, and height."""
            sight = _sight(camera, rows, columns, shape)
            facing = sight @ normal
            towards = np.where(np.abs(facing) > 1e-9, ((origin - eye) @ normal) / np.where(facing == 0, 1, facing), -1)
            hit = eye + sight[towards > 0] * towards[towards > 0, None]
            return (hit - origin) @ along / length, hit[:, 2]

        rows, columns = np.nonzero(mask)
        if len(rows) > _SAMPLE:
            keep = np.random.default_rng(0).choice(len(rows), _SAMPLE, replace=False)
            rows, columns = rows[keep], columns[keep]
        along_m, up_m = spots(rows, columns)
        if len(along_m) < 2:
            continue
        low, high = (float(v) for v in np.percentile(along_m, [2, 98]))
        top = float(np.percentile(up_m, 98))
        bottom = max(float(np.percentile(up_m, 2)), 0.0) if window else 0.0
        # Near a side of the frame: the side towards that edge is unknown (cut off, or what shows is a door
        # standing open, the office's seen edge on), and the listed width stands.
        near = int(_AROUND_OPENING * shape[1])
        for edge, strip in ((0, mask[:, :near]), (shape[1] - 1, mask[:, -near:])):
            if strip.any() and high - low < listed_width:
                edge_rows = np.nonzero(strip.any(axis=1))[0]
                at_edge = spots(edge_rows, np.full(len(edge_rows), edge))[0]
                if len(at_edge) and abs(np.median(at_edge) - high) < abs(np.median(at_edge) - low):
                    high = low + listed_width
                else:
                    low = high - listed_width
        top = min(top, walls[index].size[2] - _THICKNESS_M)
        if top <= bottom:
            continue
        cuts[index].append((oid, low, high, bottom, top, window))
        # A window's wall shows round it, so the wall need only hold it: carried 0.5 m past OfficeScan 007's left
        # window, inside the frame, its wall ran 0.5 m past the room's corner.
        past = 0.0 if window else _PAST_FRAME_M
        reach[index][0] = min(reach[index][0], low - past)
        reach[index][1] = max(reach[index][1], high + past)

    cut = []
    for wall, (origin, along, _), (start, end), wall_cuts in zip(walls, faces, reach, cuts):
        if not wall_cuts:
            cut.append(wall)
            continue
        unit = along / np.linalg.norm(along)
        shift = unit * ((start + end) / 2 - wall.size[0] / 2)
        placed = [
            Opening(
                id=oid, ends=tuple(tuple(float(v) for v in (origin + unit * side)[:2]) for side in (low, high)), top=top, bottom=bottom,
                window=window,
            )
            for oid, low, high, bottom, top, window in wall_cuts
        ]
        cut.append(wall.model_copy(update={
            "size": (float(end - start), wall.size[1], wall.size[2]),
            "translate": (float(wall.translate[0] + shift[0]), float(wall.translate[1] + shift[1]), wall.translate[2]),
            "openings": wall.openings + placed,
        }))
    return cut


def opening_spans(wall: Proxy) -> list[tuple]:
    """Each of the wall's openings as `write_wall` takes it: (from, to, bottom, top) along the slab's own x axis about
    its centre and up it, clipped to the slab, then a window's pane colour and frame (out, width, colour).

    Openings that overlap along the wall are one, from the lower bottom to the higher top, with the first one's
    pane and frame: the bedroom's survey listed its window twice.
    """
    yaw = np.radians(wall.yaw_deg)
    axis = np.array([np.cos(yaw), np.sin(yaw)])
    centre = np.array(wall.translate[:2])
    half = wall.size[0] / 2
    spans: list[tuple] = []
    for opening in wall.openings:
        sides = sorted(float((np.array(end) - centre) @ axis) for end in opening.ends)
        low, high = max(sides[0], -half), min(sides[1], half)
        if high > low:
            spans.append((low, high, opening.bottom, opening.top, opening.pane, opening.frame_out, opening.frame_width, opening.frame))
    merged: list[tuple] = []
    for span in sorted(spans, key=lambda span: span[:4]):
        if merged and span[0] <= merged[-1][1]:
            last = merged[-1]
            glazing = last[4:] if last[4] is not None else span[4:]
            merged[-1] = (last[0], max(last[1], span[1]), min(last[2], span[2]), max(last[3], span[3]), *glazing)
        else:
            merged.append(span)
    return merged


def sight_face(wall: Proxy, camera_position: np.ndarray) -> Wall:
    """The wall's face as the pose checks take it, with its doors and doorways as shares of the way along it, for lines
    of sight. A window is left out: its glass stops them as the wall does, as the depth map and the render have it."""
    origin, along, normal = wall_face(wall, camera_position)
    shares = []
    for opening in wall.openings:
        if opening.window:
            continue
        sides = sorted(float((np.array([*end, 0.0]) - origin) @ along / (along @ along)) for end in opening.ends)
        shares.append((sides[0], sides[1], opening.top))
    return Wall(origin, along, normal, wall.size[2], shares)


def glaze(wall: Proxy, image: Image.Image, points_world: np.ndarray, camera: Camera) -> Proxy:
    """The wall with each of its windows given a pane and, where the photo shows one, a frame.

    The pane is the median colour of what the photo shows in the window's cut (blinds, glass, what is past it),
    in front of the wall by no more than `_FRAME_NEAR_M`: a lamp before it is not the window. The frame is
    looked for across each side of the cut (see `_FRAME_INSIDE_M`): its outer edge is where the wall's colour
    starts, and inwards from there it is the run of steps like the two to four centimetres inside that edge, and
    unlike the wall, ending before `_FRAME_INSIDE_M` (a run that does not end is the glass or a blind). Its edge,
    width and colour are the medians over the sides that show it; a window whose sides mostly read plain has none.
    """
    windows = [opening for opening in wall.openings if opening.window]
    if not windows:
        return wall
    eye = camera.matrix[:3, 3]
    origin, along, normal = wall_face(wall, eye)
    unit = along / np.linalg.norm(along)
    photo = np.asarray(image.convert("RGB")).astype(np.float32)
    rows_of, cols_of = photo.shape[0] / camera.height, photo.shape[1] / camera.width

    def colour(a0: float, a1: float, z0: float, z1: float, behind: bool = False) -> np.ndarray | None:
        """The median colour (8-bit RGB) of the photo where it shows the stretch of the wall's face, None if unseen."""
        a, z = np.meshgrid(np.linspace(a0, a1, 40), np.linspace(z0, z1, 8))
        spots = origin + a.reshape(-1, 1) * unit + z.reshape(-1, 1) * np.array([0.0, 0.0, 1.0])
        with np.errstate(divide="ignore", invalid="ignore"):
            pixels, depth = project(spots, camera)
        pixels = np.nan_to_num(pixels, nan=-1.0, posinf=-1.0, neginf=-1.0)
        cols, rows = np.floor(pixels[:, 0]).astype(int), np.floor(pixels[:, 1]).astype(int)
        seen = (depth > 0.05) & (cols >= 0) & (cols < camera.width) & (rows >= 0) & (rows < camera.height)
        rows, cols = rows[seen], cols[seen]
        with np.errstate(invalid="ignore"):
            ahead = (points_world[rows, cols] - origin) @ normal
        shows = (ahead < _FRAME_NEAR_M) & (behind | (ahead > -_FRAME_NEAR_M))
        if shows.sum() < 5:
            return None
        return np.median(photo[(rows[shows] * rows_of).astype(int), (cols[shows] * cols_of).astype(int)], axis=0)

    glazed = []
    for opening in wall.openings:
        if not opening.window:
            glazed.append(opening)
            continue
        low, high = sorted(float((np.array([*end, 0.0]) - origin) @ unit) for end in opening.ends)
        bottom, top = opening.bottom, opening.top
        pane = colour(low, high, bottom, top, behind=True)
        # Each side: the band `d` metres outwards from it, `width` wide; inside the cut, what is past the glass counts.
        sides = {
            "low": lambda d, width: colour(low - d - width, low - d, bottom, top, d < 0),
            "high": lambda d, width: colour(high + d, high + d + width, bottom, top, d < 0),
            "bottom": lambda d, width: colour(low, high, bottom - d - width, bottom - d, d < 0),
            "top": lambda d, width: colour(low, high, top + d, top + d + width, d < 0),
        }
        steps = np.arange(-round(_FRAME_INSIDE_M / _FRAME_STEP_M), round(_FRAME_OUTSIDE_M / _FRAME_STEP_M) + 1)
        found, plain = [], 0
        for band in sides.values():
            wall_colour = band(_FRAME_WALL_M[0], _FRAME_WALL_M[1] - _FRAME_WALL_M[0])
            if wall_colour is None:
                continue
            seen = {int(k): band(k * _FRAME_STEP_M, _FRAME_STEP_M) for k in steps}
            like_wall = {k: c is not None and np.linalg.norm(c - wall_colour) < _FRAME_CONTRAST for k, c in seen.items()}
            # The outer edge: the first step from which on every seen step is the wall's.
            outer = next(
                (k for k in steps if seen[k] is not None and all(like_wall[j] for j in steps if j >= k and seen[j] is not None)), None,
            )
            if outer is None or abs(outer * _FRAME_STEP_M) > _FRAME_EDGE_M:
                continue
            inside = [seen[k] for k in range(outer - 4, outer - 1) if seen.get(k) is not None]
            if not inside:
                continue
            frame_colour = np.median(inside, axis=0)
            if np.linalg.norm(frame_colour - wall_colour) < _FRAME_CONTRAST:
                plain += 1
                continue
            # A frame ends; glass or a blind meeting the wall runs on in. The step at the edge blends frame and wall.
            width = next((outer - 1 - k for k in range(outer - 2, int(steps[0]) - 1, -1)
                          if seen[k] is None or np.linalg.norm(seen[k] - frame_colour) >= _FRAME_CONTRAST), None)
            if width is None:
                plain += 1
                continue
            found.append((outer * _FRAME_STEP_M, width * _FRAME_STEP_M, frame_colour))
        update: dict = {"pane": tuple(float(c) for c in (pane / 255) ** 2.2) if pane is not None else (0.5, 0.5, 0.5)}
        if found and len(found) >= plain:
            update |= {
                "frame_out": round(float(np.median([out for out, _, _ in found])), 3),
                "frame_width": round(float(np.median([width for _, width, _ in found])), 3),
                "frame": tuple(float(c) for c in (np.median([c for _, _, c in found], axis=0) / 255) ** 2.2),
            }
        glazed.append(opening.model_copy(update=update))
    return wall.model_copy(update={"openings": glazed})


def _on_wall_surface(
    points_world: np.ndarray, wall: Proxy, eye: np.ndarray, allowed: np.ndarray, wall_pixels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """The `allowed` pixels with a point (`valid`), and of those the ones whose points lie on the wall's surface within
    its length (`near`), with each one's place along the wall (0-1) and height. The surface is where the depth map puts
    the `wall_pixels` low on the wall (up to twice the baseboard's band), run by run along the wall (`_SURFACE_RUN_M`): a depth map can bow
    a wall by several centimetres from top to bottom (BedroomScan 007's back wall: 0-2 cm out at the top, 5-8 cm at the
    foot), more than a baseboard stands proud of it. A run with too few such pixels takes the wall's median there.
    None if too few of them show the wall at all."""
    found = _wall_surface_offsets(points_world, wall, eye, allowed, wall_pixels)
    if found is None:
        return None
    valid, off, along, height = found
    return valid, off < _BASEBOARD_NEAR_M, along, height


def wall_surface_offset(
    points_world: np.ndarray, wall: Proxy, eye: np.ndarray, allowed: np.ndarray, wall_pixels: np.ndarray,
) -> np.ndarray:
    """At each pixel, how far its point lies off the wall's surface as `_on_wall_surface` takes it (in front or
    behind); inf where it is not `allowed`, has no point, or lies past the wall's ends or out of its reach."""
    shown = np.full(points_world.shape[:2], np.inf)
    found = _wall_surface_offsets(points_world, wall, eye, allowed, wall_pixels)
    if found is not None:
        valid, off, _, _ = found
        shown[valid] = off
    return shown


def _wall_surface_offsets(
    points_world: np.ndarray, wall: Proxy, eye: np.ndarray, allowed: np.ndarray, wall_pixels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """`_on_wall_surface`'s work: the `allowed` pixels with a point (`valid`), and of those each one's distance off the
    wall's surface (inf past its ends or out of its reach), place along the wall and height."""
    origin, across, normal = wall_face(wall, eye)
    valid = allowed & np.isfinite(points_world).all(axis=-1)
    points = points_world[valid]
    offset = points - origin
    ahead, along, height = offset @ normal, offset @ across / (across @ across), points[:, 2]
    on_wall = (along >= 0) & (along <= 1) & (ahead > -_BASEBOARD_BEHIND_M) & (ahead < WALL_REACH_M)
    own = wall_pixels[valid] & on_wall & (height > _BASEBOARD_BAND_M)
    if own.sum() < _BASEBOARD_MIN_POINTS:
        return None
    surface = np.full(len(points), float(np.median(ahead[own])))
    runs = np.floor(along * np.linalg.norm(across) / _SURFACE_RUN_M)
    # The wall's own pixels nearest the floor, the band's included (a wall mask stops at its baseboard, so they are paint).
    low = wall_pixels[valid] & on_wall & (height < 2 * _BASEBOARD_BAND_M)
    for run in np.unique(runs[on_wall]):
        here = low & (runs == run)
        if here.sum() >= _SURFACE_RUN_MIN_POINTS:
            surface[runs == run] = float(np.median(ahead[here]))
    return valid, np.where(on_wall, np.abs(ahead - surface), np.inf), along, height


def wall_foot_pixels(
    points_world: np.ndarray, wall: Proxy, eye: np.ndarray, allowed: np.ndarray, wall_pixels: np.ndarray,
) -> np.ndarray:
    """The `allowed` pixels at the wall's foot: on its surface as `find_baseboard` takes it (within its length, near
    the surface the depth map shows just above), in the band a baseboard is looked for in. The segmenter's wall mask
    stops at the baseboard, taking the paint; these pixels are the wall's whether or not the baseboard's colour stands
    out enough to be found (BedroomScan 008's white on cream does not). An item standing there claims its own pixels
    and is ranked over the wall where both claim one."""
    return wall_surface_pixels(points_world, wall, eye, allowed, wall_pixels, below=_BASEBOARD_BAND_M)


def wall_surface_pixels(
    points_world: np.ndarray, wall: Proxy, eye: np.ndarray, allowed: np.ndarray, wall_pixels: np.ndarray,
    below: float = np.inf,
) -> np.ndarray:
    """The `allowed` pixels whose points lie on the wall's surface within its length (as `find_baseboard` takes it),
    lower than `below`."""
    shown = np.zeros(points_world.shape[:2], dtype=bool)
    found = _on_wall_surface(points_world, wall, eye, allowed, wall_pixels)
    if found is None:
        return shown
    valid, near, _, height = found
    shown[valid] = near & (height < below)
    return shown


def find_baseboard(
    image: Image.Image,
    points_world: np.ndarray,
    wall: Proxy,
    eye: np.ndarray,
    allowed: np.ndarray,
    wall_pixels: np.ndarray,
    wall_color: np.ndarray,
) -> BaseboardSighting:
    """What the photo shows of a baseboard along the bottom of a wall (see `BaseboardSighting`).

    Looks at the `allowed` pixels whose points lie on the wall's surface, within
    its length, in the band just above the floor. The surface is where the
    depth map puts the `wall_pixels` (the face may be set back behind it):
    a baseboard stands a centimetre or two proud of it, and the furniture
    against a wall (a nightstand, a bookshelf) further, so it is left out. The
    band is cut into 1 cm steps by height; the baseboard is the run of steps
    from the bottom up whose median colour is clearly not the wall's just above
    the band (or `wall_color`, linear RGB, if too little of that shows): a wall
    is often darker near the floor than its median. Its colour is that of its
    best-lit pixels, since a baseboard is mostly in the shade of what stands
    before it.
    """
    unseen = BaseboardSighting()
    found = _on_wall_surface(points_world, wall, eye, allowed, wall_pixels)
    if found is None:
        return unseen
    valid, near, along, height = found
    colours = np.asarray(image.convert("RGB"))[valid].astype(np.float32)
    above = near & (height >= _BASEBOARD_BAND_M) & (height < _BASEBOARD_BAND_M + 0.15)
    reference = np.median(colours[above], axis=0) if above.sum() >= _BASEBOARD_MIN_POINTS else np.asarray(wall_color) ** (1 / 2.2) * 255
    near &= (height > _BASEBOARD_STEP_M) & (height < _BASEBOARD_BAND_M)
    colours, height, along = colours[near], height[near], along[near]
    spread = len(np.unique(np.floor(along * 20))) / 20
    if len(colours) < _BASEBOARD_MIN_POINTS or spread < _BASEBOARD_SPREAD:
        return unseen

    # The run of steps from the bottom that are unlike the wall and each like the one below: the wall itself
    # can be unlike its colour higher up (lit differently under a shelf), but it does not shade off from a
    # white baseboard; the baseboard does change a little up its height (warmer where it meets the floor).
    top, below = None, None
    for step in range(1, int(round(_BASEBOARD_BAND_M / _BASEBOARD_STEP_M))):
        inside = (height >= step * _BASEBOARD_STEP_M) & (height < (step + 1) * _BASEBOARD_STEP_M)
        if inside.sum() < 5:
            if top is None:
                continue  # nothing seen this low: start from the lowest step that has points
            break
        colour = np.median(colours[inside], axis=0)
        if np.linalg.norm(colour - reference) < _BASEBOARD_CONTRAST or (below is not None and np.linalg.norm(colour - below) > _BASEBOARD_STEP_CHANGE):
            break
        top, below = (step + 1) * _BASEBOARD_STEP_M, colour
    if top is None or not _BASEBOARD_HEIGHT_M[0] <= top <= _BASEBOARD_HEIGHT_M[1]:
        # Any baseboard covers the lowest few centimetres, so plain wall there means none.
        lowest = colours[height < _BASEBOARD_HEIGHT_M[0] + 0.02]
        plain = (
            len(lowest) >= _BASEBOARD_MIN_POINTS and spread >= _PLAIN_SPREAD
            and float(np.linalg.norm(np.median(lowest, axis=0) - reference)) < _PLAIN_CONTRAST
        )
        return BaseboardSighting(plain=plain)
    band = colours[height < top]
    # Below its top edge, where the photo blends it with the wall.
    body = colours[height < max(top - 0.02, _BASEBOARD_HEIGHT_M[0])]
    if not len(body):
        body = band  # nothing seen that low: the step that ends it is all there is
    lit =body[body.sum(axis=1) >= np.percentile(body.sum(axis=1), 75)]
    color = (np.median(lit, axis=0) / 255) ** 2.2
    return BaseboardSighting(baseboard=Baseboard(height=round(float(top), 3), color=tuple(float(c) for c in color)), points=len(band))


def room_baseboards(sightings: dict[str, BaseboardSighting]) -> dict[str, Baseboard | None]:
    """One baseboard for every wall of a room, from what the photo shows of each.

    A room's walls nearly always share their baseboard, and a photo shows it
    well on one wall and poorly or not at all on the next (a bed in front, the
    shade under a desk, white trim on a light wall). So the baseboard seen over
    the most pixels is given to every wall, except one whose bottom the photo
    shows plain, in the wall's own colour down to the floor.
    """
    found = [sighting for sighting in sightings.values() if sighting.baseboard is not None]
    if not found:
        return {wall: None for wall in sightings}
    clearest = max(found, key=lambda sighting: sighting.points).baseboard
    return {wall: None if sighting.plain else clearest for wall, sighting in sightings.items()}

def surface_texture(
    image: Image.Image,
    points_world: np.ndarray,
    camera: Camera,
    origin: np.ndarray,
    across: np.ndarray,
    up: np.ndarray,
    allowed: np.ndarray | None = None,
    reach: float = _ON_SURFACE_M,
) -> tuple[Image.Image, np.ndarray]:
    """Paint a flat surface with the photo; returns the texture and its typical colour (linear RGB).

    The surface spans `origin`, `origin + across` and `origin + up`. Each texel
    takes the photo pixel it projects to, provided that pixel really shows the
    surface there and not something in front of it. Texels the photo cannot
    vouch for (hidden, or out of frame) get the surface's median colour.

    A pixel shows the surface if its 3D point is within `_ON_SURFACE_M` of it,
    or, for a surface set back behind a layer of its own points (a wall), up
    to `reach` in front of it, and if it is one of the `allowed` pixels.
    """
    width = int(np.clip(np.linalg.norm(across) * _TEXELS_PER_M, 8, _MAX_TEXTURE))
    height = int(np.clip(np.linalg.norm(up) * _TEXELS_PER_M, 8, _MAX_TEXTURE))
    u = (np.arange(width) + 0.5) / width
    v = 1 - (np.arange(height) + 0.5) / height  # image rows run downwards
    texels = origin + u[None, :, None] * across + v[:, None, None] * up

    with np.errstate(divide="ignore", invalid="ignore"):
        pixels, depth = project(texels.reshape(-1, 3), camera)
    pixels = np.nan_to_num(pixels, nan=-1.0, posinf=-1.0, neginf=-1.0)
    cols, rows = np.floor(pixels[:, 0]).astype(int), np.floor(pixels[:, 1]).astype(int)
    seen = (depth > 0.05) & (cols >= 0) & (cols < camera.width) & (rows >= 0) & (rows < camera.height)
    cols, rows = np.clip(cols, 0, camera.width - 1), np.clip(rows, 0, camera.height - 1)
    offset = points_world[rows, cols] - texels.reshape(-1, 3)
    with np.errstate(invalid="ignore"):
        if reach > _ON_SURFACE_M:
            normal = np.cross(across, up)
            normal /= np.linalg.norm(normal)
            if normal @ (camera.matrix[:3, 3] - origin) < 0:
                normal = -normal
            ahead = offset @ normal
            seen &= (ahead > -_ON_SURFACE_M) & (ahead < reach)
        else:
            seen &= np.linalg.norm(offset, axis=1) < _ON_SURFACE_M
    if allowed is not None:
        seen &= allowed[rows, cols]

    photo = np.asarray(image.convert("RGB"))
    typical = np.median(photo[rows[seen], cols[seen]], axis=0) if seen.any() else np.array([128.0, 128.0, 128.0])
    colours = np.where(seen[:, None], photo[rows, cols], typical).astype(np.uint8)
    return Image.fromarray(colours.reshape(height, width, 3)), (typical / 255) ** 2.2
