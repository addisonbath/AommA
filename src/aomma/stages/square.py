"""Squaring things up with each other: built-ins with the room's walls, large items with their neighbours.

Each box is fitted to its own depth points, so neighbours that stand square
and flush in the room come out a few degrees and centimetres apart. Two
passes take that out, each judged against the photo by the caller:

- A built-in runs along a wall, so it is turned onto the room's grid, the
  walls' direction (they are already square to each other, see `tidy_walls`),
  refitted at that heading, and its back put against the wall behind it. The
  kitchen's back base cabinets, one box over a diagonal corner cabinet and a
  straight run, fitted 28 deg off the wall they back onto.
- Large items that are box-shaped (see `squareness`) and stand next to each
  other or to a built-in are turned to the heading of the steadiest of them,
  then their fronts lined up and the gaps between their sides closed, where
  these nearly agree already.
"""

from dataclasses import dataclass, field

import numpy as np

from aomma.stages.large import OVER_SLACK_M, OVER_WITHIN_M
from aomma.stages.pose import Wall
from aomma.stages.proxy import Proxy
from aomma.stages.room import wall_face

# How box-shaped a model must be to be squared up with others: the agreement of its upright faces' directions,
# folded onto a quarter turn (1 for a box, 0 for a cylinder). Round things fell at 0.50 or under (stool 0.50,
# office chair 0.27, ottoman 0.03), boxy ones at 0.60 or over (fridge 0.81, washer 0.81, dryer 0.70, nightstands
# 0.66 and 0.90, cabinet 0.90).
SQUARE_SHARE = 0.6
# How far off the room's grid a built-in may be turned onto it. Built-ins run along walls; the kitchen's back
# base cabinets fitted 28 deg off.
BUILT_IN_WITHIN_DEG = 30.0
# A built-in's back is put against a wall whose face it stands within this of, before or behind.
BACK_TO_WALL_M = 0.15
# Items stand next to each other when their footprints come within this of each other, at the same level.
NEAR_M = 0.3
# How far apart two neighbours' headings may be for one to be turned to the other's.
ITEM_WITHIN_DEG = 10.0
# Neighbours' fronts that differ by no more than this are lined up.
FRONT_WITHIN_M = 0.1
# A gap (or overlap) between neighbours' sides no wider than this is closed.
GAP_WITHIN_M = 0.15
# A built-in's front, at the grid's heading, may run at 45 degrees at one end (a corner cabinet): over a stretch
# this long at least, and at most this long; kept where it fits the front's profile, the nearest points across
# each bin of its width, with no more than this share of a straight front's error, and the straight front is
# off by more than the noise.
CORNER_MIN_M = 0.15
CORNER_MAX_M = 0.9
CORNER_FIT_SHARE = 0.3
_PROFILE_BIN_M = 0.03
# A corner that stands out reaches back to a wall parallel to its diagonal no further than this behind it.
DIAGONAL_BACK_M = 1.2
# A wall cuts a corner (see `chamfered_corners`) where both walls' own points, those within `CHAMFER_ON_WALL_M`
# of them, stop between these distances short of it, and this many wall points stand in the gap.
CHAMFER_MIN_M = 0.15
CHAMFER_MAX_M = 1.0
CHAMFER_ON_WALL_M = 0.08
CHAMFER_MIN_POINTS = 200
_FLAT_NOISE_M = 0.03
# A squared built-in is kept if its depth points lie no more than this further from its box, on median.
BUILT_IN_KEEP_M = 0.01
# A lined-up item is kept if it loses no more than this of its outline fit.
ITEM_KEEP_FIT_DROP = 0.03


def fold_deg(angle: float) -> float:
    """An angle folded onto a quarter turn, within 45 degrees either way."""
    return (angle + 45) % 90 - 45


def squareness(normals: np.ndarray, areas: np.ndarray) -> tuple[float, float]:
    """A model's heading and how box-shaped it is, from its world-frame triangle normals and areas.

    The heading is its upright faces' direction folded onto a quarter turn,
    within 45 degrees either way; the share is how well those faces agree on it.
    """
    upright = np.abs(normals[:, 2]) < 0.3
    if not upright.any():
        return 0.0, 0.0
    angles = 4 * np.arctan2(normals[upright, 1], normals[upright, 0])
    sin, cos = (areas[upright] * np.sin(angles)).sum(), (areas[upright] * np.cos(angles)).sum()
    return float(np.degrees(np.arctan2(sin, cos) / 4)), float(np.hypot(sin, cos) / areas[upright].sum())


def room_grid_deg(walls: list[Proxy]) -> float | None:
    """The room's grid: the longest wall's direction folded onto a quarter turn; None without walls."""
    if not walls:
        return None
    return fold_deg(max(walls, key=lambda w: w.size[0]).yaw_deg)


def _axes(yaw_deg: float) -> np.ndarray:
    yaw = np.radians(yaw_deg)
    return np.array([[np.cos(yaw), np.sin(yaw)], [-np.sin(yaw), np.cos(yaw)]])


def _front_corner(along: np.ndarray, toward: np.ndarray, start: float, end: float) -> tuple[float, int, int, float] | None:
    """A 45-degree stretch at one end of a built-in's front, if its depth points show one.

    `along` is each point's place along the built-in's width (from `start` to
    `end`), `toward` how far it stands towards the camera. The front is the
    nearest points across each `_PROFILE_BIN_M` of the width. Returns the
    straight front's distance, the end (-1 at `start`, 1 at `end`), whether
    that end stands out towards the camera (1: a corner cabinet filling the
    inside corner of two runs) or is cut back (-1: a chamfered corner), and the
    stretch's length along the width, which is also its depth; None if a
    straight front fits as well, within `CORNER_FIT_SHARE` of the error.
    """
    edges = np.arange(start, end + _PROFILE_BIN_M, _PROFILE_BIN_M)
    centres, profile = [], []
    for low, high in zip(edges[:-1], edges[1:]):
        inside = (along >= low) & (along < high)
        if inside.sum() >= 5:
            centres.append((low + high) / 2)
            profile.append(np.percentile(toward[inside], 98))
    centres, profile = np.array(centres), np.array(profile)
    if len(profile) < 8:
        return None
    flat = float(((profile - np.median(profile)) ** 2).sum())
    best = None
    for side in (-1, 1):
        distance = np.abs(centres - (start if side < 0 else end))
        for length in np.arange(CORNER_MIN_M, min(CORNER_MAX_M, 0.8 * (end - start)), 0.02):
            rest = distance > length
            if rest.sum() < 3:
                continue
            front = float(np.median(profile[rest]))
            for out in (1, -1):
                error = float(((profile - front - out * np.maximum(0.0, length - distance)) ** 2).sum())
                if best is None or error < best[0]:
                    best = (error, front, side, out, float(length))
    if best is None or best[0] > CORNER_FIT_SHARE * flat or flat < len(profile) * _FLAT_NOISE_M ** 2:
        return None
    return best[1:]


def _meet(a: np.ndarray, u: np.ndarray, b: np.ndarray, v: np.ndarray) -> np.ndarray | None:
    """Where the line through `a` along `u` meets the line through `b` along `v`, in plan; None if they are parallel."""
    cross = u[0] * v[1] - u[1] * v[0]
    if abs(cross) < 1e-9:
        return None
    return a + u * ((b - a)[0] * v[1] - (b - a)[1] * v[0]) / cross


def _back_along_wall(
    tip: np.ndarray, foot: np.ndarray, back_start: np.ndarray, back_end: np.ndarray, walls: list[Wall],
) -> list[np.ndarray] | None:
    """The back of a diagonal corner, from `tip` to `foot` along the front, against a wall parallel to it.

    The wall must stand within `DIAGONAL_BACK_M` behind the diagonal, over at
    least part of it. Returns the back's corners in plan, from the tip's side:
    where a line straight back from `tip` meets the wall (or, if another wall
    comes first, where it meets that one, then where that one meets this),
    and where the wall meets the line of the straight back from `back_start`
    to `back_end`; None if there is no such wall.
    """
    diagonal = foot - tip
    backwards = np.array([-diagonal[1], diagonal[0]]) / np.linalg.norm(diagonal)
    if backwards @ (back_start + back_end - tip - foot) < 0:
        backwards = -backwards
    straight = back_end - back_start
    for corner, along, normal, _ in walls:
        corner, along, normal = corner[:2], along[:2], normal[:2]
        gap = float((tip - corner) @ normal)
        if normal @ backwards > -0.98 or not 0 < gap <= DIAGONAL_BACK_M:
            continue
        span = sorted(float((p - corner) @ along / (along @ along)) for p in (tip, foot))
        if span[1] < 0 or span[0] > 1:
            continue
        behind_tip = _meet(tip, backwards, corner, along)
        meeting = _meet(back_start, straight, corner, along)
        if behind_tip is None or meeting is None:
            continue
        # The meeting falls along the straight back, or its line out past the corner by up to `DIAGONAL_BACK_M`:
        # the wall behind the diagonal meets the back wall further out than the corner's tip.
        reach = DIAGONAL_BACK_M / np.linalg.norm(straight)
        if not -reach <= (meeting - back_start) @ straight / (straight @ straight) <= 1 + reach:
            continue
        # Straight back from the tip, another wall may come first: the kitchen's left wall, past the end of the
        # 45-degree wall. Then the back runs along that wall to where it meets this one.
        travel = float((behind_tip - tip) @ backwards)
        for other_corner, other_along, _, _ in walls:
            other_corner, other_along = other_corner[:2], other_along[:2]
            hit = _meet(tip, backwards, other_corner, other_along)
            if hit is None:
                continue
            spot = (hit - other_corner) @ other_along / (other_along @ other_along)
            ahead = float((hit - tip) @ backwards)
            if 0 < ahead < travel - 0.01 and -0.05 <= spot <= 1.05:
                junction = _meet(other_corner, other_along, corner, along)
                if junction is not None:
                    return [hit, junction, meeting]
        return [behind_tip, meeting]
    return None


def chamfered_corners(walls: list[Proxy], points: np.ndarray, eye: np.ndarray) -> tuple[list[Proxy], list[Proxy]]:
    """Walls with a 45-degree wall cutting a corner where the wall pixels show one; returns the walls and the new ones.

    Two walls square to each other meet at a corner. Where each one's own
    depth points (`points`, the wall pixels in the world frame) stop at least
    `CHAMFER_MIN_M` short of it, and at least `CHAMFER_MIN_POINTS` wall points
    stand in the gap, off both, a wall cuts the corner at 45 degrees to both:
    the kitchen's back wall turns through a short 45-degree wall to the left
    wall, which a fit of one straight wall per mask left out. It stands at
    the median distance of those points from the corner, along the 45-degree
    direction: the depth map smooths the corners at its ends, so the points'
    own line reads about 51 degrees. The two walls are cut back, or
    lengthened, to meet it.
    """
    faces = {}
    for wall in walls:
        corner, along, normal = (v[:2] for v in wall_face(wall, eye))
        faces[wall.id] = (corner, along / np.linalg.norm(along), normal)
    plan = points[:, :2]
    on = {
        wid: np.abs((plan - corner) @ normal) < CHAMFER_ON_WALL_M for wid, (corner, _, normal) in faces.items()
    }
    walls = list(walls)
    added = []
    for i in range(len(walls)):
        for j in range(i + 1, len(walls)):
            a, b = walls[i], walls[j]
            if abs((a.yaw_deg - b.yaw_deg) % 180 - 90) > 5:
                continue
            (corner_a, unit_a, normal_a), (corner_b, unit_b, normal_b) = faces[a.id], faces[b.id]
            meet = _meet(corner_a, unit_a, corner_b, unit_b)
            if meet is None or not on[a.id].any() or not on[b.id].any():
                continue
            # Along each wall, away from the corner, to the end of its own points nearest the corner.
            ends = []
            for wid, unit in ((a.id, unit_a), (b.id, unit_b)):
                reach = (plan[on[wid]] - meet) @ unit
                away = unit if np.median(reach) > 0 else -unit
                ends.append((away, float(np.percentile((plan[on[wid]] - meet) @ away, 1))))
            (away_a, short_a), (away_b, short_b) = ends
            if min(short_a, short_b) < CHAMFER_MIN_M or max(short_a, short_b) > CHAMFER_MAX_M:
                continue
            inward = (away_a + away_b) / np.linalg.norm(away_a + away_b)
            offset = (plan - meet) @ inward
            # In the gap: off both walls, within the triangle the two ends make with the corner.
            between = ~on[a.id] & ~on[b.id] & ((plan - meet) @ away_a > -CHAMFER_ON_WALL_M) & (
                (plan - meet) @ away_b > -CHAMFER_ON_WALL_M
            ) & (offset < max(short_a, short_b) / np.sqrt(2) + CHAMFER_ON_WALL_M)
            if between.sum() < CHAMFER_MIN_POINTS:
                continue
            reach = float(np.median(offset[between])) * np.sqrt(2)
            end_a, end_b = meet + away_a * reach, meet + away_b * reach
            for index, end, away in ((i, end_a, away_a), (j, end_b, away_b)):
                wall = walls[index]
                corner, unit, normal = faces[wall.id]
                length = float(np.linalg.norm(wall_face(wall, eye)[1][:2]))
                far = max((corner, corner + unit * length), key=lambda p: (p - meet) @ away)
                walls[index] = _wall_between(wall, end, far, normal)
            # The cut wall faces the room, its colour the larger neighbour's.
            larger = max((a, b), key=lambda w: w.size[0])
            added.append(_wall_between(
                larger.model_copy(update={"id": f"wall_{len(walls) + len(added) + 1}"}), end_a, end_b, inward,
            ))
    return walls + added, added


def _wall_between(wall: Proxy, start: np.ndarray, end: np.ndarray, normal: np.ndarray) -> Proxy:
    """`wall` with its room-facing face from `start` to `end`, facing along `normal` into the room."""
    along = end - start
    thickness = wall.size[1]
    centre = (start + end) / 2 - normal * thickness / 2
    return wall.model_copy(update={
        "size": (float(np.linalg.norm(along)), thickness, wall.size[2]),
        "translate": (float(centre[0]), float(centre[1]), wall.translate[2]),
        "yaw_deg": float(np.degrees(np.arctan2(along[1], along[0]))),
    })


def square_built_in(
    proxy: Proxy, points: np.ndarray, grid_deg: float, walls: list[Wall], expected_depth: float, eye: np.ndarray,
) -> Proxy | None:
    """A built-in's box turned onto the room's grid and refitted to its depth points there; None if nothing changed.

    `points` are its own depth points in the world frame. At the new heading
    its footprint is their extent (percentiles, as `fit_proxies` takes). Where
    its front runs at 45 degrees at one end (see `_front_corner`), its outline
    is the rectangle behind the straight front with that corner added or cut
    (`Proxy.footprint`): the kitchen's back run is a straight run of drawers
    and a diagonal corner cabinet filling the inside corner out to the left
    run's front. Its depth, from the straight front, is padded to
    `expected_depth` away from the camera at `eye`, and its back then put
    against the face of a wall in `walls` it stands within `BACK_TO_WALL_M` of.
    Its height is kept. None if it is within half a degree of the grid with a
    straight front, or more than `BUILT_IN_WITHIN_DEG` off the grid.
    """
    turn = fold_deg(grid_deg - proxy.yaw_deg)
    if abs(turn) > BUILT_IN_WITHIN_DEG or len(points) == 0:
        return None
    yaw = proxy.yaw_deg + turn
    axes = _axes(yaw)
    local = points[:, :2] @ axes.T
    low, high = np.percentile(local, [2, 98], axis=0)
    thin = int(np.argmin(high - low))
    wide = 1 - thin
    # Coordinates along the width, and towards the camera along the depth.
    out = -float(np.sign(((low + high) / 2 @ axes - eye[:2]) @ axes[thin]))
    toward = out * local[:, thin]
    start, end = float(low[wide]), float(high[wide])
    corner = _front_corner(local[:, wide], toward, start, end)
    if corner is None and abs(turn) < 0.5:
        return None
    front = float(np.percentile(toward, 98)) if corner is None else corner[0]
    back = min(float(np.percentile(toward, 2)), front - expected_depth)
    # Against the wall behind it: the face, square to its depth, that its back stands nearest.
    away = -out * axes[thin]
    behind = (start + end) / 2 * axes[wide] + out * back * axes[thin]
    gaps = []
    for wall_corner, wall_along, normal, _ in walls:
        across = (behind - wall_corner[:2]) @ wall_along[:2] / (wall_along[:2] @ wall_along[:2])
        if abs(normal[:2] @ away) > 0.98 and -0.5 <= across <= 1.5:
            gap = float((behind - wall_corner[:2]) @ normal[:2])
            if abs(gap) <= BACK_TO_WALL_M:
                gaps.append(gap)
    if gaps:
        # Grown or shrunk at the back, so its front stays where it was seen.
        back -= min(gaps, key=abs)

    outline = [(start, back), (end, back), (end, front), (start, front)]
    if corner is not None:
        _, side, sticks, length = corner
        if side < 0:
            outline[3:] = [(start + length, front), (start, front + sticks * length)]
        else:
            outline[2:3] = [(end, front + sticks * length), (end - length, front)]
        if sticks > 0 and gaps:
            # Backed to a wall, a corner that stands out reaches back to a wall behind its diagonal, if one stands
            # parallel to it: a corner cabinet along the kitchen's 45-degree wall.
            to_world = lambda w, o: w * axes[wide] + out * o * axes[thin]
            to_frame = lambda p: (float(p @ axes[wide]), float(out * (p @ axes[thin])))
            tip, foot = (to_world(*outline[4]), to_world(*outline[3])) if side < 0 else (to_world(*outline[2]), to_world(*outline[3]))
            backs = _back_along_wall(tip, foot, to_world(start, back), to_world(end, back), walls)
            if backs is not None:
                backs = [to_frame(p) for p in backs]
                if side < 0:
                    outline[0:1] = backs
                else:
                    outline[1:2] = backs[::-1]
    ring = np.array(outline)
    middle = (ring.min(axis=0) + ring.max(axis=0)) / 2
    size = np.zeros(2)
    size[wide], size[thin] = ring[:, 0].max() - ring[:, 0].min(), ring[:, 1].max() - ring[:, 1].min()
    centre = middle[0] * axes[wide] + out * middle[1] * axes[thin]
    footprint = None
    if corner is not None:
        shape = np.zeros_like(ring)
        shape[:, wide], shape[:, thin] = ring[:, 0] - middle[0], out * (ring[:, 1] - middle[1])
        x, y = shape[:, 0], shape[:, 1]
        if (x * np.roll(y, -1) - np.roll(x, -1) * y).sum() < 0:
            shape = shape[::-1]
        footprint = [(float(a), float(b)) for a, b in shape]
    return proxy.model_copy(update={
        "size": (float(size[0]), float(size[1]), proxy.size[2]),
        "translate": (float(centre[0]), float(centre[1]), proxy.translate[2]),
        "yaw_deg": float(yaw),
        "footprint": footprint,
    })


def outline(proxy: Proxy) -> np.ndarray:
    """A built-in's outline in plan, in its own frame, counter-clockwise: its footprint, or the rectangle of its size."""
    if proxy.footprint is not None:
        return np.array(proxy.footprint)
    hx, hy = proxy.size[0] / 2, proxy.size[1] / 2
    return np.array([(-hx, -hy), (hx, -hy), (hx, hy), (-hx, hy)])


def box_error_m(proxy: Proxy, points: np.ndarray) -> float:
    """The median distance of a built-in's depth points from its surface, in meters (its outline stood up its height)."""
    if len(points) == 0:
        return 0.0
    plan = (points[:, :2] - np.array(proxy.translate[:2])) @ _axes(proxy.yaw_deg).T
    ring = outline(proxy)
    starts, ends = ring, np.roll(ring, -1, axis=0)
    # Distance to the nearest edge of the outline, and whether a point is inside it (crossings of a ray along +x).
    edge = ends - starts
    along = np.clip(((plan[:, None] - starts) * edge).sum(-1) / (edge * edge).sum(-1), 0, 1)
    to_edge = np.linalg.norm(plan[:, None] - (starts + along[..., None] * edge), axis=-1).min(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        crossing = ((starts[:, 1] > plan[:, None, 1]) != (ends[:, 1] > plan[:, None, 1])) & (
            plan[:, None, 0] < starts[:, 0] + (plan[:, None, 1] - starts[:, 1]) * edge[:, 0] / edge[:, 1]
        )
    inside = crossing.sum(axis=1) % 2 == 1
    half = proxy.size[2] / 2
    off_height = np.abs(points[:, 2] - proxy.translate[2] - half) - half
    distance = np.where(
        inside & (off_height <= 0), np.minimum(to_edge, -off_height),
        np.hypot(np.where(inside, 0.0, to_edge), np.maximum(off_height, 0.0)),
    )
    return float(np.median(distance))


@dataclass
class Footprint:
    """A thing's box in plan, for lining up: its heading, centre, size along its heading's axes, and height range."""

    id: str
    heading_deg: float
    centre: np.ndarray
    size: np.ndarray
    z: tuple[float, float]
    # Built-ins are not moved; the others are moved to them.
    fixed: bool = False
    # Which of two to keep where it is: the steadier heading (a built-in's, then the more box-shaped model's).
    steadiness: float = 0.0
    turn_deg: float = 0.0
    shift: np.ndarray = field(default_factory=lambda: np.zeros(2))

    def corners(self, axes: np.ndarray) -> np.ndarray:
        """Its four corners, where it now stands, along `axes` (rows: the two plan directions)."""
        own = _axes(self.heading_deg + self.turn_deg)
        half = self.size / 2
        offsets = np.array([[sx * half[0], sy * half[1]] for sx in (-1, 1) for sy in (-1, 1)]) @ own
        return (self.centre + self.shift + offsets) @ axes.T


def _extent(footprint: Footprint, axes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    corners = footprint.corners(axes)
    return corners.min(axis=0), corners.max(axis=0)


def _relation(a: Footprint, b: Footprint) -> str | None:
    """How two things stand together: "beside" each other, one "over" the other, or neither (None).

    Beside: nearly square to each other, footprints within `NEAR_M`, at the same level. Over: nearly square
    to each other, one's footprint over the other's, its bottom close above the other's top (as
    `square_to_built_in` finds a cabinet over a fridge).
    """
    if abs(fold_deg(a.heading_deg - b.heading_deg)) > ITEM_WITHIN_DEG:
        return None
    axes = _axes(a.heading_deg)
    (low_a, high_a), (low_b, high_b) = _extent(a, axes), _extent(b, axes)
    gap = np.maximum(low_b - high_a, low_a - high_b)
    overlap = min(a.z[1], b.z[1]) - max(a.z[0], b.z[0])
    if gap.max() <= NEAR_M and overlap > 0.5 * min(a.z[1] - a.z[0], b.z[1] - b.z[0]):
        return "beside"
    lower, upper = (a, b) if a.z[0] <= b.z[0] else (b, a)
    if gap.max() < 0 and lower.z[1] - OVER_SLACK_M <= upper.z[0] <= lower.z[1] + OVER_WITHIN_M:
        return "over"
    return None


def line_up(footprints: list[Footprint], eye: np.ndarray) -> list[Footprint]:
    """Square up neighbours: turn each group to its steadiest member's heading, then line up fronts and close gaps.

    Things that stand together (see `_relation`) form groups. Each movable member is turned
    about its centre to the heading of the group's steadiest member. Then, for
    each pair in a group, the less steady is moved: along the axis the camera
    looks down (the pair's depth), its front to the other's if they differ by
    no more than `FRONT_WITHIN_M`; along the other, so their sides touch, if
    they stand no more than `GAP_WITHIN_M` apart or overlap. A pair one over the
    other only shares its heading. Returns the
    footprints with `turn_deg` and `shift` set; fixed ones are not moved.
    """
    things = {f.id: f for f in footprints}
    group = {f.id: f.id for f in footprints}

    def root(key: str) -> str:
        while group[key] != key:
            key = group[key]
        return key

    related = [(a, b, _relation(a, b)) for i, a in enumerate(footprints) for b in footprints[i + 1:] if not (a.fixed and b.fixed)]
    pairs = [(a, b) for a, b, how in related if how == "beside"]
    for a, b, how in related:
        if how is None:
            continue
        group[root(a.id)] = root(b.id)
    members: dict[str, list[Footprint]] = {}
    for f in footprints:
        members.setdefault(root(f.id), []).append(f)

    for together in members.values():
        if len(together) < 2:
            continue
        steadiest = max(together, key=lambda f: (f.fixed, f.steadiness))
        for f in together:
            if not f.fixed:
                f.turn_deg = fold_deg(steadiest.heading_deg - f.heading_deg)

    for a, b in pairs:
        keep, move = (a, b) if (a.fixed, a.steadiness) >= (b.fixed, b.steadiness) else (b, a)
        if move.fixed:
            continue
        axes = _axes(keep.heading_deg + keep.turn_deg)
        middle = ((keep.centre + keep.shift + move.centre + move.shift) / 2 - eye[:2]) @ axes.T
        depth = int(np.argmax(np.abs(middle)))
        side = 1 - depth
        (low_k, high_k), (low_m, high_m) = _extent(keep, axes), _extent(move, axes)
        # The front is the face towards the camera.
        towards = -np.sign(middle[depth])
        front_k, front_m = (high_k[depth], high_m[depth]) if towards > 0 else (low_k[depth], low_m[depth])
        step = np.zeros(2)
        if abs(front_k - front_m) <= FRONT_WITHIN_M:
            step[depth] = front_k - front_m
        # Which side of `keep` it stands on, and the gap between them there (negative: they overlap).
        if (low_m[side] + high_m[side]) / 2 >= (low_k[side] + high_k[side]) / 2:
            gap = low_m[side] - high_k[side]
            if abs(gap) <= GAP_WITHIN_M:
                step[side] = -gap
        else:
            gap = low_k[side] - high_m[side]
            if abs(gap) <= GAP_WITHIN_M:
                step[side] = gap
        move.shift = move.shift + step @ axes
    return list(things.values())
