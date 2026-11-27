"""The ceiling of the room shell, built as the floor is: from its own mask, cut back by the walls.

Without one, the render showed sky over the walls (the kitchen's top band of
photo, white in the render). The segmenter is asked for "ceiling"; its points
give the ceiling's height, the median of those near one level (a ceiling is
level, and the mask's edges smear into the walls and the cabinets' tops); its
outline is cut back by the walls as the floor's is (see `room_polygon`), and only
points over the floor count; its colour is the photo's over the ceiling's pixels. The walls are taken up, or down, to meet it.
"""

import numpy as np
from pydantic import BaseModel

from aomma.stages.floor import room_polygon
from aomma.stages.proxy import Proxy
from aomma.stages.world import facing, surface_normals

# A ceiling's points lie within this of its height.
ON_CEILING_M = 0.12
# A ceiling is built from at least this many points on it, and this share of the photo's pixels: under it, what
# shows is a glimpse past the room more often than its own ceiling (the office's hallway, through the doorway at the
# photo's edge, 0.27% of it; the kitchen's ceiling, 5.5%), and adds little to the render.
MIN_POINTS = 300
MIN_SHARE = 0.01
# And only between these heights above the floor.
HEIGHT_M = (2.0, 4.5)
# The slab's thickness, above its underside.
THICKNESS_M = 0.02


class Ceiling(BaseModel):
    # Its underside's height above the floor, in meters.
    height_m: float
    # Its outline in plan (world x, y, counter-clockwise).
    polygon: list[tuple[float, float]]
    # The photo's median colour over its pixels, linear RGB 0-1.
    color: tuple[float, float, float]
    # How many of the mask's points lie on it, and the walls that bound it.
    points: int
    bounded_by: list[str]


def ceiling_height(points: np.ndarray) -> tuple[float, np.ndarray] | None:
    """The ceiling's height from its mask's points ((N, 3), world frame), and which lie on it; None if too few do.

    The median of all the points, then of those within `ON_CEILING_M` of it.
    """
    points = points[np.isfinite(points).all(axis=1)]
    if len(points) < MIN_POINTS:
        return None
    height = float(np.median(points[:, 2]))
    on = np.abs(points[:, 2] - height) < ON_CEILING_M
    height = float(np.median(points[on, 2]))
    on = np.abs(points[:, 2] - height) < ON_CEILING_M
    if on.sum() < MIN_POINTS or not HEIGHT_M[0] <= height <= HEIGHT_M[1]:
        return None
    return height, on


def _inside(polygon: np.ndarray, plan: np.ndarray) -> np.ndarray:
    """Which plan points ((N, 2)) lie within the convex, counter-clockwise `polygon`, or within `ON_CEILING_M` of it."""
    inside = np.ones(len(plan), dtype=bool)
    for a, b in zip(polygon, np.roll(polygon, -1, axis=0)):
        edge = (b - a) / np.linalg.norm(b - a)
        inside &= (edge[0] * (plan[:, 1] - a[1]) - edge[1] * (plan[:, 0] - a[0])) >= -ON_CEILING_M
    return inside


def build_ceiling(
    points: np.ndarray, colors: np.ndarray, walls: list[Proxy], camera_position: np.ndarray, floor_seen: np.ndarray,
    photo_pixels: int,
) -> Ceiling | None:
    """The ceiling from its mask's world points ((N, 3)) and their photo colours ((N, 3), sRGB 0-255); None if none shows.

    Only the points over the floor's outline count, from the walls and the
    floor's own points (`floor_seen`, (M, 2)): the office photo shows no
    ceiling of its own, and the segmenter gave the hallway's, through the
    doorway, 3.12 m up. It must show over `MIN_SHARE` of the photo's
    `photo_pixels`. Its outline is cut back by the walls as the floor's is,
    judged by its own points.
    """
    keep = np.isfinite(points).all(axis=1)
    points, colors = points[keep], colors[keep]
    floor_outline, _ = room_polygon(walls, camera_position, floor_seen)
    over = _inside(floor_outline, points[:, :2])
    points, colors = points[over], colors[over]
    found = ceiling_height(points)
    if found is None:
        return None
    height, on = found
    if on.sum() < MIN_SHARE * photo_pixels:
        return None
    # Its own outline, from its own points: cut by the floor's walls, the kitchen's ceiling stopped at the line of
    # the wall behind the fridge, 0.27 m short of the window wall, and the render showed sky along it.
    polygon, bounded_by = room_polygon(walls, camera_position, points[on, :2])
    return Ceiling(
        height_m=height, polygon=[tuple(map(float, corner)) for corner in polygon],
        color=tuple(float(c) for c in (np.median(colors[on], axis=0) / 255) ** 2.2), points=int(on.sum()), bounded_by=bounded_by,
    )


def ceiling_pixels(mask: np.ndarray, points: np.ndarray, up: np.ndarray) -> np.ndarray:
    """The pixels of the segmenter's ceiling `mask` that can be a ceiling: their surface faces up or down
    (`world.facing`) and their point stands above the camera. `points` is the photo's camera-frame point map and `up`
    the world's up in the same frame.

    The segmenter's "ceiling" runs down the walls: on OfficeScan 005 it held 11% of the photo's pixels 0.3-1.9 m up a
    wall, and on OfficeScan 006 it lay on the carpet. A ceiling faces down, as a floor faces
    up, and is above the eye; neither is a size taken from these rooms.
    """
    up = up / np.linalg.norm(up)
    with np.errstate(invalid="ignore"):
        return mask & facing(surface_normals(points), up) & (points @ up > 0)
