"""Walls segmented from points on their bare surface, chosen from the scene.

The survey gives a wall nearly the box of the cabinets in front of it, and SAM, prompted with that box, segments the
cabinets (KitchenScan 008's left wall). A box can't say which thing in it is the wall; a point on the
wall's bare surface can. The points are chosen from what the run holds, not asked of the VLM, whose points missed in
the kitchens: inside the wall's box, pixels whose surface faces sideways (a wall is plumb) and that lie
outside every box the survey gave a built-in, an opening or a large item, and outside the other walls' boxes. Up to
`MAX_POINTS`, deep inside that region and spread apart.

A wall's mask is then what SAM finds from its points, plus what its box's mask found outside the listed things' boxes:
the box finds the whole of a bare wall, which a few points do not, and the listed things' boxes take the cabinets
back out of it (wall IoU: OfficeScan 005 0.835 -> 0.955, KitchenScan 008 0.67 -> 0.82, plain walls within 0.01).
A wall that gets no points (no bare surface outside the listed things) keeps its box's mask as it is.
"""

import numpy as np
from scipy.ndimage import distance_transform_edt

from aomma.stages.world import facing, surface_normals

MAX_POINTS = 3
# A point stands at least this share of the photo's width inside its region, off the edges of the boxes around it...
EDGE_SHARE = 20 / 1024
# ...and this share apart from the others.
APART_SHARE = 60 / 1024
# A wall's surface faces within this of level (its normal this far from up, either way): a wall is plumb.
_SIDEWAYS_DEG = 70


def box_mask(bbox_2d: list[float], size: tuple[int, int]) -> np.ndarray:
    """A survey box (normalised to 0-1000) as a mask of the photo's (width, height)."""
    width, height = size
    mask = np.zeros((height, width), bool)
    x1, y1, x2, y2 = (int(round(v * s / 1000)) for v, s in zip(bbox_2d, (width, height, width, height)))
    mask[max(0, y1):max(0, y2), max(0, x1):max(0, x2)] = True
    return mask


def scene_points(walls: list, things: list, points: np.ndarray, up: np.ndarray, size: tuple[int, int]) -> dict[str, list[tuple[int, int]]]:
    """Up to `MAX_POINTS` pixels (x, y) on each wall's bare surface. `walls` and `things` are survey entries (with
    `id` and `bbox_2d`); `points` the photo's camera-frame point map; `up` the world's up in that frame."""
    width, height = size
    normals = surface_normals(points)
    up = up / np.linalg.norm(up)
    sideways = np.isfinite(normals).all(-1) & ~facing(normals, up, _SIDEWAYS_DEG)
    listed = np.zeros((height, width), bool)
    for thing in things:
        listed |= box_mask(thing.bbox_2d, size)
    edge, apart = EDGE_SHARE * width, APART_SHARE * width
    yy, xx = np.ogrid[:height, :width]
    found = {}
    for wall in walls:
        others = np.zeros((height, width), bool)
        for other in walls:
            if other.id != wall.id:
                others |= box_mask(other.bbox_2d, size)
        inside = distance_transform_edt(box_mask(wall.bbox_2d, size) & sideways & ~listed & ~others)
        picked: list[tuple[int, int]] = []
        for _ in range(MAX_POINTS):
            score = inside.copy()
            for x, y in picked:
                score[(xx - x) ** 2 + (yy - y) ** 2 < apart ** 2] = 0
            y, x = np.unravel_index(int(np.argmax(score)), score.shape)
            if score[y, x] < edge:
                break
            picked.append((int(x), int(y)))
        found[wall.id] = picked
    return found


def wall_mask(from_points: np.ndarray | None, from_box: np.ndarray, things: list, size: tuple[int, int]) -> np.ndarray:
    """The wall: what its points found, plus what its box found outside every listed thing's box. With no points
    (None), its box's mask as it is: nothing says it is wrong (KitchenScan 005's left wall, cut back to its listed
    things' boxes, lost its backsplash: 0.93 -> 0.07)."""
    if from_points is None:
        return from_box
    listed = np.zeros(from_box.shape, bool)
    for thing in things:
        listed |= box_mask(thing.bbox_2d, size)
    return from_points | (from_box & ~listed)
