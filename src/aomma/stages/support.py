"""Support: make things that rest on other things actually touch them.

Each object is placed on its own, from its outline and the depth map, and both
carry a few centimetres of error. Left alone, a lamp hovers over the shelf it
stands on and a book sinks into the board under it. Once both are placed, the
lamp is moved the small distance that puts its base on the shelf's surface and
keeps it within the shelf's edges.
"""

import numpy as np

from aomma.stages.pose import rotation_z

# A resting object is moved at most this far, up or down and sideways, in meters.
# A larger gap means one of the two is badly placed, which nudging would only hide.
MAX_NUDGE_M = 0.10
# Surfaces within this angle of level can be stood on.
_LEVEL_DEG = 25
# The surface under an object is looked for within this distance of its middle, at least.
_MIN_REACH_M = 0.06
_BIN_M = 0.01


def support_height(
    facets: tuple[np.ndarray, np.ndarray, np.ndarray], low: np.ndarray, high: np.ndarray
) -> float | None:
    """Height of the surface an object stands on, given its world box (`low`, `high`) and its support.

    `facets` are the support's triangles in the world: unit normals, areas and
    centres. The surface is the highest level one under the object's middle
    that is below that middle: for a book on a lower shelf, the lower shelf, not
    the top. Returns None if nothing level lies under it.
    """
    normals, areas, centres = facets
    middle = (low + high) / 2
    reach = max(_MIN_REACH_M, float(min(high[0] - low[0], high[1] - low[1])) / 2)
    under = (
        (normals[:, 2] > np.cos(np.radians(_LEVEL_DEG)))
        & (np.linalg.norm(centres[:, :2] - middle[:2], axis=1) < reach)
        & (centres[:, 2] < middle[2])
    )
    if not under.any():
        return None
    heights, weights = centres[under, 2], areas[under]
    bins = np.floor(heights / _BIN_M).astype(int)
    totals = np.bincount(bins - bins.min(), weights=weights)
    # The highest level holding a fair share of the area: a stray facet is not a shelf.
    top = np.nonzero(totals >= 0.2 * totals.max())[0][-1]
    members = (bins - bins.min()) >= top - 1
    return float(np.average(heights[members], weights=weights[members]))


def rest_on(
    low: np.ndarray,
    high: np.ndarray,
    facets: tuple[np.ndarray, np.ndarray, np.ndarray],
    yaw_deg: float,
) -> np.ndarray:
    """The small move (x, y, z) that stands an object on its support and inside the support's edges.

    `low` and `high` are the object's world box; `facets` the support's
    triangles in the world; `yaw_deg` the direction the support faces, along
    which its edges are measured. Each part of the move is skipped if it would
    exceed MAX_NUDGE_M, or if the object is too big to fit between the edges.
    """
    move = np.zeros(3)
    height = support_height(facets, low, high)
    if height is not None and abs(height - low[2]) <= MAX_NUDGE_M:
        move[2] = height - low[2]

    # In the support's own frame its footprint is a rectangle; keep the object's inside it.
    to_frame = rotation_z(-yaw_deg)[:2, :2]
    edges = facets[2][:, :2] @ to_frame.T
    corners = np.array([[x, y] for x in (low[0], high[0]) for y in (low[1], high[1])]) @ to_frame.T
    shift = np.zeros(2)
    for axis in range(2):
        room_low, room_high = edges[:, axis].min(), edges[:, axis].max()
        own_low, own_high = corners[:, axis].min(), corners[:, axis].max()
        if own_high - own_low > room_high - room_low:
            continue
        nudge = max(room_low - own_low, 0.0) - max(own_high - room_high, 0.0)
        if abs(nudge) <= MAX_NUDGE_M:
            shift[axis] = nudge
    move[:2] = shift @ to_frame
    return move
