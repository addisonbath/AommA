"""Occlusion: work out which objects are partly hidden, and by which others."""

import numpy as np
from PIL import Image, ImageFilter
from pydantic import BaseModel

from aomma.stages.inventory import Inventory

# How far apart, in pixels, two masks may be and still count as touching.
_REACH_PX = 4
# An occluder must be at least this much nearer the camera, as a fraction of distance.
_NEARER = 0.02


class Occlusion(BaseModel):
    # Objects that hide part of this one.
    occluders: list[str]
    # Share of this object's outline that runs along an occluder, 0-1. The
    # outline there is not the object's real edge, so a high value means a
    # large part of its shape is unknown.
    hidden_edge: float


def _grow(mask: np.ndarray, pixels: int) -> np.ndarray:
    return np.asarray(Image.fromarray(mask).filter(ImageFilter.MaxFilter(2 * pixels + 1)))


def find_occlusions(
    masks: dict[str, np.ndarray], points: np.ndarray, inventory: Inventory
) -> dict[str, Occlusion]:
    """For each object, the objects in front of it and how much of its outline they cut.

    `masks` must not overlap (see `exclusive_masks`) and `points` is the
    camera-frame point map. One object hides another where their masks touch
    and it is nearer the camera there, or where it rests on the other.
    """
    depth = -points[..., 2]
    rests_on = {obj.id: obj.on for obj in inventory.objects}
    grown = {oid: _grow(mask, _REACH_PX) for oid, mask in masks.items()}

    result = {}
    for oid, mask in masks.items():
        if not mask.any():
            result[oid] = Occlusion(occluders=[], hidden_edge=0.0)
            continue
        # The outline: mask pixels with a non-mask neighbour.
        outline = mask & _grow(~mask, 1)
        occluders, cut = [], np.zeros_like(mask)
        for other, other_mask in masks.items():
            if other == oid:
                continue
            ours = mask & grown[other]  # our pixels along the shared edge
            theirs = other_mask & grown[oid]  # their pixels along it
            if ours.sum() < 20 or theirs.sum() < 20:
                continue
            our_depth, their_depth = np.nanmedian(depth[ours]), np.nanmedian(depth[theirs])
            if rests_on.get(other) == oid or their_depth < our_depth * (1 - _NEARER):
                occluders.append(other)
                cut |= outline & grown[other]
        result[oid] = Occlusion(occluders=occluders, hidden_edge=round(float(cut.sum() / outline.sum()), 3))
    return result
