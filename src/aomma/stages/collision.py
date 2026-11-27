"""Collision check: generated models passing through each other.

A scene can match the photo from its camera and still be impossible: a bed
running half a metre into the nightstand beside it, hidden behind the
nightstand's front. Each model, as placed, is turned into a solid of small
voxels (its surface samples, closed up and filled), and every two whose solids
share voxels deeper than touching are reported, with how much of the smaller
one is inside the other.
"""

import numpy as np
from scipy import ndimage

# Side of a voxel, in meters.
VOXEL = 0.03
# Gaps between surface samples up to this many voxels wide are closed before a model is filled.
_CLOSING = 2
# Solids are shrunk by this many voxels before they are compared, so that things resting
# on or against each other (a pillow on a bed, a book against a shelf) do not count.
_TOUCHING = 1
# A pair is reported when the overlap is at least this share of the smaller solid.
REPORT_SHARE = 0.05


class Solid:
    """A model's volume as voxels on the scene's grid: `grid[i, j, k]` is the voxel at `origin + (i, j, k)`."""

    def __init__(self, world: np.ndarray):
        cells = np.floor(world / VOXEL).astype(int)
        # A margin wider than the closing, so that closing never reaches the edge of the grid.
        self.origin = cells.min(axis=0) - _CLOSING - 2
        grid = np.zeros(cells.max(axis=0) - self.origin + _CLOSING + 3, dtype=bool)
        grid[tuple((cells - self.origin).T)] = True
        # Close the gaps between samples, fill what the surface encloses, and undo the closing.
        grid = ndimage.binary_fill_holes(ndimage.binary_dilation(grid, iterations=_CLOSING))
        self.grid = ndimage.binary_erosion(grid, iterations=_CLOSING)
        self.core = ndimage.binary_erosion(self.grid, iterations=_TOUCHING)
        # Overlaps are measured between the shrunken solids, so shares are of the shrunken volume too.
        self.volume = int(self.core.sum())

    def overlap(self, other: "Solid") -> int:
        """How many voxels the two solids share, beyond touching."""
        low = np.maximum(self.origin, other.origin)
        high = np.minimum(self.origin + self.core.shape, other.origin + other.core.shape)
        if np.any(high <= low):
            return 0
        mine = self.core[tuple(slice(a, b) for a, b in zip(low - self.origin, high - self.origin))]
        theirs = other.core[tuple(slice(a, b) for a, b in zip(low - other.origin, high - other.origin))]
        return int((mine & theirs).sum())


def find_collisions(objects: dict[str, np.ndarray]) -> list[dict]:
    """Every two models (world-space surface samples, by object id) that pass through each other, worst first.

    Each entry gives the two ids, the volume they share in litres, that
    volume as a share of the smaller model's, and each model's volume in litres.
    """
    solids = {object_id: Solid(world) for object_id, world in objects.items()}
    found = []
    names = sorted(solids)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            shared = solids[a].overlap(solids[b])
            smaller = min(solids[a].volume, solids[b].volume)
            if shared and smaller and shared / smaller >= REPORT_SHARE:
                found.append({
                    "objects": [a, b],
                    "litres": round(shared * VOXEL**3 * 1000, 1),
                    "share_of_smaller": round(shared / smaller, 2),
                    "volumes": {name: round(solids[name].volume * VOXEL**3 * 1000, 1) for name in (a, b)},
                })
    return sorted(found, key=lambda entry: -entry["share_of_smaller"])
