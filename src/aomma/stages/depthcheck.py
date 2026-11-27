"""Depth check: the whole scene, as placed, against the photo's depth map.

An outline fit judges each object alone, and an outline does not pin down how
big an object is or how far away. The depth map does: it measured, for every
pixel, how far the photo's surface is from the camera. A scene that is right
puts its nearest surface at that distance. So the scene's own depth, rendered
through the photo's camera (see render/depth_script.py), is compared with the
depth map pixel by pixel, at a reduced resolution.

What it cannot see is what the photo cannot: a model too long behind its
visible front is invisible to it, unless the extra shows past the object's
outline, where it stands in front of the floor or the wall.

Each object is judged only where the render shows it, or shows no named
model: where another model stands in front of it, that model's distance is
not this object's fault. A model that wrongly covers others is judged by how
much of them it covers against its own size in the photo (the garage's 1 m
slipper covered 68 times its own pixels of the washer and dryer, and they were
blamed for it; a hair dryer on the dryer, a little off, covers 0.4 times).
"""

import warnings

import numpy as np
from PIL import Image

# A pixel agrees when the scene's distance is within this factor of the measured one either way.
AGREE = 1.10


def shrink(values: np.ndarray, factor: int) -> np.ndarray:
    """A (H, W) map reduced by `factor` each way, each cell the median of its known values (NaN if none)."""
    height, width = values.shape[0] // factor * factor, values.shape[1] // factor * factor
    blocks = values[:height, :width].reshape(height // factor, factor, width // factor, factor).swapaxes(1, 2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN cells come back NaN, as wanted
        return np.nanmedian(blocks.reshape(*blocks.shape[:2], -1), axis=-1)


def check_scene(
    rendered: np.ndarray, measured: np.ndarray, masks: dict[str, np.ndarray], factor: int = 4,
    shown: dict[str, np.ndarray] | None = None,
) -> tuple[dict, np.ndarray, np.ndarray]:
    """Compare the scene's depth with the measured depth, both (H, W) along the camera's axis.

    `measured` is NaN where the depth map knows nothing, `rendered` infinite
    where the scene has no surface. Returns a summary, and the two maps at
    the reduced resolution. For each object in `masks` (the pixels where the
    photo shows it and nothing in front), and for the pixels no object
    covers ("room": floor, walls and anything the inventory missed), the
    summary gives the median ratio of the scene's distance there to the
    measured one (1 is right, above 1 the scene is too far, below too near),
    the share of those pixels that agree, and the shares too near and too far.

    `shown`, by object, is where the render shows that object's model. With
    it, an object's pixels are only those no other model covers; each object
    also gets "hidden_by_others", the share of its pixels another model
    covers, and, if its model shows at all, "over_others", how much of other
    objects' pixels its model covers, as a multiple of its own pixels.
    """
    rendered_small, measured_small = shrink(rendered, factor), shrink(measured, factor)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = rendered_small / measured_small
    known = np.isfinite(measured_small) & (measured_small > 0) & ~np.isnan(rendered_small)
    error = np.where(known, np.log(np.where(known & np.isfinite(ratio), ratio, 1e6)), 0.0)
    agree, near, far = known & (np.abs(error) < np.log(AGREE)), known & (error <= -np.log(AGREE)), known & (error >= np.log(AGREE))

    def judge(cells: np.ndarray) -> dict | None:
        cells = cells & known
        if cells.sum() < 3:
            return None
        return {
            "ratio": round(float(np.median(ratio[cells])), 2),
            "agree": round(float(agree[cells].mean()), 2),
            "too_near": round(float(near[cells].mean()), 2),
            "too_far": round(float(far[cells].mean()), 2),
            "pixels": int(cells.sum()),
        }

    small = {object_id: shrink(mask.astype(float), factor) > 0.5 for object_id, mask in masks.items()}
    drawn = {object_id: shrink(where.astype(float), factor) > 0.5 for object_id, where in (shown or {}).items()}
    objects = {}
    for object_id, cells in small.items():
        others = np.zeros_like(cells)
        for other, where in drawn.items():
            if other != object_id:
                others |= where
        if (found := judge(cells & ~others)) is None:
            continue
        if shown is not None:
            found["hidden_by_others"] = round(float((cells & others).sum() / cells.sum()), 2)
            if object_id in drawn:
                theirs = np.zeros_like(cells)
                for other, mask in small.items():
                    if other != object_id:
                        theirs |= mask
                found["over_others"] = round(float((drawn[object_id] & known & theirs & ~cells).sum() / cells.sum()), 2)
        objects[object_id] = found
    covered = np.any(list(small.values()), axis=0) if small else np.zeros_like(known)
    summary = {
        "agree": round(float(agree.sum() / known.sum()), 3) if known.any() else None,
        "room": judge(~covered),
        "objects": objects,
    }
    return summary, rendered_small, measured_small


def draw(photo: Image.Image, rendered: np.ndarray, measured: np.ndarray) -> Image.Image:
    """The photo, tinted where the scene disagrees with the depth map: blue too near, red too far, by up to half again."""
    with np.errstate(divide="ignore", invalid="ignore"):
        error = np.nan_to_num(np.log(rendered / measured), nan=0.0, posinf=np.log(1.5), neginf=-np.log(1.5))
    strength = np.where(np.abs(error) < np.log(AGREE), 0.0, np.clip(np.abs(error) / np.log(1.5), 0, 1))
    colour = np.where((error > 0)[..., None], [220, 40, 40], [40, 90, 230]).astype(float)
    small = np.asarray(photo.convert("RGB").resize((error.shape[1], error.shape[0]), Image.Resampling.BILINEAR)).astype(float)
    tinted = small * (1 - 0.7 * strength[..., None]) + colour * 0.7 * strength[..., None]
    return Image.fromarray(tinted.astype(np.uint8)).resize(photo.size, Image.Resampling.NEAREST)
