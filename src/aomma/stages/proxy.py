"""Proxy layout: stand a box in for every object, fitted to its 3D points."""

import numpy as np
from PIL import Image
from pydantic import BaseModel

from aomma.layout import Vec3
from aomma.stages.inventory import Inventory

_MIN_POINTS = 50
_YAW_STEP_DEG = 3


class Opening(BaseModel):
    """A door, doorway or window cut through a wall, from `bottom` (the floor for a door) up to `top`."""

    id: str
    # Its two sides, where they meet the wall's face in plan (world x, y): kept apart from the wall's centre and
    # length, which later steps move.
    ends: tuple[tuple[float, float], tuple[float, float]]
    top: float
    bottom: float = 0.0
    # A window is glazed: lines of sight stop at it as at the wall, and its slab has a pane (`pane`, linear RGB).
    window: bool = False
    pane: Vec3 | None = None
    # Its frame (see `room.glaze`): how far the frame's outer edge stands past the opening's sides, top and
    # bottom, how wide the frame is, and its colour (linear RGB); no colour, no frame.
    frame_out: float = 0.0
    frame_width: float = 0.0
    frame: Vec3 | None = None


class Proxy(BaseModel):
    id: str
    # Box dimensions along its own x, y, z axes, in meters.
    size: Vec3
    # Bottom-centre of the box in the world frame.
    translate: Vec3
    yaw_deg: float
    # Average colour of the object's pixels, linear RGB 0-1.
    color: Vec3
    # Its outline in plan, in its own frame (x, y about its centre, counter-clockwise), where it is not the
    # rectangle `size` gives: a corner cabinet's 45-degree front (see `square.square_built_in`). `size` holds
    # the outline's bounding rectangle.
    footprint: list[tuple[float, float]] | None = None
    # A wall's doors, doorways and windows (see `room.cut_openings`).
    openings: list[Opening] = []


def exclusive_masks(masks: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Remove from each mask the pixels of smaller objects lying on top of it.

    A shelf's mask covers the things standing on it; the smaller mask wins.
    """
    if not masks:
        return {}
    by_area = sorted(masks, key=lambda oid: int(masks[oid].sum()))
    taken = np.zeros_like(next(iter(masks.values())))
    exclusive = {}
    for oid in by_area:
        exclusive[oid] = masks[oid] & ~taken
        taken |= masks[oid]
    return exclusive


def fit_proxies(
    image: Image.Image,
    points_world: np.ndarray,
    masks: dict[str, np.ndarray],
    inventory: Inventory,
    camera_position: np.ndarray,
) -> list[Proxy]:
    """Fit an upright box to each object; objects with too few 3D points are skipped."""
    pixels = np.asarray(image).astype(np.float32) / 255
    valid = np.isfinite(points_world).all(axis=-1)
    proxies = []
    for obj in inventory.objects:
        mask = masks[obj.id] & valid
        if mask.sum() < _MIN_POINTS:
            continue
        pts = points_world[mask]

        # Percentiles throughout: depth at a mask's edge smears into the background.
        bottom, top = np.percentile(pts[:, 2], [2, 98])
        if obj.on == "floor":
            bottom = 0.0

        # Footprint: the yaw whose bounding rectangle has the smallest area.
        best = None
        for yaw in np.radians(np.arange(0, 90, _YAW_STEP_DEG)):
            axes = np.array([[np.cos(yaw), np.sin(yaw)], [-np.sin(yaw), np.cos(yaw)]])
            lo, hi = np.percentile(pts[:, :2] @ axes.T, [2, 98], axis=0)
            if best is None or np.prod(hi - lo) < np.prod(best[2] - best[1]):
                best = (axes, lo, hi, yaw)
        axes, lo, hi, yaw = best
        size_xy = hi - lo
        centre = (lo + hi) / 2 @ axes

        # The camera sees only the near faces, so the footprint is too thin
        # front to back. Pad the thin side out to the VLM's estimate, growing
        # away from the camera so the visible face stays where it was seen.
        thin = int(np.argmin(size_xy))
        expected = min(obj.size_m[0], obj.size_m[1])
        if size_xy[thin] < expected:
            away = axes[thin] * np.sign((centre - camera_position[:2]) @ axes[thin])
            centre = centre + away * (expected - size_xy[thin]) / 2
            size_xy[thin] = expected

        proxies.append(
            Proxy(
                id=obj.id,
                size=(float(size_xy[0]), float(size_xy[1]), float(max(top - bottom, 0.01))),
                translate=(float(centre[0]), float(centre[1]), float(bottom)),
                yaw_deg=float(np.degrees(yaw)),
                # Photos are sRGB; USD material colours are linear.
                color=tuple(float(c) for c in pixels[mask].mean(axis=0) ** 2.2),
            )
        )
    return proxies


def depth_scale_correction(proxies: list[Proxy], inventory: Inventory) -> float:
    """Factor to multiply the scene's depths by so that object sizes look right.

    Depth from a single photo can be off by one overall factor. The VLM knows
    roughly how big things are, so compare its estimate of each object's longest
    side with the measured one; the median over objects shrugs off bad guesses.
    """
    estimated = {obj.id: max(obj.size_m) for obj in inventory.objects}
    ratios = [estimated[proxy.id] / max(proxy.size) for proxy in proxies]
    return float(np.median(ratios)) if ratios else 1.0

