"""Gate 4: the layout as boxes. Nothing is generated.

Each small item becomes an upright box fitted to the pixels the label map gives it (`labels.png`: each pixel settled to
one item), each large item one fitted to its own mask (what lies on it is part of its shape), standing on its support:

- **floor**: its bottom on the floor (as `fit_proxies` does for anything `on` the floor);
- **a large item or a built-in**: if its own bottom is above the middle of the support's box, it stands on the
  support's level nearest its measured bottom, its measured height kept: the box's top, or a level the support's own
  points show around the item (a desk's writing surface below its hutch, a sofa's seat below its back). Lower than
  the middle it is inside the support (a shelf, a cabinet's open front), and its measured bottom is kept;
- **a wall, or a support with no box**: as measured.

Large items are fitted first, so the small items standing on them find their boxes.
"""

import numpy as np
from PIL import Image
from pydantic import BaseModel

from aomma.labels import SAME_DEPTH_M
from aomma.stages.inventory import Inventory, InventoryObject
from aomma.stages.proxy import Proxy, fit_proxies
from aomma.stages.world import facing, surface_normals

UP = np.array([0.0, 0.0, 1.0])


class Box(BaseModel):
    """An item's box (`proxy`), its class (`large` or `small`), what it stands on, and how it was stood there."""

    proxy: Proxy
    kind: str
    on: str
    # floor | top (moved onto its support's top) | level (onto a lower level of its support, seen around it) |
    # inside (within its support, as measured) | own (as measured)
    stood: str
    # How far standing it moved its bottom, in meters (+ up).
    moved_m: float = 0.0


def support_level(proxy: Proxy, support: Proxy, points: np.ndarray) -> float | None:
    """The height of the support's own surface where `proxy` stands on it, from `points` (the support's up-facing
    points, world, Z up): those around the item (its footprint grown by half its own size each way) and in the
    support's upper half, the one nearest the item's bottom taken with the others within `SAME_DEPTH_M` of it (one
    surface, as the depth spreads it). None where the support shows no such surface around the item."""
    s_bottom, s_height = support.translate[2], support.size[2]
    x, y, bottom = proxy.translate
    yaw = np.radians(proxy.yaw_deg)
    offset = points[:, :2] - (x, y)
    along = offset @ (np.cos(yaw), np.sin(yaw))
    across = offset @ (-np.sin(yaw), np.cos(yaw))
    near = (np.abs(along) <= proxy.size[0]) & (np.abs(across) <= proxy.size[1])
    heights = points[near & (points[:, 2] >= s_bottom + s_height / 2), 2]
    if not len(heights):
        return None
    nearest = heights[np.argmin(np.abs(heights - bottom))]
    return float(np.median(heights[np.abs(heights - nearest) <= SAME_DEPTH_M]))


def stand_on(proxy: Proxy, support: Proxy, points: np.ndarray | None = None) -> tuple[Proxy, str, float]:
    """The box stood on `support`'s level nearest its bottom if it sits above the support's middle: the support's top,
    or the level its up-facing `points` show around the item (`support_level`); else as measured, inside it."""
    bottom = proxy.translate[2]
    s_bottom, s_height = support.translate[2], support.size[2]
    if bottom < s_bottom + s_height / 2:
        return proxy, "inside", 0.0
    top = s_bottom + s_height
    level = support_level(proxy, support, points) if points is not None and len(points) else None
    stood, how = (level, "level") if level is not None and abs(level - bottom) < abs(top - bottom) else (top, "top")
    x, y, _ = proxy.translate
    return proxy.model_copy(update={"translate": (x, y, float(stood))}), how, float(stood - bottom)


def fit_boxes(
    image: Image.Image,
    points_world: np.ndarray,
    label_map: np.ndarray,
    labels: list[dict],
    objects: dict[str, InventoryObject],
    built_ins: list[Proxy],
    camera_position: np.ndarray,
    flat: set[str] = frozenset(),
    own_masks: dict[str, np.ndarray] | None = None,
) -> tuple[list[Box], list[str]]:
    """Every labelled large and small item's box, but those in `flat` (sheets and slabs, `stages.forms`); also the ids
    with too few 3D points to fit one.

    A large item with a mask of its own in `own_masks` (the segmenter's, before the label map settled it) is fitted to
    that mask: what lies on it (a quilt on a bed, papers on a desk) takes most of its label but is part of its shape, so
    the label alone gave a 2 x 2 m bed for a 1.95 x 1.34 m one. A small item keeps its label:
    it sits on or in something, whose pixels its mask may take in.

    A support's level is read from its own pixels (its label, and a large item's own mask) where the depth faces up."""
    supports = {p.id: p for p in built_ins}
    value = {e["id"]: e["value"] for e in labels}
    up = facing(surface_normals(points_world), UP) & np.isfinite(points_world).all(axis=-1)

    def support_points(support_id: str) -> np.ndarray:
        pixels = label_map == value[support_id] if support_id in value else np.zeros(label_map.shape, dtype=bool)
        if own_masks and support_id in own_masks:
            pixels = pixels | own_masks[support_id]
        return points_world[pixels & up]

    boxes: list[Box] = []
    skipped: list[str] = []
    for kind in ("large", "small"):
        entries = [e for e in labels if e["class"] == kind and e["id"] in objects and e["kept_px"] and e["id"] not in flat]
        inventory = Inventory(objects=[objects[e["id"]] for e in entries])
        masks = {e["id"]: own_masks[e["id"]] if kind == "large" and own_masks and e["id"] in own_masks else label_map == e["value"]
                 for e in entries}
        fitted = {p.id: p for p in fit_proxies(image, points_world, masks, inventory, camera_position)}
        for obj in inventory.objects:
            proxy = fitted.get(obj.id)
            if proxy is None:
                skipped.append(obj.id)
                continue
            if obj.on == "floor":
                box = Box(proxy=proxy, kind=kind, on=obj.on, stood="floor")
            elif obj.on in supports:
                stood, how, moved = stand_on(proxy, supports[obj.on], support_points(obj.on))
                box = Box(proxy=stood, kind=kind, on=obj.on, stood=how, moved_m=moved)
            else:
                box = Box(proxy=proxy, kind=kind, on=obj.on, stood="own")
            boxes.append(box)
            if kind == "large":
                supports[obj.id] = box.proxy
    return boxes, skipped
