"""Score a proxy layout against the photo without rendering.

Projects each box through the camera and compares its silhouette with the
object's mask. This is the cheap inner-loop metric; Blender renders are for eyes.
"""

import numpy as np
from PIL import Image, ImageDraw

from aomma.layout import Camera
from aomma.stages.proxy import Proxy

# Corner index = x + 2y + 4z over the unit cube.
_FACES = [(0, 1, 3, 2), (4, 5, 7, 6), (0, 1, 5, 4), (2, 3, 7, 6), (0, 2, 6, 4), (1, 3, 7, 5)]
_EDGES = [(0, 1), (2, 3), (4, 5), (6, 7), (0, 2), (1, 3), (4, 6), (5, 7), (0, 4), (1, 5), (2, 6), (3, 7)]


def project(points_world: np.ndarray, camera: Camera) -> tuple[np.ndarray, np.ndarray]:
    """Return (N, 2) pixel coordinates and (N,) distances in front of the camera."""
    matrix = camera.matrix
    local = (points_world - matrix[:3, 3]) @ matrix[:3, :3]
    depth = -local[:, 2]
    pixels = np.stack(
        [
            camera.width / 2 + camera.focal_px * local[:, 0] / depth,
            camera.height / 2 - camera.focal_px * local[:, 1] / depth,
        ],
        axis=1,
    )
    return pixels, depth


def box_corners(proxy: Proxy) -> np.ndarray:
    """The proxy's 8 corners in the world frame."""
    sx, sy, sz = proxy.size
    local = np.array(
        [[x * sx, y * sy, z * sz] for z in (0, 1) for y in (-0.5, 0.5) for x in (-0.5, 0.5)]
    )
    yaw = np.radians(proxy.yaw_deg)
    rotation = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
    return local @ rotation.T + np.array(proxy.translate)


def _corner_pixels(proxy: Proxy, camera: Camera) -> list[tuple[float, float]] | None:
    pixels, depth = project(box_corners(proxy), camera)
    # A box that crosses the camera plane has no sensible projection.
    if (depth <= 0.01).any():
        return None
    return [tuple(p) for p in pixels]


def silhouette(proxy: Proxy, camera: Camera) -> np.ndarray:
    """Boolean (H, W) mask of the pixels the proxy box covers."""
    canvas = Image.new("1", (camera.width, camera.height))
    corners = _corner_pixels(proxy, camera)
    if corners:
        draw = ImageDraw.Draw(canvas)
        for face in _FACES:
            draw.polygon([corners[i] for i in face], fill=1)
    return np.asarray(canvas)


def score_layout(
    proxies: list[Proxy], masks: dict[str, np.ndarray], camera: Camera
) -> dict[str, float]:
    """Intersection-over-union of each proxy's silhouette with its object's mask."""
    scores = {}
    for proxy in proxies:
        shape, mask = silhouette(proxy, camera), masks[proxy.id]
        union = (shape | mask).sum()
        scores[proxy.id] = round(float((shape & mask).sum() / union), 3) if union else 0.0
    return scores


def _outline_edges(proxy: Proxy, camera: Camera) -> tuple[list[tuple[float, float]], list[tuple[int, int]]] | None:
    """A proxy with a footprint stood up its height: its corners' pixels and its edges; None if it crosses the camera plane."""
    ring = np.array(proxy.footprint)
    count = len(ring)
    yaw = np.radians(proxy.yaw_deg)
    rotation = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
    plan = ring @ rotation.T + np.array(proxy.translate[:2])
    corners = np.array([[x, y, proxy.translate[2] + z] for z in (0.0, proxy.size[2]) for x, y in plan])
    pixels, depth = project(corners, camera)
    if (depth <= 0.01).any():
        return None
    edges = [(k, (k + 1) % count) for k in range(count)]
    edges += [(count + a, count + b) for a, b in edges] + [(k, count + k) for k in range(count)]
    return [tuple(p) for p in pixels], edges


# The wireframes' colours by what a box is: the shell's built-ins, the large items (furniture, appliances), the small items.
BUILT_IN_EDGE, LARGE_EDGE, SMALL_EDGE = (60, 210, 110), (40, 120, 255), (255, 230, 0)


def draw_wireframes(
    image: Image.Image, proxies: list[Proxy], camera: Camera, color: tuple[int, int, int] | dict[str, tuple[int, int, int]] = SMALL_EDGE,
) -> Image.Image:
    """Draw each proxy box's edges over a copy of the photo; one with a footprint as that outline stood up.

    `color` is one colour for every box, or each box's colour by its id."""
    out = image.copy()
    draw = ImageDraw.Draw(out)
    for proxy in proxies:
        if proxy.footprint is not None:
            drawn = _outline_edges(proxy, camera)
            corners, edges = drawn if drawn else (None, [])
        else:
            corners, edges = _corner_pixels(proxy, camera), _EDGES
        if not corners:
            continue
        for a, b in edges:
            draw.line([corners[a], corners[b]], fill=color.get(proxy.id, SMALL_EDGE) if isinstance(color, dict) else color, width=2)
        draw.text(min(corners, key=lambda c: c[1]), proxy.id, fill="white", stroke_width=2, stroke_fill="black")
    return out
