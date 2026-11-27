"""Flat wall art: build a picture or poster as a thin slab painted from the photo.

Image-to-3D models turn a poster seen head-on into a deep box with invented
sides. But a flat thing on a known wall needs no generating: its outline in
the photo, cast onto the wall, gives its rectangle, and the photo itself,
straightened, is its front. The frame gets no relief, and anything in front
of the picture (or a reflection in its glass) is painted onto it.
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from pydantic import BaseModel
from scipy.ndimage import gaussian_filter

from aomma.layout import Camera, Vec3
from aomma.stages.inventory import InventoryObject
from aomma.stages.pose import rotation_z
from aomma.stages.score import project

# Wall objects no deeper than this are built as slabs, and slabs are this thin at least.
MAX_DEPTH_M = 0.06
_MIN_DEPTH_M = 0.01
# Spurs on the mask's edge thinner than this share of its shorter side are ignored.
_SPUR = 0.03
_MAX_TEXTURE = 2048
# Snapping the outline to the photo: each side may move by this share of the shorter side,
# to the outermost colour step at least this strong (0-1) and this large a share of the strongest.
_SNAP_REACH = 0.07
_SNAP_MIN_STEP = 0.15
_SNAP_SHARE = 0.25
# Frame relief. A frame's inner edge is looked for within this share of the picture's shorter side,
# skipping the outermost sliver, where the mask's ragged outline lets the wall show.
_FRAME_REACH = 0.25
_FRAME_SKIP = 0.015
# The edge must be a colour step at least this strong (0-1) on every side, at widths
# that agree to within this share of their median.
_MIN_STEP = 0.12
_WIDTH_SPREAD = 0.4
# How steeply the frame's bevel tilts the surface it is drawn on.
_BEVEL = 1.5


class FlatArt(BaseModel):
    # Width along the wall, depth out of it, and height, in meters.
    size: Vec3
    # Bottom-centre of the slab, whose back rests on the wall.
    translate: Vec3
    yaw_deg: float
    # The picture's corners in the photo, in pixels: bottom-left, bottom-right, top-right, top-left.
    outline: tuple[tuple[float, float], tuple[float, float], tuple[float, float], tuple[float, float]]
    # Overlap of the slab's front with the object's mask, 0-1.
    iou: float


def is_flat_art(obj: InventoryObject) -> bool:
    return obj.on == "wall" and min(obj.size_m) <= MAX_DEPTH_M


def front_corners(art: FlatArt) -> np.ndarray:
    """World corners of the front face: bottom-left, bottom-right, top-right, top-left as a viewer sees it."""
    width, depth, height = art.size
    # The front is the slab's +Y face; facing it, +X is on the viewer's left.
    local = np.array(
        [[width / 2, depth / 2, 0], [-width / 2, depth / 2, 0], [-width / 2, depth / 2, height], [width / 2, depth / 2, height]]
    )
    return local @ rotation_z(art.yaw_deg).T + np.array(art.translate)


def find_outline(mask: np.ndarray) -> np.ndarray:
    """The four corners of a roughly upright rectangle's mask: bottom-left, bottom-right, top-right, top-left.

    Each corner is the mask pixel reaching furthest along one diagonal. Thin
    spurs on the mask's ragged edge are worn away first so they cannot pose as corners.
    """
    rows, cols = np.nonzero(mask)
    spur = max(int(min(np.ptp(rows), np.ptp(cols)) * _SPUR) | 1, 3)
    solid = Image.fromarray(mask).convert("L").filter(ImageFilter.MinFilter(spur)).filter(ImageFilter.MaxFilter(spur))
    if np.asarray(solid).any():
        rows, cols = np.nonzero(np.asarray(solid))
    # Measure each way in units of the box's own sides, so a tall shape's diagonals still pick corners.
    x, y = (cols - cols.min()) / max(np.ptp(cols), 1), (rows - rows.min()) / max(np.ptp(rows), 1)
    picks = [np.argmax(y - x), np.argmax(x + y), np.argmax(x - y), np.argmin(x + y)]
    # A pixel's far edges count as part of it.
    return np.array([[cols[i] + 0.5, rows[i] + 0.5] for i in picks])


def snap_outline(image: Image.Image, outline: np.ndarray) -> np.ndarray:
    """Move each side of `outline` onto the picture's real edge in the photo.

    Masks follow a frame's outer edge loosely, often stopping at the mat on one
    side and the wall on another. The true edge is where wall gives way to
    frame: the outermost clear colour step near the mask's side. Each side is
    placed from two such steps, one near each end, so it can also turn a little.
    """
    pixels = np.asarray(image.convert("RGB"), dtype=float) / 255
    centre = outline.mean(axis=0)
    reach = max(int(min(np.linalg.norm(outline[1] - outline[0]), np.linalg.norm(outline[2] - outline[1])) * _SNAP_REACH), 3)
    offsets = np.arange(-reach, reach + 1)
    lines = []
    for start, end in zip(outline, np.roll(outline, -1, axis=0)):
        side = end - start
        outward = np.array([side[1], -side[0]]) / np.linalg.norm(side)
        if outward @ ((start + end) / 2 - centre) < 0:
            outward = -outward
        anchors = []
        for near, far in ((0.1, 0.45), (0.55, 0.9)):
            along = start + np.linspace(near, far, 40)[:, None] * side
            spots = np.floor(along[None] + offsets[:, None, None] * outward).astype(int)
            inside = (spots[..., 0] >= 0) & (spots[..., 0] < image.width) & (spots[..., 1] >= 0) & (spots[..., 1] < image.height)
            shift = 0
            if inside.all():
                profile = pixels[spots[..., 1], spots[..., 0]].mean(axis=1)
                steps = np.array(
                    [np.linalg.norm(profile[i : i + 2].mean(axis=0) - profile[i - 2 : i].mean(axis=0)) for i in range(2, len(offsets) - 1)]
                )
                clear = np.nonzero(steps >= max(_SNAP_MIN_STEP, _SNAP_SHARE * steps.max()))[0]
                if len(clear):
                    shift = offsets[clear[-1] + 2]
            anchors.append(start + (near + far) / 2 * side + shift * outward)
        lines.append(anchors)

    # Each corner is where the side ending there meets the side starting there.
    snapped = []
    for (a, b), (c, d) in zip(np.roll(np.array(lines), 1, axis=0), np.array(lines)):
        t = np.linalg.solve(np.stack([b - a, c - d], axis=1), c - a)[0]
        snapped.append(a + t * (b - a))
    return np.array(snapped)


def fit_flat_art(
    mask: np.ndarray, camera: Camera, wall: tuple[np.ndarray, float], depth_m: float, image: Image.Image | None = None
) -> FlatArt:
    """Fit an upright rectangle on `wall` to the object's `mask`, sharpened against the photo if `image` is given.

    `wall` is a plane (unit normal into the room, offset) with normal @ p = offset.
    The mask's corners are followed along their lines of sight to the slab's
    front face, so the slab covers the mask from the camera however far away
    the wall is. The wall is never fitted perfectly, so the four points are
    not quite a rectangle: the slab takes the average of opposite sides.
    """
    normal, offset = wall
    depth_m = float(np.clip(depth_m, _MIN_DEPTH_M, MAX_DEPTH_M))
    outline = find_outline(mask)
    if image is not None:
        outline = snap_outline(image, outline)
    rays = np.stack(
        [
            (outline[:, 0] - camera.width / 2) / camera.focal_px,
            -(outline[:, 1] - camera.height / 2) / camera.focal_px,
            -np.ones(4),
        ],
        axis=1,
    ) @ camera.matrix[:3, :3].T
    position = camera.matrix[:3, 3]
    reach = (offset + depth_m - normal @ position) / (rays @ normal)
    points = position + rays * reach[:, None]

    along = np.sort(points @ np.cross(normal, (0.0, 0.0, 1.0)))
    low, high = along[:2].mean(), along[2:].mean()
    bottom, top = points[:2, 2].mean(), points[2:, 2].mean()
    centre = np.cross(normal, (0.0, 0.0, 1.0)) * (low + high) / 2 + normal * (offset + depth_m / 2)
    art = FlatArt(
        size=(float(high - low), depth_m, float(top - bottom)),
        translate=(float(centre[0]), float(centre[1]), float(bottom)),
        # Turn the slab's +Y face to look along the wall's normal.
        yaw_deg=float(np.degrees(np.arctan2(normal[1], normal[0])) - 90),
        outline=tuple((float(x), float(y)) for x, y in outline),
        iou=0.0,
    )

    seen = project(front_corners(art), camera)[0]
    if (outline.min() < 2) or (outline[:, 0].max() > camera.width - 2) or (outline[:, 1].max() > camera.height - 2):
        # The picture runs out of the photo, so its corners there are not real
        # corners. Cut the texture along the fitted rectangle's outline instead.
        art.outline = tuple((float(x), float(y)) for x, y in seen)
    drawn = Image.new("1", (camera.width, camera.height))
    ImageDraw.Draw(drawn).polygon([tuple(p) for p in seen], fill=1)
    covered = np.asarray(drawn)
    art.iou = float((covered & mask).sum() / max((covered | mask).sum(), 1))
    return art


def flat_art_texture(source: Image.Image, art: FlatArt, camera: Camera) -> Image.Image:
    """The picture cut from `source` along its outline and straightened.

    `source` is the photo at any resolution; `camera` describes it at the
    working resolution, so pixel positions are scaled between the two.
    """
    corners = np.array(art.outline) * source.width / camera.width
    bottom_left, bottom_right, top_right, top_left = corners
    width = int(np.clip(max(np.linalg.norm(bottom_right - bottom_left), np.linalg.norm(top_right - top_left)), 16, _MAX_TEXTURE))
    height = max(int(round(width * art.size[2] / art.size[0])), 16)
    if height > _MAX_TEXTURE:
        width, height = max(int(round(width * _MAX_TEXTURE / height)), 16), _MAX_TEXTURE

    # A flat rectangle seen in perspective is a homography of the texture:
    # solve the eight coefficients taking texture pixels to photo pixels.
    rows, values = [], []
    for (x, y), (u, v) in zip([(0, height), (width, height), (width, 0), (0, 0)], corners):
        rows += [[x, y, 1, 0, 0, 0, -u * x, -u * y], [0, 0, 0, x, y, 1, -v * x, -v * y]]
        values += [u, v]
    coefficients = np.linalg.solve(np.array(rows, dtype=float), np.array(values, dtype=float))
    return source.convert("RGB").transform(
        (width, height), Image.Transform.PERSPECTIVE, tuple(coefficients), Image.Resampling.BICUBIC
    )


def frame_widths(texture: Image.Image) -> tuple[int, int, int, int] | None:
    """Width in pixels of the frame along the left, top, right and bottom of a straightened picture.

    A frame is a band of its own colour round the border, so going inwards from
    each side the colour steps where the frame ends. A side can show several
    steps (a sliver of wall, an inner border, the print's own edge), so the
    frame is the outermost width at which all four sides have a step. Returns
    None when there is no such width.
    """
    pixels = np.asarray(texture.convert("RGB"), dtype=float) / 255
    reach = int(min(pixels.shape[:2]) * _FRAME_REACH)
    skip = max(int(min(pixels.shape[:2]) * _FRAME_SKIP), 1)
    window = max(reach // 20, 2)
    if reach < skip + 4 * window:
        return None
    sides = []
    # Turn the picture so each side in turn is the top; average the middle of each row.
    for turns in (3, 0, 1, 2):
        side = np.rot90(pixels, turns)
        margin = side.shape[1] // 5
        profile = side[:reach, margin : side.shape[1] - margin].mean(axis=1)
        first = skip + window
        steps = np.array(
            [
                np.linalg.norm(profile[d : d + window].mean(axis=0) - profile[d - window : d].mean(axis=0))
                for d in range(first, reach - window)
            ]
        )
        peaks = [
            i + first
            for i in range(1, len(steps) - 1)
            if steps[i] >= _MIN_STEP and steps[i] >= steps[i - 1] and steps[i] > steps[i + 1]
        ]
        sides.append(np.array(peaks))
    if any(len(peaks) == 0 for peaks in sides):
        return None
    for width in sides[0]:
        matches = [int(peaks[np.argmin(np.abs(peaks - width))]) for peaks in sides]
        if max(matches) - min(matches) <= _WIDTH_SPREAD * np.median(matches):
            return tuple(matches)
    return None


def frame_normal_map(texture: Image.Image) -> Image.Image | None:
    """A normal map that makes the picture's frame stand proud of what it holds, or None if no frame is found."""
    widths = frame_widths(texture)
    if widths is None:
        return None
    left, top, right, bottom = widths
    height = np.ones((texture.height, texture.width), dtype=np.float32)
    height[top : texture.height - bottom, left : texture.width - right] = 0
    # Soften the step into a bevel a fraction of the frame wide.
    bevel = max(min(widths) / 4, 1.0)
    height = gaussian_filter(height, bevel)
    down, across = np.gradient(height * bevel * _BEVEL)
    # Tangent space: red to the right, green up the picture (rows run down), blue out of it.
    normal = np.stack([-across, down, np.ones_like(height)], axis=-1)
    normal /= np.linalg.norm(normal, axis=-1, keepdims=True)
    return Image.fromarray(((normal + 1) / 2 * 255).round().astype(np.uint8))
