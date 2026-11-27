"""Gate 4: what each small item is built as, from what the label map measured. Nothing generated.

- **sheet**: more of its pixels lie on a surface (`labels`' layers) than stand off it. It adds nothing to that
  surface's shape: a paper on a fridge door, a poster flush on a wall, a blind in its window's plane. Painted from the
  photo onto the surface (`fit_flat`, `flat_texture`), cut out along its pixels; no box, no model.
- **box**: anything else. A box at gate 4, a generated model at gate 5.

Decided by measurement alone: not by the listing's "on wall", the VLM's guessed size or the item's kind (the plan's
"Reasoning, not rules per kind of thing"; `flat.is_flat_art` decides by those). Which share is larger, not every
pixel, decides a sheet: MoGe gives a printed picture relief it does not have, so a flat poster keeps blocks of its own
pixels.

**Not decided yet: slabs** (a thin thing standing off its surface, parallel to it: a framed picture, a blind hanging in
front of its window, a tray). Tried as "its own plane parallel to the surface, within the surface's own
spread, and thinner than its face", and it does not hold: the claimant beside an item is often not one
plane (a desk's top and front, a window showing the outdoors), so its spread reads 5-60 cm and a tomato or a bowl
passes as flat. The measures are kept on each box's `Form` (standoff, spread, tilt, face) for that work.
"""

import numpy as np
from PIL import Image
from pydantic import BaseModel
from scipy.ndimage import binary_dilation, binary_fill_holes

from aomma.labels import _MIN_OWN, _OWN_SPREAD_SHARE, _WINDOWS, _tolerance
from aomma.layout import Camera, Vec3
from aomma.stages.proxy import Proxy
from aomma.stages.room import wall_face

SHEET, BOX = "sheet", "box"
# A sheet's front is lifted this far off its surface, only so the renderer draws it over the surface rather than
# through it (a rendering offset, below any depth source's precision; it decides nothing).
SHEET_LIFT_M = 0.003
_MAX_TEXTURE = 2048


class Form(BaseModel):
    """What one small item is built as, and the measures that decided it."""

    id: str
    form: str
    why: str
    kept_px: int
    layer_px: int
    # The claimant it lies on or stands off, the plane fitted there (world, unit normal toward the camera, a point on
    # it), and that surface's own spread.
    surface: str | None = None
    surface_class: str | None = None
    normal: Vec3 | None = None
    point: Vec3 | None = None
    tolerance_m: float | None = None
    # How far its own points stand off the surface (median), and how far 90% of them lie from the plane through that.
    standoff_m: float | None = None
    spread_m: float | None = None
    # Its face's sides along the surface, and how far tilting its own plane onto the surface's moves depth across it.
    face_m: tuple[float, float] | None = None
    tilt_m: float | None = None


class Flat(BaseModel):
    """A sheet as built: its front's world corners (bottom-left, bottom-right, top-right, top-left as seen from the
    camera), what it lies on (`built_on`: a box's or a wall's id, None for the measured plane), how far its middle
    lies in front of the plane measured from points, and its corners in the photo."""

    id: str
    form: str
    corners: list[Vec3]
    built_on: str | None
    off_measured_m: float
    normal: Vec3
    outline_px: list[tuple[float, float]]
    size_m: tuple[float, float]


def _plane(points: np.ndarray, eye: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Least-squares plane through `points`: a unit normal facing `eye`, and their centroid."""
    centre = points.mean(axis=0)
    normal = np.linalg.svd(points - centre, full_matrices=False)[2][2]
    if normal @ (eye - centre) < 0:
        normal = -normal
    return normal, centre


def _surface_plane(on: np.ndarray, near: np.ndarray, points: np.ndarray, known: np.ndarray, tolerance: float,
                   eye: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """The plane of the surface's pixels `on` within `near`: fitted, then fitted again to the points within the
    surface's own `tolerance` of the first (a handle, a door's gap or a leak does not tilt it)."""
    at = on & near & known
    if at.sum() < _MIN_OWN:
        return None
    pts = points[at]
    normal, centre = _plane(pts, eye)
    close = np.abs((pts - centre) @ normal) <= tolerance
    if close.sum() >= _MIN_OWN:
        normal, centre = _plane(pts[close], eye)
    return normal, centre


def decide_forms(label_map: np.ndarray, layer_map: np.ndarray, labels: list[dict], points_world: np.ndarray,
                 eye: np.ndarray) -> dict[str, Form]:
    """Each small item's form. `labels` is labels.json's list; `layer_map` is `debug/labels_layers.png`."""
    known = np.isfinite(points_world).all(axis=-1)
    p = np.where(known[..., None], points_world, 0.0)
    width = label_map.shape[1]
    sizes = [max(3, int(round(width * share)) | 1) for share in _WINDOWS]
    by_value = {e["value"]: e for e in labels}

    tolerances: dict[int, float | None] = {}

    def tolerance_of(value: int) -> float | None:
        if value not in tolerances:
            tolerances[value] = _tolerance((label_map == value) & known, p, sizes)
        return tolerances[value]

    forms = {}
    for e in labels:
        if e["class"] != "small":
            continue
        value, kept, layer_px = e["value"], int(e["kept_px"]), int(e.get("layer_px", 0))
        if not kept and not layer_px:
            continue
        mine, layered = label_map == value, layer_map == value
        whole = mine | layered
        rows, cols = np.nonzero(whole)
        # Around the item: its own extent again on every side.
        h, w = np.ptp(rows) + 1, np.ptp(cols) + 1
        near = np.zeros(whole.shape, bool)
        near[max(0, rows.min() - h):rows.max() + h + 1, max(0, cols.min() - w):cols.max() + w + 1] = True

        if layer_px > kept:
            # The surface it lies on: whichever claimant keeps most of its layer pixels.
            under = label_map[layered]
            surface = int(np.bincount(under[under > 0]).argmax()) if (under > 0).any() else 0
        else:
            # The surface it stands off: the claimant (not another small item) most along its border.
            ring = binary_dilation(whole, iterations=2) & ~whole
            around = [v for v in label_map[ring] if v and by_value[v]["class"] != "small"]
            surface = int(np.bincount(around).argmax()) if around else 0
        form = Form(id=e["id"], form=BOX, why="", kept_px=kept, layer_px=layer_px)
        tolerance = tolerance_of(surface) if surface else None
        if not surface or tolerance is None:
            form.why = "no surface beside it to measure against"
            forms[e["id"]] = form
            continue
        # A layer's pixels lie on the surface by construction, so they help fit its plane.
        plane = _surface_plane((label_map == surface) | layered, near, p, known, tolerance, eye)
        if plane is None:
            form.why = f"too little of {by_value[surface]['id']} near it to fit a plane"
            forms[e["id"]] = form
            continue
        normal, centre = plane
        form.surface, form.tolerance_m = by_value[surface]["id"], round(tolerance, 4)
        form.surface_class = by_value[surface]["class"]
        form.normal, form.point = tuple(float(x) for x in normal), tuple(float(x) for x in centre)
        own = mine & known
        if own.sum() >= _MIN_OWN:
            off = (p[own] - centre) @ normal
            standoff = float(np.median(off))
            spread = float(np.quantile(np.abs(off - standoff), _OWN_SPREAD_SHARE))
            form.standoff_m, form.spread_m = round(standoff, 4), round(spread, 4)
            mine_pts = p[own]
            right = np.linalg.svd(np.eye(3) - np.outer(normal, normal))[0][:, 0]
            up = np.cross(normal, right)
            face = (float(np.ptp(mine_pts @ right)), float(np.ptp(mine_pts @ up)))
            own_normal, _ = _plane(mine_pts, eye)
            sine = float(np.linalg.norm(np.cross(own_normal, normal)))
            form.face_m = (round(face[0], 4), round(face[1], 4))
            form.tilt_m = round(sine * max(face), 4)
        if layer_px > kept:
            form.form, form.why = SHEET, f"{layer_px} of its {layer_px + kept} px lie on {form.surface}"
        elif form.standoff_m is None:
            form.why = "too few points of its own"
        else:
            form.why = (f"{kept} of its {layer_px + kept} px stand off {form.surface}: {100 * form.standoff_m:.1f} cm "
                        f"(spread {100 * form.spread_m:.1f} cm, tilt {100 * form.tilt_m:.1f} cm, the surface's own "
                        f"{100 * tolerance:.1f} cm)")
        forms[e["id"]] = form
    return forms


def sheet_pixels(mask: np.ndarray | None, label_map: np.ndarray, layer_map: np.ndarray, value: int,
                 surface_value: int | None) -> np.ndarray:
    """The pixels a sheet's picture is cut out along: its own mask wherever the label map gives the pixel to it or to
    its surface (`mask`; without one, only what the label map gives it, its kept and layer pixels), and the holes those
    close round. Depth gives the surface much of an item lying on it (BedroomScan 006's blind: 42 of its 50 thousand
    pixels to its window, settled by depth), and those are still its picture. Where the mask
    leaks onto the surface, the sheet shows the surface's own photo on the surface in the same place, which cannot be
    seen. A pixel given to anything else, standing in front of it, is left out."""
    own = (label_map == value) | (layer_map == value)
    if mask is not None and surface_value:
        own |= mask & (label_map == surface_value)
    return binary_fill_holes(own)


def _rays(camera: Camera, pixels: np.ndarray) -> np.ndarray:
    """World directions through pixel positions (x, y), as `flat.fit_flat_art` casts them."""
    cam = np.stack([(pixels[:, 0] - camera.width / 2) / camera.focal_px,
                    -(pixels[:, 1] - camera.height / 2) / camera.focal_px, -np.ones(len(pixels))], axis=1)
    return cam @ camera.matrix[:3, :3].T


def project_px(camera: Camera, points: np.ndarray) -> np.ndarray:
    """World points to pixel positions (x, y) in the photo."""
    cam = (points - camera.matrix[:3, 3]) @ camera.matrix[:3, :3]
    return np.stack([camera.focal_px * cam[:, 0] / -cam[:, 2] + camera.width / 2,
                     -camera.focal_px * cam[:, 1] / -cam[:, 2] + camera.height / 2], axis=1)


def built_on(form: Form, built: dict[str, Proxy], walls: list[Proxy]) -> Proxy | None:
    """What a sheet's surface is built as: its box (a built-in's, a large or small item's), the wall holding it where
    it is an opening, or, where it is a wall, the wall whose room-side face lies nearest the plane measured there.
    None where nothing built stands for it (a floor; a surface with no box)."""
    if form.surface in built:
        return built[form.surface]
    holding = [w for w in walls if any(o.id == form.surface for o in w.openings)]
    if holding:
        return holding[0]
    if form.surface_class != "wall" or not walls or form.point is None:
        return None
    eye_side = np.array(form.point) + np.array(form.normal)  # a point on the room's side of the measured plane
    return min(walls, key=lambda w: _to_face(np.array(form.point), *wall_face(w, eye_side)))


def _to_face(point: np.ndarray, origin: np.ndarray, along: np.ndarray, normal: np.ndarray) -> float:
    """How far `point` lies from a wall's room-side face (`room.wall_face`), its width along `along`, of any height."""
    t = np.clip((point - origin) @ along / (along @ along), 0.0, 1.0)
    offset = point - (origin + along * t)
    return float(np.linalg.norm(offset - offset[2] * np.array([0.0, 0.0, 1.0])))


def built_face(form: Form, box: Proxy) -> tuple[np.ndarray, np.ndarray]:
    """The face of `box` a sheet goes on: of its six, the one whose outward normal is nearest the measured plane's
    (`form.normal`, toward the camera). Returned as that normal and a point on the face. The measured plane only says
    which surface and which way it faces; the sheet lies on what is built, or the box (or a wall's pane) hides part of
    it wherever the two planes cross."""
    sx, sy, sz = box.size
    yaw = np.radians(box.yaw_deg)
    turn = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
    centre = np.array(box.translate) + np.array([0.0, 0.0, sz / 2])
    faces = [(turn @ axis * sign, centre + turn @ axis * sign * half)
             for axis, half in ((np.array([1.0, 0, 0]), sx / 2), (np.array([0, 1.0, 0]), sy / 2), (np.array([0, 0, 1.0]), sz / 2))
             for sign in (1.0, -1.0)]
    return max(faces, key=lambda face: float(face[0] @ np.array(form.normal)))


def fit_flat(form: Form, pixels: np.ndarray, camera: Camera, box: Proxy | None = None) -> Flat | None:
    """The sheet's front: the rectangle that holds every one of its `pixels` (a mask), axes along the camera's right and
    up as they lie in its plane, `SHEET_LIFT_M` off the face of `box` it goes on (`built_on`, `built_face`), or off the
    measured plane where nothing is built for its surface."""
    normal, point = (built_face(form, box) if box is not None else (np.array(form.normal), np.array(form.point)))
    front = point + normal * SHEET_LIFT_M
    eye = camera.matrix[:3, 3]
    rows, cols = np.nonzero(pixels)
    # Each pixel's four corners, so the rectangle covers the pixels, not only their centres.
    xy = np.concatenate([np.stack([cols + dx, rows + dy], axis=1) for dx in (0, 1) for dy in (0, 1)]).astype(float)
    rays = _rays(camera, xy)
    facing = rays @ normal
    reach = ((front - eye) @ normal) / np.where(np.abs(facing) > 1e-9, facing, np.nan)
    hit = np.isfinite(reach) & (reach > 0)
    if hit.sum() < 4:
        return None
    on_plane = eye + rays[hit] * reach[hit, None]
    right = camera.matrix[:3, 0] - (camera.matrix[:3, 0] @ normal) * normal
    if np.linalg.norm(right) < 1e-6:
        return None
    right /= np.linalg.norm(right)
    up = np.cross(normal, right)
    if up @ camera.matrix[:3, 1] < 0:
        up = -up
    u, v = (on_plane - front) @ right, (on_plane - front) @ up
    corners = np.array([front + right * a + up * b for a, b in
                        ((u.min(), v.min()), (u.max(), v.min()), (u.max(), v.max()), (u.min(), v.max()))])
    centre = corners.mean(axis=0)
    return Flat(id=form.id, form=form.form, corners=[tuple(float(x) for x in c) for c in corners],
                built_on=box.id if box is not None else None,
                off_measured_m=float((centre - np.array(form.point)) @ np.array(form.normal)),
                normal=tuple(float(x) for x in normal), outline_px=[(float(x), float(y)) for x, y in project_px(camera, corners)],
                size_m=(float(np.ptp(u)), float(np.ptp(v))))


def flat_texture(source: Image.Image, pixels: np.ndarray, flat: Flat, camera: Camera) -> Image.Image:
    """The front's picture: the photo straightened from the front's outline, transparent off the item's `pixels`."""
    scale = source.width / camera.width
    corners = np.array(flat.outline_px) * scale
    sides = [np.linalg.norm(corners[(i + 1) % 4] - corners[i]) for i in range(4)]
    width = int(np.clip(max(sides[0], sides[2]), 16, _MAX_TEXTURE))
    height = int(np.clip(round(width * flat.size_m[1] / max(flat.size_m[0], 1e-6)), 16, _MAX_TEXTURE))
    # A flat rectangle seen in perspective is a homography of the texture (as `flat.flat_art_texture`).
    rows, values = [], []
    for (x, y), (a, b) in zip([(0, height), (width, height), (width, 0), (0, 0)], corners):
        rows += [[x, y, 1, 0, 0, 0, -a * x, -a * y], [0, 0, 0, x, y, 1, -b * x, -b * y]]
        values += [a, b]
    coefficients = tuple(np.linalg.solve(np.array(rows, float), np.array(values, float)))
    colour = source.convert("RGB").transform((width, height), Image.Transform.PERSPECTIVE, coefficients,
                                             Image.Resampling.BICUBIC)
    mask = Image.fromarray(pixels.astype(np.uint8) * 255).resize(source.size, Image.NEAREST)
    alpha = mask.transform((width, height), Image.Transform.PERSPECTIVE, coefficients, Image.Resampling.BILINEAR)
    colour.putalpha(alpha)
    return colour
