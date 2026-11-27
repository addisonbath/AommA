"""The floor of the room shell: what it is covered with, where each covering lies, and a texture for each.

The floor was one slab of one colour, 12 m square: flat taupe for the
bedroom's carpet, one dark grey for the garage's black mats and grey concrete,
and running on past the walls. Now:

- a short survey of its own lists the floor's materials, each with a box, and
  each is segmented from its box, as walls are. The largest is the floor;
  the rest (a rug, mats) are coverings lying on it, each a thin rectangle;
- each material's texture is drawn by an image editor from a clean square of
  it seen from straight above, cut out of the photo where nothing stands on
  it, then made to tile and brought to the material's colour in the photo.
  A repeated piece of the photo itself showed its seams and copied whatever
  lay on the floor;
- the floor stops at the walls: it is cut back by each wall's plane, unless
  that would cut away floor the photo shows (a wall in a recess).
"""

import io
import json
import re
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image
from pydantic import BaseModel
from scipy import ndimage

from aomma.backends.comfy import ComfyClient
from aomma.backends.vlm import VLM
from aomma.layout import Camera
from aomma.ledger import Ledger
from aomma.stages.image import to_jpeg
from aomma.stages.inventory import InventoryObject, parse_inventory
from aomma.stages.proxy import Proxy
from aomma.stages.room import wall_face
from aomma.stages.score import project

PROMPT = """\
You are surveying the floor in a photo so that it can be rebuilt in 3D.

List each material the floor is covered with that the photo shows: the floor itself
(wood boards, carpet, tiles, concrete, vinyl) and anything that covers part of it and
lies flat on it, such as a rug, a runner or floor mats. One entry per material, so
matching mats laid side by side are one entry. Do not list furniture, a pet bed,
clothes or anything else that merely lies on the floor, nor floor seen only through a
doorway into another room.

Reply with JSON only, in the form {"floors": [...]}, the floor itself first. Each entry has:
- "id": unique snake_case name, such as "carpet" or "rubber_mats"
- "kind": "floor" for the floor itself, "covering" for a rug or mats lying on it
- "bbox_2d": [x1, y1, x2, y2] bounding box normalised to 0-1000 (x across the width, y down the height)
- "material": at most 8 words: its colour, material and pattern, such as "light oak boards, narrow"
"""

SCHEMA = {
    "type": "object",
    "properties": {
        "floors": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "kind": {"type": "string", "enum": ["floor", "covering"]},
                    "bbox_2d": {"type": "array", "items": {"type": "number"}},
                    "material": {"type": "string"},
                },
                "required": ["id", "kind", "bbox_2d", "material"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["floors"],
    "additionalProperties": False,
}

# A covering is this thick: it lies on the floor, and its top is drawn this far above it.
COVERING_M = 0.01
# The floor reaches this far from the camera's foot where no wall stops it.
_REACH_M = 8.0
# A wall's plane does not bound the floor if it would cut away more than this share of the floor the photo shows.
_MAX_CUT = 0.02
# A covering is kept if the photo shows at least this much of it, in square meters; less is taken as the floor.
_MIN_COVERING_M2 = 0.1
# The texture's source square: at most this side, and at least this, from where the photo sees the
# material nearest (sharpest), on a grid of this cell size.
_MAX_TILE_M = 1.0
_MIN_TILE_M = 0.25
_CELL_M = 0.004
# A cell of the source square is clean if the photo's point there lies within this of the floor.
_ON_FLOOR_M = 0.05
# A floor's grain within this many degrees of a wall's heading is turned to it (boards laid square to the room).
_SNAP_DEG = 10.0
# A texture is made to tile by blending it, over this share of its side from each edge, with itself shifted by half.
_BLEND = 0.25
# A material has a grain if its name says so; its texture is then drawn with the grain running across. How
# strongly the photo's lines run one way did not tell: the carpets and concrete scored 0.18-0.35, and the
# simple photo's boards, far off and with a book on them, 0.11.
_GRAINED = re.compile(r"\b(board|plank|wood|laminate|parquet|tile|strip)", re.IGNORECASE)


class FloorSurvey(BaseModel):
    # Each material, the floor first; the description is the material.
    floors: list[InventoryObject]
    # "floor" or "covering", by id.
    kinds: dict[str, str] = {}
    # How well each one's mask fits its box (see `MaskInfo`); one nowhere near it is not used.
    box_iou: dict[str, float] = {}


class FloorRegion(BaseModel):
    id: str
    material: str
    kind: str
    # Corners in the world frame's x, y, counter-clockwise; a convex polygon.
    polygon: list[tuple[float, float]]
    top_m: float
    # The material's colour in the photo, linear RGB 0-1.
    color: tuple[float, float, float]
    # The texture's file name in assets/, the side in meters one copy of it covers, and its heading.
    texture: str | None = None
    tile_m: float | None = None
    angle_deg: float = 0.0


RETRY_NOTE = """

Your last reply could not be read ({problem}). Reply with JSON only, as above:
each entry in "floors" an object with "id", "kind", "bbox_2d" and "material"."""


def parse_floor_survey(text: str) -> FloorSurvey:
    """Parse a model reply, tolerating code fences; ids are made unique."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"no JSON in model reply: {text[:200]!r}")
    entries = json.loads(match.group()).get("floors", [])
    if not all(isinstance(entry, dict) for entry in entries):
        raise ValueError(f"floors not given as objects: {entries[:3]!r}")
    kinds = [str(entry.get("kind", "floor")) for entry in entries]
    plain = [
        {"id": entry["id"], "bbox_2d": entry["bbox_2d"], "description": entry.get("material", ""), "size_m": [1.0, 1.0, 0.01], "on": "floor"}
        for entry in entries
    ]
    floors = parse_inventory(json.dumps({"objects": plain})).objects
    return FloorSurvey(floors=floors, kinds={floor.id: kind for floor, kind in zip(floors, kinds)})


def take_floor_survey(image, vlm: VLM, ledger: Ledger) -> tuple[FloorSurvey, str]:
    """Return the parsed floor survey and the raw reply (kept for debugging; both replies, if it was asked twice).

    A local model, without structured output, has listed the floors as bare
    names. Asked the same again at temperature 0 it says the same, so the
    second question says what was wrong.
    """
    ledger.check_budget()
    text, usage = vlm.ask(to_jpeg(image), PROMPT, SCHEMA, label="floor survey")
    ledger.record("floor survey", usage)
    try:
        return parse_floor_survey(text), text
    except ValueError as error:
        problem = str(error)
    ledger.check_budget()
    again, usage = vlm.ask(to_jpeg(image), PROMPT + RETRY_NOTE.format(problem=problem), SCHEMA, label="floor survey, again")
    ledger.record("floor survey", usage)
    return parse_floor_survey(again), f"{text}\n\n--- asked again ({problem}) ---\n\n{again}"


def assign_floor(floor_pixels: np.ndarray, masks: dict[str, np.ndarray], kinds: dict[str, str]) -> tuple[np.ndarray, list[str]]:
    """Which material each floor pixel shows: an (H, W) array of indices into the returned ids, -1 off the floor.

    A covering's mask takes its pixels from the floor's, the smaller from the
    larger where they overlap; what no mask takes is the floor's. The floor is
    the material listed as "floor" that takes the most pixels (with none, the
    largest), first among the ids.
    """
    order = sorted(masks, key=lambda oid: -int((masks[oid] & floor_pixels).sum()))
    floors = [oid for oid in order if kinds.get(oid) == "floor"] or order[:1]
    ids = (floors[:1] or ["floor"]) + [oid for oid in order if oid not in floors[:1]]
    labels = np.where(floor_pixels, 0, -1)
    # Largest first, so a smaller mask lying inside a larger one wins; the floor only keeps what no covering takes.
    for oid in sorted(ids[1:], key=lambda oid: -int(masks[oid].sum())):
        labels[floor_pixels & masks[oid]] = ids.index(oid)
    return labels, ids


def clip_polygon(polygon: np.ndarray, point: np.ndarray, normal: np.ndarray) -> np.ndarray:
    """The part of the convex `polygon` ((N, 2), in order) where (p - point) @ normal >= 0."""
    clipped = []
    for a, b in zip(polygon, np.roll(polygon, -1, axis=0)):
        side_a, side_b = (a - point) @ normal, (b - point) @ normal
        if side_a >= 0:
            clipped.append(a)
        if (side_a >= 0) != (side_b >= 0):
            clipped.append(a + (b - a) * side_a / (side_a - side_b))
    return np.array(clipped).reshape(-1, 2)


def room_polygon(walls: list[Proxy], camera_position: np.ndarray, seen: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """The floor's outline: a square `_REACH_M` around the camera, cut back by each wall's plane.

    The cut runs along the wall's back face, so the floor reaches under it and
    no gap shows at its foot. A wall whose plane would cut away more than
    `_MAX_CUT` of the `seen` floor points ((N, 2)) does not bound the room past
    its own length (a wall in a recess), and is not used. Returns the polygon
    and the ids of the walls that bound it.
    """
    foot = camera_position[:2]
    polygon = foot + _REACH_M * np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])
    used = []
    for wall in walls:
        origin, _, normal = wall_face(wall, camera_position)
        back = origin[:2] - normal[:2] * wall.size[1]
        if len(seen) and ((seen - back) @ normal[:2] < 0).mean() > _MAX_CUT:
            continue
        polygon = clip_polygon(polygon, back, normal[:2])
        used.append(wall.id)
    return polygon, used


def covering_rectangle(points: np.ndarray, wall_headings: list[float]) -> tuple[np.ndarray, float]:
    """The smallest rectangle around a covering's floor points ((N, 2)), laid square to the longest wall if there is one.

    A rug or mats are laid square to the room. What the photo shows of one is
    cut by what stands on it: the garage's mats, under the treadmill and a
    basket, gave a rectangle turned 39 degrees. Its sides are taken from the
    1st to the 99th percentile of the points along each axis, so a stray point
    does not stretch it. Returns its corners, counter-clockwise, and its
    heading in degrees.
    """
    def extent(angle: float) -> tuple[float, np.ndarray, np.ndarray]:
        axes = np.array([[np.cos(np.radians(angle)), np.sin(np.radians(angle))], [-np.sin(np.radians(angle)), np.cos(np.radians(angle))]])
        local = points @ axes.T
        low, high = np.percentile(local, 1, axis=0), np.percentile(local, 99, axis=0)
        return float(np.prod(high - low)), np.array([low, [high[0], low[1]], high, [low[0], high[1]]]) @ axes, axes

    angle = wall_headings[0] % 90 if wall_headings else min(np.arange(0.0, 90.0, 1.0), key=lambda a: extent(a)[0])
    return extent(angle)[1], float(angle)


def top_down(
    image: Image.Image, points_world: np.ndarray, camera: Camera, region: np.ndarray, angle_deg: float, cell: float = _CELL_M
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """The photo seen from straight above over one floor material's `region` pixels, on a grid turned by `angle_deg`.

    Returns the colours (H, W, 3, uint8), which cells are clean (the photo
    shows this material there, its point on the floor), each cell's distance
    from the camera's foot, and the grid's frame: rows of [origin, step along
    a row, step down a column] in world x, y.
    """
    turn = np.radians(angle_deg)
    across, down = np.array([np.cos(turn), np.sin(turn)]), np.array([-np.sin(turn), np.cos(turn)])
    seen = points_world[region][:, :2]
    low = np.percentile(seen @ np.stack([across, down]).T, 1, axis=0)
    high = np.percentile(seen @ np.stack([across, down]).T, 99, axis=0)
    cell = max(cell, float((high - low).max()) / 1200)
    cols, rows = int((high[0] - low[0]) / cell) + 1, int((high[1] - low[1]) / cell) + 1
    origin = low[0] * across + low[1] * down
    grid = origin + (np.arange(cols)[None, :, None] + 0.5) * cell * across + (np.arange(rows)[:, None, None] + 0.5) * cell * down
    cells = np.concatenate([grid.reshape(-1, 2), np.zeros((rows * cols, 1))], axis=1)

    with np.errstate(divide="ignore", invalid="ignore"):
        pixels, depth = project(cells, camera)
    pixels = np.nan_to_num(pixels, nan=-1.0, posinf=-1.0, neginf=-1.0)
    x, y = np.floor(pixels[:, 0]).astype(int), np.floor(pixels[:, 1]).astype(int)
    inside = (depth > 0.05) & (x >= 0) & (x < camera.width) & (y >= 0) & (y < camera.height)
    x, y = np.clip(x, 0, camera.width - 1), np.clip(y, 0, camera.height - 1)
    with np.errstate(invalid="ignore"):
        clean = inside & region[y, x] & (np.linalg.norm(points_world[y, x] - cells, axis=1) < _ON_FLOOR_M)
    photo = np.asarray(image.convert("RGB"))
    colours = _bilinear(photo, pixels).reshape(rows, cols, 3)
    distance = np.linalg.norm(cells[:, :2] - camera.matrix[:2, 3], axis=1).reshape(rows, cols)
    return colours, clean.reshape(rows, cols), distance, np.stack([origin, across * cell, down * cell])


def _bilinear(photo: np.ndarray, pixels: np.ndarray) -> np.ndarray:
    """The photo's colours at fractional pixel positions ((N, 2), x then y), clamped to its edges."""
    height, width = photo.shape[:2]
    x = np.clip(pixels[:, 0] - 0.5, 0, width - 1)
    y = np.clip(pixels[:, 1] - 0.5, 0, height - 1)
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    x1, y1 = np.minimum(x0 + 1, width - 1), np.minimum(y0 + 1, height - 1)
    fx, fy = (x - x0)[:, None], (y - y0)[:, None]
    top = photo[y0, x0] * (1 - fx) + photo[y0, x1] * fx
    bottom = photo[y1, x0] * (1 - fx) + photo[y1, x1] * fx
    return (top * (1 - fy) + bottom * fy).astype(np.uint8)


def clean_square(clean: np.ndarray, distance: np.ndarray, cell: float) -> tuple[int, int, int] | None:
    """The largest square of clean cells, at most `_MAX_TILE_M` a side, as (row, col, side in cells).

    It is looked for first among the nearer half of the clean cells, which the
    photo shows sharpest, and among all of them if no square there is at least
    `_MIN_TILE_M` a side. None if there is no clean cell at all.
    """
    if not clean.any():
        return None
    limit = max(1, int(_MAX_TILE_M / cell))
    for allowed in ((clean & (distance <= np.median(distance[clean]))).astype(np.uint8), clean.astype(np.uint8)):
        # A square of side k fits wherever a k x k minimum filter is still clean; the largest k is found by halving.
        low, high = 0, limit
        while low < high:
            side = (low + high + 1) // 2
            if ndimage.minimum_filter(allowed, size=side, mode="constant", cval=False).any():
                low = side
            else:
                high = side - 1
        if low and (low * cell >= _MIN_TILE_M or allowed.sum() == clean.sum()):
            fits = ndimage.minimum_filter(allowed, size=low, mode="constant", cval=False)
            r, c = np.argwhere(fits)[0]
            # The filter's window around (r, c) starts side // 2 before it.
            return int(r) - low // 2, int(c) - low // 2, low
    return None


def grain(colours: np.ndarray, clean: np.ndarray) -> tuple[float, float]:
    """Which way the lines in a top-down view run over its `clean` cells: (degrees from across, coherence 0-1).

    From the structure tensor: the direction the brightness changes most is
    across the lines. A wood floor's boards and grain give a high coherence;
    carpet and concrete a low one.
    """
    grey = ndimage.gaussian_filter(colours.astype(float).mean(axis=2), 1.0)
    down, across = np.gradient(grey)
    inner = ndimage.binary_erosion(clean, iterations=2)
    if inner.sum() < 100:
        return 0.0, 0.0
    xx, yy, xy = (across[inner] ** 2).sum(), (down[inner] ** 2).sum(), (across[inner] * down[inner]).sum()
    coherence = float(np.hypot(xx - yy, 2 * xy) / max(xx + yy, 1e-9))
    # The steepest change is at this angle (down the rows positive); the lines run square to it.
    steepest = 0.5 * np.degrees(np.arctan2(2 * xy, xx - yy))
    return float((steepest + 90) % 180), coherence


def even_light(texture: np.ndarray) -> np.ndarray:
    """`texture` ((N, N, 3), sRGB uint8) with its broad shading taken out, keeping its pattern and median.

    The editor draws a floor lit from one side: one half of the texture came
    out darker than the other, and each repeat of it showed as a block.
    """
    linear = (texture / 255.0) ** 2.2
    # The shading is taken as a quadratic across the texture, fitted to each channel: smooth enough to leave
    # the boards' own tones, and fitted out to the edges, where a blur would be pulled towards the middle.
    rows, cols = np.mgrid[0 : texture.shape[0], 0 : texture.shape[1]]
    y, x = (rows / texture.shape[0] - 0.5).ravel(), (cols / texture.shape[1] - 0.5).ravel()
    terms = np.stack([np.ones_like(x), x, y, x * x, x * y, y * y], axis=1)
    flat = linear.reshape(-1, 3)
    broad = terms @ np.linalg.lstsq(terms, flat, rcond=None)[0]
    linear = (flat / np.maximum(broad, 1e-4) * np.median(flat, axis=0)).reshape(linear.shape)
    return (np.clip(linear, 0, 1) ** (1 / 2.2) * 255).round().astype(np.uint8)


def make_tileable(texture: np.ndarray) -> np.ndarray:
    """`texture` ((N, N, 3)) made to repeat without a seam.

    Near its edges it is blended into a copy of itself shifted by half its
    side, whose edges meet: the shifted copy's left column continues its right
    one. The blend shows as a faint doubling near the edges, which a regular
    pattern (boards) shows more than an irregular one (carpet, concrete).
    """
    side = texture.shape[0]
    shifted = np.roll(texture, (side // 2, side // 2), axis=(0, 1))
    position = (np.arange(side) + 0.5) / side
    ramp = np.clip(np.minimum(position, 1 - position) / _BLEND, 0, 1)
    weight = np.minimum(ramp[:, None], ramp[None, :])[..., None]
    return (texture * weight + shifted * (1 - weight)).astype(texture.dtype)


def match_colour(texture: np.ndarray, color: np.ndarray) -> np.ndarray:
    """`texture` (sRGB, uint8) scaled so that its median is `color` (linear RGB 0-1).

    The editor relights what it draws, so the texture keeps its pattern and
    takes the photo's colour for the material.
    """
    linear = (texture / 255.0) ** 2.2
    linear = linear * (color / np.maximum(np.median(linear.reshape(-1, 3), axis=0), 1e-4))
    return (np.clip(linear, 0, 1) ** (1 / 2.2) * 255).round().astype(np.uint8)


def draw_texture(
    comfy: ComfyClient, workflow_path: Path, crop: Image.Image, material: str, name: str, grained: bool, seed: int = 1
) -> Image.Image:
    """Redraw a top-down crop of a floor `material` as an even, empty texture of it; `grained`: its lines run across.

    The workflow is a ComfyUI API-format graph with nodes named "input_image"
    (LoadImage), "input_prompt" (text) and "output_image".
    """
    workflow = json.loads(workflow_path.read_text())
    workflow["input_image"]["inputs"]["image"] = comfy.upload_image(to_jpeg(crop), f"aomma_floor_{name}.jpg")
    workflow["input_prompt"]["inputs"]["text"] = (
        f"Seamless texture of {material}, seen from directly above. Same material, same scale and same colour as "
        f"the photo. Evenly lit, with no shadows or patches of sunlight, and nothing lying on it."
        + (" Its boards, grain and lines run straight from left to right, as in the photo." if grained else "")
    )
    workflow["noise"]["inputs"]["noise_seed"] = seed
    outputs = comfy.run(workflow, label=f"floor texture {name}")
    return Image.open(io.BytesIO(comfy.fetch(outputs["output_image"]["images"][0]))).convert("RGB")


Drawer = Callable[[Image.Image, str, str, bool], Image.Image]


def floor_regions(
    run_dir: Path,
    image: Image.Image,
    points_world: np.ndarray,
    camera: Camera,
    walls: list[Proxy],
    floor_pixels: np.ndarray,
    survey: FloorSurvey | None,
    masks: dict[str, np.ndarray],
    draw: Drawer | None,
) -> list[FloorRegion]:
    """The floor and its coverings, with textures in assets/, from a floor survey's `masks` over the `floor_pixels`.

    Without a survey (the old pipeline), the floor is one material of one
    colour. A texture is drawn with `draw` (crop, material, name, grained) only if the
    run's floors/ has none for that material yet, so a rebuild reuses it;
    without `draw` and with none saved, the material is one colour.
    """
    photo = np.asarray(image.convert("RGB"))
    eye = camera.matrix[:3, 3]
    kinds = survey.kinds if survey else {}
    materials = {floor.id: floor.description for floor in survey.floors} if survey else {}
    labels, ids = assign_floor(floor_pixels, masks if survey else {}, kinds)
    seen = points_world[floor_pixels][:, :2]
    outline, bounded_by = room_polygon(walls, eye, seen)
    headings = [wall.yaw_deg % 90 for wall in sorted(walls, key=lambda w: -w.size[0])]
    angle = headings[0] if headings else 0.0

    regions = []
    for index, oid in enumerate(ids):
        region = labels == index
        if index and region.sum() < 50:
            labels[region] = 0
            continue
        points = points_world[region][:, :2]
        if index:
            corners, heading = covering_rectangle(points, headings)
            area = _polygon_area(corners)
            if area < _MIN_COVERING_M2:
                print(f"{oid}: only {area:.2f} m² seen, taken as the floor", flush=True)
                labels[region] = 0
                continue
            polygon, top = corners, COVERING_M
        else:
            polygon, top, heading = outline, 0.0, angle
        regions.append(FloorRegion(
            id=oid, material=materials.get(oid, "floor"), kind="floor" if not index else "covering",
            polygon=[tuple(map(float, corner)) for corner in polygon], top_m=top, angle_deg=heading,
            color=tuple(float(c) for c in (np.median(photo[region], axis=0) / 255) ** 2.2) if region.sum() >= 50 else (0.2, 0.2, 0.2),
        ))
    # Coverings dropped above are the floor's now: give it its colour again from all of its pixels.
    if regions and (labels == 0).sum() >= 50:
        regions[0].color = tuple(float(c) for c in (np.median(photo[labels == 0], axis=0) / 255) ** 2.2)
    print(f"floor: {', '.join(r.id for r in regions)}; bounded by {', '.join(bounded_by) or 'no wall'}", flush=True)

    if survey is None:
        return regions
    saved = run_dir / "floors"
    saved.mkdir(exist_ok=True)
    for region in regions:
        index = ids.index(region.id)
        texture, meta = saved / f"{region.id}.png", saved / f"{region.id}.json"
        # A floor out of view (placed from standard tops) has no pixels to draw from: it keeps one colour.
        if not texture.exists() and draw is not None and (labels == index).any():
            # Drawn with its grain running across, if it has one, turned square to the room if nearly so; the
            # grain is read where the photo sees the floor nearest, as the far floor is blurred along the view.
            colours, clean, distance, frame = top_down(image, points_world, camera, labels == index, region.angle_deg)
            if not clean.any():
                continue
            lines, coherence = grain(colours, clean & (distance <= np.median(distance[clean])))
            grained = bool(_GRAINED.search(region.material))
            heading = region.angle_deg
            if grained:
                heading = (region.angle_deg + lines) % 180
                for wall_heading in headings:
                    turn = (wall_heading - heading + 45) % 90 - 45
                    if abs(turn) <= _SNAP_DEG:
                        heading = (heading + turn) % 180
                        break
                colours, clean, distance, frame = top_down(image, points_world, camera, labels == index, heading)
            cell = float(np.linalg.norm(frame[1]))
            square = clean_square(clean, distance, cell)
            if square is None:
                continue
            row, col, side = square
            crop = Image.fromarray(colours[row : row + side, col : col + side]).resize((1024, 1024), Image.Resampling.LANCZOS)
            crop.save(saved / f"{region.id}_source.jpg", quality=92)
            drawn = np.asarray(draw(crop, region.material, region.id, grained).resize((1024, 1024), Image.Resampling.LANCZOS))
            Image.fromarray(make_tileable(even_light(drawn))).save(texture)
            meta.write_text(json.dumps({"tile_m": side * cell, "angle_deg": heading, "grain_coherence": round(coherence, 3)}))
            print(
                f"{region.id}: texture drawn from a {side * cell:.2f} m square"
                + (f", grain at {heading:.0f}° (coherence {coherence:.2f})" if grained else ", no grain"),
                flush=True,
            )
        if not texture.exists():
            continue
        tile = json.loads(meta.read_text())
        coloured = match_colour(np.asarray(Image.open(texture).convert("RGB")), np.array(region.color))
        name = f"floor_{region.id}.png"
        Image.fromarray(coloured).save(run_dir / "assets" / name)
        region.texture, region.tile_m, region.angle_deg = name, float(tile["tile_m"]), float(tile["angle_deg"])
    return regions


def _polygon_area(polygon: np.ndarray) -> float:
    x, y = polygon[:, 0], polygon[:, 1]
    return float(abs(x @ np.roll(y, -1) - y @ np.roll(x, -1)) / 2)
