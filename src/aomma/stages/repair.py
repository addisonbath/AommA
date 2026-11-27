"""Occlusion repair: redraw a partly hidden object as a complete one before modelling it.

An image-to-3D model shown half a nightstand builds half a nightstand. An
image-editing model can first redraw the object whole, on a plain background,
from a crop that still shows its surroundings. The result is a plausible
completion, not the truth: hidden parts are invented, and the object may come
back at a different size or position in the frame.
"""

import io
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

from aomma.backends.comfy import ComfyClient
from aomma.stages.image import to_jpeg

# Objects whose outline is cut by occluders for at least this share are repaired.
REPAIR_THRESHOLD = 0.25
_CONTEXT_PADDING = 1.3
# How far past where the object can be a redrawn one may reach, as a share of the
# mask's longer side: the redraw can come back a little shifted or larger.
_REACH = 0.03
# Mean brightness (0-1) of a redrawn object above which it is drawn again on black.
_LIGHT = 0.75


def context_crop(source: Image.Image, mask: np.ndarray, size: int = 1024) -> tuple[Image.Image, int]:
    """A square crop around the object that keeps its surroundings visible.

    Returns the crop and the side of the square it covers in `source`, in pixels.
    """
    left, top, right, bottom = Image.fromarray(mask).resize(source.size).getbbox()
    side = round(max(right - left, bottom - top) * _CONTEXT_PADDING)
    x, y = (left + right - side) // 2, (top + bottom - side) // 2
    # `crop` pads with black where the square hangs over the photo's edge.
    return source.crop((x, y, x + side, y + side)).resize((size, size), Image.Resampling.LANCZOS), side


def repair_object(
    comfy: ComfyClient,
    workflow_path: Path,
    crop: Image.Image,
    description: str,
    holds_things: bool,
    name: str,
    seed: int = 1,
    backdrop: str = "white",
    to_floor: bool = False,
) -> tuple[Image.Image, np.ndarray]:
    """Redraw the object in `crop` complete and alone; returns the picture and its mask.

    `to_floor` says the photo's edge cuts off its lower part (the crop is black past it), which it is drawn down to.

    `holds_things` says the object is hidden by things standing on or in it
    (a shelf by its books), which are then cleared away. `backdrop` is the
    colour it is drawn on (see `is_light`).

    The workflow is a ComfyUI API-format graph with nodes named "input_image"
    (LoadImage), "input_prompt" (text), "output_image" and "output_mask".
    """
    workflow = json.loads(workflow_path.read_text())
    workflow["input_image"]["inputs"]["image"] = comfy.upload_image(to_jpeg(crop), f"aomma_repair_{name}.jpg")
    # Short and blunt works best, and naming the things to remove backfires: the
    # model keeps half of them. Told a shelf is empty, it clears it. The last
    # sentence is there because it also embellishes (a bed gains a headboard).
    subject = f"Empty {description[0].lower()}{description[1:]}" if holds_things else f"{description}, alone"
    workflow["input_prompt"]["inputs"]["text"] = (
        f"{subject}{', whole, down to the floor it stands on' if to_floor else ''}. "
        f"Nothing on top of it, nothing inside it, nothing underneath it, nothing in front of it. "
        f"Same object, same position, size and camera angle, with nothing added. Plain {backdrop} background."
    )
    workflow["noise"]["inputs"]["noise_seed"] = seed
    outputs = comfy.run(workflow, label=f"repair {name}")
    picture = Image.open(io.BytesIO(comfy.fetch(outputs["output_image"]["images"][0]))).convert("RGB")
    mask = np.asarray(Image.open(io.BytesIO(comfy.fetch(outputs["output_mask"]["images"][0]))).convert("L")) > 127
    return picture, mask


# The colour the holes are painted in a cut-out sent to `fill_contents`: plain black would read as background.
_HOLE_MARK = (255, 0, 255)


def blobs(area: np.ndarray, grow: int) -> np.ndarray:
    """`area` as plain shapes: each connected part replaced by its convex hull, grown by `grow` pixels.

    Asked to fill a hole shaped like what stood there (a tray, a cup), a fill model draws that thing back; a plain
    blob over it gives the shape away no more than a flat strip does."""
    from scipy.spatial import ConvexHull, QhullError
    from PIL import ImageDraw

    labels, count = ndimage.label(area)
    canvas = Image.new("L", (area.shape[1], area.shape[0]))
    draw = ImageDraw.Draw(canvas)
    for part in range(1, count + 1):
        ys, xs = np.nonzero(labels == part)
        points = np.stack([xs, ys], axis=1).astype(float)
        try:
            hull = points[ConvexHull(points).vertices]
        except (QhullError, ValueError):  # a part too thin or small to have a hull
            hull = points
        if len(hull) >= 3:
            draw.polygon([tuple(p) for p in hull], fill=255)
        else:
            draw.point([tuple(p) for p in hull], fill=255)
    shapes = np.asarray(canvas) > 0
    return ndimage.binary_dilation(shapes, iterations=grow) if grow else shapes


def fill_area(
    comfy: ComfyClient, workflow_path: Path, picture: Image.Image, area: np.ndarray, prompt: str, name: str, seed: int = 1,
    blank: int = 128,
    feather: int = 0,
) -> tuple[Image.Image, np.ndarray]:
    """`picture` with only `area` drawn again by a fill model (FLUX.1 Fill) to `prompt`, everything else kept as it
    is; returns the picture and the matting's mask of what stands out in it.

    The editor `repair_object` uses draws the object anew, and draws another (Coffee's back chair) or moves it
    (Coffee's table, notched by the cut that follows). A fill model draws
    nothing outside `area`, so what the photo sees of the object stays where it is."""
    # What was there is removed, not just masked: the model is given nothing of it to copy back.
    cleared = np.asarray(picture.convert("RGB")).copy()
    cleared[area] = blank
    workflow = json.loads(workflow_path.read_text())
    lossless = io.BytesIO()  # PNG, so no JPEG ringing carries the old edges into the blank
    if "input_mask" in workflow:
        Image.fromarray(cleared).save(lossless, format="PNG")
        workflow["input_image"]["inputs"]["image"] = comfy.upload_image(lossless.getvalue(), f"aomma_fill_{name}.png")
        buffer = io.BytesIO()
        Image.fromarray(area).convert("RGB").save(buffer, format="PNG")
        workflow["input_mask"]["inputs"]["image"] = comfy.upload_image(buffer.getvalue(), f"aomma_fill_{name}_mask.png")
    else:
        # As Comfy's own Flux.1 Fill template: one picture, the area to draw left transparent; LoadImage's mask
        # output is its transparent pixels, so no mask picture is sent.
        # `feather` px: the mask fades out over a band of the photo round the area, not over its blanked pixels,
        # so the fill blends into real pixels and the blank does not bleed into a seam at the edge.
        if feather:
            alpha = (np.clip(ndimage.distance_transform_edt(~area) / feather, 0, 1) * 255).astype(np.uint8)
        else:
            alpha = np.where(area, 0, 255).astype(np.uint8)
        Image.fromarray(np.dstack([cleared, alpha]), "RGBA").save(lossless, format="PNG")
        workflow["input_image"]["inputs"]["image"] = comfy.upload_image(lossless.getvalue(), f"aomma_fill_{name}.png")
    workflow["input_prompt"]["inputs"]["text"] = prompt
    workflow["sampler"]["inputs"]["seed"] = seed
    outputs = comfy.run(workflow, label=f"fill {name}")
    out = Image.open(io.BytesIO(comfy.fetch(outputs["output_image"]["images"][0]))).convert("RGB").resize(picture.size)
    mask = np.asarray(Image.open(io.BytesIO(comfy.fetch(outputs["output_mask"]["images"][0]))).convert("L").resize(picture.size)) > 127
    return out, mask


def fill_contents(
    comfy: ComfyClient,
    workflow_path: Path,
    cutout: Image.Image,
    holes: np.ndarray,
    outline: np.ndarray,
    description: str,
    name: str,
    seed: int = 1,
) -> tuple[Image.Image, np.ndarray]:
    """Fill the holes the things inside an object leave in its cut-out; returns the picture and its mask.

    A shelf cut out of the photo has holes where its books were, and the
    generator builds them as holes in its back and shelves. Redrawn whole from
    the photo instead (`repair_object`), it comes back as a shelf, not this one.
    Here the editor is shown the cut-out with the holes painted in a colour
    of their own and asked to fill only those, so what shows of the object
    is kept. `holes` and `outline` (the mask with its holes filled) are in the
    cut-out's frame; the picture's mask is kept within the outline (and `_REACH`).
    The low bookshelf's filled model fit 0.79, the best of four ways tried.
    """
    marked = np.asarray(cutout.convert("RGB")).copy()
    marked[holes] = _HOLE_MARK
    workflow = json.loads(workflow_path.read_text())
    workflow["input_image"]["inputs"]["image"] = comfy.upload_image(to_jpeg(Image.fromarray(marked)), f"aomma_fill_{name}.jpg")
    workflow["input_prompt"]["inputs"]["text"] = (
        f"{description}. The magenta areas are where things were taken out of it. Fill them with the object's own "
        "empty inside: its back panel, shelves and sides, in the same material and light. Make its top edge "
        "straight where things stood on it. Nothing on it or inside it. "
        "Change nothing else: same object, same position, size and camera angle. Plain black background."
    )
    workflow["noise"]["inputs"]["noise_seed"] = seed
    outputs = comfy.run(workflow, label=f"fill {name}")
    picture = Image.open(io.BytesIO(comfy.fetch(outputs["output_image"]["images"][0]))).convert("RGB").resize(cutout.size)
    mask = np.asarray(Image.open(io.BytesIO(comfy.fetch(outputs["output_mask"]["images"][0]))).convert("L").resize(cutout.size)) > 127
    reach = ndimage.distance_transform_edt(~outline) <= _REACH * cutout.width
    return picture, mask & reach


def is_light(picture: Image.Image, area: np.ndarray) -> bool:
    """Whether the redrawn object in `area` of `picture` is light.

    The matting after the redraw cannot tell a light object from a white
    background: a white marble countertop came back as little more than its
    edge. Such an object is drawn again on black. The photo's own brightness
    does not tell, as the editor relights what it draws: that countertop, in
    shade in the photo, is no brighter there than the room around it.
    """
    return float((np.asarray(picture.convert("L")) / 255)[area].mean()) > _LIGHT


def where_it_can_be(mask: np.ndarray, occluders: list[np.ndarray]) -> np.ndarray:
    """Where in the photo a redrawn object may lie: where it is seen or behind what hides it.

    Told to draw an object alone, the editor keeps what it is joined to (a
    countertop comes back with the cabinets under it and the stool in front),
    and the matting takes everything that is not background; this is what to
    keep of it. The inventory's box is no guide here: it can be drawn round
    only what is seen and would cut off the hidden parts.
    """
    region = np.logical_or.reduce([mask, *occluders])
    return ndimage.distance_transform_edt(~region) <= _REACH * max(mask.shape)


def extend_below(source: Image.Image, mask: np.ndarray, rows: int, outline_px: np.ndarray) -> tuple[Image.Image, np.ndarray, np.ndarray]:
    """The photo and the mask carried `rows` mask rows past the photo's bottom edge, black there, so a redraw can draw
    what the edge cuts off; and the part of that strip where the item can be: inside `outline_px`, its box's corners
    in the mask's frame (x, y). Returns the taller photo, the taller mask and that area (the taller mask's shape)."""
    from PIL import ImageDraw

    height, width = mask.shape
    source_rows = round(rows * source.height / height)
    taller = Image.new("RGB", (source.width, source.height + source_rows))
    taller.paste(source.convert("RGB"), (0, 0))
    grown = np.zeros((height + rows, width), dtype=bool)
    grown[:height] = mask
    canvas = Image.new("L", (width, height + rows))
    ImageDraw.Draw(canvas).polygon([tuple(c) for c in _convex(outline_px)], fill=255)
    area = np.asarray(canvas) > 0
    area[:height] = False
    return taller, grown, area


def _convex(points: np.ndarray) -> np.ndarray:
    """The convex hull of 2D points, in order round it."""
    from scipy.spatial import ConvexHull

    points = np.asarray(points, dtype=float)
    return points[ConvexHull(points).vertices]


def in_crop(area: np.ndarray, mask: np.ndarray, size: int = 1024) -> np.ndarray:
    """`area`, a boolean array in the photo's frame, in the frame of `context_crop(..., mask, size)`."""
    square, _ = context_crop(Image.fromarray(area).convert("L"), mask, size)
    return np.asarray(square) > 127


def crop_fov_deg(side_px: float, focal_px: float) -> float:
    """Horizontal field of view of a square crop `side_px` wide, given the focal length in pixels."""
    return math.degrees(2 * math.atan(side_px / (2 * focal_px)))
