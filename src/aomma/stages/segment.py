"""Segmentation stage: a pixel mask per inventory object, from SAM 3 in ComfyUI."""

import colorsys
import io
import json
import re
from typing import Literal

import numpy as np
from PIL import Image, ImageDraw
from pydantic import BaseModel

from aomma.backends.comfy import ComfyClient, ComfyError
from aomma.stages.inventory import Inventory

SAM_CHECKPOINT = "sam3.1_multiplex_fp16.safetensors"
# Below this agreement with the VLM's box, a box-prompted mask is retried with text.
MIN_BOX_IOU = 0.7
# Below this, even after the retries, the mask is not of the object at all: the garage's right slipper
# got scraps spread over the treadmill and the dryer (0.008), modelled as a 1 m blob in front of them.
# Of 226 masks in the early runs, the next lowest was 0.235.
MIN_USABLE_BOX_IOU = 0.15
# An object's box should be mostly covered by its mask and the masks of things in
# front of it. Below half, more of the box is unexplained than explained: it holds
# something the masks have missed, several things listed as one or a mask that caught
# only a part. Chosen on three photos, where any value from 0.4 to 0.6 caught the same
# failures; to be checked again against more.
MIN_FILL = 0.5

Box = tuple[float, float, float, float]  # x1, y1, x2, y2 in pixels


class MaskInfo(BaseModel):
    # "faces": a built-in's mask with its faces' (`stages.faces`).
    prompt: Literal["box", "text", "description", "faces"]
    # How well the mask's bounding box matches the VLM's box, 0-1. A low value
    # means the two models disagree about where the object is.
    box_iou: float
    area_px: int
    # Share of the VLM's box covered by this mask and the masks of smaller objects, 0-1
    # (see `box_fills`). None in runs made before this was measured.
    fill: float | None = None


def segment(
    comfy: ComfyClient, image_name: str, size: tuple[int, int], inventory: Inventory
) -> tuple[dict[str, np.ndarray], dict[str, MaskInfo]]:
    """Return a boolean (H, W) mask and its quality info for every object."""
    width, height = size
    boxes: dict[str, Box] = {
        obj.id: (
            obj.bbox_2d[0] * width / 1000,
            obj.bbox_2d[1] * height / 1000,
            obj.bbox_2d[2] * width / 1000,
            obj.bbox_2d[3] * height / 1000,
        )
        for obj in inventory.objects
    }

    # Box prompts are precise for most objects but sometimes grab only a part
    # (the picture inside a frame) or nothing; a short text prompt recovers those.
    # If the name misleads (asking for "hummingbird art" finds the bird, not the
    # framed picture), the VLM's fuller description is the last resort.
    texts = {
        "text": {obj.id: re.sub(r"[_\d]+", " ", obj.id).strip() for obj in inventory.objects},
        "description": {obj.id: obj.description for obj in inventory.objects},
    }
    def quality(oid: str, mask: np.ndarray, info: MaskInfo) -> float:
        """A good mask has the box's outline and covers what is in it."""
        return info.box_iou * _covered(boxes[oid], mask)

    masks, infos = _run(comfy, image_name, size, boxes, "box", list(boxes), {})
    for prompt in ("text", "description"):
        retry = [
            oid for oid, info in infos.items()
            if info.box_iou < MIN_BOX_IOU or _covered(boxes[oid], masks[oid]) < MIN_FILL
        ]
        if not retry:
            break
        retry_masks, retry_infos = _run(comfy, image_name, size, boxes, prompt, retry, texts[prompt])
        for oid in retry:
            if quality(oid, retry_masks[oid], retry_infos[oid]) > quality(oid, masks[oid], infos[oid]):
                masks[oid], infos[oid] = retry_masks[oid], retry_infos[oid]
    return masks, infos


def describe_mask(bbox_2d: tuple[float, float, float, float], mask: np.ndarray, prompt: str) -> MaskInfo:
    """Quality info for a mask put together outside `segment`, against a box normalised to 0-1000."""
    height, width = mask.shape
    box = (bbox_2d[0] * width / 1000, bbox_2d[1] * height / 1000, bbox_2d[2] * width / 1000, bbox_2d[3] * height / 1000)
    return MaskInfo(prompt=prompt, box_iou=round(_box_iou(box, mask), 3), area_px=int(mask.sum()))


def box_fills(inventory: Inventory, masks: dict[str, np.ndarray]) -> dict[str, float]:
    """For each object, the share of its box covered by its own mask and the masks of smaller objects.

    Smaller objects count because they are what sits on or in front of a
    bigger one: a shelf's box is mostly books. What is left uncovered should
    be little, the gaps round an irregular shape. A box left mostly uncovered
    means the masks have missed something in it.
    """
    area = {oid: int(mask.sum()) for oid, mask in masks.items()}
    fills = {}
    for obj in inventory.objects:
        own = masks[obj.id]
        height, width = own.shape
        box = (obj.bbox_2d[0] * width / 1000, obj.bbox_2d[1] * height / 1000, obj.bbox_2d[2] * width / 1000, obj.bbox_2d[3] * height / 1000)
        covering = own.copy()
        for oid, mask in masks.items():
            if oid != obj.id and area[oid] < area[obj.id]:
                covering |= mask
        fills[obj.id] = round(_covered(box, covering), 3)
    return fills


def judge_second_look(
    inventory: Inventory,
    masks: dict[str, np.ndarray],
    object_id: str,
    relisted: Inventory,
    part_masks: dict[str, np.ndarray],
) -> tuple[str, np.ndarray | None]:
    """Decide what to do with a suspect entry after its contents were listed and segmented again.

    `relisted` is the inventory with the entry replaced by its parts, and
    `part_masks` their masks. Returns one of:
    - ("split", None): several parts, each mask filling its own box. Take the parts.
    - ("redrawn", None): one part that fills its box better than before. Take it.
    - ("grouped", mask): the parts do not separate cleanly, but together with
      the old mask they cover more of the entry's box. Keep the entry with this mask.
    - ("kept", None): nothing covers the box better. Change nothing.
    """
    before = box_fills(inventory, masks)[object_id]
    others = {oid: mask for oid, mask in masks.items() if oid != object_id}
    fills = box_fills(relisted, others | part_masks)
    if all(fills[part] >= MIN_FILL for part in part_masks) and (len(part_masks) > 1 or fills[next(iter(part_masks))] > before):
        return ("split" if len(part_masks) > 1 else "redrawn"), None
    together = masks[object_id] | np.any(list(part_masks.values()), axis=0)
    if box_fills(inventory, others | {object_id: together})[object_id] > before:
        return "grouped", together
    return "kept", None


def _covered(box: Box, mask: np.ndarray) -> float:
    """Share of `box` that `mask` covers."""
    x1, y1, x2, y2 = (int(round(v)) for v in box)
    x1, y1 = max(x1, 0), max(y1, 0)
    inside = mask[y1:y2, x1:x2]
    return float(inside.mean()) if inside.size else 0.0


def segment_floor(comfy: ComfyClient, image_name: str, size: tuple[int, int]) -> np.ndarray:
    """Return a boolean (H, W) mask of the visible floor (may be empty)."""
    return segment_surface(comfy, image_name, size, "floor")


def segment_surface(comfy: ComfyClient, image_name: str, size: tuple[int, int], text: str) -> np.ndarray:
    """Return a boolean (H, W) mask of every part of the room `text` names, such as "floor" or "wall" (may be empty)."""
    outputs = comfy.run(
        {
            "image": {"class_type": "LoadImage", "inputs": {"image": image_name}},
            "sam": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": SAM_CHECKPOINT}},
            "text": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["sam", 1], "text": text}},
            "detect": {
                "class_type": "SAM3_Detect",
                "inputs": {
                    "model": ["sam", 0],
                    "image": ["image", 0],
                    "conditioning": ["text", 0],
                    "threshold": 0.5,
                    "refine_iterations": 2,
                    "individual_masks": False,
                },
            },
            "mask": {"class_type": "MaskToImage", "inputs": {"mask": ["detect", 0]}},
            "out": {"class_type": "PreviewImage", "inputs": {"images": ["mask", 0]}},
        },
        label=f"{text} mask {image_name}",
    )
    width, height = size
    surface = np.zeros((height, width), dtype=bool)
    for file in outputs.get("out", {}).get("images", []):
        surface |= np.asarray(Image.open(io.BytesIO(comfy.fetch(file))).convert("L")) > 127
    return surface


def _run(
    comfy: ComfyClient,
    image_name: str,
    size: tuple[int, int],
    boxes: dict[str, Box],
    prompt: Literal["box", "text", "description"],
    ids: list[str],
    texts: dict[str, str],
) -> tuple[dict[str, np.ndarray], dict[str, MaskInfo]]:
    """Segment `ids` with one kind of prompt: each object's box, or its entry in `texts`.

    An object for which nothing is found gets an empty mask. If the job fails
    anyway, the objects are tried one at a time, so one object cannot take the
    others' masks (and the run) down with it.
    """
    width, height = size
    workflow: dict = {
        "image": {"class_type": "LoadImage", "inputs": {"image": image_name}},
        "sam": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": SAM_CHECKPOINT}},
        # Leads every object's batch of masks, so that a batch is never empty:
        # saving an empty batch fails the whole job.
        "blank": {"class_type": "EmptyImage", "inputs": {"width": width, "height": height, "batch_size": 1, "color": 0}},
    }
    for oid in ids:
        detect = {
            "model": ["sam", 0],
            "image": ["image", 0],
            "threshold": 0.5,
            "refine_iterations": 2,
            "individual_masks": True,
        }
        if prompt == "box":
            x1, y1, x2, y2 = boxes[oid]
            workflow[f"box_{oid}"] = {
                "class_type": "PrimitiveBoundingBox",
                "inputs": {
                    "x": round(x1),
                    "y": round(y1),
                    "width": max(1, round(x2 - x1)),
                    "height": max(1, round(y2 - y1)),
                },
            }
            detect["bboxes"] = [f"box_{oid}", 0]
        else:
            workflow[f"text_{oid}"] = {
                "class_type": "CLIPTextEncode",
                "inputs": {"clip": ["sam", 1], "text": texts[oid]},
            }
            detect["conditioning"] = [f"text_{oid}", 0]
        workflow[f"detect_{oid}"] = {"class_type": "SAM3_Detect", "inputs": detect}
        workflow[f"mask_{oid}"] = {
            "class_type": "MaskToImage",
            "inputs": {"mask": [f"detect_{oid}", 0]},
        }
        workflow[f"batch_{oid}"] = {
            "class_type": "ImageBatch",
            "inputs": {"image1": ["blank", 0], "image2": [f"mask_{oid}", 0]},
        }
        # PreviewImage writes to ComfyUI's temp folder, so nothing piles up there.
        workflow[f"out_{oid}"] = {
            "class_type": "PreviewImage",
            "inputs": {"images": [f"batch_{oid}", 0]},
        }
    try:
        outputs = comfy.run(workflow, label=f"{len(ids)} masks by {prompt} {image_name}")
    except ComfyError as error:
        if len(ids) == 1:
            print(f"{ids[0]}: no mask by {prompt} ({error})", flush=True)
            return {ids[0]: np.zeros((height, width), dtype=bool)}, {ids[0]: MaskInfo(prompt=prompt, box_iou=0.0, area_px=0)}
        print(f"{len(ids)} masks by {prompt} failed; trying them one at a time", flush=True)
        masks, infos = {}, {}
        for oid in ids:
            one_mask, one_info = _run(comfy, image_name, size, boxes, prompt, [oid], texts)
            masks.update(one_mask)
            infos.update(one_info)
        return masks, infos

    masks, infos = {}, {}
    for oid in ids:
        best, best_iou = np.zeros((height, width), dtype=bool), 0.0
        # A text prompt can match several instances; keep the one at the VLM's box.
        # The first image is the blank that leads the batch.
        for file in outputs.get(f"out_{oid}", {}).get("images", [])[1:]:
            candidate = np.asarray(Image.open(io.BytesIO(comfy.fetch(file))).convert("L")) > 127
            iou = _box_iou(boxes[oid], candidate)
            if iou > best_iou:
                best, best_iou = candidate, iou
        masks[oid] = best
        infos[oid] = MaskInfo(prompt=prompt, box_iou=round(best_iou, 3), area_px=int(best.sum()))
    return masks, infos


def segment_points(
    comfy: ComfyClient, image_name: str, size: tuple[int, int], points: dict[str, list[tuple[int, int]]], boxes: dict[str, Box]
) -> dict[str, np.ndarray]:
    """A mask for each id with points, from positive point prompts alone (pixels, x across, y down): of SAM's
    candidates, the one holding most of the points, then nearest the id's box. Ids with no points are left out."""
    width, height = size
    asked = [oid for oid, pts in points.items() if pts]
    if not asked:
        return {}
    workflow: dict = {
        "image": {"class_type": "LoadImage", "inputs": {"image": image_name}},
        "sam": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": SAM_CHECKPOINT}},
        "blank": {"class_type": "EmptyImage", "inputs": {"width": width, "height": height, "batch_size": 1, "color": 0}},
    }
    for oid in asked:
        workflow[f"pos_{oid}"] = {"class_type": "PrimitiveString",
                                  "inputs": {"value": json.dumps([{"x": x, "y": y} for x, y in points[oid]])}}
        workflow[f"detect_{oid}"] = {"class_type": "SAM3_Detect", "inputs": {
            "model": ["sam", 0], "image": ["image", 0], "threshold": 0.5, "refine_iterations": 2,
            "individual_masks": True, "positive_coords": [f"pos_{oid}", 0]}}
        workflow[f"mask_{oid}"] = {"class_type": "MaskToImage", "inputs": {"mask": [f"detect_{oid}", 0]}}
        workflow[f"batch_{oid}"] = {"class_type": "ImageBatch", "inputs": {"image1": ["blank", 0], "image2": [f"mask_{oid}", 0]}}
        workflow[f"out_{oid}"] = {"class_type": "PreviewImage", "inputs": {"images": [f"batch_{oid}", 0]}}
    outputs = comfy.run(workflow, label=f"{len(asked)} masks by points {image_name}")
    masks = {}
    for oid in asked:
        best, best_score = np.zeros((height, width), dtype=bool), -1.0
        for file in outputs.get(f"out_{oid}", {}).get("images", [])[1:]:
            candidate = np.asarray(Image.open(io.BytesIO(comfy.fetch(file))).convert("L")) > 127
            held = float(np.mean([candidate[min(y, height - 1), min(x, width - 1)] for x, y in points[oid]]))
            score = held + 0.01 * _box_iou(boxes[oid], candidate)
            if score > best_score:
                best, best_score = candidate, score
        masks[oid] = best
    return masks


def _box_iou(box: Box, mask: np.ndarray) -> float:
    """Intersection-over-union of `box` with the bounding box of `mask`."""
    if not mask.any():
        return 0.0
    rows, cols = np.nonzero(mask)
    m = (cols.min(), rows.min(), cols.max() + 1, rows.max() + 1)
    iw = max(0.0, min(box[2], m[2]) - max(box[0], m[0]))
    ih = max(0.0, min(box[3], m[3]) - max(box[1], m[1]))
    union = (box[2] - box[0]) * (box[3] - box[1]) + (m[2] - m[0]) * (m[3] - m[1]) - iw * ih
    return float(iw * ih / union)


def draw_masks(image: Image.Image, masks: dict[str, np.ndarray]) -> Image.Image:
    """Tint each object's mask a different colour on a copy of the image."""
    pixels = np.asarray(image).astype(np.float32)
    # Largest first, so small objects sitting on big ones stay visible.
    ordered = sorted(masks.items(), key=lambda item: -int(item[1].sum()))
    colors = {}
    for i, (oid, mask) in enumerate(ordered):
        rgb = colorsys.hsv_to_rgb((i * 0.618) % 1.0, 0.9, 1.0)
        colors[oid] = tuple(int(c * 255) for c in rgb)
        pixels[mask] = pixels[mask] * 0.45 + np.array(colors[oid]) * 0.55
    out = Image.fromarray(pixels.astype(np.uint8))
    draw = ImageDraw.Draw(out)
    for oid, mask in ordered:
        if mask.any():
            rows, cols = np.nonzero(mask)
            draw.text((cols.min() + 2, rows.min() + 2), oid, fill="white", stroke_width=2, stroke_fill="black")
    return out
