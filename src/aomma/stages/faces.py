"""A built-in's mask from its faces, one at a time.

SAM asked for a built-in as a whole often takes one face of it: the counter's top and not the white doors under it
(KitchenScan 007's `base_cabinets`, 6% of its box), and text prompts for it do no better. So a built-in whose mask
covers less than half its box (`segment.MIN_FILL`) is looked at again: the VLM, shown a close-up, lists the faces of
that built-in the photo shows (its top, front, sides, doors and drawer fronts), each with a box, and never what stands
on it or the built-ins beside it. Each face is segmented from its own box and kept within it, and the built-in's mask
becomes its old mask and its faces together, if that covers more of its box.
"""

import json
import re

import numpy as np
from PIL import Image
from pydantic import BaseModel

from aomma.backends.vlm import VLM
from aomma.ledger import Ledger
from aomma.stages.image import to_jpeg
from aomma.stages.inventory import Inventory, InventoryObject

FACES_PROMPT = """\
This is a close-up from a photo that is being rebuilt as a 3D scene. In it is part of the room's fabric, catalogued
as "{description}". A segmenter asked for it as a whole found only part of it, so it is to be found face by face.

List each face of this one built-in thing that the close-up shows: its top, its front, its sides or end panels, and,
where it has them, its doors and drawer fronts (as one face per run of them, or each, as you see them best). Give a
tight box round each face as it shows, partly hidden faces included.

Do not list anything standing on it, hanging from it or stored on it (appliances, dishes, food, decorations), nor the
walls, the floor or any other built-in beside it. If it shows only one face, list that one.

Reply with JSON only, in the form {{"faces": [...]}}. Each entry has:
- "id": unique snake_case name, such as "top" or "front_doors"
- "bbox_2d": [x1, y1, x2, y2] bounding box normalised to 0-1000 within this close-up (x across the width, y down the height)
- "description": at most 12 words: which face it is, its colour and material
"""

SCHEMA = {
    "type": "object",
    "properties": {
        "faces": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "bbox_2d": {"type": "array", "items": {"type": "number"}},
                    "description": {"type": "string"},
                },
                "required": ["id", "bbox_2d", "description"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["faces"],
    "additionalProperties": False,
}

# How much of the photo around the built-in's box the VLM is shown, as a multiple of the box (as `split_entry`'s).
_VIEW = 1.5


class Face(BaseModel):
    id: str
    # Normalised 0-1000 in the whole photo.
    bbox_2d: tuple[float, float, float, float]
    description: str


def parse_faces(text: str) -> list[dict]:
    """The faces in a model reply, tolerating code fences and a bare list."""
    match = re.search(r"[\[{].*[\]}]", text, re.DOTALL)
    if not match:
        return []
    data = json.loads(match.group())
    faces = data if isinstance(data, list) else data.get("faces", [])
    return [f for f in faces if isinstance(f, dict) and len(f.get("bbox_2d", [])) == 4]


def view_of(image: Image.Image, bbox_2d: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    """The close-up shown for a box (pixels): the box grown to `_VIEW` times its size, within the photo."""
    x1, y1, x2, y2 = (v * side / 1000 for v, side in zip(bbox_2d, (image.width, image.height) * 2))
    grow_x, grow_y = (x2 - x1) * (_VIEW - 1) / 2, (y2 - y1) * (_VIEW - 1) / 2
    return max(x1 - grow_x, 0), max(y1 - grow_y, 0), min(x2 + grow_x, image.width), min(y2 + grow_y, image.height)


def to_photo(view: tuple[float, float, float, float], bbox_2d, size: tuple[int, int]) -> tuple[float, float, float, float]:
    """A box normalised within the close-up `view`, normalised within the whole photo of `size`."""
    ax, ay, bx, by = (min(max(float(v), 0.0), 1000.0) for v in bbox_2d)
    return (
        (view[0] + ax / 1000 * (view[2] - view[0])) * 1000 / size[0],
        (view[1] + ay / 1000 * (view[3] - view[1])) * 1000 / size[1],
        (view[0] + bx / 1000 * (view[2] - view[0])) * 1000 / size[0],
        (view[1] + by / 1000 * (view[3] - view[1])) * 1000 / size[1],
    )


def list_faces(image: Image.Image, entry: InventoryObject, vlm: VLM, ledger: Ledger) -> tuple[list[Face], str]:
    """The faces of the built-in `entry` the VLM finds in a close-up, their boxes in the whole photo; and its reply."""
    view = view_of(image, entry.bbox_2d)
    ledger.check_budget()
    text, usage = vlm.ask(to_jpeg(image.crop(view)), FACES_PROMPT.format(description=entry.description), SCHEMA,
                          label=f"faces of {entry.id}")
    ledger.record("faces", usage)
    faces, taken = [], set()
    for found in parse_faces(text):
        base = re.sub(r"\W+", "_", str(found.get("id", "face")).lower()).strip("_") or "face"
        face_id, n = base, 2
        while face_id in taken:
            face_id, n = f"{base}_{n}", n + 1
        taken.add(face_id)
        box = to_photo(view, found["bbox_2d"], image.size)
        if box[2] > box[0] and box[3] > box[1]:
            faces.append(Face(id=face_id, bbox_2d=box, description=str(found.get("description", ""))))
    return faces, text


def as_inventory(entry: InventoryObject, faces: list[Face]) -> Inventory:
    """The faces as entries to segment, each named `<built-in>__<face>`."""
    return Inventory(objects=[
        InventoryObject(id=f"{entry.id}__{face.id}", bbox_2d=face.bbox_2d, description=face.description,
                        size_m=entry.size_m, on=entry.on)
        for face in faces
    ])


def within_box(mask: np.ndarray, bbox_2d: tuple[float, float, float, float]) -> np.ndarray:
    """The mask's pixels inside its box (normalised 0-1000): a face's mask is kept to the face it was asked for."""
    height, width = mask.shape
    x1, y1, x2, y2 = bbox_2d
    inside = np.zeros_like(mask)
    inside[int(np.floor(y1 * height / 1000)):int(np.ceil(y2 * height / 1000)),
           int(np.floor(x1 * width / 1000)):int(np.ceil(x2 * width / 1000))] = True
    return mask & inside


def from_faces(mask: np.ndarray, faces: Inventory, face_masks: dict[str, np.ndarray]) -> np.ndarray:
    """The built-in's old `mask` and each face's mask within the face's own box, together."""
    together = mask.copy()
    for face in faces.objects:
        if face.id in face_masks:
            together |= within_box(face_masks[face.id], face.bbox_2d)
    return together

