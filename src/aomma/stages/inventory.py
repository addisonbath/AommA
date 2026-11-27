"""Inventory stage: ask a VLM what objects are in the photo and where."""

import json
import re

from PIL import Image, ImageDraw
from pydantic import BaseModel

from aomma.backends.vlm import VLM
from aomma.layout import Vec3
from aomma.ledger import Ledger
from aomma.stages.image import to_jpeg

PROMPT = """\
You are cataloguing a photo so it can be rebuilt as a 3D scene. Each entry you list
will become one 3D model, placed on its own.

List every movable physical object you can see.
- One entry is one thing that moves as a single piece. Things that could be picked up
  separately are separate entries, even when they match, touch or overlap.
- The one exception is many small things packed together so tightly that they read as
  a single mass. List such a mass as one entry and say how many things are in it.
- Do not list the room itself or anything built into it.
- Do not list a part of an object as an entry of its own.
- At most 30 entries; when there are more, keep the largest and most prominent.

Reply with JSON only, in the form {"objects": [...]}. Each entry has:
- "id": unique snake_case name
- "bbox_2d": [x1, y1, x2, y2] bounding box normalised to 0-1000 (x across the width, y down the height)
- "description": at most 12 words covering colour, material and shape
- "size_m": [width, depth, height] estimated real-world size in meters
- "on": the id of the object it rests on or hangs from, or "floor" or "wall"
- "count": how many separate things the entry covers: 1, unless it is a packed mass
"""

SCHEMA = {
    "type": "object",
    "properties": {
        "objects": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "bbox_2d": {"type": "array", "items": {"type": "number"}},
                    "description": {"type": "string"},
                    "size_m": {"type": "array", "items": {"type": "number"}},
                    "on": {"type": "string"},
                    "count": {"type": "integer"},
                },
                "required": ["id", "bbox_2d", "description", "size_m", "on"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["objects"],
    "additionalProperties": False,
}


class InventoryObject(BaseModel):
    id: str
    # Normalised 0-1000, origin top-left.
    bbox_2d: tuple[float, float, float, float]
    description: str
    size_m: Vec3
    on: str
    # How many separate things the entry covers: more than one for a packed mass.
    count: int = 1


class Inventory(BaseModel):
    objects: list[InventoryObject]


def parse_inventory(text: str) -> Inventory:
    """Parse a model reply, tolerating code fences and a bare top-level list."""
    match = re.search(r"[\[{].*[\]}]", text, re.DOTALL)
    if not match:
        raise ValueError(f"no JSON in model reply: {text[:200]!r}")
    data = json.loads(match.group())
    if isinstance(data, list):
        data = {"objects": data}
    inventory = Inventory.model_validate(data)

    # Ids become USD prim names, so force them to be unique identifiers.
    renamed: dict[str, str] = {}
    for obj in inventory.objects:
        base = re.sub(r"\W+", "_", obj.id.lower()).strip("_") or "object"
        if base[0].isdigit():
            base = f"object_{base}"
        new_id, n = base, 2
        while new_id in renamed.values():
            new_id, n = f"{base}_{n}", n + 1
        renamed[obj.id] = new_id
        obj.id = new_id
    for obj in inventory.objects:
        obj.on = renamed.get(obj.on, obj.on)
    return inventory


def take_inventory(image: Image.Image, vlm: VLM, ledger: Ledger) -> tuple[Inventory, str]:
    """Return the parsed inventory and the raw reply (kept for debugging)."""
    ledger.check_budget()
    text, usage = vlm.ask(to_jpeg(image), PROMPT, SCHEMA, label="inventory")
    ledger.record("inventory", usage)
    return parse_inventory(text), text


SPLIT_PROMPT = """\
This is a close-up from a photo that is being rebuilt as a 3D scene. It was catalogued
as a single entry: "{description}". That may be wrong: it may be several separate
things, each of which needs an entry of its own because each becomes its own 3D model.

Look again. List every separate thing of that kind you can see, including any partly
hidden behind the others, with a tight box round each. If it really is one thing, list
it once. Do not list what they rest on.

Reply with JSON only, in the form {{"objects": [...]}}. Each entry has:
- "id": unique snake_case name
- "bbox_2d": [x1, y1, x2, y2] bounding box normalised to 0-1000 within this close-up (x across the width, y down the height)
- "description": at most 12 words covering colour, material and shape, and what tells it apart from the others
- "size_m": [width, depth, height] estimated real-world size in meters
- "on": "{on}"
"""

# An entry covering this many things or fewer is listed thing by thing; a larger mass stays one entry.
SPLIT_UP_TO = 6
# How much of the photo around a lumped entry's box the VLM is shown, as a multiple of the box.
_SPLIT_VIEW = 1.5


def split_entry(image: Image.Image, inventory: Inventory, object_id: str, vlm: VLM, ledger: Ledger) -> list[InventoryObject]:
    """Show the VLM a close-up of one entry and have it list again what is there, replacing the entry.

    Used when an entry is suspect: it claims to cover a few things, or the
    masks leave most of its box uncovered. Several things listed as one suit no
    single mask or model. Returns the new entries, already in `inventory` in
    the old one's place; one entry back means the VLM stands by it, with a
    fresh box.
    """
    group = next(obj for obj in inventory.objects if obj.id == object_id)
    x1, y1, x2, y2 = (v * side / 1000 for v, side in zip(group.bbox_2d, (image.width, image.height) * 2))
    grow_x, grow_y = (x2 - x1) * (_SPLIT_VIEW - 1) / 2, (y2 - y1) * (_SPLIT_VIEW - 1) / 2
    view = (max(x1 - grow_x, 0), max(y1 - grow_y, 0), min(x2 + grow_x, image.width), min(y2 + grow_y, image.height))

    ledger.check_budget()
    text, usage = vlm.ask(
        to_jpeg(image.crop(view)), SPLIT_PROMPT.format(description=group.description, on=group.on), SCHEMA,
        label=f"second look at {object_id}",
    )
    ledger.record("split", usage)
    found = parse_inventory(text).objects
    if not found:
        return [group]  # the VLM saw nothing to list: leave the entry as it was

    taken = {obj.id for obj in inventory.objects} - {object_id}
    for entry in found:
        # Boxes come back relative to the close-up; ids must not collide with the rest of the photo's.
        ax, ay, bx, by = entry.bbox_2d
        entry.bbox_2d = (
            (view[0] + ax / 1000 * (view[2] - view[0])) * 1000 / image.width,
            (view[1] + ay / 1000 * (view[3] - view[1])) * 1000 / image.height,
            (view[0] + bx / 1000 * (view[2] - view[0])) * 1000 / image.width,
            (view[1] + by / 1000 * (view[3] - view[1])) * 1000 / image.height,
        )
        entry.on = group.on
        base, number = entry.id, 2
        while entry.id in taken:
            entry.id, number = f"{base}_{number}", number + 1
        taken.add(entry.id)

    # Whatever rested on the lump now rests on what the lump rested on.
    at = inventory.objects.index(group)
    inventory.objects[at : at + 1] = found
    for obj in inventory.objects:
        if obj.on == object_id:
            obj.on = group.on
    return found


def draw_inventory(image: Image.Image, inventory: Inventory) -> Image.Image:
    """Overlay each object's box and id on a copy of the image."""
    out = image.copy()
    draw = ImageDraw.Draw(out)
    sx, sy = out.width / 1000, out.height / 1000
    for obj in inventory.objects:
        x1, y1, x2, y2 = obj.bbox_2d
        box = (x1 * sx, y1 * sy, x2 * sx, y2 * sy)
        draw.rectangle(box, outline=(255, 40, 40), width=2)
        label = draw.textbbox((box[0] + 3, box[1] + 2), obj.id)
        draw.rectangle(label, fill=(255, 40, 40))
        draw.text((box[0] + 3, box[1] + 2), obj.id, fill=(255, 255, 255))
    return out
