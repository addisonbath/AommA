"""Critic stage: look at the finished scene beside the photo and say what is wrong.

The pipeline's own scores measure outlines. They cannot tell that a poster
shows its blank back, or that a "set of pillows" is really one pillow. A VLM
can, if asked one narrow question at a time: each object is shown as a
close-up from the photo next to the same close-up from the render.
"""

import json
import re
from typing import Literal

import numpy as np
from PIL import Image, ImageDraw
from pydantic import BaseModel

from aomma.backends.vlm import VLM
from aomma.ledger import Ledger
from aomma.stages.image import to_jpeg
from aomma.stages.inventory import SCHEMA, Inventory, InventoryObject, draw_inventory, parse_inventory

Verdict = Literal["ok", "missing", "wrong_object", "several_objects", "wrong_shape", "wrong_side", "wrong_place"]
# Complaints the review's own measurements make, which the critic is never asked about.
# The last three come from the shell's measures, on a staged run's large items.
MeasuredVerdict = Literal["behind_wall", "covers_others", "wrong_size", "collides", "left_out", "off_support", "off_square"]

PROMPT = """\
Two pictures of the same spot in a room, side by side. LEFT: a photo. RIGHT: a 3D
reconstruction of the room, rendered from the same camera position.

The object to judge is "{name}": {description}. In the photo it is in the middle of
the left picture; the reconstruction should show it in the middle of the right one.

Compare that one object across the two pictures and choose the verdict that fits best:
- "ok": the reconstruction shows the same object with about the same shape, size and
  place. Differences in colour, lighting, sharpness or fine detail do not matter.
- "missing": the object is not there in the reconstruction.
- "wrong_object": something else is shown in its place, or only a piece of it.
- "several_objects": what the photo shows there is several separate objects, and the
  reconstruction has made one lump of them.
- "wrong_shape": it is the right object, but its shape is badly off: stretched,
  melted, broken, or with large parts that are not in the photo.
- "wrong_side": it is the right object, but turned so that it shows its back, its
  underside or a blank face.
- "wrong_place": it is the right object and shape, but clearly shifted, or much too
  big or too small.

Reply with JSON only: {{"verdict": "...", "note": "one short sentence on what you see"}}
"""

SCHEMA_VERDICT = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": list(Verdict.__args__)},
        "note": {"type": "string"},
    },
    "required": ["verdict", "note"],
    "additionalProperties": False,
}

MISSING_PROMPT = """\
This photo is being rebuilt as a 3D scene. Every object catalogued so far has a red
box and a label. Each catalogued object becomes one 3D model.

List any movable physical object that is plainly visible but has no box of its own.
- Do not list the room itself or anything built into it.
- Do not list a part of an object that is already boxed.
- Do not list things too small to make out.
If nothing is missing, reply with an empty list.

Reply with JSON only, in the form {"objects": [...]}. Each entry has:
- "id": unique snake_case name
- "bbox_2d": [x1, y1, x2, y2] bounding box normalised to 0-1000 (x across the width, y down the height)
- "description": at most 12 words covering colour, material and shape
- "size_m": [width, depth, height] estimated real-world size in meters
- "on": the id of the object it rests on or hangs from, or "floor" or "wall"
"""

# The close-up shows this multiple of the object's own extent, and is this many pixels tall.
_VIEW = 1.6
_VIEW_PX = 448


class Finding(BaseModel):
    id: str
    verdict: Verdict | MeasuredVerdict
    note: str = ""
    # Who complained: the critic, looking at close-ups, or the scene checks, measuring.
    source: Literal["critic", "checks"] = "critic"


def object_view(photo: Image.Image, render: Image.Image, mask: np.ndarray) -> Image.Image:
    """The same square close-up from the photo (left) and the render (right), centred on the object."""
    return close_ups([(photo, "PHOTO"), (render, "RECONSTRUCTION")], mask)


def close_ups(pictures: list[tuple[Image.Image, str]], mask: np.ndarray) -> Image.Image:
    """The same square close-up from each picture, side by side under its title, centred on the object."""
    rows, cols = np.nonzero(mask)
    centre = ((cols.min() + cols.max() + 1) / 2, (rows.min() + rows.max() + 1) / 2)
    side = max(cols.max() + 1 - cols.min(), rows.max() + 1 - rows.min(), 32) * _VIEW
    box = tuple(round(v) for v in (centre[0] - side / 2, centre[1] - side / 2, centre[0] + side / 2, centre[1] + side / 2))
    gap, banner = 12, 24
    view = Image.new("RGB", (len(pictures) * (_VIEW_PX + gap) - gap, _VIEW_PX + banner), "white")
    draw = ImageDraw.Draw(view)
    for index, (picture, title) in enumerate(pictures):
        # `crop` pads with black where the square hangs over the picture's edge.
        close_up = picture.convert("RGB").crop(box).resize((_VIEW_PX, _VIEW_PX), Image.Resampling.LANCZOS)
        view.paste(close_up, (index * (_VIEW_PX + gap), banner))
        draw.text((index * (_VIEW_PX + gap) + 6, 6), title, fill="black")
    return view


def parse_finding(object_id: str, text: str) -> Finding:
    """Parse the critic's reply; anything unreadable counts as no complaint."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    try:
        data = json.loads(match.group()) if match else {}
        return Finding(id=object_id, verdict=data.get("verdict", "ok"), note=str(data.get("note", ""))[:300])
    except (json.JSONDecodeError, ValueError):
        return Finding(id=object_id, verdict="ok", note="(unreadable reply)")


def critique(
    photo: Image.Image,
    render: Image.Image,
    obj: InventoryObject,
    mask: np.ndarray,
    vlm: VLM,
    ledger: Ledger,
    ticket: int | None = None,
) -> tuple[Finding, Image.Image]:
    """Judge one object's reconstruction against the photo; returns the finding and the close-up shown.

    `ticket` is a place in the job queue taken earlier for this judgement.
    """
    ledger.check_budget()
    view = object_view(photo, render, mask)
    text, usage = vlm.ask(
        to_jpeg(view), PROMPT.format(name=obj.id.replace("_", " "), description=obj.description), SCHEMA_VERDICT,
        label=f"judge {obj.id}", ticket=ticket,
    )
    ledger.record("critic", usage)
    return parse_finding(obj.id, text), view


def find_unlisted(photo: Image.Image, inventory: Inventory, vlm: VLM, ledger: Ledger) -> list[InventoryObject]:
    """Ask the VLM for objects in the photo that the inventory has no entry for.

    Returns new entries with ids that do not collide with the inventory's; the
    inventory itself is not changed.
    """
    ledger.check_budget()
    text, usage = vlm.ask(to_jpeg(draw_inventory(photo, inventory)), MISSING_PROMPT, SCHEMA, label="look for unlisted objects")
    ledger.record("unlisted", usage)
    taken = {obj.id for obj in inventory.objects}
    found = parse_inventory(text).objects
    for entry in found:
        base, number = entry.id, 2
        while entry.id in taken:
            entry.id, number = f"{base}_{number}", number + 1
        taken.add(entry.id)
        if entry.on not in taken and entry.on not in ("floor", "wall"):
            entry.on = "floor"
    return found
