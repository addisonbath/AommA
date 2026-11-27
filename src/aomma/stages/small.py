"""Small items' masks (gate 3): a listing per support, each item masked and carried back to the photo.

Small is everything that is not the shell, a built-in or a large item. No model is made here: only the listing and the
masks, which `aomma labels` then takes in as class `small`.

1. **Supports.** Every built-in and large item, whatever its kind: anything may hold something, and its listing
   says when nothing is there (`supports`).
2. **A listing per support.** A close-up crop of the full-resolution photo around the support's mask, grown upward by
   the support's own height to take in what stands on it, and one VLM question: what is on or in this? The prompt is
   given what is already listed, so nothing is listed twice. Then one pass over the whole photo for what stands on the
   floor or hangs or leans on a wall.
3. **Masks.** Segmented in the close-up, at its resolution, and carried back to the photo. Where two supports' crops
   found the same thing (their masks overlapping by more than half), it is kept once, on the support it sits lower in
   the crop of.

Writes `small_survey.json` (the items, each with `on`: its support's id, `floor` or `wall`), `masks/<id>.png`,
`debug/small_masks.jpg`, `debug/small_inventory.jpg` (every item's box over the photo), and `debug/small/<support>.jpg` (each crop with what was listed in it).
"""

import json
import re
from pathlib import Path

import numpy as np
from PIL import Image
from pydantic import BaseModel

from aomma.stages.image import to_jpeg
from aomma.stages.inventory import Inventory, InventoryObject, draw_inventory, parse_inventory
from aomma.stages.segment import draw_masks, segment

# Large items that hold things on or in them.
# A crop is segmented at most this size along its longer side, which is what the segmenter takes in comfortably.
CROP_MAX_PX = 1536
# Two masks are the same thing if this share of the smaller lies in the larger.
SAME_THING = 0.5

PROMPT_SUPPORT = """\
This is a close-up from a photo of a {room_type}, around {support}: {description}.
These are already listed and must not be listed again:
{listed}

List every small thing that stands on, lies on, or sits inside {support}: anything not part of the room,
not a piece of furniture and not an appliance standing on the floor. Books, boxes, lamps, plants, bowls, bottles,
cushions, pillows and bedding, clothes, toys, electronics, countertop appliances, decorations. Things packed together
that can't be told apart (a row of books) are one entry, with "count" how many there are.
- List only what this close-up shows on or in {support}; leave out things further back or on other furniture.
- Do not list {support} itself or any part of it (its drawers, doors, shelves, legs).

Reply with JSON only, in the form {{"objects": [...]}}. Each entry has:
- "id": unique snake_case name
- "bbox_2d": [x1, y1, x2, y2] bounding box in this close-up, normalised to 0-1000
- "description": at most 12 words covering colour, material and shape
- "size_m": [width, depth, height] estimated real-world size in meters
- "count": how many separate things the entry covers
"""

PROMPT_ROOM = """\
This is a photo of a {room_type}. These are already listed and must not be listed again:
{listed}

List every other small thing that stands on the floor, or hangs or leans on a wall: anything not part of the room,
not a piece of furniture, not an appliance and not a rug. Such as floor lamps, guitars, floor cushions, pet beds,
baskets, bins, plants in pots, bags, boxes, shoes, pictures, mirrors, clocks, wall hangings, clothes on hooks and
curtains. A window's blinds are part of the window, not a thing to list. Leave out anything seen through a doorway
or a window, and anything standing on furniture.

Reply with JSON only, in the form {{"objects": [...]}}. Each entry has:
- "id": unique snake_case name
- "bbox_2d": [x1, y1, x2, y2] bounding box normalised to 0-1000
- "description": at most 12 words covering colour, material and shape
- "size_m": [width, depth, height] estimated real-world size in meters
- "count": how many separate things the entry covers
- "on": "floor" or "wall"
"""

_ENTRY = {
    "id": {"type": "string"},
    "bbox_2d": {"type": "array", "items": {"type": "number"}},
    "description": {"type": "string"},
    "size_m": {"type": "array", "items": {"type": "number"}},
    "count": {"type": "integer"},
}


def _schema(with_on: bool) -> dict:
    props = _ENTRY | ({"on": {"type": "string"}} if with_on else {})
    return {
        "type": "object",
        "properties": {"objects": {"type": "array", "items": {
            "type": "object", "properties": props, "required": list(props), "additionalProperties": False}}},
        "required": ["objects"],
        "additionalProperties": False,
    }


class SmallSurvey(BaseModel):
    objects: list[InventoryObject]
    # The supports looked at, each with its crop in the photo (pixels of input.jpg), and what each pass found.
    supports: dict[str, list[int]] = {}
    found: dict[str, list[str]] = {}
    # Things two crops both found, and the support each was kept on.
    merged: dict[str, str] = {}


def supports(survey: dict, large: dict) -> list[InventoryObject]:
    """Every built-in and large item: anything may hold something on or in it (a basket on a wall cabinet, clothes on
    a chair), and its listing says when nothing is there. Not chosen by kind: the survey's kinds are a VLM's guess,
    which varies between draws (KitchenScan 007's survey called every built-in `other`, so its counter was never
    looked at)."""
    return [InventoryObject(**o) for o in survey.get("built_ins", []) + large.get("objects", [])]


def crop_box(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    """The support's box, grown upward by 1.5 times its height and sideways by a fifth of its width, within the photo.

    The wider of the two crops tried: 85% of the kitchens' small things against 76% for a box grown by its
    height and a tenth of its width, with no more things listed that are not small items."""
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return None
    h, w = mask.shape
    x0, x1, y0, y1 = int(xs.min()), int(xs.max()) + 1, int(ys.min()), int(ys.max()) + 1
    side = (x1 - x0) // 5
    return max(0, x0 - side), max(0, y0 - int(1.5 * (y1 - y0))), min(w, x1 + side), y1


def parse_small(text: str, on: str | None, taken: set[str]) -> list[InventoryObject]:
    """The listed things, each `on` the support (or its own "floor"/"wall"), ids made unique against `taken`."""
    # An object, or a bare list of the entries (the room's pass sometimes answers so).
    match = re.search(r"[\[{].*[\]}]", text, re.DOTALL)
    if not match:
        raise ValueError(f"no JSON in model reply: {text[:200]!r}")
    data = json.loads(re.sub(r",(\s*[\]}])", r"\1", match.group()))
    if isinstance(data, list):
        data = {"objects": data}
    entries = [e for e in data.get("objects", []) if len(e.get("bbox_2d", [])) == 4 and len(e.get("size_m", [])) == 3]
    for e in entries:
        # The local VLM sometimes gives the description as a list of strings.
        if isinstance(e.get("description"), list):
            e["description"] = " ".join(str(d) for d in e["description"])
        e["on"] = on or (e.get("on") if e.get("on") in ("floor", "wall") else "floor")
        e["count"] = max(1, int(e.get("count") or 1))
    objects = parse_inventory(json.dumps({"objects": entries})).objects
    for obj in objects:
        base, number = re.sub(r"[^a-z0-9_]", "_", obj.id.lower()) or "thing", 2
        obj.id = base
        while obj.id in taken:
            obj.id, number = f"{base}_{number}", number + 1
        taken.add(obj.id)
    return objects


def _listed(objects: list[InventoryObject]) -> str:
    return "\n".join(f'- "{o.id}": {o.description}' for o in objects) or "- none"


def _in_photo(obj: InventoryObject, box: tuple[int, int, int, int], size: tuple[int, int]) -> InventoryObject:
    """A crop's entry with its box in the whole photo's 0-1000 frame."""
    x0, y0, x1, y1 = box
    w, h = size
    b = obj.bbox_2d
    to_x = lambda v: (x0 + v * (x1 - x0) / 1000) * 1000 / w
    to_y = lambda v: (y0 + v * (y1 - y0) / 1000) * 1000 / h
    return obj.model_copy(update={"bbox_2d": (to_x(b[0]), to_y(b[1]), to_x(b[2]), to_y(b[3]))})


def survey_small(run_dir: Path, vlm, ledger, comfy) -> SmallSurvey:
    """List and segment the small items of a run whose shell and large items are made."""
    image = Image.open(run_dir / "input.jpg").convert("RGB")
    full = Image.open(run_dir / "photo.jpg").convert("RGB") if (run_dir / "photo.jpg").exists() else image
    size = image.size
    scale = full.size[0] / size[0]
    survey = json.loads((run_dir / "survey.json").read_text())
    large = json.loads((run_dir / "large_survey.json").read_text()) if (run_dir / "large_survey.json").exists() else {}
    room_type = survey.get("room_type") or "room"
    # Made again: an earlier listing's masks go first, so they are neither kept nor taken as already listed.
    if (run_dir / "small_survey.json").exists():
        for old in json.loads((run_dir / "small_survey.json").read_text()).get("objects", []):
            (run_dir / "masks" / f"{old['id']}.png").unlink(missing_ok=True)
    known = [InventoryObject(**o) for o in survey.get("built_ins", []) + large.get("objects", [])]
    taken = {p.stem for p in (run_dir / "masks").glob("*.png")} | {o.id for o in known}
    (run_dir / "debug" / "small").mkdir(parents=True, exist_ok=True)

    result = SmallSurvey(objects=[])
    masks: dict[str, np.ndarray] = {}
    # For each thing, how low it sits in its crop (its mask's bottom, as a share of the crop's height).
    lowness: dict[str, float] = {}
    for support in supports(survey, large):
        path = run_dir / "masks" / f"{support.id}.png"
        if not path.exists():
            continue
        support_mask = np.asarray(Image.open(path).convert("L").resize(size, Image.NEAREST)) > 127
        box = crop_box(support_mask)
        if box is None:
            continue
        result.supports[support.id] = list(box)
        crop = full.crop(tuple(round(v * scale) for v in box))
        crop.thumbnail((CROP_MAX_PX, CROP_MAX_PX))
        prompt = PROMPT_SUPPORT.format(room_type=room_type, support=f'"{support.id}"', description=support.description,
                                       listed=_listed(known + result.objects))
        ledger.check_budget()
        text, usage = vlm.ask(to_jpeg(crop), prompt, _schema(False), label=f"small items on {support.id}")
        ledger.record("small items", usage)
        (run_dir / "debug" / "small" / f"{support.id}.txt").write_text(text)
        found = parse_small(text, support.id, taken)
        result.found[support.id] = [o.id for o in found]
        print(f"on {support.id}: {', '.join(o.id for o in found) or '-'}", flush=True)
        if not found:
            continue
        crop_inventory = Inventory(objects=found)
        draw_inventory(crop, crop_inventory).save(run_dir / "debug" / "small" / f"{support.id}.jpg", quality=88)
        name = comfy.upload_image(to_jpeg(crop), f"aomma_{run_dir.name}_small_{support.id}.jpg")
        crop_masks, _ = segment(comfy, name, crop.size, crop_inventory)
        x0, y0, x1, y1 = box
        for obj in found:
            placed = np.zeros((size[1], size[0]), bool)
            small = Image.fromarray(crop_masks[obj.id].astype(np.uint8) * 255).resize((x1 - x0, y1 - y0), Image.NEAREST)
            placed[y0:y1, x0:x1] = np.asarray(small) > 127
            ys = np.nonzero(placed)[0]
            if not len(ys):
                continue
            masks[obj.id] = placed
            lowness[obj.id] = (ys.max() - y0) / max(1, y1 - y0)
            result.objects.append(_in_photo(obj, box, size))

    # The floor and the walls, over the whole photo.
    prompt = PROMPT_ROOM.format(room_type=room_type, listed=_listed(known + result.objects))
    ledger.check_budget()
    text, usage = vlm.ask(to_jpeg(image), prompt, _schema(True), label="small items on the floor and walls")
    ledger.record("small items", usage)
    (run_dir / "debug" / "small" / "room.txt").write_text(text)
    found = parse_small(text, None, taken)
    result.found["room"] = [o.id for o in found]
    print(f"on the floor and walls: {', '.join(f'{o.id} ({o.on})' for o in found) or '-'}", flush=True)
    if found:
        room_inventory = Inventory(objects=found)
        draw_inventory(image, room_inventory).save(run_dir / "debug" / "small" / "room.jpg", quality=88)
        whole = full.copy()
        whole.thumbnail((CROP_MAX_PX, CROP_MAX_PX))
        name = comfy.upload_image(to_jpeg(whole), f"aomma_{run_dir.name}_small_room.jpg")
        room_masks, _ = segment(comfy, name, whole.size, room_inventory)
        for obj in found:
            mask = np.asarray(Image.fromarray(room_masks[obj.id].astype(np.uint8) * 255).resize(size, Image.NEAREST)) > 127
            if mask.any():
                masks[obj.id] = mask
                lowness[obj.id] = -1.0  # a support's crop shows a thing closer: it wins over the room's pass
                result.objects.append(obj)

    # One thing found twice is kept once, on the support it sits lower in the crop of.
    order = sorted(masks, key=lambda oid: -lowness[oid])
    kept: list[str] = []
    for oid in order:
        same = next((k for k in kept if (masks[oid] & masks[k]).sum() > SAME_THING * min(masks[oid].sum(), masks[k].sum())), None)
        if same:
            result.merged[oid] = same
        else:
            kept.append(oid)
    result.objects = [o for o in result.objects if o.id in kept]
    for oid in kept:
        Image.fromarray(masks[oid].astype(np.uint8) * 255).save(run_dir / "masks" / f"{oid}.png")
    draw_masks(image, {oid: masks[oid] for oid in kept}).save(run_dir / "debug" / "small_masks.jpg", quality=88)
    draw_inventory(image, Inventory(objects=result.objects)).save(run_dir / "debug" / "small_inventory.jpg", quality=88)
    (run_dir / "small_survey.json").write_text(result.model_dump_json(indent=2))
    return result
