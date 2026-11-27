"""Room shell, the first stage of a staged run: is the photo indoors, and what is the room built of.

The rest of the scene stands inside the shell, so it is found first and on
its own: the floor, the walls, and what is built into the room (counters,
cabinet runs, shelving fixed to a wall). Built-ins are part of the room, not
objects: each becomes a box fitted to the depth map and painted from the
photo, as the walls are, and is never sent to image-to-3D. Walls are fitted
to the pixels a segmenter calls wall, not to whatever no object covers: in
the kitchen run, the unlisted lower cabinets were fitted as two walls 18 cm
apart in front of the real one.
"""

import json
import re

import numpy as np
from pydantic import BaseModel

from aomma.backends.vlm import VLM
from aomma.ledger import Ledger
from aomma.stages.image import to_jpeg
from aomma.stages.inventory import InventoryObject, parse_inventory
from aomma.stages.pose import Wall, through_openings
from aomma.stages.proxy import Proxy

PROMPT = """\
You are surveying a photo so that the room in it can be rebuilt in 3D before any of
its furniture or loose objects are.

1. Say whether the photo is taken indoors, and what kind of room it shows.
2. List each wall you can see: one entry for each flat stretch of wall, so a wall
   that turns a corner is two entries. Its box covers all of that stretch the photo
   shows, including what hangs on it or stands in front of it.
3. List what is built into the room: things fixed in place that are part of the
   building rather than furniture, such as kitchen counters, base and wall cabinet
   runs, built-in shelving, a kitchen island, a fixed sink, a fireplace, a bathtub. A
   run of matching cabinets side by side is one entry; a counter is its own entry,
   resting on the cabinets below it. Do not list walls, the floor or the ceiling, nor
   what is part of a wall's surface (tiles, a backsplash, paint, trim), nor doors,
   windows or blinds, nor free-standing furniture or appliances.
4. List the doors, doorways and windows you can see. A window's blinds are part of
   the window: its box covers them too, also where they hang past the glass.

Reply with JSON only, in the form
{"indoor": true, "room_type": "...", "walls": [...], "built_ins": [...], "openings": [...]}.
Each entry in "walls" has:
- "id": unique snake_case name, such as "left_wall"
- "bbox_2d": [x1, y1, x2, y2] bounding box normalised to 0-1000 (x across the width, y down the height)
- "description": at most 12 words: its colour and what is on it
Each entry in "built_ins" and "openings" has:
- "id": unique snake_case name
- "kind": for a built-in one of "counter", "island", "base_cabinets", "wall_cabinets",
  "shelving", "sink", "fireplace", "bathtub", "other"; for an opening one of "door", "doorway", "window"
- "bbox_2d": as for a wall
- "description": at most 12 words covering colour, material and shape
- "size_m": [width, depth, height] estimated real-world size in meters
- "on": the id of the built-in it rests on, or "floor" or "wall"
"""

_WALL = {
    "type": "object",
    "properties": {"id": {"type": "string"}, "bbox_2d": {"type": "array", "items": {"type": "number"}}, "description": {"type": "string"}},
    "required": ["id", "bbox_2d", "description"],
    "additionalProperties": False,
}
_ENTRY = {
    "type": "object",
    "properties": {
        "id": {"type": "string"},
        "kind": {"type": "string"},
        "bbox_2d": {"type": "array", "items": {"type": "number"}},
        "description": {"type": "string"},
        "size_m": {"type": "array", "items": {"type": "number"}},
        "on": {"type": "string"},
    },
    "required": ["id", "kind", "bbox_2d", "description", "size_m", "on"],
    "additionalProperties": False,
}
SCHEMA = {
    "type": "object",
    "properties": {
        "indoor": {"type": "boolean"},
        "room_type": {"type": "string"},
        "walls": {"type": "array", "items": _WALL},
        "built_ins": {"type": "array", "items": _ENTRY},
        "openings": {"type": "array", "items": _ENTRY},
    },
    "required": ["indoor", "room_type", "walls", "built_ins", "openings"],
    "additionalProperties": False,
}

# The kinds a built-in may have, as the prompt above lists them (`relook` asks for the same).
BUILT_IN_KINDS = ("counter", "island", "base_cabinets", "wall_cabinets", "shelving", "sink", "fireplace", "bathtub", "other")
# Heights above the floor that vary little from room to room, by kind: the top of a kitchen counter
# (and of an island, and of base cabinets without their counter) and of a door.
STANDARD_TOP_M = {"counter": 0.91, "island": 0.91, "base_cabinets": 0.88, "door": 2.03, "doorway": 2.03}
# A scale is believed only if it puts a hand-held camera this high off the floor.
CAMERA_HEIGHT_M = (1.0, 1.9)


class Survey(BaseModel):
    indoor: bool
    room_type: str
    # Each flat stretch of wall the photo shows; only id, box and description mean anything.
    walls: list[InventoryObject]
    built_ins: list[InventoryObject]
    # Doors and windows: not built, since the wall's painting shows them, but measured, to scale the depth map.
    openings: list[InventoryObject]
    # The kind of each built-in and opening (see `STANDARD_TOP_M`).
    kinds: dict[str, str] = {}


def parse_survey(text: str) -> Survey:
    """Parse a model reply, tolerating code fences; ids are made unique across all three lists."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"no JSON in model reply: {text[:200]!r}")
    data = json.loads(match.group())
    walls = [{**wall, "size_m": [1.0, 0.05, 2.4], "on": "floor"} for wall in data.get("walls", [])]
    lists = [walls, data.get("built_ins", []), data.get("openings", [])]
    kinds = [str(entry.get("kind", "other")) for entry in lists[1] + lists[2]]
    plain = [{key: value for key, value in entry.items() if key != "kind"} for entry in sum(lists, [])]
    # One inventory parse over all of them, so an id is unique among all of them.
    entries = parse_inventory(json.dumps({"objects": plain})).objects
    split = (len(lists[0]), len(lists[0]) + len(lists[1]))
    return Survey(
        indoor=bool(data.get("indoor", True)), room_type=str(data.get("room_type", "")),
        walls=entries[: split[0]], built_ins=entries[split[0] : split[1]], openings=entries[split[1] :],
        kinds={entry.id: kind for entry, kind in zip(entries[split[0] :], kinds)},
    )


def take_survey(image, vlm: VLM, ledger: Ledger) -> tuple[Survey, str]:
    """Return the parsed survey and the raw reply (kept for debugging)."""
    ledger.check_budget()
    text, usage = vlm.ask(to_jpeg(image), PROMPT, SCHEMA, label="survey")
    ledger.record("survey", usage)
    return parse_survey(text), text


def scale_from_heights(survey: Survey, proxies: list[Proxy], camera_height_m: float) -> tuple[float, list[str], str]:
    """The depth map's scale from the tops of things whose height is standard (see `STANDARD_TOP_M`).

    `proxies` are boxes fitted at the depth map's own scale, and
    `camera_height_m` the camera's height at that scale. A thing whose top the
    photo's upper edge cuts off is not used. Returns the scale (1 when nothing
    can be used or the result puts the camera at an unlikely height), the ids
    used, and a note of why.
    """
    entries = {entry.id: entry for entry in survey.built_ins + survey.openings}
    ratios, used = [], []
    for proxy in proxies:
        standard = STANDARD_TOP_M.get(survey.kinds.get(proxy.id, ""))
        top = proxy.translate[2] + proxy.size[2]
        if standard is None or top <= 0.1 or entries[proxy.id].bbox_2d[1] <= 5:
            continue
        ratios.append(standard / top)
        used.append(proxy.id)
    if not ratios:
        return 1.0, [], "nothing of a standard height: left as measured"
    scale = float(np.median(ratios))
    if not CAMERA_HEIGHT_M[0] <= camera_height_m * scale <= CAMERA_HEIGHT_M[1]:
        return 1.0, [], f"x{scale:.2f} would put the camera {camera_height_m * scale:.2f} m up: left as measured"
    return scale, used, f"from the tops of {', '.join(used)}"


def floor_from_heights(survey: Survey, proxies: list[Proxy], camera_height_m: float) -> tuple[float, list[str]] | None:
    """How far below the floor taken the real one lies, from the tops of things whose height is standard (see
    `STANDARD_TOP_M`), for a floor out of view: the lowest level surface stood in for it, which in a kitchen is
    often a counter's top (KitchenScan 008's put the camera 0.5-0.8 m up).

    `proxies` are boxes fitted over the floor taken, and `camera_height_m` the camera's height over it. A thing
    whose top the photo's upper edge cuts off is not used. Returns the drop (negative if the real floor is higher)
    and the ids used, or None when nothing can be used or the result puts the camera at an unlikely height.
    """
    entries = {entry.id: entry for entry in survey.built_ins + survey.openings}
    drops, used = [], []
    for proxy in proxies:
        standard = STANDARD_TOP_M.get(survey.kinds.get(proxy.id, ""))
        if standard is None or entries[proxy.id].bbox_2d[1] <= 5:
            continue
        drops.append(standard - (proxy.translate[2] + proxy.size[2]))
        used.append(proxy.id)
    if not drops:
        return None
    drop = float(np.median(drops))
    if not CAMERA_HEIGHT_M[0] <= camera_height_m + drop <= CAMERA_HEIGHT_M[1]:
        return None
    return drop, used


def wall_agreement(world: np.ndarray, walls: list[Wall], eye: np.ndarray, agree: float = 1.10) -> dict | None:
    """How well the walls stand where the photo's wall pixels are, measured along each line of sight.

    `world` are the depth map's points for the pixels the photo shows as wall.
    For each, the nearest wall face its line of sight crosses should be about
    as far as the point: "agree" within `agree` either way, "too_near" when a
    wall stands in front of what the photo shows (a cabinet fitted as a wall),
    "too_far" when the nearest wall is beyond it, and "missed" when the line
    of sight meets no wall at all. Shares of the points; None if there are none.
    """
    if not len(world):
        return None
    # Distance to the nearest wall crossing, as a multiple of the distance to the point.
    nearest = np.full(len(world), np.inf)
    for wall in walls:
        corner, along, normal, height = wall
        towards = (world - eye) @ normal
        eye_side = (eye - corner) @ normal
        with np.errstate(divide="ignore", invalid="ignore"):
            share = -eye_side / towards
        hit = eye + share[:, None] * (world - eye)
        across = (hit - corner) @ along / (along @ along)
        on_face = (towards < 0) & (share > 0) & (across >= 0) & (across <= 1) & (hit[:, 2] >= 0) & (hit[:, 2] <= height)
        on_face &= ~through_openings(wall, across, hit[:, 2])
        nearest = np.where(on_face, np.minimum(nearest, share), nearest)
    missed = ~np.isfinite(nearest)
    near = ~missed & (nearest < 1 / agree)
    far = ~missed & (nearest > agree)
    return {
        "agree": round(float((~missed & ~near & ~far).mean()), 3),
        "too_near": round(float(near.mean()), 3),
        "too_far": round(float(far.mean()), 3),
        "missed": round(float(missed.mean()), 3),
        "points": int(len(world)),
    }
