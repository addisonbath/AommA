"""Large items, the second stage of a staged run: the furniture and appliances, placed inside the room's shell.

The shell (stage 1) gives the floor, the walls and the built-ins. The large
items are listed with the shell's survey in hand, so built-ins are not listed
again, and each says what it stands on: the floor, a wall, or a built-in.
Their models are generated as in the old pipeline, but placed within the
shell's limits while being posed, rather than fixed afterwards: each stands on
its support, is turned square to the room when nearly so, never stands past a
wall, and is backed against a wall it nearly touches. One that stands far
past a wall was placed at the wrong distance, and is posed again, held to the
depth map; one whose own depth points stand past the walls is in another
room, and is left out.
"""

import json
import re

import numpy as np
from pydantic import BaseModel
from scipy.ndimage import binary_fill_holes

from aomma.backends.vlm import VLM
from aomma.ledger import Ledger
from aomma.stages.image import to_jpeg
from aomma.stages.inventory import InventoryObject, parse_inventory
from aomma.stages.pose import (
    _WALL_SLACK_M, Pose, Wall, _upright_faces_angle_deg, euler_xyz_deg, lowest, matrix_from_euler_xyz_deg, rotation_z,
)
from aomma.stages.proxy import Proxy

PROMPT = """\
You are cataloguing a photo of a {room_type} so it can be rebuilt as a 3D scene. The room
itself is already built: its walls, floor, doors and windows, and these built-ins:
{built_ins}

List only the large items: the furniture and the appliances, such as beds, sofas,
armchairs, chairs, tables, desks, dressers, nightstands, free-standing cabinets and
shelving units, benches, washers, dryers, fridges, stoves, treadmills. Each will become
one 3D model, placed on its own.
- Do not list the built-ins above, the walls, floor, doors or windows.
- Do not list anything in another room, seen through a doorway, an opening or a window.
- Do not list small things: what can be carried in one hand, what stands on a piece of
  furniture, cushions, bedding, lamps, plants, baskets, rugs, pictures, clothes.
- Do not list anything that stands on a built-in or on another piece of furniture, whatever
  its size: microwaves, toaster ovens, coffee makers, kettles, blenders, mixers.
- One entry is one piece of furniture; matching pieces side by side are separate entries.

Reply with JSON only, in the form {{"objects": [...]}}. Each entry has:
- "id": unique snake_case name
- "kind": one of {kinds}
- "bbox_2d": [x1, y1, x2, y2] bounding box normalised to 0-1000 (x across the width, y down the height)
- "description": at most 12 words covering colour, material and shape
- "size_m": [width, depth, height] estimated real-world size in meters
- "on": "floor", "wall" for something hung or fixed on a wall, or the id of the built-in it stands on
"""

KINDS = (
    "bed", "sofa", "armchair", "chair", "table", "desk", "dresser", "nightstand", "cabinet", "shelving", "bench",
    "appliance", "exercise", "other",
)

SCHEMA = {
    "type": "object",
    "properties": {
        "objects": {
            "type": "array",
            "items": {
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
            },
        }
    },
    "required": ["objects"],
    "additionalProperties": False,
}

# A model whose upright faces are within this angle of a wall's direction is turned square to it, unless the
# photo shows the item standing at an angle; and an item whose faces in the photo are this near it stands square.
SQUARE_WITHIN_DEG = 10.0
# An item's faces in the photo say how it is turned (see `seen_turn_deg`) only if they agree on a heading this
# well, over this many depth cells. Square items read 0.2-5.2 deg off the walls at 0.74-0.96, the office
# cabinet 35.2 at 0.97; round things, and the right nightstand, 53 cells seen, 0.12-0.46.
SEEN_SHARE = 0.6
SEEN_MIN_CELLS = 500
_SEEN_STEP_PX = 4
_SEEN_ERODE_PX = 6
# A model whose back is within this distance of a wall is moved back against it.
BACK_WITHIN_M = 0.15
# Kinds of large item that things stand inside, left to stage 3: a shelf's books, a cabinet's open shelves.
HOLDS_KINDS = ("shelving", "cabinet", "dresser", "nightstand")
# Holes enclosed by such an item's mask are its contents once they are at least this share of its outline.
# The low bookshelf's books left 0.48; real openings elsewhere (a stool's handle, a treadmill's frame) 0.05-0.07.
CONTENTS_SHARE = 0.2
# A model further than this past a wall is not pushed out sideways: its distance from the camera is wrong
# (the floor placed it from a bottom the photo hides), so it is posed again, held to the depth map.
# Bringing it nearer along the line of sight, shrunk to keep its outline, did not work: the office desk
# came out at 0.61 of its size and dropped 0.42 m to the floor (fit 0.87 -> 0.08), the fridge 0.27 m.
CLEAR_WITHIN_M = 0.3
# An item more than this share of whose own depth points stand past a wall is in another room.
BEYOND_SHARE = 0.5
# Holding a model to the shell that loses more than this of the fit it had before (judged with no walls in
# the way) is undone, and the model left out: the wall or the model is wrong, and the review decides which.
UNDO_FIT_DROP = 0.3
# How much the search that poses a model again counts against a candidate: the |log| ratio of its distance to the
# depth map's, the share of it behind a wall, and the |log| of its size over the listed size. Depth 1 with size 1
# got the office desk's distance right (1.02 of the photo's), but shrank it to 0.63.
REPOSE_DEPTH_WEIGHT = 1.0
# How far the search may turn the model from the heading that turns the side painted from the photo to the camera:
# not at all. A box's outline from the side fits as well as from the front, and the outline drew the kitchen's
# fridge away from it every time: searched over every heading, side-on; near the heading it was solved with, its
# front 45 degrees from the camera; within 15 degrees of the painted side, to the edge of that, 15 degrees away.
REPOSE_TURN_WITHIN_DEG = 0.0
REPOSE_WALL_WEIGHT = 2.0
REPOSE_SIZE_WEIGHT = 0.5
# A model on the floor whose seen surface stands further than the depth map puts its pixels, or nearer, by more than
# this ratio is at the wrong distance too, and is posed again the same way: a model wider in view than the item, fitted
# to its outline and stood on the floor, stands further back. OfficeScan 007's office chair, ~13% wider than the real
# one, stood at 1.07-1.12 of the depth map's distance and 0.22-0.26 m too far. Every item the re-pose helped
# read 1.07 or more (0.90 the other way); items placed right up to 1.050 (006's desk, moved 7 cm wrong at 1.05).
OFF_DEPTH_RATIO = 1.06
# The fewest of its mask's pixels the model must cover for that ratio to count.
OFF_DEPTH_MIN_PIXELS = 50
# A floor-standing item's mask that stops short of its box's bottom by more than this share of the box's height has
# its bottom hidden, and is redrawn whole: the office desk's mask stopped at its drawers, 0.22 short, leaving out
# the leg the box reaches; every other large item on the floor came within 0.04.
BOTTOM_SHORT = 0.1
# An item the photo's side edge cuts, whose measured box's longest side is under this share of its listed one, is not
# stretched onto that box: the box is cut too. The office desk's was 0.63 (0.95 m of 1.5), and stretched onto it
# the redrawn desk came out 1.03 m long. It is posed with at least `CUT_SIZE_WEIGHT` on its listed size.
CUT_BOX_SHARE = 0.75
CUT_SIZE_WEIGHT = 1.0
# A built-in stands over an item when its bottom is no lower than this below the item's box top, nor higher
# than `OVER_WITHIN_M` above it, and `OVER_SHARE` of the item's depth points lie within its width. Close above,
# as the kitchen's cabinet over the fridge (0.12 m above its box): the Garage's plank, 0.35 m over the dryer,
# is on the wall, not boxing it in.
OVER_SLACK_M = 0.15
OVER_WITHIN_M = 0.25
OVER_SHARE = 0.6
# A box squared to a built-in is kept only if its depth comes out within this factor of the listed depth.
SQUARED_DEPTH_WITHIN = 1.6
# How far a model may be stretched onto a box squared to a built-in, per dimension. Measured on every side, the
# box is trusted further than one measured from what shows: the kitchen's fridge model is 1.61 m deep, its
# alcove 0.67, and the 1.6 limit left it 0.95 deep, through the wall behind and out past its doors.
SQUARED_MAX_STRETCH = 2.5


class LargeSurvey(BaseModel):
    objects: list[InventoryObject]
    kinds: dict[str, str] = {}
    # How well each mask covers its box (see `segment.box_fills`), and its overlap with the box.
    box_iou: dict[str, float] = {}


class ShellMoves(BaseModel):
    """What holding a model to the shell did to its pose."""

    # Raised (or lowered, if negative) onto its support, in meters.
    settled_m: float = 0.0
    # Turned about the vertical to lie square to a wall, in degrees.
    turned_deg: float = 0.0
    # Moved out of the walls it stood past, in meters.
    cleared_m: float = 0.0
    # Moved back against a wall it nearly touched, in meters.
    backed_m: float = 0.0
    # The move back it was not given, as it fit its outline worse there (see `cli._hold_to_shell`), in meters.
    not_backed_m: float = 0.0
    # It stood more than `CLEAR_WITHIN_M` past a wall, at the wrong distance: it is to be posed again.
    far_past: bool = False
    # Posed again, held to the depth map, its listed size and the walls.
    reposed: bool = False
    # Its distance over the depth map's, where off by more than `OFF_DEPTH_RATIO`: it is to be posed again.
    off_depth: float | None = None
    # Stretched onto the box the depth map measured for it, where that fit its outline better.
    fitted_to_box: bool = False
    # The built-in standing over it, whose outline its box was squared to (see `square_to_built_in`).
    squared_to: str | None = None
    # Its own depth points stand past the walls: it is in another room, and left out of the scene.
    beyond_walls: bool = False
    # The moves lost too much of its fit and were undone; it is left out: the wall or the model is wrong.
    wall_or_model_wrong: bool = False


def parse_large(text: str, taken: set[str], built_ins: set[str] = frozenset()) -> LargeSurvey:
    """Parse a model reply; ids are made unique, and distinct from `taken` (the shell's ids).

    What stands on any of `built_ins` (their ids) is left out, whatever the built-in's kind and the thing's size: a
    thing held up by another is the small items' (`small.supports` looks on every built-in), and a built-in's kind is
    a VLM's guess.
    """
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"no JSON in model reply: {text[:200]!r}")
    # qwen once left a box a number short, with a trailing comma ([0, 592, 302,]): the comma is dropped
    # here, and an entry without a whole box below.
    data = json.loads(re.sub(r",(\s*[\]}])", r"\1", match.group()))
    entries = data.get("objects", []) if isinstance(data, dict) else data
    for entry in [e for e in entries if len(e.get("bbox_2d", [])) != 4]:
        print(f"{entry.get('id')}: left out, its box is not four numbers ({entry.get('bbox_2d')})", flush=True)
    entries = [entry for entry in entries if len(entry.get("bbox_2d", [])) == 4]
    kinds = [str(entry.get("kind", "other")) for entry in entries]
    plain = [{key: value for key, value in entry.items() if key != "kind"} for entry in entries]
    objects = parse_inventory(json.dumps({"objects": plain})).objects
    kinds = [kind for obj, kind in zip(objects, kinds) if obj.on not in built_ins]
    objects = [obj for obj in objects if obj.on not in built_ins]
    shell = set(taken)
    for obj in objects:
        base, number = obj.id, 2
        while obj.id in taken:
            obj.id, number = f"{base}_{number}", number + 1
        taken.add(obj.id)
    # A support the shell does not know is taken as the floor: stage 2 places nothing on stage 2's own items.
    for obj in objects:
        if obj.on not in ("floor", "wall") and obj.on not in shell:
            obj.on = "floor"
    return LargeSurvey(objects=objects, kinds={obj.id: kind for obj, kind in zip(objects, kinds)})


def take_large(
    image, room_type: str, built_ins: list[InventoryObject], taken: set[str], vlm: VLM, ledger: Ledger,
) -> tuple[LargeSurvey, str]:
    """Return the parsed list of large items and the raw reply (kept for debugging)."""
    listed = "\n".join(f'- "{o.id}": {o.description}, box {[round(v) for v in o.bbox_2d]}' for o in built_ins) or "- none"
    prompt = PROMPT.format(room_type=room_type or "room", built_ins=listed, kinds=", ".join(f'"{k}"' for k in KINDS))
    ledger.check_budget()
    text, usage = vlm.ask(to_jpeg(image), prompt, SCHEMA, label="large items")
    ledger.record("large items", usage)
    try:
        return parse_large(text, set(taken), {o.id for o in built_ins}), text
    except json.JSONDecodeError as error:
        # The reply is lost otherwise: debug/large_survey_raw.txt is written only once it parses.
        raise ValueError(f"large items reply is not JSON ({error}): {text!r}") from error


def _place(pose: Pose, points: np.ndarray) -> np.ndarray:
    return pose.scale * points @ matrix_from_euler_xyz_deg(pose.rotate_xyz_deg).T + np.array(pose.translate)


def _moved(pose: Pose, by: np.ndarray) -> Pose:
    return pose.model_copy(update={"translate": tuple(float(v) for v in np.array(pose.translate) + by)})


def _along_face(world: np.ndarray, wall: Wall) -> np.ndarray:
    """Which points lie in front of or behind the wall's face, within its length."""
    corner, along, _, _ = wall
    across = (world - corner) @ along / (along @ along)
    return (across >= 0) & (across <= 1)


def _past_walls(world: np.ndarray, walls: list[Wall]) -> float:
    """How far the model stands past the face of any wall, in meters; 0 if it stands clear of them all."""
    furthest = 0.0
    for wall in walls:
        corner, _, normal, _ = wall
        beyond = -((world - corner) @ normal)
        past = _along_face(world, wall) & (beyond > _WALL_SLACK_M)
        if past.any():
            furthest = max(furthest, float(beyond[past].max()))
    return furthest


def settle_on_support(pose: Pose, points: np.ndarray, support_top: float | None, trim: bool) -> tuple[Pose, float]:
    """Move a model up or down so its lowest surface (see `lowest`) stands on `support_top`; returns it and the rise."""
    if support_top is None:
        return pose, 0.0
    rise = support_top - lowest(_place(pose, points)[:, 2], trim)
    return _moved(pose, np.array([0.0, 0.0, rise])), float(rise)


def share_past_walls(world: np.ndarray, walls: list[Wall]) -> float:
    """The share of the points that stand more than the slack past the face of some wall."""
    past = np.zeros(len(world), dtype=bool)
    for wall in walls:
        corner, _, normal, _ = wall
        past |= _along_face(world, wall) & (-((world - corner) @ normal) > _WALL_SLACK_M)
    return float(past.mean()) if len(world) else 0.0


def contents_id(object_id: str) -> str:
    """The name of the mask of what stands inside an item (see `contents_holes`), listed with its occluders."""
    return f"{object_id}_contents"


def contents_holes(mask: np.ndarray, kind: str | None) -> np.ndarray | None:
    """Where things stand inside a large item, as the holes its mask encloses; None if it has none to speak of.

    A shelf's mask leaves out its books, and they are not placed until stage 3:
    until then the holes are what stands in front of it, and are judged as an
    occluder's pixels are, not as the shelf's absence. Only for `HOLDS_KINDS`,
    since a chair's back or a stool's handle is a real opening. Where the
    photo's edge cuts the item, the edge closes its holes too: the Garage's
    utility sink, cut by the right edge, holds a bin on its shelf in a hole
    open only to the frame.
    """
    if kind not in HOLDS_KINDS:
        return None
    closed = mask.copy()
    for edge in (closed[:, 0], closed[:, -1], closed[0, :], closed[-1, :]):
        touching = np.nonzero(edge)[0]
        if len(touching):
            edge[touching[0]:touching[-1] + 1] = True
    filled = binary_fill_holes(closed)
    holes = filled & ~mask
    return holes if holes.sum() >= CONTENTS_SHARE * filled.sum() else None


def bottom_short(mask: np.ndarray, bbox_2d: list[float]) -> float:
    """How far the mask's lowest row stops above the bottom of the VLM's box (0-1000 across the photo), as a share
    of the box's height; at or under 0 where it reaches the box's bottom."""
    top, bottom = bbox_2d[1] / 1000 * mask.shape[0], bbox_2d[3] / 1000 * mask.shape[0]
    rows = np.nonzero(mask.any(axis=1))[0]
    return float((bottom - (rows.max() + 1)) / max(bottom - top, 1.0)) if len(rows) else 0.0


def cut_below_frame(mask: np.ndarray, corners_px: np.ndarray) -> int:
    """How many rows (in the mask's pixels) of a floor-standing item the photo's bottom edge cuts off, from its box's
    corners projected into the mask's frame (x, y; y may run past the photo); 0 where the mask does not reach the
    bottom edge or the box ends within `BOTTOM_SHORT` of its projected height below it. The VLM's box stops at the
    frame too, so `bottom_short` cannot see this: Coffee's round table showed only its top, its pedestal below."""
    if not mask[-1].any():
        return 0
    top, bottom = float(corners_px[:, 1].min()), float(corners_px[:, 1].max())
    rows = bottom - mask.shape[0]
    return int(np.ceil(rows)) if rows > BOTTOM_SHORT * max(bottom - top, 1.0) else 0


def below_mask(mask: np.ndarray, bbox_2d: list[float]) -> np.ndarray:
    """The part of the VLM's box below the mask's lowest row: where the hidden bottom of an item that is
    `bottom_short` lies, so that a redraw may keep what it draws there."""
    height, width = mask.shape
    left, right = round(bbox_2d[0] / 1000 * width), round(bbox_2d[2] / 1000 * width)
    bottom = round(bbox_2d[3] / 1000 * height)
    rows = np.nonzero(mask.any(axis=1))[0]
    area = np.zeros_like(mask, dtype=bool)
    if len(rows):
        area[rows.max() + 1:bottom, left:right] = True
    return area


def cut_by_frame(mask: np.ndarray, box_size: tuple[float, float, float], listed: tuple[float, float, float]) -> bool:
    """Whether the photo's side edge cuts an item, and with it the box measured for it (see `CUT_BOX_SHARE`)."""
    on_edge = mask[:, 0].any() or mask[:, -1].any()
    return bool(on_edge and max(box_size[:2]) < CUT_BOX_SHARE * max(listed[:2]))


def hold_to_shell(
    pose: Pose,
    points: np.ndarray,
    facets: tuple[np.ndarray, ...] | None,
    walls: list[Wall],
    support_top: float | None,
    trim: bool = True,
    seen: np.ndarray | None = None,
    square: bool = True,
    back: bool = True,
) -> tuple[Pose, ShellMoves]:
    """Move a solved model of a large item to where the shell allows it to stand.

    In order: stood on its support (`support_top`, the floor's 0 or a
    built-in's top; None for an item on a wall); turned about the vertical to
    lie square to the wall nearest in direction, if within
    `SQUARE_WITHIN_DEG` and not `square` False (the photo shows the item
    standing at an angle; see `seen_turn_deg`); moved out along each wall's normal so that none of it
    stands past a wall's face (a hard limit, where the old `wall_weight` was a
    penalty); and moved back against a wall it stands within `BACK_WITHIN_M`
    of, unless `back` is False. A model more than `CLEAR_WITHIN_M` past a wall is instead returned
    stood on its support only: marked `beyond_walls` if more than
    `BEYOND_SHARE` of `seen`, the item's own depth points in the world frame,
    stand past a wall too, and otherwise `far_past`, for the caller to pose
    again. Without `seen` it is pushed out as a smaller overlap is.
    The pose's scores are left as they were: the caller judges the new pose
    against the photo.
    """
    moves = ShellMoves()
    pose, rise = settle_on_support(pose, points, support_top, trim)
    moves.settled_m = round(rise, 3)

    if walls:
        rotation = matrix_from_euler_xyz_deg(pose.rotate_xyz_deg)
        if facets is not None and square:
            angle = _upright_faces_angle_deg(facets, rotation)
            # Each wall's direction, folded onto a quarter turn as the faces' angle is.
            turns = [(np.degrees(np.arctan2(along[1], along[0])) - angle + 45) % 90 - 45 for _, along, _, _ in walls]
            turn = min(turns, key=abs)
            if abs(turn) <= SQUARE_WITHIN_DEG:
                world = _place(pose, points)
                centre = (world.min(axis=0) + world.max(axis=0)) / 2
                centre[2] = 0.0
                spin = rotation_z(turn)
                pose = pose.model_copy(update={
                    "rotate_xyz_deg": euler_xyz_deg(spin @ rotation),
                    "translate": tuple(float(v) for v in centre + spin @ (np.array(pose.translate) - centre)),
                })
                moves.turned_deg = round(float(turn), 1)

        if seen is not None and _past_walls(_place(pose, points), walls) > CLEAR_WITHIN_M:
            if share_past_walls(seen, walls) > BEYOND_SHARE:
                moves.beyond_walls = True
            else:
                moves.far_past = True
            return pose, moves

        # Past a wall: two passes, since clearing one wall of a corner can push into the other.
        for _ in range(2):
            for wall in walls:
                corner, _, normal, _ = wall
                world = _place(pose, points)
                beyond = -((world - corner) @ normal)
                past = _along_face(world, wall) & (beyond > _WALL_SLACK_M)
                if past.any():
                    push = float(beyond[past].max())
                    pose = _moved(pose, normal * push)
                    moves.cleared_m = round(moves.cleared_m + push, 3)

        # Backed against the nearest wall it nearly touches, unless that would put it past another.
        if not back:
            return pose, moves
        world = _place(pose, points)
        gaps = []
        for wall in walls:
            corner, _, normal, _ = wall
            facing = _along_face(world, wall)
            if facing.sum() > 0.5 * len(world):
                gaps.append((float(((world[facing] - corner) @ normal).min()), normal))
        if gaps:
            gap, normal = min(gaps, key=lambda g: g[0])
            if 0 < gap <= BACK_WITHIN_M:
                backed = _moved(pose, -normal * gap)
                moved = _place(backed, points)
                if not any((_along_face(moved, w) & (-((moved - w[0]) @ w[2]) > _WALL_SLACK_M)).any() for w in walls):
                    pose, moves.backed_m = backed, round(gap, 3)
    return pose, moves


def square_to_built_in(
    box: Proxy,
    world: np.ndarray,
    mask: np.ndarray,
    built_ins: list[Proxy],
    listed: tuple[float, float, float],
    eye: np.ndarray,
) -> tuple[Proxy, str] | None:
    """The item's box rebuilt from the outline of a built-in standing over it; with that built-in's id, or None.

    An item boxed in under a built-in, as a fridge under the cabinet over it, shares that built-in's outline: it
    is turned as the built-in is, its back stands where the built-in's does, against the wall, and it fits below
    it. The built-in's plain front gives its heading better than the item's own face (the kitchen's fridge,
    papered in magnets, read 69 deg on its doors and 71 on its drawer; the cabinet over it 72.3); and its back
    gives the item's depth, which the camera cannot see: `fit_proxies` pads that out to the listed depth. The
    front and sides are the item's own depth points (`world`, the world-frame points per pixel; `mask`, its
    pixels), except a side the photo's frame cuts off, which is extended to the listed width (`listed`, width,
    depth, height) from the side that shows. The camera stands at `eye`. None if no built-in stands over the
    item, or the depth comes out more than `SQUARED_DEPTH_WITHIN` from the listed one.
    """
    seen = world[mask]
    keep = np.isfinite(seen).all(axis=1)
    seen = seen[keep]
    if len(seen) == 0:
        return None
    top = box.translate[2] + box.size[2]
    best = None
    for built_in in built_ins:
        bottom = built_in.translate[2]
        if not top - OVER_SLACK_M <= bottom <= top + OVER_WITHIN_M:
            continue
        yaw = np.radians(built_in.yaw_deg)
        depth_axis, width_axis = np.array([np.cos(yaw), np.sin(yaw)]), np.array([-np.sin(yaw), np.cos(yaw)])
        centre = np.array(built_in.translate[:2])
        across = (seen[:, :2] - centre) @ width_axis
        share = float((np.abs(across) <= built_in.size[1] / 2 + _WALL_SLACK_M).mean())
        if share >= OVER_SHARE and (best is None or share > best[0]):
            best = (share, built_in, depth_axis, width_axis, centre)
    if best is None:
        return None
    _, built_in, depth_axis, width_axis, centre = best

    # Out of the built-in's front, towards the camera; its back is half its depth behind its centre.
    out = -depth_axis * np.sign((centre - eye[:2]) @ depth_axis)
    back = -built_in.size[0] / 2
    front = float(np.percentile((seen[:, :2] - centre) @ out, 98))
    depth = front - back
    listed_width, listed_depth = max(listed[0], listed[1]), min(listed[0], listed[1])
    if not 1 / SQUARED_DEPTH_WITHIN <= depth / listed_depth <= SQUARED_DEPTH_WITHIN:
        return None

    across = (seen[:, :2] - centre) @ width_axis
    low, high = np.percentile(across, [2, 98])
    short = listed_width - (high - low)
    if short > 0:
        # Which end of the item the frame cuts, if either: where its points at the frame's edge lie.
        cut_low = cut_high = False
        for column in (0, mask.shape[1] - 1):
            edge = world[:, column][mask[:, column]]
            edge = edge[np.isfinite(edge).all(axis=1)]
            if len(edge):
                at = float(np.median((edge[:, :2] - centre) @ width_axis))
                cut_low, cut_high = cut_low or at - low < high - at, cut_high or at - low >= high - at
        if cut_low != cut_high:
            low, high = (low - short, high) if cut_low else (low, high + short)

    middle = centre + out * (front + back) / 2 + width_axis * (low + high) / 2
    height = min(box.size[2], built_in.translate[2] - box.translate[2])
    return box.model_copy(update={
        "size": (float(depth), float(high - low), float(height)),
        "translate": (float(middle[0]), float(middle[1]), box.translate[2]),
        "yaw_deg": built_in.yaw_deg,
    }), built_in.id


def distance_ratio(world: np.ndarray, mask: np.ndarray, distances: np.ndarray, camera_matrix: np.ndarray, focal_px: float) -> float | None:
    """The median, over the mask's pixels the model covers, of its nearest surface's distance along the camera's axis
    over the depth map's (`distances`, NaN where unknown); None where it covers fewer than `OFF_DEPTH_MIN_PIXELS`."""
    height, width = mask.shape
    local = (world - camera_matrix[:3, 3]) @ camera_matrix[:3, :3]
    depth = -local[:, 2]
    ahead = depth > 0.05
    cols = np.floor(width / 2 + focal_px * local[ahead, 0] / depth[ahead]).astype(int)
    rows = np.floor(height / 2 - focal_px * local[ahead, 1] / depth[ahead]).astype(int)
    inside = (cols >= 0) & (cols < width) & (rows >= 0) & (rows < height)
    nearest = np.full(mask.shape, np.inf)
    np.minimum.at(nearest, (rows[inside], cols[inside]), depth[ahead][inside])
    judged = mask & np.isfinite(nearest) & np.isfinite(distances) & (distances > 0)
    if judged.sum() < OFF_DEPTH_MIN_PIXELS:
        return None
    return float(np.median(nearest[judged] / distances[judged]))


def seen_turn_deg(world: np.ndarray, mask: np.ndarray, walls: list[Wall]) -> float | None:
    """How far the item's own faces in the photo turn from the nearest wall's direction, within 45 degrees either way.

    Read from the depth map, not the model, so it tells a badly turned model from a thing that really stands
    at an angle (the office cabinet, 35 deg off the walls, its model squared to them by the review). `world` is
    the world-frame points per pixel, `mask` the item's pixels: the depth surface's normals inside the mask,
    `_SEEN_ERODE_PX` off its outline, where depth runs onto what stands behind, folded onto a quarter turn by
    area (`squareness`). Within `SQUARE_WITHIN_DEG`, the item stands square to the walls. None without walls,
    or where the faces do not agree on a heading (`SEEN_SHARE`, over `SEEN_MIN_CELLS`): a round thing, or one
    too little of which shows.
    """
    from scipy.ndimage import binary_erosion

    from aomma.stages.square import fold_deg, squareness  # square imports this module

    if not walls:
        return None
    step = _SEEN_STEP_PX
    inner = binary_erosion(mask, iterations=_SEEN_ERODE_PX)
    corner, right, down = world[:-step, :-step], world[:-step, step:], world[step:, :-step]
    keep = inner[:-step, :-step] & inner[:-step, step:] & inner[step:, :-step]
    cross = np.cross(right - corner, down - corner)[keep]
    cross = cross[np.isfinite(cross).all(axis=1)]
    areas = np.linalg.norm(cross, axis=1)
    cross, areas = cross[areas > 0], areas[areas > 0]
    if len(areas) < SEEN_MIN_CELLS:
        return None
    heading, share = squareness(cross / areas[:, None], areas)
    if share < SEEN_SHARE:
        return None
    return round(float(min((fold_deg(heading - np.degrees(np.arctan2(along[1], along[0]))) for _, along, _, _ in walls), key=abs)), 1)


def square_turn_deg(
    facets: tuple[np.ndarray, ...], rotation: np.ndarray, walls: list[Wall], seen_turn: float = 0.0
) -> float | None:
    """The turn about the vertical, within 45 degrees either way, that turns a placed model's upright faces onto the
    nearest wall's direction turned by `seen_turn` (0, square to the wall; see `seen_turn_deg`).

    None without walls, or for a model that is not box-shaped (see `squareness`):
    a round thing's faces agree on no direction, so any angle read from them is noise.
    """
    from aomma.stages.square import SQUARE_SHARE, squareness  # square imports this module

    if not walls:
        return None
    angle, share = squareness(facets[0] @ rotation.T, facets[1])
    if share < SQUARE_SHARE:
        return None
    return round(float(min(
        ((np.degrees(np.arctan2(along[1], along[0])) + seen_turn - angle + 45) % 90 - 45 for _, along, _, _ in walls), key=abs,
    )), 1)


def off_square_deg(facets: tuple[np.ndarray, ...], rotation: np.ndarray, walls: list[Wall]) -> float | None:
    """How far a placed model is turned from square to the walls, in degrees (see `square_turn_deg`)."""
    turn = square_turn_deg(facets, rotation, walls)
    return None if turn is None else abs(turn)
