"""`aomma ref-shell`: what each pixel of an input shot is, as far as the room's shell goes, from the scan alone
(gate 1's reference).

RoomPlan's walls, floor, doors, windows and openings give the room's surfaces, but not to the centimetre: some of
KitchenScan's walls stand 5-15 cm behind where RoomPlan puts them. So each surface is first
moved onto the shot's own LiDAR, by one fact about rooms: **nothing solid stands behind a wall, floor or ceiling.** Of
the shot's sure LiDAR points near a surface (within `SNAP_BAND_M`, inside its extents, outside its doors and windows),
the surface is the deepest dense layer (`deepest_layer`): furniture adds layers in front of it, never behind; a spread
of points with no layer (a surface seen edge on) is not one. The plane is then fitted through that layer (`fit_surface`):
a wall may turn about the vertical by up to `MAX_TILT`, as a shot may see it a degree or two off RoomPlan's; a floor
or ceiling stays level. A surface with no such layer is unseen, and its pixels
are not scored. RoomPlan has no ceiling: it is taken level, at the walls' median top, and snapped the same way.

Each pixel of the LiDAR map (its own size, 256 x 192; sure pixels only) is then:
- `floor`, `ceiling` or `wall:<i>`: its point within `ON_M` of that snapped surface (the nearest, at a corner);
- `opening:<j>`: inside a door, window or opening of a wall, at the wall's plane or beyond it;
- `fixture` or `appliance`: inside one of RoomPlan's fixture boxes (`fixtures`: cabinets in a run, sinks, appliances;
  storage standing alone is furniture, gate 2's), grown by `BOX_MARGIN_M`, on no surface and behind no seen wall;
- `in_front`: in front of every surface, in no fixture: furniture and things;
- `unsure`: behind a snapped surface outside any opening (a mirror's reflection, the LiDAR's own error), or on a
  surface that was not seen.
No run, and no segmenter, has a hand in it.

Writes `refs/<scan>/<shot>/shell_labels.png` (the codes below) and `shell_labels.json` (what each code is, and each
surface's snap), for every input shot that passed the scan check. A shot whose pose failed gate 0 gets one too, drawn
by that pose, so its judges carry a `pose_off` note.
"""

import json
from pathlib import Path

import numpy as np
from PIL import Image

from aomma.capture import REFS, Scan, View, _matrix, to_frame
from aomma.posecheck import _inside_polygon
from aomma.scancheck import usable_shots
from aomma.stages.roomplan import APPLIANCE_KINDS, FIXTURE_KINDS, TOUCH_M, _gap, category

# How far a surface may be moved onto the shot's LiDAR. Past this, a layer is something else's (KitchenScan's upper
# cabinets stand about 30 cm out of their wall).
SNAP_BAND_M = 0.25
# A point this close to a snapped surface lies on it: about the LiDAR's own spread at a few metres.
ON_M = 0.04
# Fixture boxes are grown by this, as RoomPlan's boxes are a few cm off what they hold.
BOX_MARGIN_M = 0.05
# A layer holds at least this many points within `_LAYER_HALF_M` of its centre...
MIN_LAYER_POINTS = 100
# ...and this many times what lies `_FLANK_M` before and behind it: a surface seen edge on spreads its points over
# every distance and is not a layer.
LAYER_CONTRAST = 3.0
_LAYER_HALF_M = 0.02
_FLANK_M = 0.10  # past the ~10 cm a turned wall smears its layer over (KitchenScan 008 wall_0)
# Past a wall's edges, a point may lie this far and still count for it.
_EDGE_M = 0.1

# A snapped wall may turn this far from RoomPlan's about the vertical (as a slope, m per m: about 3 degrees), to follow
# a wall the shot sees a degree or two off RoomPlan's (BedroomScan 010's back wall).
MAX_TILT = 0.05

UNSCORED, IN_FRONT, FIXTURE, UNSURE, APPLIANCE, FLOOR, CEILING = 0, 1, 2, 3, 4, 10, 11
WALL_BASE, OPENING_BASE = 20, 120


def deepest_layer(distances: np.ndarray) -> float | None:
    """The deepest dense layer among signed distances (m, positive toward the room) within the band, refined as the
    median of the points within `_LAYER_HALF_M` of it; None if there is none."""
    d = distances[np.abs(distances) <= SNAP_BAND_M]
    if len(d) < MIN_LAYER_POINTS:
        return None
    step = 0.005
    centres = np.arange(-SNAP_BAND_M, SNAP_BAND_M + step / 2, step)
    # The surroundings are counted past the band's ends too: cut there, a layer near an end would stand out from
    # nothing (KitchenScan 005's wall_7, seen edge on, at +15 cm).
    ordered = np.sort(distances[np.abs(distances) <= SNAP_BAND_M + _FLANK_M + _LAYER_HALF_M])
    def window(at: np.ndarray) -> np.ndarray:
        return np.searchsorted(ordered, at + _LAYER_HALF_M) - np.searchsorted(ordered, at - _LAYER_HALF_M)

    counts = window(centres)
    # Against its own surroundings, not the whole band: a wall seen in fewer points than the cabinets in front of it
    # is still a layer (KitchenScan's wall_9).
    flanks = (window(centres - _FLANK_M) + window(centres + _FLANK_M)) / 2
    dense = (counts >= MIN_LAYER_POINTS) & (counts >= LAYER_CONTRAST * np.maximum(flanks, 1))
    if not dense.any():
        return None
    # The deepest run of dense centres, and its fullest centre.
    first = int(np.argmax(dense))
    last = first
    while last + 1 < len(dense) and dense[last + 1]:
        last += 1
    centre = centres[first + int(np.argmax(counts[first:last + 1]))]
    near = d[np.abs(d - centre) <= _LAYER_HALF_M]
    return float(np.median(near))


def _surfaces(room: dict) -> list[dict]:
    """The room's walls, floors and a level ceiling, each as a frame (local z its normal) with its extents and holes."""
    cut = [c for kind in ("doors", "windows", "openings") for c in room.get(kind, [])]
    found = []
    walls = room.get("walls", [])
    for index, wall in enumerate(walls):
        found.append({"id": f"wall_{index}", "kind": "wall", "code": WALL_BASE + index, "transform": _matrix(wall["transform"]),
                      "size": wall["dimensions"][:2], "corners": [],
                      "holes": [(j, c) for j, c in enumerate(cut) if c.get("parentIdentifier") == wall["identifier"]]})
    for floor in room.get("floors", [])[:1]:
        transform = _matrix(floor["transform"])
        found.append({"id": "floor", "kind": "floor", "code": FLOOR, "transform": transform, "size": floor["dimensions"][:2],
                      "corners": floor.get("polygonCorners") or [], "holes": []})
        if walls:
            # Level, at the walls' median top: each wall's centre is half its height above its foot.
            tops = [_matrix(w["transform"])[1, 3] + w["dimensions"][1] / 2 for w in walls]
            ceiling = transform.copy()
            ceiling[:3, 3] = transform[:3, 3] + np.array([0.0, float(np.median(tops)) - transform[1, 3], 0.0])
            found.append({"id": "ceiling", "kind": "ceiling", "code": CEILING, "transform": ceiling,
                          "size": floor["dimensions"][:2], "corners": floor.get("polygonCorners") or [], "holes": []})
    return found


def _signed(points: np.ndarray, surface: dict, camera: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[np.ndarray], np.ndarray]:
    """Each point's signed distance from the surface (positive toward the camera), whether it lies within the surface's
    extents, per hole whether it lies within that hole's rectangle, and its place in the surface's plane (x, y)."""
    local = to_frame(points, np.linalg.inv(surface["transform"]))
    side = np.sign(to_frame(camera[None], np.linalg.inv(surface["transform"]))[0, 2]) or 1.0
    width, height = surface["size"]
    if surface["corners"]:
        within = _inside_polygon(local[:, :2], surface["corners"])
    else:
        within = (np.abs(local[:, 0]) <= width / 2 + _EDGE_M) & (np.abs(local[:, 1]) <= height / 2 + _EDGE_M)
    holes = []
    for _, hole in surface["holes"]:
        h = to_frame(points, np.linalg.inv(_matrix(hole["transform"])))
        holes.append((np.abs(h[:, 0]) <= hole["dimensions"][0] / 2) & (np.abs(h[:, 1]) <= hole["dimensions"][1] / 2))
    return local[:, 2] * side, within, holes, local[:, :2]


def _seen_through(points: np.ndarray, surface: dict, camera: np.ndarray) -> np.ndarray:
    """Whether the camera's line of sight to each point crosses the surface's plane within its extents and outside its
    holes: a point behind the plane that the camera sees past the wall's end (a fridge beside an archway's pier) is not
    behind the wall, though it lies behind its plane."""
    inverse = np.linalg.inv(surface["transform"])
    local, eye = to_frame(points, inverse), to_frame(camera[None], inverse)[0]
    with np.errstate(divide="ignore", invalid="ignore"):
        t = eye[2] / (eye[2] - local[:, 2])
    cross = eye + t[:, None] * (local - eye)
    width, height = surface["size"]
    if surface["corners"]:
        inside = _inside_polygon(cross[:, :2], surface["corners"])
    else:
        inside = (np.abs(cross[:, 0]) <= width / 2 + _EDGE_M) & (np.abs(cross[:, 1]) <= height / 2 + _EDGE_M)
    inside &= (t > 0) & (t < 1)
    world = to_frame(cross, surface["transform"])
    for _, hole in surface["holes"]:
        h = to_frame(world, np.linalg.inv(_matrix(hole["transform"])))
        inside &= ~((np.abs(h[:, 0]) <= hole["dimensions"][0] / 2) & (np.abs(h[:, 1]) <= hole["dimensions"][1] / 2))
    return inside


def fit_surface(d: np.ndarray, xy: np.ndarray, start: float, turns: bool) -> tuple[float, float]:
    """The plane d = c + a x through the points near the layer found at `start` (x along a wall), refitted three times
    on the points within `ON_M` of it, its slope held within `MAX_TILT`. Only a wall `turns`, and only about the
    vertical: walls are plumb, and a floor or ceiling is level (ARKit's up is gravity). Returns (c, a)."""
    c, a = start, 0.0
    for _ in range(3):
        near = np.abs(d - (c + a * xy[:, 0])) <= ON_M
        if near.sum() < MIN_LAYER_POINTS:
            break
        if turns:
            a = float(np.clip(np.polyfit(xy[near, 0], d[near], 1)[0], -MAX_TILT, MAX_TILT))
        c = float(np.median(d[near] - a * xy[near, 0]))
    return c, a


def _plan(element: dict) -> np.ndarray:
    """A RoomPlan box's corners in plan (ARKit's x, z), (4, 2)."""
    transform = _matrix(element["transform"])
    w, _, d = element["dimensions"]
    corners = np.array([[sx * w / 2, 0.0, sz * d / 2] for sx, sz in ((1, 1), (1, -1), (-1, -1), (-1, 1))])
    return to_frame(corners, transform)[:, [0, 2]]


def fixtures(room: dict) -> list[tuple[dict, int]]:
    """RoomPlan's boxes that belong to the shell's reference, each with its code: `APPLIANCE` for an appliance (a run may
    hold it as a built-in or as a large item: both are right), `FIXTURE` for the rest. Storage counts only when it
    touches another fixture box (a run of cabinets); alone it is furniture (the office cabinet, the bedroom dresser),
    left to gate 2. This is phase 2's storage rule less its survey half, which a run's own listing decides."""
    boxes = [o for o in room.get("objects", []) if category(o) in FIXTURE_KINDS]
    plans = [_plan(o) for o in boxes]
    found = []
    for i, element in enumerate(boxes):
        kind = category(element)
        if kind == "storage" and not any(j != i and _gap(plans[i], plans[j]) < TOUCH_M for j in range(len(boxes))):
            continue
        found.append((element, APPLIANCE if kind in APPLIANCE_KINDS else FIXTURE))
    return found


def _in_fixture(points: np.ndarray, room: dict) -> np.ndarray:
    """Each point's fixture code (`FIXTURE` or `APPLIANCE`), 0 where it lies in none."""
    code = np.zeros(len(points), np.uint8)
    for element, kind in fixtures(room):
        local = to_frame(points, np.linalg.inv(_matrix(element["transform"])))
        half = np.asarray(element["dimensions"]) / 2 + BOX_MARGIN_M
        code[(np.abs(local) <= half).all(axis=1) & (code == 0)] = kind
    return code


def shell_labels(scan: Scan, view: View) -> tuple[np.ndarray, dict]:
    """The shot's reference shell labels at its LiDAR map's size, and what each surface's snap was."""
    depth, conf = scan.depth(view)
    labels = np.full(depth.shape, UNSCORED, np.uint8)
    points, pixels = scan.lidar_world(view)
    camera = view.depth_to_world[:3, 3]
    room = scan.room
    code = np.full(len(points), IN_FRONT, np.uint8)
    best = np.full(len(points), np.inf)
    behind = np.zeros(len(points), bool)
    behind_wall = np.zeros(len(points), bool)
    unseen = np.zeros(len(points), bool)
    in_hole = np.zeros(len(points), bool)
    hole_code = np.zeros(len(points), np.uint8)
    snaps = {}
    for surface in _surfaces(room):
        d, within, holes, xy = _signed(points, surface, camera)
        solid = within.copy()
        for h in holes:
            solid &= ~h
        layer = deepest_layer(d[solid])
        snap = {"code": surface["code"], "kind": surface["kind"], "points": int((solid & (np.abs(d) <= SNAP_BAND_M)).sum()),
                "moved_m": None, "tilt": None}
        snaps[surface["id"]] = snap
        if layer is None:
            # Not seen: what lies on RoomPlan's plane can't be told from what stands near it.
            unseen |= solid & (np.abs(d) <= ON_M)
            continue
        c, a = fit_surface(d[solid], xy[solid], layer, surface["kind"] == "wall")
        off = d - (c + a * xy[:, 0])
        seen_here = solid & (np.abs(off) <= ON_M)
        # How far the surface moved where the shot sees it: + toward the room (the shot sees it in front of RoomPlan's).
        snap["moved_m"] = round(float(np.median(c + a * xy[seen_here, 0])) if seen_here.any() else c, 4)
        snap["tilt"] = round(a, 4)
        on = solid & (np.abs(off) <= ON_M) & (np.abs(off) < best)
        code[on], best[on] = surface["code"], np.abs(off[on])
        behind |= solid & (off < -ON_M)
        if surface["kind"] == "wall":
            behind_wall |= solid & (off < -ON_M) & _seen_through(points, surface, camera)
        for (j, _), h in zip(surface["holes"], holes):
            through = within & h & (off <= ON_M)
            in_hole |= through
            hole_code[through] = OPENING_BASE + j
    on_surface = np.isfinite(best)
    code[~on_surface & in_hole] = hole_code[~on_surface & in_hole]
    rest = ~on_surface & ~in_hole
    fixture = _in_fixture(points, room)
    # A fixture stands in the room, in front of its wall: a point behind a wall the shot sees is not in it, however
    # far the box (grown by `BOX_MARGIN_M`) reaches. RoomPlan's walls have no thickness, so a wall's end face at an
    # archway lies behind the wall's plane (KitchenScan 008's jamb, in the fridge's box).
    in_box = rest & (fixture > 0) & ~behind_wall
    code[in_box] = fixture[in_box]
    code[rest & (behind | unseen) & (code == IN_FRONT)] = UNSURE
    labels[pixels[:, 0], pixels[:, 1]] = code
    return labels, snaps


def reference_shell(scan: Scan, refs: Path = REFS) -> dict[str, dict]:
    """Shell labels for every input shot that passed the scan check, its pose good or not; writes them beside the
    shot's reference."""
    usable = usable_shots(scan.name, refs, poses=False)
    room = scan.room
    cut = [c for kind in ("doors", "windows", "openings") for c in room.get(kind, [])]
    meaning = {UNSCORED: "unscored (no sure LiDAR)", IN_FRONT: "in front", FIXTURE: "fixture", APPLIANCE: "appliance", UNSURE: "unsure",
               FLOOR: "floor", CEILING: "ceiling"}
    meaning |= {WALL_BASE + i: f"wall_{i}" for i in range(len(room.get("walls", [])))}
    meaning |= {OPENING_BASE + j: f"opening_{j} ({category(c) if c.get('category') else 'opening'})" for j, c in enumerate(cut)}
    made = {}
    for view in scan.shots:
        if view.role != "input" or not view.depth_size or (usable is not None and view.name not in usable):
            continue
        labels, snaps = shell_labels(scan, view)
        folder = refs / scan.name / view.name
        folder.mkdir(parents=True, exist_ok=True)
        Image.fromarray(labels).save(folder / "shell_labels.png")
        scored = labels != UNSCORED
        shares = {meaning.get(int(c), str(c)): round(float((labels[scored] == c).mean()), 4) for c in np.unique(labels[scored])}
        made[view.name] = {"codes": {str(k): v for k, v in meaning.items()}, "snaps": snaps, "shares": shares,
                           "on": {"on_m": ON_M, "snap_band_m": SNAP_BAND_M, "box_margin_m": BOX_MARGIN_M}}
        (folder / "shell_labels.json").write_text(json.dumps(made[view.name], indent=2))
    return made
