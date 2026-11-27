"""The shell from RoomPlan.

RoomPlan gives the room in ARKit's world (+y up): walls, doors, windows and openings, the floor, and object boxes. A
run's world is Z up with the floor at 0 and its camera over the origin (`stages.world.fit_camera_to_world`). Here the
camera's height comes from RoomPlan's floor, put right by the shot's own LiDAR where it sees the floor, and every wall
and built-in is carried into the run's world by the shot's pose. A wall the shot's LiDAR sees is moved onto it: the run
is given only its own shot's measurements and RoomPlan, never the scan's mesh, which holds the reference shots' LiDAR.

Of RoomPlan's objects only the fixtures are given (`FIXTURE_KINDS`); the furniture boxes are `depth+objects`' input.
RoomPlan's `storage` is both built-in cabinets and furniture (the office cabinet, the bedroom dresser), so a storage
box is built in only where it stands in a run with other fixtures, or where the survey lists a built-in in it
(`is_built_in`). Checked by eye on OfficeScan, BedroomScan and KitchenScan.

`depth+objects` (phase 3, build step 8) also gives the furniture boxes, and stage 2 takes a box as a listed large item's
proxy where they match (`match_objects`): the box's category fits the item's kind and most of the item's own points lie
in it, and none of them reach past it.
"""

import numpy as np

from aomma.stages.proxy import Opening, Proxy

# RoomPlan's categories of things fixed in a room. Storage is decided by `is_built_in`.
FIXTURE_KINDS = ("storage", "sink", "stove", "oven", "refrigerator", "dishwasher", "washerDryer", "fireplace", "bathtub", "toilet")
# Appliances a photo's stage 2 lists and generates as large items when it sees them.
APPLIANCE_KINDS = ("stove", "oven", "refrigerator", "dishwasher", "washerDryer")
# RoomPlan's furniture: `depth+objects`' input, with the fixtures `is_built_in` leaves to stage 2.
FURNITURE_KINDS = ("bed", "sofa", "chair", "table", "television")
# Which of RoomPlan's categories may be the box of a large item of each of stage 2's kinds (`stages.large.KINDS`):
# RoomPlan's vocabulary set against stage 2's. An item of a kind not here (exercise, other) has no box.
BOX_KINDS = {
    "bed": ("bed",), "sofa": ("sofa",), "armchair": ("sofa", "chair"), "chair": ("chair",), "bench": ("chair", "sofa"),
    "table": ("table",), "desk": ("table",), "nightstand": ("table", "storage"),
    "dresser": ("storage",), "cabinet": ("storage",), "shelving": ("storage",),
    "appliance": APPLIANCE_KINDS + ("television",),
}
# A box is an item's when more than this share of the item's points lie in it (grown by `IN_BOX_M`).
MATCH_SHARE = 0.5
# Of an item's points, those past these percentiles along each of the box's axes say how far it reaches: past the box
# grown by `IN_BOX_M` on any side, the box is refused (a mask's stray pixels on the wall behind are not counted).
REACH_PERCENTILES = (5, 95)
# A box is seen by the shot if this many of its LiDAR points lie in it.
MIN_SEEN_POINTS = 50
# A wall slab's thickness, behind the face RoomPlan gives (as `stages.room`'s fitted walls).
WALL_THICKNESS_M = 0.05
# Boxes this close in plan stand in one run.
TOUCH_M = 0.05
# A survey built-in is in a box if the median of its mask's points lies within the box and this margin.
IN_BOX_M = 0.10
# The shot's LiDAR moves a wall or the floor only with this many points on it, and only this far.
MIN_SNAP_POINTS = 200
MAX_SNAP_M = 0.10
# A LiDAR point is on a wall if it is within this of RoomPlan's plane, inside its span and above its foot and below
# its top by this much (skirting, furniture against it, the ceiling's edge).
SNAP_BAND_M = 0.10
SNAP_INSET_M = 0.30


def given_room(room: dict) -> dict:
    """What `depth+roomplan` is given of a RoomPlan room: everything but the furniture's boxes."""
    return room | {"objects": [o for o in room.get("objects", []) if category(o) in FIXTURE_KINDS]}


def walls_only(room: dict) -> dict:
    """What `depth+walls` is given of a RoomPlan room: its walls, floor, doors, windows and openings, no boxes."""
    return room | {"objects": []}


def category(element: dict) -> str:
    return next(iter(element["category"]))


def matrix(element: dict) -> np.ndarray:
    """A RoomPlan element's 4x4 transform (ARKit's column-major list)."""
    return np.array(element["transform"], float).reshape(4, 4).T


def floor_y(room: dict) -> float:
    """The floor's height in ARKit's world: RoomPlan's floor, else the walls' feet."""
    if room.get("floors"):
        return float(matrix(room["floors"][0])[1, 3])
    return float(np.median([matrix(w)[1, 3] - w["dimensions"][1] / 2 for w in room["walls"]]))


def floor_polygon(room: dict, scan_to_run: np.ndarray) -> np.ndarray | None:
    """RoomPlan's floor outline in the run's plan (N, 2), or None."""
    if not room.get("floors") or not room["floors"][0].get("polygonCorners"):
        return None
    floor = room["floors"][0]
    corners = np.array([[x, y, 0.0, 1.0] for x, y, *_ in floor["polygonCorners"]]) @ matrix(floor).T
    return (corners @ scan_to_run.T)[:, :2]


def inside(point: np.ndarray, polygon: np.ndarray) -> bool:
    """Whether a plan point lies inside a polygon (ray casting)."""
    x, y = point
    hit = False
    for (x0, y0), (x1, y1) in zip(polygon, np.roll(polygon, -1, axis=0)):
        if (y0 > y) != (y1 > y) and x < x0 + (y - y0) * (x1 - x0) / (y1 - y0):
            hit = not hit
    return hit


def camera_height(shot_to_world: np.ndarray, room: dict) -> float:
    """The camera's height above RoomPlan's floor, along ARKit's up."""
    return float(shot_to_world[1, 3] - floor_y(room))


def _in_run(element: dict, scan_to_run: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """An element's centre and its x axis (along it), in the run's world."""
    m = scan_to_run @ matrix(element)
    along = m[:3, 0]
    along = along / np.linalg.norm(along)
    return m[:3, 3], along


def walls_from_room(room: dict, scan_to_run: np.ndarray, eye: np.ndarray) -> list[Proxy]:
    """RoomPlan's walls as slabs in the run's world, their room-facing face on RoomPlan's plane, with their doors,
    windows and openings cut through them.

    The room's side of a wall is the side RoomPlan's floor outline lies on, so a wall of an L-shaped room faces
    the right way though the camera is not in front of it; with no outline, the camera's side.
    """
    polygon = floor_polygon(room, scan_to_run)
    walls, faces = [], {}
    for n, element in enumerate(room["walls"], start=1):
        centre, along = _in_run(element, scan_to_run)
        length, height = element["dimensions"][0], element["dimensions"][1]
        plan_along = along[:2] / np.linalg.norm(along[:2])
        normal = np.array([-plan_along[1], plan_along[0]])
        towards = (polygon is not None and inside(centre[:2] + 0.15 * normal, polygon)) or (
            polygon is None and normal @ (eye[:2] - centre[:2]) > 0)
        if not towards:
            normal = -normal
        middle = centre[:2] - normal * WALL_THICKNESS_M / 2
        wall_id = f"wall_{n}"
        faces[element["identifier"]] = (wall_id, centre, plan_along, normal, length)
        # Carried past both ends by as far as a snap may move its neighbour, so a corner stays closed when one of its
        # walls is moved onto the LiDAR and the other is not (BedroomScan 009: 5.5 cm).
        walls.append(Proxy(
            id=wall_id, size=(float(length + 2 * (MAX_SNAP_M + WALL_THICKNESS_M)), WALL_THICKNESS_M, float(height)),
            translate=(float(middle[0]), float(middle[1]), 0.0),
            yaw_deg=float(np.degrees(np.arctan2(plan_along[1], plan_along[0]))), color=(0.8, 0.8, 0.8),
        ))
    by_id = {wall.id: wall for wall in walls}
    for kind in ("doors", "windows", "openings"):
        for n, element in enumerate(room.get(kind, []), start=1):
            parent = faces.get(element.get("parentIdentifier"))
            if parent is None:
                continue
            wall_id, wall_centre, plan_along, _, _ = parent
            centre, _ = _in_run(element, scan_to_run)
            width, height = element["dimensions"][0], element["dimensions"][1]
            # Its sides on the wall's face (RoomPlan's plane), measured along the wall from the wall's centre.
            s = float((centre[:2] - wall_centre[:2]) @ plan_along)
            ends = tuple(tuple(float(v) for v in wall_centre[:2] + plan_along * (s + side * width / 2)) for side in (-1, 1))
            window = kind == "windows"
            bottom = max(0.0, float(centre[2] - height / 2)) if window else 0.0
            wall = by_id[wall_id]
            by_id[wall_id] = wall.model_copy(update={"openings": wall.openings + [Opening(
                id=f"{kind[:-1]}_{n}", ends=ends, top=float(centre[2] + height / 2), bottom=bottom, window=window,
            )]})
    return [by_id[wall.id] for wall in walls]


def boxes_from_room(room: dict, scan_to_run: np.ndarray, kinds: tuple[str, ...] = FIXTURE_KINDS) -> list[tuple[str, Proxy]]:
    """RoomPlan's boxes of `kinds` (the fixtures by default) as (category, upright box) in the run's world, named by
    category and number among them."""
    found = []
    for n, element in enumerate((o for o in room.get("objects", []) if category(o) in kinds), start=1):
        centre, along = _in_run(element, scan_to_run)
        width, height, depth = element["dimensions"]
        found.append((category(element), Proxy(
            id=f"{category(element)}_{n}", size=(float(width), float(depth), float(height)),
            translate=(float(centre[0]), float(centre[1]), float(centre[2] - height / 2)),
            yaw_deg=float(np.degrees(np.arctan2(along[1], along[0]))), color=(0.5, 0.5, 0.5),
        )))
    return found


def footprint(box: Proxy, margin: float = 0.0) -> np.ndarray:
    """A box's corners in plan (4, 2), grown by `margin`."""
    yaw = np.radians(box.yaw_deg)
    a, c = np.array([np.cos(yaw), np.sin(yaw)]), np.array([-np.sin(yaw), np.cos(yaw)])
    half_a, half_c = box.size[0] / 2 + margin, box.size[1] / 2 + margin
    middle = np.array(box.translate[:2])
    return np.array([middle + sa * half_a * a + sc * half_c * c for sa, sc in ((1, 1), (1, -1), (-1, -1), (-1, 1))])


def _gap(p: np.ndarray, q: np.ndarray) -> float:
    """The least distance between two convex outlines in plan (0 where they overlap)."""
    if any(inside(x, q) for x in p) or any(inside(x, p) for x in q):
        return 0.0

    def to_edges(points, outline):
        best = np.inf
        for a, b in zip(outline, np.roll(outline, -1, axis=0)):
            ab = b - a
            t = np.clip(((points - a) @ ab) / (ab @ ab), 0, 1)
            best = min(best, float(np.linalg.norm(points - (a + t[:, None] * ab), axis=1).min()))
        return best
    return min(to_edges(p, q), to_edges(q, p))


def in_box(box: Proxy, point: np.ndarray, margin: float = IN_BOX_M) -> bool:
    """Whether a point (run world) lies in the box grown by `margin`, in plan and in height: a range hood or a wall
    cabinet hangs over the box below it."""
    point = np.asarray(point)
    return inside(point[:2], footprint(box, margin)) and box.translate[2] - margin <= point[2] <= box.translate[2] + box.size[2] + margin


def unlisted_seen(found: list[dict], boxes: list, points: np.ndarray) -> tuple[list[str], list[str]]:
    """RoomPlan's boxes no listed item took (`match_objects`' `found`), split into those the shot sees (`MIN_SEEN_POINTS`
    of its LiDAR `points`, run world, in the box) and those it does not: only the first are what stage 2's list missed.
    Counted over the whole room, BedroomScan 008-010 listed the bed they do not show."""
    by_box = {box.id: box for _, box in boxes}
    free = [f["box"] for f in found if f["item"] is None]
    seen = [b for b in free if in_box_points(by_box[b], points).sum() >= MIN_SEEN_POINTS]
    return seen, [b for b in free if b not in seen]


def in_box_points(box: Proxy, points: np.ndarray) -> np.ndarray:
    """Which of `points` (..., 3; run world) lie inside the box."""
    yaw = np.radians(box.yaw_deg)
    rel = points[..., :2] - np.array(box.translate[:2])
    along = rel @ np.array([np.cos(yaw), np.sin(yaw)])
    across = rel @ np.array([-np.sin(yaw), np.cos(yaw)])
    with np.errstate(invalid="ignore"):
        return ((np.abs(along) <= box.size[0] / 2) & (np.abs(across) <= box.size[1] / 2)
                & (points[..., 2] >= box.translate[2]) & (points[..., 2] <= box.translate[2] + box.size[2]))


def _grown(box: Proxy, margin: float = IN_BOX_M) -> Proxy:
    return box.model_copy(update={
        "size": (box.size[0] + 2 * margin, box.size[1] + 2 * margin, box.size[2] + 2 * margin),
        "translate": (box.translate[0], box.translate[1], box.translate[2] - margin),
    })


def reaches_past(box: Proxy, points: np.ndarray, margin: float = IN_BOX_M) -> str | None:
    """Which side of the box (grown by `margin`) an item's points reach past, judged at `REACH_PERCENTILES`, or None."""
    yaw = np.radians(box.yaw_deg)
    rel = points[:, :2] - np.array(box.translate[:2])
    low, high = REACH_PERCENTILES
    for name, values, half in (
        ("along", rel @ np.array([np.cos(yaw), np.sin(yaw)]), box.size[0] / 2),
        ("across", rel @ np.array([-np.sin(yaw), np.cos(yaw)]), box.size[1] / 2),
    ):
        lo, hi = np.percentile(values, [low, high])
        if lo < -half - margin or hi > half + margin:
            return f"{name}, {max(-half - lo, hi - half):.2f} m past"
    top = float(np.percentile(points[:, 2], high))
    if top > box.translate[2] + box.size[2] + margin:
        return f"above, {top - box.translate[2] - box.size[2]:.2f} m past"
    return None


def match_objects(items: dict[str, tuple[str, np.ndarray]], boxes: list[tuple[str, Proxy]]) -> list[dict]:
    """Listed large items paired with RoomPlan's boxes, one to one.

    `items` maps an item's id to its kind and its own points (run world). A box may be an item's when its category
    is one of `BOX_KINDS[kind]` and more than `MATCH_SHARE` of the item's points lie in it, grown by `IN_BOX_M`; pairs are
    taken most shared first. A paired box is refused (the item keeps its own proxy) when the item's points reach past it
    (`reaches_past`): its own LiDAR says the box is wrong. Returns one record per paired item (item, box, category,
    share, refused: None or why) and one per box no item took (item None).
    """
    pairs = []
    for oid, (kind, points) in items.items():
        if not len(points):
            continue
        for index, (cat, box) in enumerate(boxes):
            if cat not in BOX_KINDS.get(kind, ()):
                continue
            share = float(in_box_points(_grown(box), points).mean())
            if share > MATCH_SHARE:
                pairs.append((share, oid, index))
    found, taken_items, taken_boxes = [], set(), set()
    for share, oid, index in sorted(pairs, key=lambda p: -p[0]):
        if oid in taken_items or index in taken_boxes:
            continue
        taken_items.add(oid)
        taken_boxes.add(index)
        cat, box = boxes[index]
        found.append({"item": oid, "box": box.id, "category": cat, "share": round(share, 3),
                      "refused": reaches_past(box, items[oid][1])})
    found += [{"item": None, "box": box.id, "category": cat} for index, (cat, box) in enumerate(boxes) if index not in taken_boxes]
    return found


def is_built_in(index: int, boxes: list[tuple[str, Proxy]], survey_points: dict[str, np.ndarray], seen: set[str] = frozenset()) -> str | None:
    """Why box `index` is built in, or None (left to stage 2).

    Storage is built in when it touches another fixture box (a run of cabinets, a counter round a sink or stove),
    or when the median of a survey built-in's points lies in it (a kitchen island stands alone); alone it is
    furniture (the office cabinet, the bedroom dresser, 0.91 m tall). An appliance the shot sees (its id in
    `seen`) is stage 2's, which lists and generates it, unless the survey holds it as a built-in; one the shot
    does not see is part of the room. Sinks, fireplaces, baths and toilets always are.
    """
    kind, box = boxes[index]
    held = next((oid for oid, points in survey_points.items() if len(points) and in_box(box, np.median(points, axis=0))), None)
    if kind in APPLIANCE_KINDS:
        if box.id not in seen:
            return f"{kind}, out of the shot"
        return f"{kind}, the survey's {held}" if held else None
    if kind != "storage":
        return kind
    mine = footprint(box)
    for other, (other_kind, other_box) in enumerate(boxes):
        if other != index and _gap(mine, footprint(other_box)) < TOUCH_M:
            return f"in a run with {other_box.id}"
    return f"the survey's {held}" if held else None


def snap_offset(points: np.ndarray, origin: np.ndarray, along: np.ndarray, normal: np.ndarray, length: float, height: float) -> tuple[float, int]:
    """How far a wall's face should move along its `normal` to lie on the shot's LiDAR `points` (run world), and on
    how many points: the median offset of those near its plane, inside its span and height; 0 with too few."""
    rel = points - origin
    s, off, z = rel[:, :2] @ along, rel[:, :2] @ normal, points[:, 2]
    near = (np.abs(off) < SNAP_BAND_M) & (s > SNAP_INSET_M) & (s < length - SNAP_INSET_M) & (z > SNAP_INSET_M) & (z < height - SNAP_INSET_M)
    if near.sum() < MIN_SNAP_POINTS:
        return 0.0, int(near.sum())
    return float(np.clip(np.median(off[near]), -MAX_SNAP_M, MAX_SNAP_M)), int(near.sum())


def snap_walls(walls: list[Proxy], points: np.ndarray, room_side: dict[str, np.ndarray]) -> tuple[list[Proxy], dict[str, tuple[float, int]]]:
    """Each wall moved onto the shot's LiDAR where it sees it (`snap_offset`), its openings with it. `room_side`
    holds each wall's unit normal into the room (plan)."""
    moved, record = [], {}
    for wall in walls:
        normal = room_side[wall.id]
        yaw = np.radians(wall.yaw_deg)
        along = np.array([np.cos(yaw), np.sin(yaw)])
        face = np.array(wall.translate[:2]) + normal * wall.size[1] / 2
        origin = face - along * wall.size[0] / 2
        shift, count = snap_offset(points, np.r_[origin, 0.0], along, normal, wall.size[0], wall.size[2])
        record[wall.id] = (shift, count)
        if shift == 0.0:
            moved.append(wall)
            continue
        step = normal * shift
        openings = [o.model_copy(update={"ends": tuple(tuple(float(v) for v in np.add(end, step)) for end in o.ends)}) for o in wall.openings]
        moved.append(wall.model_copy(update={
            "translate": (float(wall.translate[0] + step[0]), float(wall.translate[1] + step[1]), wall.translate[2]),
            "openings": openings,
        }))
    return moved, record


def room_sides(walls: list[Proxy], room: dict, scan_to_run: np.ndarray, eye: np.ndarray) -> dict[str, np.ndarray]:
    """Each wall's unit normal into the room in plan, as `walls_from_room` chose it: from the slab's centre to its face."""
    polygon = floor_polygon(room, scan_to_run)
    sides = {}
    for wall in walls:
        yaw = np.radians(wall.yaw_deg)
        normal = np.array([-np.sin(yaw), np.cos(yaw)])
        probe = np.array(wall.translate[:2]) + normal * (wall.size[1] / 2 + 0.15)
        towards = inside(probe, polygon) if polygon is not None else normal @ (eye[:2] - np.array(wall.translate[:2])) > 0
        sides[wall.id] = normal if towards else -normal
    return sides
