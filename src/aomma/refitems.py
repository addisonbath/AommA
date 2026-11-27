"""Reference items: each large item's box, from the reference shots alone.

In each reference shot the large items are listed (the stage 2 prompt, with no
built-ins given, so cabinets and counters are listed too) and segmented, and each
mask's sure LiDAR points are put in the scan's world. An item seen in several shots
is one item when the kinds agree and the shots see the same surfaces; its box comes from all
of its points. Where three or more shots see it and one shares almost no surface with the rest
(one bad mask), that shot is dropped and the item flagged. The box is taken from what the
shots saw, so a side none of them saw is short: the reference shots are spread round
the room for this.

Neither the run nor RoomPlan has a hand in it, so it judges every condition.
Writes `refs/<scan>/items.json` and `refs/<scan>/<shot>/masks/<item>.png`.
"""

import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import binary_erosion
from scipy.spatial import cKDTree

from aomma.capture import REFS, Scan, View
from aomma.scancheck import usable_shots

# Kinds that are one item when the VLM names them differently from two shots.
_KIND_GROUPS = ({"chair", "armchair"}, {"table", "desk"}, {"cabinet", "shelving", "dresser", "nightstand"})
# Two shots' items are one when they see the same surfaces: this share of either's points lies within `_SHARED_M` of the
# other's. Measured on the three protocol scans' references: sightings of different items share
# none (0.00, even 0.15-0.40 m apart: the range hood over the stove, the pouf by the guitar), sightings of one item 0.11
# and up. Their points' centres, the rule before, lie up to 1 m apart on a 2 m bed, as each shot sees its own faces.
_SHARED_M = 0.05
_SHARED_SHARE = 0.1
# How far `scoring.match_to_roomplan` looks for a model's box from a RoomPlan box's, in plan.
_SAME_ITEM_M = 0.6
# A mask with fewer sure LiDAR points than this says nothing of where its item is.
_MIN_POINTS = 30
# ARKit's confidence taken for an item's points: medium as well as sure. Past about 2 m most of a shot is medium
# (OfficeScan 004's sofa, 2.2 m away: 13 sure points, 730 medium).
_CONFIDENCE = 1
# Masks are eroded by this share of the image's longer side, so a pixel past an item's edge (the wall behind) is not its.
_ERODE_SHARE = 1 / 200
# An item whose box is under this share of its listed size on any side (the longer plan side against the listed longer,
# the shorter against the shorter, the height against the height) is a face of it, not the item: the reference shots saw
# one side only. Its centre lies on that face, so it is not scored by its box. OfficeScan's green sofa came out 0.14, its
# cabinet 0.24 and its crate 0.26; the whole items 0.43 (the amplifier) and up.
PARTIAL_SHARE = 0.35


def kind_group(kind: str) -> str:
    return next(("/".join(sorted(group)) for group in _KIND_GROUPS if kind in group), kind)


def lift(scan: Scan, view: View, mask: np.ndarray) -> np.ndarray:
    """The view's LiDAR points (world, of `_CONFIDENCE` or more) whose pixel centre falls in `mask` (any size, the view's
    aspect), eroded."""
    eroded = binary_erosion(mask, iterations=max(1, round(max(mask.shape) * _ERODE_SHARE)))
    points, pixels = scan.lidar_world(view, _CONFIDENCE)
    rows, cols = scan.depth(view)[0].shape
    y = ((pixels[:, 0] + 0.5) * mask.shape[0] / rows).astype(int)
    x = ((pixels[:, 1] + 0.5) * mask.shape[1] / cols).astype(int)
    return points[eroded[y, x]]


def box(points: np.ndarray) -> dict:
    """The upright box round `points` (ARKit's world, +y up) with the least area in plan, its extents the 2nd to 98th
    percentiles: centre, size (along its heading, up, across) and heading (degrees about +y from world +x, 0-180)."""
    plan = points[:, [0, 2]]
    best = None
    for angle in range(0, 90):
        turn = np.radians(angle)
        along = plan @ [np.cos(turn), -np.sin(turn)]
        across = plan @ [np.sin(turn), np.cos(turn)]
        (a0, a1), (c0, c1) = np.percentile(along, [2, 98]), np.percentile(across, [2, 98])
        if best is None or (a1 - a0) * (c1 - c0) < best[0]:
            best = ((a1 - a0) * (c1 - c0), angle, (a0, a1), (c0, c1))
    _, angle, (a0, a1), (c0, c1) = best
    # The heading is along the longer side. Turned 90 degrees more, along is the old across reversed, across the old along.
    if c1 - c0 > a1 - a0:
        angle, (a0, a1), (c0, c1) = angle + 90, (-c1, -c0), (a0, a1)
    turn = np.radians(angle)
    middle_along, middle_across = (a0 + a1) / 2, (c0 + c1) / 2
    # Back from the turned axes to the world's: along is (cos, -sin) in (x, z), across is (sin, cos).
    x = middle_along * np.cos(turn) + middle_across * np.sin(turn)
    z = -middle_along * np.sin(turn) + middle_across * np.cos(turn)
    low, high = np.percentile(points[:, 1], [0.5, 98])
    return {
        "centre": [round(float(x), 3), round(float((low + high) / 2), 3), round(float(z), 3)],
        "size": [round(float(a1 - a0), 3), round(float(high - low), 3), round(float(c1 - c0), 3)],
        "heading_deg": round(float(angle % 180), 1),
    }


def partial(size: list[float], listed: list[float]) -> bool:
    """Whether a box (along, up, across) is only a face of an item listed as (width, depth, height): under
    `PARTIAL_SHARE` of it on any side."""
    plan, listed_plan = sorted([size[0], size[2]], reverse=True), sorted(listed[:2], reverse=True)
    shares = [found / guess for found, guess in zip(plan, listed_plan)] + [size[1] / listed[2]]
    return min(shares) < PARTIAL_SHARE


def roomplan_boxes(room: dict) -> list[dict]:
    """RoomPlan's object boxes in `box`'s form (ARKit's world): category, centre, size (along, up, across), heading."""
    found = []
    for obj in room.get("objects", []):
        matrix = np.array(obj["transform"], float).reshape(4, 4).T
        width, height, depth = obj["dimensions"]
        x_axis = matrix[:3, 0]
        found.append({
            "category": next(iter(obj["category"])), "centre": matrix[:3, 3].round(3).tolist(),
            "size": [round(width, 3), round(height, 3), round(depth, 3)],
            "heading_deg": round(float(np.degrees(np.arctan2(-x_axis[2], x_axis[0])) % 180), 1),
        })
    return found


def shared(points: np.ndarray, others: np.ndarray) -> float:
    """The share of `points` within `_SHARED_M` of any of `others`."""
    if not len(points) or not len(others):
        return 0.0
    found, _ = cKDTree(others).query(points, distance_upper_bound=_SHARED_M)
    return float((found <= _SHARED_M).mean())


def merge(sightings: list[dict]) -> list[dict]:
    """Sightings (a shot's item: shot, id, kind, description, points, and its listed size if known) grouped into items,
    a bad shot left out of each; an item whose box is only a face of it (see `partial`) is marked.

    Two sightings, of one kind group and from different shots, see one item when they share surface (`shared`, either
    way, at least `_SHARED_SHARE`). Pairs are joined most shared first, and two items are joined through any such pair
    unless a shot would then see the item twice: a thin item's sightings from opposite sides share nothing, but each
    shares with a shot from its edge (OfficeScan's blue guitar, shots 000 and 002 through 003). Of three or more shots
    of an item, one whose points barely lie on the others' (under `_SHARED_SHARE`) is a bad mask: left out, the item
    flagged. Not caught: a mask that takes in the item and something beside it, as its points do lie on the item's.
    """
    pairs = []
    for i, a in enumerate(sightings):
        for j in range(i + 1, len(sightings)):
            b = sightings[j]
            if a["shot"] == b["shot"] or kind_group(a["kind"]) != kind_group(b["kind"]):
                continue
            share = max(shared(a["points"], b["points"]), shared(b["points"], a["points"]))
            if share >= _SHARED_SHARE:
                pairs.append((share, i, j))
    item_of = list(range(len(sightings)))
    for _, i, j in sorted(pairs, reverse=True):
        a, b = item_of[i], item_of[j]
        if a == b:
            continue
        shots_a = {sightings[k]["shot"] for k, n in enumerate(item_of) if n == a}
        if shots_a & {sightings[k]["shot"] for k, n in enumerate(item_of) if n == b}:
            continue
        item_of = [a if n == b else n for n in item_of]
    groups = [[s for s, n in zip(sightings, item_of) if n == m] for m in dict.fromkeys(item_of)]
    items, taken = [], set()
    for group in groups:
        odd = []
        if len(group) >= 3:
            odd = [s["shot"] for s in group if shared(s["points"], np.concatenate([o["points"] for o in group if o is not s])) < _SHARED_SHARE]
        kept = [s for s in group if s["shot"] not in odd]
        name, number = group[0]["id"], 2
        while name in taken:
            name, number = f"{group[0]['id']}_{number}", number + 1
        taken.add(name)
        item = {
            "id": name, "kind": group[0]["kind"], "description": group[0]["description"],
            "shots": {s["shot"]: s["id"] for s in kept}, "left_out": odd, "flagged": bool(odd),
            **box(np.concatenate([s["points"] for s in kept])),
        }
        listed = [s["listed"] for s in kept if s.get("listed")]
        if listed:
            item["listed_size"] = np.median(listed, axis=0).round(3).tolist()
            item["partial"] = partial(item["size"], item["listed_size"])
        items.append(item)
    return items


def reference_items(scan: Scan, config_path: Path, vlm_name: str | None = None, refs: Path = REFS, shots: str = "reference") -> list[dict]:
    """List, segment and lift the large items in the scan's reference shots (or with `shots="all"`, every shot that
    passed the scan check); writes and returns refs/<scan>/items.json."""
    from aomma.cli import comfy_client, load_config, vlm_for
    from aomma.ledger import Ledger
    from aomma.stages.image import load_working_image, to_jpeg
    from aomma.stages.inventory import Inventory
    from aomma.stages.large import take_large
    from aomma.stages.segment import draw_masks, segment

    usable = usable_shots(scan.name, refs)
    if usable is None:
        raise SystemExit(f"{scan.name} has not been checked: run `aomma scan-check {scan.folder}` first")
    views = [v for v in scan.shots if v.name in usable and (shots == "all" or v.role == "reference") and v.depth_size]
    config = load_config(config_path)
    folder = refs / scan.name
    vlm = vlm_for(config, vlm_name, folder)
    ledger = Ledger(folder / "usage.jsonl", config.max_billed_tokens)
    comfy = comfy_client(config.comfy.light[0], folder)
    sightings = []
    for view in views:
        image, _ = load_working_image(view.image, config.image_max_side)
        listed, raw = take_large(image, "room", [], set(), vlm, ledger)
        out = folder / view.name / "masks"
        out.mkdir(parents=True, exist_ok=True)
        (out / "listed.json").write_text(listed.model_dump_json(indent=2))
        if not listed.objects:
            continue
        name = comfy.upload_image(to_jpeg(image), f"aomma_ref_{scan.name}_{view.name}.jpg")
        masks, _ = segment(comfy, name, image.size, Inventory(objects=listed.objects))
        draw_masks(image, masks).save(out / "masks.jpg", quality=85)
        for obj in listed.objects:
            points = lift(scan, view, masks[obj.id])
            print(f"  shot {view.name}: {obj.id} ({listed.kinds[obj.id]}), {len(points)} LiDAR points", flush=True)
            Image.fromarray(masks[obj.id]).save(out / f"{obj.id}.png")
            if len(points) >= _MIN_POINTS:
                sightings.append({"shot": view.name, "id": obj.id, "kind": listed.kinds[obj.id], "description": obj.description,
                                  "points": points, "listed": list(obj.size_m)})
    items = merge(sightings)
    (folder / "items.json").write_text(json.dumps({"scan": str(scan.folder), "shots": [v.name for v in views], "items": items}, indent=2))
    return items


def merge_saved(scan: Scan, refs: Path = REFS) -> list[dict]:
    """The items made again from the lists and masks a `reference_items` run saved, merged by today's `merge`, with no
    listing or segmenting: only the merge (or the lift) is new. A shot no longer usable (its pose failed gate 0 since)
    is left out. Writes and returns refs/<scan>/items.json."""
    from aomma.stages.large import LargeSurvey

    folder = refs / scan.name
    usable = usable_shots(scan.name, refs)
    shots = [s for s in json.loads((folder / "items.json").read_text())["shots"] if usable is None or s in usable]
    views = {v.name: v for v in scan.shots}
    sightings = []
    for shot in shots:
        listed_path = folder / shot / "masks" / "listed.json"
        if not listed_path.exists():
            continue
        listed = LargeSurvey.model_validate_json(listed_path.read_text())
        for obj in listed.objects:
            points = lift(scan, views[shot], item_mask(scan.name, shot, obj.id, refs))
            if len(points) >= _MIN_POINTS:
                sightings.append({"shot": shot, "id": obj.id, "kind": listed.kinds[obj.id], "description": obj.description,
                                  "points": points, "listed": list(obj.size_m)})
    items = merge(sightings)
    (folder / "items.json").write_text(json.dumps({"scan": str(scan.folder), "shots": shots, "items": items}, indent=2))
    return items


def load_items(scan_name: str, refs: Path = REFS) -> list[dict]:
    path = refs / scan_name / "items.json"
    return json.loads(path.read_text())["items"] if path.exists() else []


def item_mask(scan_name: str, shot: str, sighting_id: str, refs: Path = REFS) -> np.ndarray:
    return np.asarray(Image.open(refs / scan_name / shot / "masks" / f"{sighting_id}.png")) > 0


def reference_small(scan: Scan, config_path: Path, vlm_name: str | None = None, refs: Path = REFS) -> list[dict]:
    """Gate 3's reference: the small items in the scan's reference shots, listed as a run lists them (`stages.small`:
    one close-up listing per support, every item the large reference listed in that shot taken as a support, built-ins
    included as it lists them, and one pass over the floor and walls), segmented, lifted by the shot's LiDAR and
    merged across shots as the large items are (`merge`, all of one kind: only shared surface joins two sightings).

    Each shot's listing is made in refs/<scan>/<shot>/small/ (a run's files: input.jpg, photo.jpg, survey.json,
    large_survey.json, masks/). A mask with under `_MIN_POINTS` LiDAR points gives no box and is reported (`few_points`).
    Writes and returns refs/<scan>/small_items.json."""
    import shutil

    from aomma.cli import comfy_client, load_config, vlm_for
    from aomma.ledger import Ledger
    from aomma.stages.image import load_working_image
    from aomma.stages.small import survey_small

    folder = refs / scan.name
    usable = usable_shots(scan.name, refs)
    shots = [s for s in json.loads((folder / "items.json").read_text())["shots"] if usable is None or s in usable]
    views = {v.name: v for v in scan.shots}
    config = load_config(config_path)
    vlm = vlm_for(config, vlm_name, folder)
    ledger = Ledger(folder / "usage.jsonl", config.max_billed_tokens)
    comfy = comfy_client(config.comfy.light[0], folder)
    sightings, few = [], []
    for shot in shots:
        view, listed = views[shot], folder / shot / "masks" / "listed.json"
        if not listed.exists():
            continue
        work = folder / shot / "small"
        shutil.rmtree(work, ignore_errors=True)
        (work / "masks").mkdir(parents=True)
        image, _ = load_working_image(view.image, config.image_max_side)
        image.save(work / "input.jpg", quality=95)
        shutil.copyfile(view.image, work / "photo.jpg")
        (work / "survey.json").write_text(json.dumps({"room_type": "room", "built_ins": []}))
        shutil.copyfile(listed, work / "large_survey.json")
        for obj in json.loads(listed.read_text()).get("objects", []):
            if (folder / shot / "masks" / f"{obj['id']}.png").exists():
                shutil.copyfile(folder / shot / "masks" / f"{obj['id']}.png", work / "masks" / f"{obj['id']}.png")
        print(f"== shot {shot}", flush=True)
        found = survey_small(work, vlm, ledger, comfy)
        for obj in found.objects:
            mask = np.asarray(Image.open(work / "masks" / f"{obj.id}.png").convert("L")) > 127
            points = lift(scan, view, mask)
            print(f"  shot {shot}: {obj.id} (on {obj.on}), {len(points)} LiDAR points", flush=True)
            if len(points) >= _MIN_POINTS:
                sightings.append({"shot": shot, "id": obj.id, "kind": "small", "description": obj.description,
                                  "points": points, "listed": list(obj.size_m), "on": obj.on})
            else:
                few.append({"shot": shot, "id": obj.id, "description": obj.description, "points": len(points)})
    items = merge(sightings)
    on = {(s["shot"], s["id"]): s["on"] for s in sightings}
    for item in items:
        item["on"] = {shot: on[(shot, sid)] for shot, sid in item["shots"].items()}
    (folder / "small_items.json").write_text(json.dumps(
        {"scan": str(scan.folder), "shots": shots, "items": items, "few_points": few}, indent=2))
    return items
