"""`aomma score`: capture runs judged at views of their scan they were not given.

Each run's scene is put in its scan's world and rendered at the views (`views`):

- the reference shots ("reference"); the scan's other input shots, not given to a
  single-shot run ("shot"); and keyframes ("keyframe"), a few from each group;
- grouped by distance from the nearest view the run was given: "near" (within
  `NEAR_M` and `NEAR_DEG`; left out but for reference shots, as the run saw much the
  same), "middle" (within `MIDDLE_M` and `MIDDLE_DEG`) and "far". Starting values;
- only shots that passed the scan check.

Measures, a row each in `results.jsonl`:

- "depth": the scene's depth against the view's sure LiDAR, pixel by pixel, at the
  LiDAR's size: the share that agrees (within `depthcheck.AGREE` either way), is too
  near, too far, or missing (no surface), and the median error in metres. Over all
  pixels, and those a view the run was given saw ("seen") and did not ("unseen"): a
  LiDAR point is seen when it lies where that view's own LiDAR found a surface.
- "depth_model": the same over the pixels each model shows, and "room" where the
  render shows no named model.
- "outline": each reference item's mask in the view against the run's model that
  is matched to (intersection over union, at the render's size).
- "item_box": each reference item's box (`refitems`) against that model's, placed in
  the scan's world: centre error, in plan too, size by axis, heading error. An item
  whose box is only the face the reference shots saw ("partial") gets no box measures.
- "roomplan_box", only for conditions not given RoomPlan: each of RoomPlan's object
  boxes against the model nearest it, the same measures. An extra, as the plan allows:
  RoomPlan is Apple's reading of the scan, not a measurement.

A row carries the run, scan, shot, condition, seed offset and the code the run was last changed
with, so rows from many runs and commits sit in one file.

**Which scene** (`scene`; every round scores one, so a gain at one gate that costs the whole scene shows): "scene"
scores the run's `scene.usda` and models (`layout.json`); "boxes" scores gate 4's box layout (`boxes.usda`,
`boxes_layout.json`), each item by its box (`boxes.json`) and each built-in by its proxy. A layout-only run's
`scene.usda` holds only the shell, so its whole scene is the boxes. Each row says which.
"""

import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

from aomma.capture import CONDITIONS, REFS, Scan, View, _manifest, to_frame
from aomma.layout import SceneLayout
from aomma.refitems import _SAME_ITEM_M, box, item_mask, load_items, roomplan_boxes
from aomma.scancheck import pose_failures, usable_shots
from aomma.stages import depthcheck
from aomma.views import LIDAR_FACTOR, Rendered, render_views, run_scan, run_to_scan

NEAR_M, NEAR_DEG = 0.25, 15.0
MIDDLE_M, MIDDLE_DEG = 1.0, 45.0
# A point is seen by a view when that view's LiDAR depth there is within this, or this share, of the point's.
SEEN_M, SEEN_SHARE = 0.05, 0.03
# Keyframes scored from each group, spread over the walk.
KEYFRAMES_PER_GROUP = 4
# An item whose best model covers less of it than this has no model.
MIN_OUTLINE_IOU = 0.05
# Surface samples per model for its box.
_SAMPLES = 4000


def given_views(run_dir: Path, scan: Scan, shot: View) -> list[View]:
    """The views the run was given: its shot, for every single-shot condition."""
    return [shot]


def group(view: View, given: list[View]) -> str:
    apart, turned = min((view.distance_to(g) for g in given), key=lambda d: d[0])
    if apart <= NEAR_M and turned <= NEAR_DEG:
        return "near"
    return "middle" if apart <= MIDDLE_M and turned <= MIDDLE_DEG else "far"


def pick_views(scan: Scan, given: list[View], keyframes: int, refs: Path = REFS) -> list[tuple[View, str, str]]:
    """(view, source, group) for every view to score at."""
    usable = usable_shots(scan.name, refs)
    names = {g.name for g in given if g.kind == "shot"}
    picked = []
    for view in scan.shots:
        if view.name in names or (usable is not None and view.name not in usable) or view.depth_size is None:
            continue
        where = group(view, given)
        source = "reference" if view.role == "reference" else "shot"
        if source == "reference" or where != "near":
            picked.append((view, source, where))
    by_group = defaultdict(list)
    for view in scan.keyframes:
        if view.depth_size:
            by_group[group(view, given)].append(view)
    for where in ("middle", "far"):
        found = by_group[where]
        for index in np.linspace(0, len(found) - 1, min(keyframes, len(found))).round().astype(int) if found else []:
            picked.append((found[index], "keyframe", where))
    return picked


def seen_by(scan: Scan, view: View, given: list[View]) -> np.ndarray:
    """Which of the view's LiDAR pixels a given view saw: (rows, cols) bool, False where the LiDAR is unsure."""
    depth, _ = scan.depth(view)
    seen = np.zeros(depth.shape, bool)
    points, pixels = scan.lidar_world(view)
    for other in given:
        at, distance = other.project(points)
        measured = scan.depth_at(other, at)
        with np.errstate(invalid="ignore"):
            hit = (distance > 0) & (np.abs(measured - distance) <= np.maximum(SEEN_M, SEEN_SHARE * distance))
        seen[pixels[hit, 0], pixels[hit, 1]] = True
    return seen


def to_lidar(rendered: Rendered) -> tuple[np.ndarray, np.ndarray]:
    """The render at its view's LiDAR size: each pixel's depth the median of its block, its model the block's centre's."""
    rows, cols = rendered.depth.shape[0] // LIDAR_FACTOR, rendered.depth.shape[1] // LIDAR_FACTOR
    blocks = rendered.depth.reshape(rows, LIDAR_FACTOR, cols, LIDAR_FACTOR).swapaxes(1, 2).reshape(rows, cols, -1)
    with np.errstate(invalid="ignore"):
        depth = np.median(blocks, axis=-1)
    return depth, rendered.ids[LIDAR_FACTOR // 2 :: LIDAR_FACTOR, LIDAR_FACTOR // 2 :: LIDAR_FACTOR]


def depth_stats(scene: np.ndarray, lidar: np.ndarray, cells: np.ndarray) -> dict | None:
    """How the scene's depth stands against the LiDAR's over `cells`."""
    if cells.sum() < 3:
        return None
    scene, lidar = scene[cells], lidar[cells]
    missing = ~np.isfinite(scene)
    with np.errstate(divide="ignore", invalid="ignore"):
        error = np.log(scene / lidar)
    agree = ~missing & (np.abs(error) < np.log(depthcheck.AGREE))
    return {
        "pixels": int(cells.sum()),
        "agree": round(float(agree.mean()), 3),
        "too_near": round(float((~missing & (error <= -np.log(depthcheck.AGREE))).mean()), 3),
        "too_far": round(float((~missing & (error >= np.log(depthcheck.AGREE))).mean()), 3),
        "missing": round(float(missing.mean()), 3),
        "median_error_m": round(float(np.median(np.abs(scene[~missing] - lidar[~missing]))), 3) if (~missing).any() else None,
        "ratio": round(float(np.median(scene[~missing] / lidar[~missing])), 3) if (~missing).any() else None,
    }


def item_names(run_dir: Path) -> set[str]:
    """The run's built-ins and large items: what a reference item may be matched with."""
    names = set()
    for file in ("inventory.json", "large_survey.json"):
        if (run_dir / file).exists():
            names |= {o["id"] for o in json.loads((run_dir / file).read_text())["objects"]}
    return names


SCENES = {"scene": ("scene.usda", "layout.json"), "boxes": ("boxes.usda", "boxes_layout.json")}


def _corners(proxy: dict) -> np.ndarray:
    (sx, sy, sz), (tx, ty, tz), yaw = proxy["size"], proxy["translate"], np.radians(proxy["yaw_deg"])
    corners = np.array([[x, y, z] for x in (-sx / 2, sx / 2) for y in (-sy / 2, sy / 2) for z in (0, sz)])
    turn = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
    return corners @ turn.T + [tx, ty, tz]


def model_points(run_dir: Path, run_to_world: np.ndarray, scene: str = "scene") -> dict[str, np.ndarray]:
    """Each placed item's surface (generated models) or box corners (built-ins; with `scene` "boxes", every item's
    gate 4 box), in the scan's world."""
    from aomma.cli import _placed_models

    layout = SceneLayout.model_validate_json((run_dir / SCENES[scene][1]).read_text())
    shown = {o.name for o in layout.objects if not o.left_out}
    if scene == "boxes":
        boxes = json.loads((run_dir / "boxes.json").read_text())["boxes"] if (run_dir / "boxes.json").exists() else []
        found = {b["proxy"]["id"]: _corners(b["proxy"]) for b in boxes if b["proxy"]["id"] in shown}
    else:
        found = _placed_models(run_dir, layout, _SAMPLES)
    if (run_dir / "proxies.json").exists():
        for proxy in json.loads((run_dir / "proxies.json").read_text()):
            # Every item has a proxy; only a built-in, which has no model or box, is scored by its proxy.
            if proxy["id"] not in shown or proxy["id"] in found:
                continue
            found[proxy["id"]] = _corners(proxy)
    return {name: to_frame(points, run_to_world) for name, points in found.items()}


def heading_error_deg(model: dict, item: dict) -> float:
    """Between the two boxes' headings, folded to 0-90; to 0-45 where either footprint is nearly square."""
    angle = abs(model["heading_deg"] - item["heading_deg"]) % 180
    angle = min(angle, 180 - angle)
    square = any(min(b["size"][0], b["size"][2]) > 0.85 * max(b["size"][0], b["size"][2]) for b in (model, item))
    return round(float(min(angle, 90 - angle) if square else angle), 1)


def compare_boxes(model: dict, item: dict) -> dict:
    apart = np.subtract(model["centre"], item["centre"])
    return {
        "centre_m": round(float(np.linalg.norm(apart)), 3),
        "plan_m": round(float(np.linalg.norm(apart[[0, 2]])), 3),
        "height_m": round(float(apart[1]), 3),
        "size_ratio": [round(m / i, 2) if i > 0 else None for m, i in zip(sorted(model["size"][::2], reverse=True) + [model["size"][1]],
                                                                         sorted(item["size"][::2], reverse=True) + [item["size"][1]])],
        "heading_deg": heading_error_deg(model, item),
    }


def code_of(run_dir: Path) -> dict:
    """The code the run was last changed with: its code.jsonl's last line, else its manifest's."""
    manifest = _manifest(run_dir)
    found = {"commit": manifest.get("commit"), "diff": manifest.get("diff")}
    if (run_dir / "code.jsonl").exists():
        lines = (run_dir / "code.jsonl").read_text().splitlines()
        if lines:
            last = json.loads(lines[-1])
            found = {"commit": last.get("commit"), "diff": last.get("diff")}
    return found


def score_run(run_dir: Path, out: Path, keyframes: int = KEYFRAMES_PER_GROUP, refs: Path = REFS, label: str = "",
              scene: str = "scene") -> list[dict]:
    """Every measure of one run's `scene` ("scene" or "boxes", see the module's notes) at every view picked for it,
    each row carrying `label` (as "before review"); debug pictures go to out/<run>/ (out/<run>-<label>/ with a label,
    with "-boxes" after it for the box layout)."""
    which = scene  # `scene` is reused below for each view's rendered depth
    scan, shot = run_scan(run_dir)
    manifest = _manifest(run_dir)
    given = given_views(run_dir, scan, shot)
    picked = pick_views(scan, given, keyframes, refs)
    if not picked:
        return []
    debug = out / "-".join(filter(None, [run_dir.name, label.replace(" ", "_"), "boxes" if which == "boxes" else ""]))
    rendered = render_views(run_dir, [view for view, _, _ in picked], debug / "renders", SCENES[which][0])
    base = {"run": str(run_dir), "label": label, "scene": which, "scan": scan.name, "shot": shot.name, "condition": manifest.get("condition", "photo"),
            "seed_offset": manifest.get("seed_offset", 0), **code_of(run_dir)}
    candidates = item_names(run_dir)
    items = load_items(scan.name, refs)
    rows = []
    overlaps: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    outlines = []
    for (view, source, where), (key, seen_from) in zip(picked, rendered.items()):
        at = {**base, "view": key, "source": source, "group": where}
        lidar, conf = scan.depth(view)
        sure = (conf == 2) & np.isfinite(lidar) & (lidar > 0)
        scene, ids = to_lidar(seen_from)
        seen = seen_by(scan, view, given)
        for part, cells in (("all", sure), ("seen", sure & seen), ("unseen", sure & ~seen)):
            if (stats := depth_stats(scene, lidar, cells)) is not None:
                rows.append({**at, "measure": "depth", "pixels_of": part, **stats})
        for index, name in [(0, "room")] + list(enumerate(seen_from.names, start=1)):
            if (stats := depth_stats(scene, lidar, sure & (ids == index))) is not None:
                rows.append({**at, "measure": "depth_model", "model": name, **stats})
        with np.errstate(divide="ignore", invalid="ignore"):
            measured = np.where(sure, lidar, np.nan)
        depthcheck.draw(Image.open(view.image), scene, measured).resize(seen_from.depth.shape[::-1]).save(debug / f"{key}_depth.jpg", quality=85)

        # Outlines: each item this view has a mask of, against every model; scored once items and models are matched.
        for item in items:
            if view.kind != "shot" or view.name not in item["shots"]:
                continue
            mask = item_mask(scan.name, view.name, item["shots"][view.name], refs)
            mask = np.asarray(Image.fromarray(mask).resize(seen_from.depth.shape[::-1], Image.Resampling.NEAREST))
            ious = {}
            for name in seen_from.names:
                if name not in candidates:
                    continue
                shows = seen_from.shows(name)
                ious[name] = float((shows & mask).sum() / max((shows | mask).sum(), 1))
                overlaps[item["id"]][name] += ious[name]
            outlines.append((at, item, ious))

    matched = match_items(overlaps)
    for at, item, ious in outlines:
        name, iou = matched.get(item["id"]), ious.get(matched.get(item["id"]), 0.0)
        rows.append({**at, "measure": "outline", "item": item["id"], "kind": item["kind"],
                     "model": name if iou >= MIN_OUTLINE_IOU else None, "iou": round(iou, 3)})

    # Boxes: each item against the model matched to it; an item whose box is only a face of it is not (see `partial`).
    placed = model_points(run_dir, run_to_scan(run_dir, shot), which) if items or not took_roomplan(base["condition"]) else {}
    for item in items:
        name = matched.get(item["id"])
        row = {**base, "measure": "item_box", "item": item["id"], "kind": item["kind"], "flagged": item["flagged"],
               "partial": item.get("partial", False), "model": name}
        if name in placed and not row["partial"]:
            row |= compare_boxes(box(placed[name]), item)
        rows.append(row)
    # An extra, where RoomPlan was not an input: RoomPlan's object boxes against the models nearest them.
    if not took_roomplan(base["condition"]):
        boxes = {name: box(points) for name, points in placed.items()}
        for found, name in match_to_roomplan(roomplan_boxes(scan.room), boxes):
            row = {**base, "measure": "roomplan_box", "object": found["category"], "model": name}
            if name is not None:
                row |= compare_boxes(boxes[name], found)
            rows.append(row)
    return rows


def took_roomplan(condition: str) -> bool:
    """Whether a condition gave its runs any of RoomPlan's boxes, so that its items must not be judged by them. A run
    given only RoomPlan's walls (`depth+walls`) never saw a box, and is."""
    return any(given in ("roomplan_room", "roomplan_objects") for given in CONDITIONS.get(condition, ()))


def match_to_roomplan(found: list[dict], boxes: dict[str, dict]) -> list[tuple[dict, str | None]]:
    """Each RoomPlan box paired with the model whose box centre is nearest it in plan, within `_SAME_ITEM_M`, one to one,
    the nearest pairs first; None where no model is left that near. RoomPlan's categories are not used: it called
    OfficeScan's crate a chair."""
    pairs = sorted(
        (float(np.linalg.norm(np.subtract(f["centre"], b["centre"])[[0, 2]])), i, name)
        for i, f in enumerate(found) for name, b in boxes.items()
    )
    paired: dict[int, str] = {}
    for distance, i, name in pairs:
        if distance <= _SAME_ITEM_M and i not in paired and name not in paired.values():
            paired[i] = name
    return [(f, paired.get(i)) for i, f in enumerate(found)]


def match_items(overlaps: dict[str, dict[str, float]]) -> dict[str, str]:
    """Pair items with models one to one, the pair overlapping most (IoU summed over the views) first.

    A model covering two items is one of them, not both. A pair under `MIN_OUTLINE_IOU` is no match.
    """
    pairs = sorted(((overlap, item, model) for item, models in overlaps.items() for model, overlap in models.items()), reverse=True)
    matched: dict[str, str] = {}
    for overlap, item, model in pairs:
        if overlap >= MIN_OUTLINE_IOU and item not in matched and model not in matched.values():
            matched[item] = model
    return matched


def score(run_dirs: list[Path], out: Path, keyframes: int = KEYFRAMES_PER_GROUP, refs: Path = REFS, label: str = "",
          scene: str = "scene") -> Path:
    """Score each run's `scene` ("scene" or "boxes"), adding its rows to out/results.jsonl, each carrying `label`."""
    out.mkdir(parents=True, exist_ok=True)
    results = out / "results.jsonl"
    for run_dir in run_dirs:
        print(f"== {run_dir}", flush=True)
        scan, shot = run_scan(run_dir)
        rows = score_run(run_dir, out, keyframes, refs, label, scene)
        # Gate 0: the shot's model is laid into the scan by its pose, so where the pose is off its scores are not to be trusted.
        if why := pose_failures(scan.name, refs).get(shot.name):
            print(f"  pose off (gate 0; scores not to be trusted): {why}", flush=True)
            rows = [row | {"pose_off": why} for row in rows]
        with results.open("a") as f:
            for row in rows:
                f.write(json.dumps(row | {"scored": f"{datetime.now():%Y-%m-%d %H:%M:%S}"}) + "\n")
        for line in summary(rows):
            print(line, flush=True)
    return results


def summary(rows: list[dict]) -> list[str]:
    """A few lines per run: depth agreement by source and group, seen and unseen, and the items."""
    lines = []
    depth = [r for r in rows if r["measure"] == "depth"]
    for source in ("reference", "shot", "keyframe"):
        for where in ("near", "middle", "far"):
            for part in ("seen", "unseen"):
                found = [r for r in depth if r["source"] == source and r["group"] == where and r["pixels_of"] == part]
                if found:
                    pixels = sum(r["pixels"] for r in found)
                    agree = sum(r["agree"] * r["pixels"] for r in found) / pixels
                    missing = sum(r["missing"] * r["pixels"] for r in found) / pixels
                    error = np.median([r["median_error_m"] for r in found if r["median_error_m"] is not None] or [np.nan])
                    lines.append(f"  depth at {len(found)} {source} views, {where}, {part}: {agree:.0%} agree, {missing:.0%} missing, "
                                 f"median error {error:.2f} m")
    for row in (r for r in rows if r["measure"] == "item_box"):
        if row.get("model") is None:
            lines.append(f"  {row['item']} ({row['kind']}): no model")
        elif row.get("centre_m") is None:
            outline = [r["iou"] for r in rows if r["measure"] == "outline" and r["item"] == row["item"]]
            lines.append(f"  {row['item']} ({row['kind']}) as {row['model']}: the reference is a face of it, not scored by its box"
                         + (f", outline {np.mean(outline):.2f}" if outline else ""))
        else:
            outline = [r["iou"] for r in rows if r["measure"] == "outline" and r["item"] == row["item"]]
            lines.append(f"  {row['item']} ({row['kind']}) as {row['model']}: centre {row['centre_m']:.2f} m off "
                         f"({row['plan_m']:.2f} in plan), size x{row['size_ratio']}, heading {row['heading_deg']} deg off, "
                         f"outline {np.mean(outline):.2f}" + (" [flagged]" if row["flagged"] else ""))
    return lines
