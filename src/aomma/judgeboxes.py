"""`aomma judge-boxes`: a run's box layout (`aomma boxes`) against the reference items (gate 4).

**The references**: the large items (`refs/<scan>/items.json`, `aomma ref-items`) and the small ones
(`small_items.json`, `aomma ref-items --small`), boxes from the reference shots' LiDAR in the scan's world. The large
reference lists cabinets and counters too, which a run builds as built-ins (the shell's `proxies.json`).

**Matching**: an item's pixels in the input shot are its sure LiDAR points within `refitems._SHARED_M` of the item's own
reference points (`item_points`, its box grown by `refshell.ON_M` bounding the search) and not on the shell reference's
floor, walls, ceiling or openings; a pixel two items' points reach counts for both (books in a shelf's box), and the one-to-one matching settles them. An item both references
list (a guitar the large one lists, the small one again) is left unmatched once its twin takes the run item. At the LiDAR map's size, the run's label most common in each 4 x 4 block (a small item kept as a
layer on its owner counts as its own). Each reference item with `MIN_PIXELS` or more is matched, one to one, the largest
overlap first, to the run item holding most of its pixels: like with like first (a large reference item to a large run
item or a built-in, a small one to a small item), then across. A run item the run made a sheet (`forms.json`) has no box.

**Measures**, per matched pair whose run item has a box (`scoring.compare_boxes`, in the scan's world by the shot's pose,
`views.run_to_scan`): centre, plan and height errors (m), size ratios (run over reference: the plan's longer side, its
shorter, the height), heading (degrees). A reference item that is only a face of the item (`partial`) is matched but its
box is not compared: its centre lies on that face; nor is one left out by eye (`refs/<scan>/items_by_eye.json`, a
reference fault seen on the reference shots' masks). Each run box also says how it was stood (`boxes.json`: on the floor,
on its support's top, inside it, or as measured). `verdict` holds gate 4's bars (`BARS`, `OBLONG`). Writes `<run>/judge_boxes.json`.
"""

import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import maximum_filter, minimum_filter
from scipy.spatial import cKDTree

from aomma.capture import REFS, Scan, to_frame
from aomma.judgeshell import run_on_grid
from aomma.judgesmall import _inside
from aomma.refitems import _SHARED_M, lift
from aomma.refshell import CEILING, FLOOR, ON_M, UNSCORED, UNSURE, WALL_BASE
from aomma.scancheck import pose_failures
from aomma.scoring import compare_boxes
from aomma.views import run_scan, run_to_scan

MIN_PIXELS = 20
# Gate 4's bars (set on the baseline, before any fix), on large items compared; small items
# are reported only (their reference is thin).
BARS = {"plan_m": 0.30, "height_m": 0.15, "size_ratio": (0.67, 1.5), "heading_deg": 20.0}
# Heading is judged only where the reference's plan is oblong, its longer side this many times its shorter: a round or
# square thing's heading says little (OfficeScan 005's swivel chair, 39 degrees).
OBLONG = 1.2


def world_box(proxy: dict, run_to_world: np.ndarray) -> dict:
    """A run's box (`Proxy`: bottom-centre `translate`, `size` along its x, y and up, `yaw_deg` about up, run world Z up)
    in `refitems.box`'s form in the scan's world (+y up): centre, size (along, up, across), heading (degrees, 0-180)."""
    (sx, sy, sz), (tx, ty, tz), yaw = proxy["size"], proxy["translate"], np.radians(proxy["yaw_deg"])
    centre = to_frame(np.array([[tx, ty, tz + sz / 2]]), run_to_world)[0]
    axis = run_to_world[:3, :3] @ np.array([np.cos(yaw), np.sin(yaw), 0.0])
    heading = float(np.degrees(np.arctan2(-axis[2], axis[0])))
    along, across = sx, sy
    if across > along:
        along, across, heading = across, along, heading + 90
    return {"centre": centre.round(3).tolist(), "size": [round(along, 3), round(sz, 3), round(across, 3)],
            "heading_deg": round(heading % 180, 1)}


def item_points(scan: Scan, item: dict, refs: Path = REFS) -> np.ndarray:
    """The reference item's own LiDAR points: each of its sightings' saved masks lifted by that reference shot
    (`refitems.lift`), as its box was made."""
    views = {v.name: v for v in scan.shots}
    found = []
    for shot, sighting in item["shots"].items():
        path = refs / scan.name / shot / ("small/masks" if item["class"] == "small" else "masks") / f"{sighting}.png"
        if path.exists() and shot in views:
            found.append(lift(scan, views[shot], np.asarray(Image.open(path).convert("L")) > 127))
    return np.concatenate(found) if found else np.zeros((0, 3))


def _reference_pixels(scan: Scan, view, shell: np.ndarray, items: list[dict], refs: Path = REFS) -> list[np.ndarray]:
    """Each item's LiDAR pixels in the view: sure points within `refitems._SHARED_M` of the item's own reference points
    (`item_points`) and not on the shell. By its points, not its box: a box's empty corners take in what stands behind
    a hidden item (KitchenScan 008's step stool, behind the island: its box caught the sink cabinet's door). Items may
    share pixels (books on a shelf; a guitar the large reference lists and the small one again): the one-to-one
    matching settles them."""
    points, pixels = scan.lidar_world(view)
    code = shell[pixels[:, 0], pixels[:, 1]]
    eligible = ~((code == FLOOR) | (code == CEILING) | (code >= WALL_BASE)) & (code != UNSCORED) & (code != UNSURE)
    found = []
    for item in items:
        mask = np.zeros(shell.shape, bool)
        own = item_points(scan, item, refs)
        near = eligible & _inside(points, item, ON_M)
        if len(own) and near.any():
            gap, _ = cKDTree(own).query(points[near], distance_upper_bound=_SHARED_M)
            hit = np.flatnonzero(near)[gap <= _SHARED_M]
            mask[pixels[hit, 0], pixels[hit, 1]] = True
        found.append(mask)
    return found


def judge(run_dir: Path, refs: Path = REFS) -> dict:
    scan, shot = run_scan(run_dir)
    folder = refs / scan.name
    items = [{**i, "class": "large"} for i in json.loads((folder / "items.json").read_text())["items"]]
    if (folder / "small_items.json").exists():
        items += [{**i, "class": "small"} for i in json.loads((folder / "small_items.json").read_text())["items"]]
    # Reference items looked at and left out by eye (`items_by_eye.json`): matched, but their boxes not compared.
    by_eye = json.loads((folder / "items_by_eye.json").read_text()).get("left_out", {}) if (folder / "items_by_eye.json").exists() else {}
    shell = np.asarray(Image.open(folder / shot.name / "shell_labels.png"))
    ref = _reference_pixels(scan, shot, shell, items, refs)
    saved = {e["value"]: e for e in json.loads((run_dir / "labels.json").read_text())["labels"]}
    values = run_on_grid(np.asarray(Image.open(run_dir / "labels.png")), (shell.shape[1], shell.shape[0]))
    if (run_dir / "debug" / "labels_layers.png").exists():
        layers = run_on_grid(np.asarray(Image.open(run_dir / "debug" / "labels_layers.png")), (shell.shape[1], shell.shape[0]))
        values = np.where(layers > 0, layers, values)
    boundary = maximum_filter(shell, size=3) != minimum_filter(shell, size=3)
    scored = (shell != UNSCORED) & (shell != UNSURE) & ~boundary

    boxes = {b["proxy"]["id"]: b for b in json.loads((run_dir / "boxes.json").read_text())["boxes"]} \
        if (run_dir / "boxes.json").exists() else {}
    built = {p["id"]: p for p in json.loads((run_dir / "proxies.json").read_text())} if (run_dir / "proxies.json").exists() else {}
    to_world = run_to_scan(run_dir, shot)

    seen, overlaps, masks = {}, [], {}
    for i, item in enumerate(items):
        mine = scored & ref[i]
        n = int(mine.sum())
        if n < MIN_PIXELS:
            continue
        seen[item["id"]] = n
        masks[item["id"]] = mine
        for value in np.unique(values[mine]):
            if value in saved and saved[value]["class"] in ("large", "small", "builtin", "other"):
                overlaps.append((int((mine & (values == value)).sum()), item["id"], saved[value]["id"]))
    # Like with like first: a large reference item to a large run item or a built-in, a small one to a small item, each
    # the largest overlap first; then what is left across (an item one lister calls large, the other small). By pixels
    # alone a bed's reference went to the throw on it, which holds more of the bed's box than the bed's own label does.
    run_class = {e["id"]: e["class"] for e in saved.values()}
    ref_class = {item["id"]: item["class"] for item in items}
    alike = lambda item_id, run_id: (run_class[run_id] == "small") == (ref_class[item_id] == "small")
    matched, used = {}, set()
    for same in (True, False):
        for count, item_id, run_id in sorted(overlaps, reverse=True):
            if item_id not in matched and run_id not in used and alike(item_id, run_id) == same:
                matched[item_id] = (run_id, count)
                used.add(run_id)
    forms = json.loads((run_dir / "forms.json").read_text()) if (run_dir / "forms.json").exists() else {}
    sheets = {f["id"] for f in forms.get("forms", []) if f.get("form") != "box"}

    by_id = {item["id"]: item for item in items}
    rows = {}
    for item_id, n in seen.items():
        item = by_id[item_id]
        row = {"class": item["class"], "description": item["description"], "pixels": n, "partial": item.get("partial", False),
               "run_item": None, "share": 0.0}
        if item_id in by_eye:
            row["left_out"] = by_eye[item_id]
        if item_id in matched:
            run_id, count = matched[item_id]
            row |= {"run_item": run_id, "share": round(count / n, 4)}
            proxy = boxes[run_id]["proxy"] if run_id in boxes else built.get(run_id)
            row["run_class"] = boxes[run_id]["kind"] if run_id in boxes else ("builtin" if run_id in built else None)
            if run_id in boxes:
                row["stood"] = boxes[run_id]["stood"]
            if proxy is None and run_id in sheets:
                row["sheet"] = True
            elif proxy is None:
                row["no_box"] = True
            else:
                row["run_box"] = world_box(proxy, to_world)
                if not row["partial"] and "left_out" not in row:
                    row |= compare_boxes(row["run_box"], item)
                    plan = sorted([item["size"][0], item["size"][2]])
                    row["oblong"] = bool(plan[1] >= OBLONG * plan[0])
        rows[item_id] = row
    # An unmatched item lying mostly in the pixels of a matched item of the other reference is that item listed twice
    # (the large reference's `guitar`, the small one's `acoustic_guitar`): its twin, not a miss.
    for item_id, row in rows.items():
        if row["run_item"] is not None:
            continue
        twins = [(int((masks[item_id] & masks[o]).sum()), o) for o, r in rows.items()
                 if r["run_item"] is not None and r["class"] != row["class"]]
        if twins and (best := max(twins))[0] > seen[item_id] / 2:
            row["twin_of"] = best[1]
    return {"scan": scan.name, "shot": shot.name, "items": rows,
            "unmatched": [i for i, r in rows.items() if r["run_item"] is None and "twin_of" not in r]}


def verdict(r: dict, items: dict[str, dict] | None = None) -> dict:
    """Gate 4's bars on each compared large item: what failed, and whether all passed."""
    failed = {}
    for item_id, f in r["items"].items():
        if f["class"] != "large" or "centre_m" not in f:
            continue
        why = []
        if f["plan_m"] > BARS["plan_m"]:
            why.append(f"plan {f['plan_m']:.2f} m")
        if abs(f["height_m"]) > BARS["height_m"]:
            why.append(f"height {f['height_m']:+.2f} m")
        low, high = BARS["size_ratio"]
        if any(v is None or not low <= v <= high for v in f["size_ratio"]):
            why.append(f"size x{'/'.join(str(v) for v in f['size_ratio'])}")
        if f.get("oblong") and f["heading_deg"] > BARS["heading_deg"]:
            why.append(f"heading {f['heading_deg']:.0f} deg")
        if why:
            failed[item_id] = why
    return {"bars": {**BARS, "size_ratio": list(BARS["size_ratio"]), "oblong": OBLONG}, "failed": failed, "ok": not failed}


def judge_run(run_dir: Path, refs: Path = REFS) -> dict:
    result = judge(run_dir, refs)
    result["verdict"] = verdict(result)
    scan, shot = run_scan(run_dir)
    if why := pose_failures(scan.name, refs).get(shot.name):
        result["pose_off"] = why
    (run_dir / "judge_boxes.json").write_text(json.dumps(result, indent=2))
    return result


def describe(run_dir: Path, r: dict) -> str:
    rows = r["items"]
    twins = sum("twin_of" in f for f in rows.values())
    v = r.get("verdict") or verdict(r)
    head = "PASS" if v["ok"] else "FAIL: " + "; ".join(f"{i} {', '.join(w)}" for i, w in v["failed"].items())
    lines = [f"{run_dir.name}: {head}; {len(rows) - twins - len(r['unmatched'])} of {len(rows) - twins} reference items matched"
             + (f"; unmatched: {', '.join(r['unmatched'])}" if r["unmatched"] else "")
             + (f" (pose off, not to be trusted: {r['pose_off']})" if "pose_off" in r else "")]
    for item_id, f in rows.items():
        head = f"  {item_id} ({f['class']}, {f['pixels']} px): "
        if f.get("twin_of"):
            lines.append(head + f"the same thing as {f['twin_of']} in the other reference")
        elif f["run_item"] is None:
            lines.append(head + "NO RUN ITEM")
        elif f.get("sheet"):
            lines.append(head + f"{f['run_item']} {100 * f['share']:.0f}%, a sheet painted on its surface (no box)")
        elif f.get("no_box"):
            lines.append(head + f"{f['run_item']} {100 * f['share']:.0f}%, no box")
        elif "left_out" in f:
            lines.append(head + f"{f['run_item']} {100 * f['share']:.0f}% (reference left out by eye: box not compared)")
        elif f["partial"]:
            lines.append(head + f"{f['run_item']} {100 * f['share']:.0f}% (reference is a face: box not compared)")
        else:
            lines.append(head + f"{f['run_item']} ({f.get('run_class')}{', ' + f['stood'] if f.get('stood') else ''}) {100 * f['share']:.0f}%:"
                         f" centre {f['centre_m']:.2f} m (plan {f['plan_m']:.2f}, height {f['height_m']:+.2f}),"
                         f" size x{'/'.join(str(v) for v in f['size_ratio'])}, heading {f['heading_deg']:.0f} deg")
    return "\n".join(lines)
