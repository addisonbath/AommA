"""`aomma ref-small` and `aomma judge-small`: the small items of an input shot against those the reference shots found
(gate 3).

**The reference** (`small_labels`): each small reference item (`refs/<scan>/small_items.json`, `aomma ref-items --small`)
is a box from the reference shots' LiDAR. A pixel of the input shot's LiDAR map is the item's when its point lies in
that box, grown by `refshell.ON_M` (the LiDAR's own spread), and the shell reference does not put it on the floor, a
wall, the ceiling or an opening. A pixel in two items' boxes is `SHARED` and not scored. Like gate 2's, it says which
item a pixel shows, not its outline. Writes `refs/<scan>/<shot>/small_labels.png` and `.json`.

**The judge**, at the LiDAR map's size as gates 1 and 2 (the run's most common label in each 4 x 4 block; only pixels
off the references' boundaries):
- per reference item in view (`MIN_PIXELS` or more scored): the small run item holding most of its pixels (one kept
  as a layer on the pixel's owner counts), and its share; apart, the large run item holding most of it (an item one
  lister calls large, a quilt a bed's mask takes in), never taken for the small one; `missed` when no item holds any,
  with what the run calls its pixels instead (shell, built-in, unknown);
- per run small item, off its own outline (gate 2's rule): its share on the shell reference's floor, walls, ceiling or
  openings: a leak, unless nearly all of it is (`FLAT_SHARE`), when it lies flat on the surface (a blind, a picture);
- the label map's `unknown` share of the photo (`labels.json`; bar 10%);
- where the shot has hand-marked small things (`refmarked`), the small listing's recall of
  them, reported, not barred: no bar is agreed for it yet.
`verdict` holds gate 3's bars (`BAR_MIN_PIXELS`). Writes `<run>/judge_small.json`.
"""

import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import maximum_filter, minimum_filter

from aomma.capture import REFS, Scan, View, to_frame
from aomma.judgeshell import _reference_class, run_on_grid
from aomma.refmarked import recall as marked_recall
from aomma.refshell import CEILING, FLOOR, ON_M, OPENING_BASE, UNSCORED, UNSURE, WALL_BASE
from aomma.scancheck import pose_failures, usable_shots

SHARED = 255
# Gate 3's bars (set on the baseline, before any fix): the label map's unknown share at most
# `labels.UNKNOWN_BAR`; no reference item in view (`BAR_MIN_PIXELS` or more scored) held by nothing, an item held by a
# large run item counting as listed (gate 4 places it). Leaks are reported, not barred: at the LiDAR's resolution a small
# item is mostly outline, and the shell reference can't tell a low or leaning thing from the surface under it.
BAR_MIN_PIXELS = 20
# A small item this much on the shell reference's surfaces lies flat on them (a picture on a wall, a blind in its
# window): the reference can't tell it from the surface, as gate 1's `items_on_shell`. Reported apart, not a leak.
FLAT_SHARE = 0.9
MIN_PIXELS = 5
_SHELL = ("floor", "wall", "ceiling", "opening")


def _inside(points: np.ndarray, item: dict, margin: float) -> np.ndarray:
    """Whether each world point lies in the item's box (`refitems.box`: centre, size along/up/across, heading about +y)."""
    turn = np.radians(item["heading_deg"])
    offset = points - np.asarray(item["centre"])
    along = offset[:, 0] * np.cos(turn) - offset[:, 2] * np.sin(turn)
    across = offset[:, 0] * np.sin(turn) + offset[:, 2] * np.cos(turn)
    half = np.asarray(item["size"]) / 2 + margin
    return (np.abs(along) <= half[0]) & (np.abs(offset[:, 1]) <= half[1]) & (np.abs(across) <= half[2])


def small_labels(scan: Scan, view: View, shell: np.ndarray, items: list[dict]) -> tuple[np.ndarray, list[dict]]:
    labels = np.zeros(shell.shape, np.uint8)
    points, pixels = scan.lidar_world(view)
    code = shell[pixels[:, 0], pixels[:, 1]]
    on_shell = (code == FLOOR) | (code == CEILING) | (code >= WALL_BASE)
    eligible = ~on_shell & (code != UNSCORED) & (code != UNSURE)
    count = np.zeros(len(points), int)
    which = np.zeros(len(points), int)
    for i, item in enumerate(items):
        inside = _inside(points, item, ON_M) & eligible
        count += inside
        which[inside] = i + 1
    one = count == 1
    labels[pixels[one, 0], pixels[one, 1]] = which[one]
    labels[pixels[count > 1, 0], pixels[count > 1, 1]] = SHARED
    found = [{"value": i + 1, "id": item["id"], "description": item["description"], "pixels": int((labels == i + 1).sum())}
             for i, item in enumerate(items)]
    return labels, found


def reference_small(scan: Scan, refs: Path = REFS) -> dict[str, list[dict]]:
    """Small-item labels for every input shot that has a shell reference; written beside it."""
    path = refs / scan.name / "small_items.json"
    if not path.exists():
        raise SystemExit(f"{scan.name}: no small reference items; run `aomma ref-items {scan.folder} --small` first")
    items = json.loads(path.read_text())["items"]
    usable = usable_shots(scan.name, refs, poses=False)
    made = {}
    for view in scan.shots:
        if view.role != "input" or not view.depth_size or (usable is not None and view.name not in usable):
            continue
        folder = refs / scan.name / view.name
        if not (folder / "shell_labels.png").exists():
            raise SystemExit(f"{scan.name} {view.name}: no shell reference; run `aomma ref-shell {scan.folder}` first")
        labels, found = small_labels(scan, view, np.asarray(Image.open(folder / "shell_labels.png")), items)
        Image.fromarray(labels).save(folder / "small_labels.png")
        (folder / "small_labels.json").write_text(json.dumps({"items": found, "margin_m": ON_M}, indent=2))
        made[view.name] = [f for f in found if f["pixels"]]
    return made


def judge(run_dir: Path, refs: Path = REFS) -> dict:
    run = json.loads((run_dir / "run.json").read_text())
    scan, shot = Path(run["scan"]).name, run["shot"]
    folder = refs / scan / shot
    if not (folder / "small_labels.png").exists():
        raise SystemExit(f"{run_dir}: no small-item reference for {scan} {shot}; run `aomma ref-small scans/{scan}`")
    shell = np.asarray(Image.open(folder / "shell_labels.png"))
    ref = np.asarray(Image.open(folder / "small_labels.png"))
    ref_items = {i["value"]: i for i in json.loads((folder / "small_labels.json").read_text())["items"]}
    saved = json.loads((run_dir / "labels.json").read_text())
    by_value = {e["value"]: e for e in saved["labels"]}
    values = run_on_grid(np.asarray(Image.open(run_dir / "labels.png")), (shell.shape[1], shell.shape[0]))
    run_class = np.full(values.shape, "unknown", dtype=object)
    run_id = np.full(values.shape, "", dtype=object)
    for value, entry in by_value.items():
        kind = entry["class"]
        run_class[values == value] = {"covering": "floor", "other": "large"}.get(kind, kind)
        run_id[values == value] = entry["id"]
    # Each pixel's small item: the label map's, or a small item kept as a layer on the pixel's owner.
    small_id = np.where(run_class == "small", run_id, "").astype(object)
    if (run_dir / "debug" / "labels_layers.png").exists():
        layers = run_on_grid(np.asarray(Image.open(run_dir / "debug" / "labels_layers.png")), (shell.shape[1], shell.shape[0]))
        for value in np.unique(layers[layers > 0]):
            small_id[layers == value] = by_value[int(value)]["id"]
    boundary = (maximum_filter(shell, size=3) != minimum_filter(shell, size=3)) | (maximum_filter(ref, size=3) != minimum_filter(ref, size=3))
    scored = (shell != UNSCORED) & (shell != UNSURE) & ~boundary & (ref != SHARED)

    found = {}
    for value, item in ref_items.items():
        mine = scored & (ref == value)
        n = int(mine.sum())
        if n < MIN_PIXELS:
            continue
        # The small item holding most of it, a layer counting as its own (a paper on a fridge door is kept as a layer on
        # the door, `labels.SETTLED_LAYER`); and, apart, the large item holding most of it (an item the lister calls large
        # on one shot, a quilt the bed's mask takes in). A large item never passes for the small one: behind a wrongly
        # placed reference box it would (OfficeScan 005's cabinet beside the electric guitar).
        held = {rid: int((mine & (small_id == rid)).sum()) for rid in np.unique(small_id[mine & (small_id != "")])}
        large = {rid: int((mine & (run_id == rid)).sum()) for rid in np.unique(run_id[mine & (run_class == "large")])}
        best = max(held, key=held.get) if held else None
        big = max(large, key=large.get) if large else None
        found[item["id"]] = {
            "description": item["description"], "pixels": n, "item": best,
            "share": round(held[best] / n, 4) if best else 0.0,
            "large_item": big, "large_share": round(large[big] / n, 4) if big else 0.0,
            "shares": {kind: round(float((mine & np.isin(run_class, kinds)).sum() / n), 4) for kind, kinds in
                       (("small", ("small",)), ("large", ("large",)), ("builtin", ("builtin",)), ("shell", _SHELL), ("unknown", ("unknown",)))},
        }
    ref_class = _reference_class(shell)
    outline = maximum_filter(values, size=3) != minimum_filter(values, size=3)
    items = {}
    for value, entry in by_value.items():
        if entry["class"] != "small":
            continue
        mine = scored & (values == value) & ~outline
        n = int(mine.sum())
        if n < MIN_PIXELS:
            continue
        items[entry["id"]] = {"pixels": n, "on_shell": round(float((mine & np.isin(ref_class, _SHELL)).sum() / n), 4)}
    return {
        "scan": scan, "shot": shot,
        "reference_items": found,
        "missed": [i for i, f in found.items() if f["item"] is None],
        "items": items,
        "unknown": saved["report"]["unknown"],
        "marked": marked_recall(run_dir, folder),
    }


def verdict(r: dict) -> dict:
    from aomma.labels import UNKNOWN_BAR

    failed = {}
    if r["unknown"] > UNKNOWN_BAR:
        failed["unknown"] = r["unknown"]
    missed = [i for i, f in r["reference_items"].items()
              if f["pixels"] >= BAR_MIN_PIXELS and f["item"] is None and f["large_share"] < 0.5]
    if missed:
        failed["missed"] = missed
    return {"unknown_bar": UNKNOWN_BAR, "min_pixels": BAR_MIN_PIXELS, "failed": failed, "ok": not failed}


def judge_run(run_dir: Path, refs: Path = REFS) -> dict:
    result = judge(run_dir, refs)
    result["verdict"] = verdict(result)
    run = json.loads((run_dir / "run.json").read_text())
    if why := pose_failures(Path(run["scan"]).name, refs).get(run["shot"]):
        result["pose_off"] = why
    (run_dir / "judge_small.json").write_text(json.dumps(result, indent=2))
    return result


def describe(run_dir: Path, r: dict) -> str:
    found = r["reference_items"]
    v = r.get("verdict") or verdict(r)
    head = "PASS" if v["ok"] else "FAIL: " + "; ".join(
        f"unknown {100 * x:.1f}%" if k == "unknown" else f"missed {', '.join(x)}" for k, x in v["failed"].items())
    lines = [f"{run_dir.name}: {head}; {len(found) - len(r['missed'])} of {len(found)} reference items held by a small run item"
             f"; unknown {100 * r['unknown']:.1f}% of the photo"
             + (f"; missed: {', '.join(r['missed'])}" if r["missed"] else "")
             + (f" (pose off, not to be trusted: {r['pose_off']})" if "pose_off" in r else "")]
    for i, f in found.items():
        s = f["shares"]
        lines.append(f"  {i} ({f['pixels']} px, {f['description']}): {f['item'] or 'NO SMALL ITEM'}"
                     + (f" {100 * f['share']:.0f}%" if f["item"] else "")
                     + (f"; large {f['large_item']} {100 * f['large_share']:.0f}%" if f["large_item"] else "")
                     + f" | built-in {100 * s['builtin']:.0f}%, shell {100 * s['shell']:.0f}%, unknown {100 * s['unknown']:.0f}%")
    if m := r.get("marked"):
        lines.append(f"  hand-marked small things: {m['found']} of {m['real']} found ({100 * (m['recall'] or 0):.0f}%)"
                     + (f"; {m['not_small_listed']} listed marked not small" if m["not_small_listed"] else "")
                     + (f"; {len(m['new'])} listed matching nothing marked" if m["new"] else "")
                     + (f"; missed: {', '.join(m['missed'])}" if m["missed"] else ""))
    flat = [i for i, f in r["items"].items() if f["on_shell"] >= FLAT_SHARE]
    leaks = [f"{i} {100 * f['on_shell']:.0f}%" for i, f in r["items"].items() if 0.1 <= f["on_shell"] < FLAT_SHARE]
    if leaks:
        lines.append("  small items 10%+ on the shell: " + ", ".join(leaks))
    if flat:
        lines.append("  flat on the shell (a blind, a picture; not a leak): " + ", ".join(flat))
    return "\n".join(lines)
