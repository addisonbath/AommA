"""`aomma judge-large`: a run's label map against its shot's large-item reference (`aomma ref-large`), pixel by pixel
(gate 2).

At the LiDAR map's size, as `judgeshell`: the run's most common label in each 4 x 4 block; only pixels the references
are sure of, off the boundaries of both the shell and the large-item reference.

Per RoomPlan box in view (`MIN_PIXELS` or more of its pixels scored):
- `item`: the run's item matched to the box, one to one, the largest overlap first (a large item; for an appliance,
  a built-in too; failing that, a small item holding most of it, `small_item`), and its `share` of the box's pixels: under 1, the item's mask is partial or another mask took part of it (the bed). A box whose
  pixels no item holds is a missed listing.
- `shares`: what the run calls the box's pixels: that item, other large items, built-ins, small items, shell, unknown.
Per run large item, off its own outline (a pixel whose 3 x 3 neighbourhood holds another run label):
- `on_shell`: the share of its pixels the shell reference calls wall, floor, ceiling or an opening: a leak.
- `in_other_box`: the share inside another box than the one it holds most of: it took a neighbour's pixels.
- `outside_boxes`: the share in front of the shell but in no box (things RoomPlan has no box for, a bed's pillows
  above its low box): reported, not a fault.
And `things_unknown`: of everything in front of the shell, the share no mask claims.
`verdict` holds gate 2's bars (`BARS`): no box missed, each box at least 90% its item's and small items', no item more
than 10% on the shell; boxes and items under `BAR_MIN_PIXELS` are not judged.
Writes `<run>/judge_large.json`.
"""

import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import maximum_filter, minimum_filter

from aomma.capture import REFS
from aomma.judgeshell import _reference_class, run_on_grid
from aomma.refshell import APPLIANCE, IN_FRONT, UNSCORED, UNSURE
from aomma.refthings import SHARED
from aomma.scancheck import pose_failures
from aomma.stages.roomplan import APPLIANCE_KINDS

MIN_PIXELS = 50
# Gate 2's bars (set after the baseline, before any fix).
# Each judged only on boxes and items with `BAR_MIN_PIXELS` or more scored pixels.
BARS = {
    "missed": 0,  # boxes in view no item holds, at most
    # Of each box's pixels, its item's plus small items' at least: a quilt on a bed, papers on a desk are things of
    # their own, not the item missing.
    "complete": 0.90,
    "on_shell": 0.10,  # of each large item's pixels, at most
}
BAR_MIN_PIXELS = 100
_SHELL = ("floor", "wall", "ceiling", "opening")
# The run's classes as gate 2 reads them: a mask no list names (`other`) counts as a large item.
_CLASS = {"covering": "floor", "other": "large"}


def judge(run_dir: Path, refs: Path = REFS) -> dict:
    run = json.loads((run_dir / "run.json").read_text())
    scan, shot = Path(run["scan"]).name, run["shot"]
    folder = refs / scan / shot
    if not (folder / "large_labels.png").exists():
        raise SystemExit(f"{run_dir}: no large-item reference for {scan} {shot}; run `aomma ref-large scans/{scan}`")
    shell = np.asarray(Image.open(folder / "shell_labels.png"))
    boxes_map = np.asarray(Image.open(folder / "large_labels.png"))
    boxes = {b["value"]: b for b in json.loads((folder / "large_labels.json").read_text())["boxes"]}
    saved = json.loads((run_dir / "labels.json").read_text())
    by_value = {e["value"]: e for e in saved["labels"]}
    values = run_on_grid(np.asarray(Image.open(run_dir / "labels.png")), (shell.shape[1], shell.shape[0]))
    run_class = np.full(values.shape, "unknown", dtype=object)
    run_id = np.full(values.shape, "", dtype=object)
    for value, entry in by_value.items():
        run_class[values == value] = _CLASS.get(entry["class"], entry["class"])
        run_id[values == value] = entry["id"]
    ref_class = _reference_class(shell)
    boundary = (maximum_filter(shell, size=3) != minimum_filter(shell, size=3)) | (maximum_filter(boxes_map, size=3) != minimum_filter(boxes_map, size=3))
    scored = (shell != UNSCORED) & (shell != UNSURE) & ~boundary & (boxes_map != SHARED)
    in_front = scored & ((shell == IN_FRONT) | (shell == APPLIANCE))

    found_boxes, overlaps = {}, []
    for value, box in boxes.items():
        mine = scored & (boxes_map == value)
        n = int(mine.sum())
        if n < MIN_PIXELS:
            continue
        holders = ("large", "builtin") if box["category"] in APPLIANCE_KINDS else ("large",)
        for rid in np.unique(run_id[mine & np.isin(run_class, holders)]):
            overlaps.append((int((mine & (run_id == rid)).sum()), box["id"], rid))
        shares = {kind: round(float((mine & (run_class == kind)).sum() / n), 4) for kind in ("large", "builtin", "small", "unknown")}
        shares["shell"] = round(float((mine & np.isin(run_class, _SHELL)).sum() / n), 4)
        found_boxes[box["id"]] = {"category": box["category"], "pixels": n, "item": None, "share": 0.0, "shares": shares}
    # One item to one box, the largest overlap first: a desk can't hold its chair's box too.
    used = set()
    for count, box_id, rid in sorted(overlaps, reverse=True):
        if found_boxes[box_id]["item"] is None and rid not in used:
            found_boxes[box_id]["item"] = rid
            found_boxes[box_id]["share"] = round(count / found_boxes[box_id]["pixels"], 4)
            used.add(rid)
    # A box no large item holds is still listed if one small item holds most of it: the lister classed it small
    # (OfficeScan 006's crate, `wooden_box`, large on 007). Small items count toward completeness already.
    for box_id, found in found_boxes.items():
        if found["item"] is not None:
            continue
        mine = scored & (boxes_map == next(v for v, b in boxes.items() if b["id"] == box_id))
        held = {rid: int((mine & (run_id == rid)).sum()) for rid in np.unique(run_id[mine & (run_class == "small")])}
        if held and (best := max(held, key=held.get)) and held[best] > found["pixels"] / 2:
            found.update(item=best, share=round(held[best] / found["pixels"], 4), small_item=True)
    # An item's own outline is not judged for leaks, as the references' boundaries are not: where a thing meets the
    # floor, wall or ceiling, or a thin leg stands on the floor, one LiDAR pixel spans both.
    outline = maximum_filter(values, size=3) != minimum_filter(values, size=3)
    items = {}
    for value, entry in by_value.items():
        if _CLASS.get(entry["class"], entry["class"]) != "large":
            continue
        mine = scored & (values == value) & ~outline
        n = int(mine.sum())
        if n < MIN_PIXELS:
            continue
        held = [b for b, f in found_boxes.items() if f["item"] == entry["id"]]
        own = {boxes[v]["value"] for v in boxes if boxes[v]["id"] in held}
        items[entry["id"]] = {
            "pixels": n, "boxes": held,
            "on_shell": round(float((mine & np.isin(ref_class, _SHELL)).sum() / n), 4),
            "in_other_box": round(float((mine & (boxes_map > 0) & ~np.isin(boxes_map, list(own))).sum() / n), 4),
            "outside_boxes": round(float((mine & in_front & (boxes_map == 0)).sum() / n), 4),
        }
    return {
        "scan": scan, "shot": shot,
        "boxes": found_boxes,
        "items": items,
        "missed": [b for b, f in found_boxes.items() if f["item"] is None],
        "things_unknown": round(float((in_front & (run_class == "unknown")).sum() / max(1, in_front.sum())), 4),
        "in_front_pixels": int(in_front.sum()),
    }


def verdict(r: dict) -> dict:
    """Gate 2's bars against one run's measures: what failed on which box or item, and whether all passed."""
    failed = {}
    missed = [b for b in r["missed"] if r["boxes"][b]["pixels"] >= BAR_MIN_PIXELS]
    if len(missed) > BARS["missed"]:
        failed["missed"] = missed
    def complete(f: dict) -> float:
        return f["shares"]["small"] + (0.0 if f.get("small_item") else f["share"])

    short = {b: round(complete(f), 4) for b, f in r["boxes"].items()
             if f["pixels"] >= BAR_MIN_PIXELS and f["item"] is not None and complete(f) < BARS["complete"]}
    if short:
        failed["complete"] = short
    leaks = {i: f["on_shell"] for i, f in r["items"].items() if f["pixels"] >= BAR_MIN_PIXELS and f["on_shell"] > BARS["on_shell"]}
    if leaks:
        failed["on_shell"] = leaks
    return {"bars": BARS, "min_pixels": BAR_MIN_PIXELS, "failed": failed, "ok": not failed}


def judge_run(run_dir: Path, refs: Path = REFS) -> dict:
    result = judge(run_dir, refs)
    result["verdict"] = verdict(result)
    # Gate 0: the reference is drawn by the shot's pose, so where the pose is off the measures are not to be trusted.
    run = json.loads((run_dir / "run.json").read_text())
    if why := pose_failures(Path(run["scan"]).name, refs).get(run["shot"]):
        result["pose_off"] = why
    (run_dir / "judge_large.json").write_text(json.dumps(result, indent=2))
    return result


def describe(run_dir: Path, r: dict) -> str:
    if "left_out" in r:  # judged by an older version, when a shot whose pose failed was not judged
        return f"{run_dir.name}: left out (gate 0): {r['left_out']}"
    v = r.get("verdict") or verdict(r)
    head = "PASS" if v["ok"] else "FAIL: " + "; ".join(
        f"{k} {', '.join(v['failed'][k]) if k == 'missed' else ', '.join(f'{n} {100 * x:.0f}%' for n, x in v['failed'][k].items())}"
        for k in v["failed"])
    lines = [f"{run_dir.name}: {head}; things unknown {100 * r['things_unknown']:.0f}% of {r['in_front_pixels']} px in front"
             + (f"; missed: {', '.join(r['missed'])}" if r["missed"] else "")
             + (f" (pose off, not to be trusted: {r['pose_off']})" if "pose_off" in r else "")]
    for b, f in r["boxes"].items():
        s = f["shares"]
        lines.append(f"  {b} ({f['pixels']} px): {f['item'] or 'NO ITEM'}{' (small)' if f.get('small_item') else ''} {100 * f['share']:.0f}%, small {100 * s['small']:.0f}%"
                     f" | other large {100 * (s['large'] - (f['share'] if f['item'] and not f.get('small_item') else 0)):.0f}%, built-in {100 * s['builtin']:.0f}%,"
                     f" shell {100 * s['shell']:.0f}%, unknown {100 * s['unknown']:.0f}%")
    for i, f in r["items"].items():
        lines.append(f"  item {i} ({f['pixels']} px): on shell {100 * f['on_shell']:.0f}%, in another box {100 * f['in_other_box']:.0f}%,"
                     f" in front in no box {100 * f['outside_boxes']:.0f}%")
    return "\n".join(lines)
