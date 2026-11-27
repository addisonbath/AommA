"""`aomma judge-models`: a run's posed models against the reference items (gate 5).

Each model's box is the upright box round its posed surface (`scoring.model_points`, `refitems.box`), in the scan's world.
The reference item it stands for is the one gate 4 matched to the item (`judgeboxes.judge`: by the reference's own
points in the input shot, like with like first), and it is judged by gate 4's bars (`judgeboxes.BARS`). Also, how far
the model lies from the item's own gate 4 box (`aomma boxes`): centre apart, and the share of its surface inside the
box grown by `INSIDE_M`. Only items whose gate 4 box passed are judged (`judged`); the rest are reported.

Small items' models (`aomma small-models`) are judged against their own gate 4 box, not the reference (whose small
items are too weak to bar, gates 3-4; their comparison is reported): each model must be placed (not left out of the
scene), have `SMALL_BARS["inside"]` of its surface inside its box grown by `INSIDE_M`, and a height within
`SMALL_BARS["height_ratio"]` of the box's (bars set on the baseline, before any fix). Writes
`<run>/judge_models.json`.
"""

import json
from pathlib import Path

import numpy as np

from aomma.capture import REFS
from aomma.judgeboxes import BARS, OBLONG, judge as judge_boxes, verdict as box_verdict, world_box
from aomma.layout import SceneLayout
from aomma.judgesmall import _inside
from aomma.refitems import box
from aomma.scancheck import pose_failures
from aomma.scoring import compare_boxes, model_points
from aomma.views import run_scan, run_to_scan

# A model's surface this near its gate 4 box counts as inside it: a box is fitted to its pixels' 2nd-98th percentiles.
INSIDE_M = 0.05
# Small items' models against their own gate 4 box.
SMALL_BARS = {"inside": 0.5, "height_ratio": (0.67, 1.5)}


def _fails(f: dict, item: dict) -> list[str]:
    why = []
    if f["plan_m"] > BARS["plan_m"]:
        why.append(f"plan {f['plan_m']:.2f} m")
    if abs(f["height_m"]) > BARS["height_m"]:
        why.append(f"height {f['height_m']:+.2f} m")
    low, high = BARS["size_ratio"]
    if any(v is None or not low <= v <= high for v in f["size_ratio"]):
        why.append(f"size x{'/'.join(str(v) for v in f['size_ratio'])}")
    plan = sorted([item["size"][0], item["size"][2]])
    if plan[1] >= OBLONG * plan[0] and f["heading_deg"] > BARS["heading_deg"]:
        why.append(f"heading {f['heading_deg']:.0f} deg")
    return why


def judge(run_dir: Path, refs: Path = REFS) -> dict:
    scan, shot = run_scan(run_dir)
    folder = refs / scan.name
    items = {i["id"]: i for i in json.loads((folder / "items.json").read_text())["items"]}
    if (folder / "small_items.json").exists():
        items |= {i["id"]: i for i in json.loads((folder / "small_items.json").read_text())["items"]}
    boxes = judge_boxes(run_dir, refs)
    failed_box = box_verdict(boxes)["failed"]
    placed = model_points(run_dir, run_to_scan(run_dir, shot))
    rows = {}
    for item_id, f in boxes["items"].items():
        run_id = f.get("run_item")
        if run_id is None or run_id not in placed or "run_box" not in f:
            continue
        points = placed[run_id]
        if len(points) < 10:
            continue
        model_box = box(points)
        row = {"class": f["class"], "run_item": run_id, "model_box": model_box, "gate4_box": f["run_box"],
               "from_gate4_m": round(float(np.linalg.norm(np.subtract(model_box["centre"], f["run_box"]["centre"]))), 3),
               "inside_gate4": round(float(_inside(points, f["run_box"], INSIDE_M).mean()), 3),
               "judged": "centre_m" in f and item_id not in failed_box}
        if "centre_m" in f:
            row |= compare_boxes(model_box, items[item_id])
            row["fails"] = _fails(row, items[item_id])
        rows[item_id] = row
    small = small_models(run_dir, run_to_scan(run_dir, shot), placed)
    failed = {i: r["fails"] for i, r in rows.items() if r["judged"] and r["class"] == "large" and r.get("fails")}
    failed |= {f"{i} (small)": r["fails"] for i, r in small.items() if r["fails"]}
    return {"scan": scan.name, "shot": shot.name, "items": rows, "small_models": small,
            "verdict": {"failed": failed, "ok": not failed}}


def small_models(run_dir: Path, run_to_world: np.ndarray, placed: dict[str, np.ndarray]) -> dict[str, dict]:
    """Each small item's generated model against its own gate 4 box (`SMALL_BARS`). A small item with no model (a box
    or a sheet: too thin for one) is not judged."""
    if not (run_dir / "boxes.json").exists():
        return {}
    boxes = {b["proxy"]["id"]: b for b in json.loads((run_dir / "boxes.json").read_text())["boxes"] if b["kind"] == "small"}
    scene = SceneLayout.model_validate_json((run_dir / "layout.json").read_text())
    rows = {}
    for placement in scene.objects:
        is_model = placement.asset.endswith(".usdc") or "_fitted" in placement.asset
        if placement.name not in boxes or not is_model or (run_dir / "assets" / placement.name / "flat.json").exists():
            continue
        points = placed.get(placement.name)
        if placement.left_out or points is None or len(points) < 10:
            rows[placement.name] = {"left_out": True, "fails": ["left out of the scene"]}
            continue
        gate4, model_box = world_box(boxes[placement.name]["proxy"], run_to_world), box(points)
        row = {"model_box": model_box, "gate4_box": gate4,
               "from_gate4_m": round(float(np.linalg.norm(np.subtract(model_box["centre"], gate4["centre"]))), 3),
               "inside_gate4": round(float(_inside(points, gate4, INSIDE_M).mean()), 3),
               "height_ratio": round(model_box["size"][1] / gate4["size"][1], 2) if gate4["size"][1] > 0 else None}
        why = []
        if row["inside_gate4"] < SMALL_BARS["inside"]:
            why.append(f"{100 * row['inside_gate4']:.0f}% inside its box")
        low, high = SMALL_BARS["height_ratio"]
        if row["height_ratio"] is None or not low <= row["height_ratio"] <= high:
            why.append(f"height x{row['height_ratio']}")
        rows[placement.name] = row | {"fails": why}
    return rows


def judge_run(run_dir: Path, refs: Path = REFS) -> dict:
    result = judge(run_dir, refs)
    scan, shot = run_scan(run_dir)
    if why := pose_failures(scan.name, refs).get(shot.name):
        result["pose_off"] = why
    (run_dir / "judge_models.json").write_text(json.dumps(result, indent=2))
    return result


def describe(run_dir: Path, r: dict) -> str:
    v = r["verdict"]
    head = "PASS" if v["ok"] else "FAIL: " + "; ".join(f"{i} {', '.join(w)}" for i, w in v["failed"].items())
    lines = [f"{run_dir.name}: {head}; {len(r['items'])} models matched to reference items"
             + (f" (pose off, not to be trusted: {r['pose_off']})" if "pose_off" in r else "")]
    for item_id, f in r["items"].items():
        head = f"  {item_id} ({f['class']}) -> {f['run_item']}: "
        if "plan_m" in f:
            head += (f"plan {f['plan_m']:.2f} m, height {f['height_m']:+.2f}, size x{'/'.join(str(v) for v in f['size_ratio'])},"
                     f" heading {f['heading_deg']:.0f} deg; ")
        lines.append(head + f"{f['from_gate4_m']:.2f} m from its gate 4 box, {100 * f['inside_gate4']:.0f}% inside it"
                     + ("" if f["judged"] else " (not judged: its gate 4 box failed or was not compared)"))
    if small := r.get("small_models"):
        bad = sum(bool(f["fails"]) for f in small.values())
        lines.append(f"  small models against their gate 4 boxes: {len(small) - bad} of {len(small)} pass")
    return "\n".join(lines)
