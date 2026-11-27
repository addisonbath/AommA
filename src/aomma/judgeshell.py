"""`aomma judge-shell`: a run's label map (`aomma labels`) against its shot's shell reference (`aomma ref-shell`), pixel by
pixel (gate 1).

Both are compared at the LiDAR map's size (256 x 192): each of its pixels takes the run's label most common in the 4 x 4
block of the photo under it. Only pixels the reference is sure of are scored (not `unscored`, not `unsure`), and not
those on a boundary of the reference (a pixel whose 3 x 3 neighbourhood holds another reference code), where one LiDAR
pixel spans both sides.

The run's classes are read as the reference's: `floor` and `covering` are floor; `large`, `small` and `other` are items.
What the reference cannot tell: a flat thing on a wall or floor (a poster, a rug) lies on that surface, so the reference
calls it shell. A run that gives it to an item is not wrong, so those pixels are counted apart (`items_on_shell`),
not against the shell.

Measures, each a share of the reference's pixels of that kind:
- `iou`: per shell class (floor, wall, ceiling, opening), the run's against the reference's, with both pixel counts.
  A class the reference holds none of and the run does (a ceiling mask on a wall) scores 0. `false`: the run's class
  where the reference has another, as a share of all scored pixels.
- `walls`: each reference wall matched to the run's wall that covers most of it; its IoU, and whether another run wall
  takes 20% or more of it (`split`). A run wall covering two reference walls is a `merged`.
- `shell_on_things`: of what stands in front of the shell (`in_front`), the share the run calls shell. The worst fault:
  a wall or floor mask over furniture.
- `shell_missed`: of the reference's shell, the share the run leaves `unknown`.
- `items_on_shell`: of the reference's shell, the share the run gives to an item (flat things; not a fault by itself).
- `fixtures`: of the reference's fixtures, the shares the run calls built-in, shell, item and unknown; appliances may be
  built-in or item, both right.
Each share carries how many pixels it is of; under `MIN_PIXELS` it is left out (None). `verdict` holds each of gate 1's
bars (`BARS`), a bar the shot doesn't show enough of to judge left as None (not a failure).
Writes `<run>/judge_shell.json`.
"""

import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import maximum_filter, minimum_filter

from aomma.capture import REFS
from aomma.refshell import APPLIANCE, CEILING, FIXTURE, FLOOR, IN_FRONT, OPENING_BASE, UNSCORED, UNSURE, WALL_BASE
from aomma.scancheck import pose_failures

# A reference wall another run wall takes this share of is split between them.
SPLIT_SHARE = 0.2
# A share taken over fewer pixels than this says nothing, and is left out (KitchenScan 008 has a handful in front).
MIN_PIXELS = 100

# Gate 1's bars (set after the baseline, before any fix). Openings are reported, not judged (glass gives little
# sure LiDAR); shell the run leaves unknown is gate 3's (mostly flat things not yet listed).
BARS = {
    "floor_iou": 0.90,  # at least
    "wall_iou": 0.85,  # at least
    "false_ceiling": 0.01,  # at most, of the scored pixels
    # At most, of the scored pixels: floor where the reference has none of it (KitchenScan 011's backsplash, 0.59%, where
    # no floor is in view). The other 15 shots are at 0.23% or less.
    "false_floor": 0.005,
    "shell_on_things": 0.02,  # at most, where the reference has `SHELL_ON_THINGS_MIN` pixels in front
    "fixtures_builtin": 0.80,  # at least
}
SHELL_ON_THINGS_MIN = 500

SHELL = ("floor", "wall", "ceiling", "opening")
# The pictures the Labels tab draws (`debug/shell_reference.png`, `debug/shell_against.png`), at the LiDAR map's size.
REFERENCE_CODES = {"floor": 1, "wall": 2, "ceiling": 3, "opening": 4, "fixture": 5, "appliance": 6, "in_front": 7}
AGAINST_CODES = {"agree": 1, "shell_on_things": 2, "wrong_surface": 3, "shell_missed": 4, "items_on_shell": 5,
                 "fixture_not_builtin": 6}
_RUN_CLASS = {"floor": "floor", "covering": "floor", "wall": "wall", "ceiling": "ceiling", "opening": "opening",
              "builtin": "builtin", "large": "item", "small": "item", "other": "item"}


def _reference_class(code: np.ndarray) -> np.ndarray:
    out = np.full(code.shape, "", dtype=object)
    out[code == FLOOR] = "floor"
    out[code == CEILING] = "ceiling"
    out[(code >= WALL_BASE) & (code < OPENING_BASE)] = "wall"
    out[code >= OPENING_BASE] = "opening"
    out[code == FIXTURE] = "fixture"
    out[code == APPLIANCE] = "appliance"
    out[code == IN_FRONT] = "in_front"
    return out


def run_on_grid(labels: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """The run's label values at the LiDAR map's (width, height): the most common in each block."""
    width, height = size
    h, w = labels.shape
    by, bx = h // height, w // width
    blocks = labels[: by * height, : bx * width].reshape(height, by, width, bx).transpose(0, 2, 1, 3).reshape(height, width, -1)
    out = np.zeros((height, width), labels.dtype)
    for value in np.unique(blocks):
        count = (blocks == value).sum(axis=-1)
        better = count > (blocks == out[..., None]).sum(axis=-1)
        out[better] = value
    return out


def judge(run_dir: Path, refs: Path = REFS) -> dict:
    run = json.loads((run_dir / "run.json").read_text())
    scan, shot = Path(run["scan"]).name, run["shot"]
    ref_path = refs / scan / shot / "shell_labels.png"
    if not ref_path.exists():
        raise SystemExit(f"{run_dir}: no shell reference for {scan} {shot}; run `aomma ref-shell scans/{scan}`")
    code = np.asarray(Image.open(ref_path))
    saved = json.loads((run_dir / "labels.json").read_text())
    by_value = {e["value"]: e for e in saved["labels"]}
    values = run_on_grid(np.asarray(Image.open(run_dir / "labels.png")), (code.shape[1], code.shape[0]))
    run_class = np.full(values.shape, "unknown", dtype=object)
    run_id = np.full(values.shape, "", dtype=object)
    for value, entry in by_value.items():
        run_class[values == value] = _RUN_CLASS.get(entry["class"], "item")
        run_id[values == value] = entry["id"]
    ref_class = _reference_class(code)
    boundary = maximum_filter(code, size=3) != minimum_filter(code, size=3)
    scored = (code != UNSCORED) & (code != UNSURE) & ~boundary

    def share(of: np.ndarray, what: np.ndarray) -> dict:
        n = int((scored & of).sum())
        return {"share": round(float((scored & of & what).sum() / n), 4) if n >= MIN_PIXELS else None, "of": n}

    iou = {}
    for kind in SHELL:
        a, b = scored & (ref_class == kind), scored & (run_class == kind)
        if a.sum() >= MIN_PIXELS or b.sum() >= MIN_PIXELS:
            iou[kind] = {"iou": round(float((a & b).sum() / (a | b).sum()), 4), "reference": int(a.sum()), "run": int(b.sum()),
                         # The run's X where the reference has something else, as a share of every scored pixel.
                         "false": round(float((b & ~a).sum() / scored.sum()), 4)}
    ref_shell = np.isin(ref_class, SHELL)
    run_shell = np.isin(run_class, SHELL)
    walls, owners = {}, {}
    for wall_code in np.unique(code[scored & (ref_class == "wall")]):
        mine = scored & (code == wall_code)
        taken = {rid: int((mine & (run_id == rid) & (run_class == "wall")).sum()) for rid in np.unique(run_id[mine & (run_class == "wall")])}
        if not taken:
            walls[f"wall_{int(wall_code) - WALL_BASE}"] = {"pixels": int(mine.sum()), "run_wall": None, "iou": 0.0, "split": False}
            continue
        best = max(taken, key=taken.get)
        theirs = scored & (run_id == best)
        walls[f"wall_{int(wall_code) - WALL_BASE}"] = {
            "pixels": int(mine.sum()), "run_wall": best,
            "iou": round(float((mine & theirs).sum() / (mine | theirs).sum()), 4),
            "split": bool(sum(n >= SPLIT_SHARE * mine.sum() for n in taken.values()) > 1),
        }
        owners.setdefault(best, []).append(f"wall_{int(wall_code) - WALL_BASE}")
    fixture = ref_class == "fixture"
    appliance = ref_class == "appliance"
    _save_pictures(run_dir, code, ref_class, run_class, scored)
    return {
        "scan": scan, "shot": shot,
        "scored_pixels": int(scored.sum()),
        "iou": iou,
        "walls": walls,
        "merged": {rid: refs_ for rid, refs_ in owners.items() if len(refs_) > 1},
        "shell_on_things": share(ref_class == "in_front", run_shell),
        "shell_missed": share(ref_shell, run_class == "unknown"),
        "items_on_shell": share(ref_shell, run_class == "item"),
        "fixtures": {
            "builtin": share(fixture, run_class == "builtin"), "shell": share(fixture, run_shell),
            "item": share(fixture, run_class == "item"), "unknown": share(fixture, run_class == "unknown"),
        },
        "appliances": {
            "builtin_or_item": share(appliance, np.isin(run_class, ("builtin", "item"))), "shell": share(appliance, run_shell),
            "unknown": share(appliance, run_class == "unknown"),
        },
    }


def _save_pictures(run_dir: Path, code: np.ndarray, ref_class: np.ndarray, run_class: np.ndarray, scored: np.ndarray) -> None:
    """The reference's classes and where the run disagrees with it, for the Labels tab."""
    reference = np.zeros(code.shape, np.uint8)
    for kind, value in REFERENCE_CODES.items():
        reference[ref_class == kind] = value
    reference[code == UNSURE] = 8
    shell = np.isin(ref_class, SHELL)
    run_shell = np.isin(run_class, SHELL)
    against = np.zeros(code.shape, np.uint8)
    agree = (ref_class == run_class) | ((ref_class == "in_front") & np.isin(run_class, ("item", "builtin"))) \
        | ((ref_class == "fixture") & (run_class == "builtin")) | ((ref_class == "appliance") & np.isin(run_class, ("builtin", "item")))
    against[agree] = AGAINST_CODES["agree"]
    against[(ref_class == "in_front") & run_shell] = AGAINST_CODES["shell_on_things"]
    against[shell & run_shell & (ref_class != run_class)] = AGAINST_CODES["wrong_surface"]
    against[shell & (run_class == "unknown")] = AGAINST_CODES["shell_missed"]
    against[shell & np.isin(run_class, ("item", "builtin"))] = AGAINST_CODES["items_on_shell"]
    against[(ref_class == "fixture") & (run_class != "builtin")] = AGAINST_CODES["fixture_not_builtin"]
    against[~scored] = 0
    (run_dir / "debug").mkdir(exist_ok=True)
    Image.fromarray(reference).save(run_dir / "debug" / "shell_reference.png")
    Image.fromarray(against).save(run_dir / "debug" / "shell_against.png")


def verdict(r: dict) -> dict:
    """Gate 1's bars against one run's measures: each bar's value and whether it passed (None where the shot does not
    show enough to say), and whether all passed."""
    iou = r["iou"]
    checks = {
        "floor_iou": (iou.get("floor", {}).get("iou") if iou.get("floor", {}).get("reference", 0) >= MIN_PIXELS else None, ">="),
        "wall_iou": (iou.get("wall", {}).get("iou") if iou.get("wall", {}).get("reference", 0) >= MIN_PIXELS else None, ">="),
        "false_ceiling": (iou.get("ceiling", {}).get("false", 0.0), "<="),
        "false_floor": (iou.get("floor", {}).get("false", 0.0), "<="),
        "shell_on_things": (r["shell_on_things"]["share"] if r["shell_on_things"]["of"] >= SHELL_ON_THINGS_MIN else None, "<="),
        "fixtures_builtin": (r["fixtures"]["builtin"]["share"], ">="),
    }
    found = {}
    for name, (value, way) in checks.items():
        ok = None if value is None else (value >= BARS[name] if way == ">=" else value <= BARS[name])
        found[name] = {"value": value, "bar": BARS[name], "ok": ok}
    return {"checks": found, "ok": all(c["ok"] is not False for c in found.values())}


def judge_run(run_dir: Path, refs: Path = REFS) -> dict:
    result = judge(run_dir, refs)
    result["verdict"] = verdict(result)
    # Gate 0: the reference is drawn by the shot's pose, so where the pose is off the measures are not to be trusted.
    run = json.loads((run_dir / "run.json").read_text())
    if why := pose_failures(Path(run["scan"]).name, refs).get(run["shot"]):
        result["pose_off"] = why
    (run_dir / "judge_shell.json").write_text(json.dumps(result, indent=2))
    return result


def describe(run_dir: Path, r: dict) -> str:
    if "left_out" in r:  # judged by an older version, when a shot whose pose failed was not judged
        return f"{run_dir.name}: left out (gate 0): {r['left_out']}"
    pct = lambda v: "-" if v["share"] is None else f"{100 * v['share']:.0f}%"
    iou = ", ".join(f"{k} {v['iou']:.2f} ({v['reference']}/{v['run']} px)" for k, v in r["iou"].items())
    walls = ", ".join(f"{w} {v['iou']:.2f}" + (" split" if v["split"] else "") for w, v in r["walls"].items())
    merged = f"; merged: {r['merged']}" if r["merged"] else ""
    f = r["fixtures"]
    failed = [f"{k} {v['value']:.3f} (bar {v['bar']})" for k, v in r["verdict"]["checks"].items() if v["ok"] is False]
    head = "PASS" if r["verdict"]["ok"] else "FAIL: " + ", ".join(failed)
    if "pose_off" in r:
        head += f" (pose off, not to be trusted: {r['pose_off']})"
    return (f"{run_dir.name}: {head}\n  IoU {iou}\n  walls {walls}{merged}\n"
            f"  shell on things {pct(r['shell_on_things'])}, shell missed {pct(r['shell_missed'])}, items on shell {pct(r['items_on_shell'])}\n"
            f"  fixtures: built-in {pct(f['builtin'])}, shell {pct(f['shell'])}, item {pct(f['item'])}, unknown {pct(f['unknown'])}")
