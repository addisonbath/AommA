"""Missed listings: a panoptic segmenter's flags, each looked at again by the VLM.

Runs after the small items and the labels (`aomma relook`), on the run itself. Mask2Former segments the photo
(`aomma.panoptic`, in its own pinned environment). Each segment of `MIN_SHARE` of the photo or more, half or more of
it unknown before growth (`unclaimed`), is flagged; the segmenter's class is never read. Each flag is
shown to the VLM as a close-up of the photo, the segment outlined and the rest dimmed, with what is already listed
there named: is there a thing inside the outline that is not listed yet, and what role and kind does it have (the
kind, from its role's survey's list: a large item kept as `other` lost its
nightstand or appliance)? A `room` role was tried and cost real things, so a shell part it lists (BedroomScan 010's
window sliver, as a built-in) is kept; it was found harmless at gates 1 and 2. What it lists is masked by SAM 3 from its box (`stages.segment.segment`); two new entries whose masks overlap by more than
half of the smaller (`SAME_THING`) are one thing, the first kept. Kept entries go into the survey for their role
(marked `"relook": true`), their masks into masks/, and the run is labelled again.

Writes `panoptic.npz`, `relook.json` (every flag, the reply, the entries and their masks' measures, unknown before
growth before and after), `debug/relook/<segment>.jpg` (what the VLM was shown) and `debug/relook.jpg` (flags outlined
in orange, each kept entry tinted, a flag with nothing kept marked `none`). Run again, it first takes out its earlier
entries and labels the run without them.
"""

import json
import re
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import binary_dilation

from aomma.labels import SETTLED_GROWN, label_run
from aomma.ledger import Ledger
from aomma.stages.image import to_jpeg
from aomma.stages.inventory import Inventory, InventoryObject
from aomma.stages.large import KINDS as LARGE_KINDS
from aomma.stages.segment import segment
from aomma.stages.shell import BUILT_IN_KINDS

MIN_SHARE = 0.001  # a segment under 0.1% of the photo is not read
UNCLAIMED = 0.5  # half or more unknown before growth
VIEW = 1.5  # the close-up is this multiple of the segment's box, as `inventory.split_entry`
MIN_VIEW = 0.25  # ... and at least this share of the photo's longer side, so a small flag is seen in its setting
SAME_THING = 0.5  # as `stages.small`
SHELL = {"floor", "ceiling", "covering", "wall", "opening"}
SURVEYS = {"built_in": ("survey.json", "built_ins"), "large": ("large_survey.json", "objects"),
           "small": ("small_survey.json", "objects")}
ROLES = tuple(SURVEYS)
# The kinds each role's survey gives (a small item has none). A kept entry's kind goes into its survey's `kinds`, as
# the survey's own entries' do, so a large item or built-in found here is read like one listed first time.
KINDS = {"built_in": BUILT_IN_KINDS, "large": LARGE_KINDS, "small": ("other",)}
OUTLINE = (255, 0, 255)
DIM = 0.6  # what lies outside the outline, dimmed to this, still seen as the setting

PANOPTIC_ENV = ["uv", "run", "--no-project", "--with", "torch==2.15.0", "--with", "transformers==5.24.0",
                "--with", "timm==1.0.31", "--with", "pillow", "--with", "numpy", "python"]

PROMPT = """\
This is a close-up from a photo of a {room_type} that is being rebuilt as a 3D scene. Everything in the
scene has to be listed to be built. The room's walls, floor, ceiling, doors and windows are already built,
and these are already listed and must not be listed again:
{listed}

Look inside the magenta outline (what lies outside it is dimmed). Nothing listed covers most of it. Is there a physical thing there that
is not listed yet?
- If there is, list it; if the outline takes in several separate things, list each. Its box covers all
  of the thing, also where it runs past the outline.
- Give each one a role:
  - "built_in": fixed in place, part of the building rather than furniture, such as counters, cabinet
    runs, built-in shelving, a kitchen island, a fixed sink, a fireplace, a bathtub.
  - "large": a piece of furniture or an appliance standing on the floor, such as a bed, sofa, chair,
    table, desk, dresser, free-standing cabinet or shelving unit, fridge, stove, washer.
  - "small": anything else: what stands on, lies on or hangs from furniture, a built-in or a wall, and
    what stands on the floor and can be carried, such as lamps, plants, cushions, bottles, boxes, pictures
    and curtains.
- List nothing if the outline is only the room itself (a wall, the floor, the ceiling, a door, a window
  with its blinds, paint, tiles, trim), part of something already listed, a shadow or a reflection, or
  something seen through a doorway or a window.

Reply with JSON only, in the form {{"objects": [...]}}, an empty list if there is nothing to add. Each entry has:
- "id": unique snake_case name
- "role": "built_in", "large" or "small"
- "kind": for a built-in one of {built_in_kinds}; for a large item one of {large_kinds};
  for a small item "other"
- "bbox_2d": [x1, y1, x2, y2] bounding box in this close-up, normalised to 0-1000
- "description": at most 12 words covering colour, material and shape
- "size_m": [width, depth, height] estimated real-world size in meters
- "on": the id of the listed thing it rests on or hangs from, or "floor" or "wall"
- "count": how many separate things the entry covers
"""

SCHEMA = {
    "type": "object",
    "properties": {
        "objects": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "role": {"type": "string", "enum": list(ROLES)},
                    "kind": {"type": "string"},
                    "bbox_2d": {"type": "array", "items": {"type": "number"}},
                    "description": {"type": "string"},
                    "size_m": {"type": "array", "items": {"type": "number"}},
                    "on": {"type": "string"},
                    "count": {"type": "integer"},
                },
                "required": ["id", "role", "kind", "bbox_2d", "description", "size_m", "on"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["objects"],
    "additionalProperties": False,
}


def predict(runs: list[Path]) -> None:
    """Mask2Former's segments for each run that has none yet (`aomma.panoptic`, in its pinned environment)."""
    todo = [run for run in runs if not (run / "panoptic.npz").exists()]
    if todo:
        subprocess.run([*PANOPTIC_ENV, str(Path(__file__).parent.parent / "panoptic.py"), *map(str, todo)], check=True)


def flags(run_dir: Path) -> list[dict]:
    """The `unclaimed` segments: id, mask, share of the photo, share unknown before growth."""
    segments = np.load(run_dir / "panoptic.npz")["segments"]
    labels = np.array(Image.open(run_dir / "labels.png"))
    grown = np.array(Image.open(run_dir / "debug" / "labels_settled.png")) == SETTLED_GROWN
    before = (labels == 0) | grown
    found = []
    for seg_id in np.unique(segments[segments > 0]):
        seg = segments == seg_id
        share = float(seg.mean())
        if share >= MIN_SHARE and before[seg].mean() >= UNCLAIMED:
            found.append({"segment": int(seg_id), "mask": seg, "share": round(share, 4),
                          "unknown_before": round(float(before[seg].mean()), 3)})
    return found


def view_of(mask: np.ndarray) -> tuple[int, int, int, int]:
    """The close-up round a mask, in photo pixels (input.jpg's)."""
    h, w = mask.shape
    rows, cols = np.nonzero(mask)
    cx, cy = (cols.min() + cols.max() + 1) / 2, (rows.min() + rows.max() + 1) / 2
    least = MIN_VIEW * max(h, w)
    half_w = max((cols.max() + 1 - cols.min()) * VIEW, least) / 2
    half_h = max((rows.max() + 1 - rows.min()) * VIEW, least) / 2
    return (int(max(cx - half_w, 0)), int(max(cy - half_h, 0)), int(min(cx + half_w, w)), int(min(cy + half_h, h)))


def close_up(photo: Image.Image, mask: np.ndarray, view: tuple[int, int, int, int]) -> Image.Image:
    """The full-resolution photo in the view, the mask's edge drawn on it and what lies outside it dimmed (where the
    mask runs to the photo's edge, the outline alone is open on that side)."""
    sx, sy = photo.width / mask.shape[1], photo.height / mask.shape[0]
    crop = photo.crop((round(view[0] * sx), round(view[1] * sy), round(view[2] * sx), round(view[3] * sy))).convert("RGB")
    edge = binary_dilation(mask, iterations=2) & ~mask
    edge = edge[view[1]:view[3], view[0]:view[2]]
    edge_img = Image.fromarray(edge.astype(np.uint8) * 255).resize(crop.size, Image.NEAREST)
    thick = binary_dilation(np.asarray(edge_img) > 0, iterations=max(1, round(sx)))
    inside = np.asarray(Image.fromarray(mask[view[1]:view[3], view[0]:view[2]].astype(np.uint8) * 255)
                        .resize(crop.size, Image.NEAREST)) > 0
    out = np.asarray(crop).astype(float)
    out[~inside] *= DIM
    out = out.astype(np.uint8)
    out[thick] = OUTLINE
    return Image.fromarray(out)


def listed_in(run_dir: Path, view: tuple[int, int, int, int]) -> list[tuple[str, str]]:
    """The run's labels (not the shell) that hold pixels in the view: (id, class)."""
    labels = np.array(Image.open(run_dir / "labels.png"))[view[1]:view[3], view[0]:view[2]]
    meta = json.loads((run_dir / "labels.json").read_text())["labels"]
    here = set(np.unique(labels).tolist())
    return [(e["id"], e["class"]) for e in meta if e["value"] in here and e["class"] not in SHELL]


def parse(text: str) -> list[dict]:
    match = re.search(r"[\[{].*[\]}]", text, re.DOTALL)
    if not match:
        return []
    data = json.loads(match.group())
    if isinstance(data, list):
        data = {"objects": data}
    return [o for o in data.get("objects", []) if o.get("role") in ROLES and len(o.get("bbox_2d", [])) == 4]


def _quoted(kinds: tuple[str, ...]) -> str:
    return ", ".join(f'"{k}"' for k in kinds)


def take_out_earlier(run_dir: Path) -> bool:
    """Remove a previous relook's entries (survey entries marked `relook`, and their masks); True if there were any."""
    removed = False
    for name, key in SURVEYS.values():
        path = run_dir / name
        if not path.exists():
            continue
        data = json.loads(path.read_text())
        mine = [e for e in data.get(key, []) if e.get("relook")]
        if mine:
            data[key] = [e for e in data[key] if not e.get("relook")]
            for e in mine:
                data.get("kinds", {}).pop(e["id"], None)
            path.write_text(json.dumps(data, indent=2))
            for e in mine:
                (run_dir / "masks" / f"{e['id']}.png").unlink(missing_ok=True)
            removed = True
    return removed


def add_to_surveys(run_dir: Path, entries: list[dict]) -> None:
    """Each kept entry into the run's survey for its role, which is where `labels.classes` reads the class from."""
    for role, (name, key) in SURVEYS.items():
        mine = [e for e in entries if e["role"] == role]
        if not mine:
            continue
        path = run_dir / name
        data = json.loads(path.read_text()) if path.exists() else {key: []}
        data.setdefault(key, []).extend({k: e[k] for k in ("id", "bbox_2d", "description", "size_m", "on", "count")}
                                        | {"kind": e.get("kind", "other"), "relook": True} for e in mine)
        if role != "small":
            data.setdefault("kinds", {}).update({e["id"]: e.get("kind", "other") for e in mine})
        path.write_text(json.dumps(data, indent=2))


def relook(run_dir: Path, vlm, ledger: Ledger, comfy) -> dict:
    """The missed-listings look on one run whose small items are listed (see the module's docstring)."""
    if take_out_earlier(run_dir) or not (run_dir / "labels.json").exists():
        label_run(run_dir)
    image = Image.open(run_dir / "input.jpg").convert("RGB")
    photo = Image.open(run_dir / "photo.jpg") if (run_dir / "photo.jpg").exists() else image
    if abs(photo.width / photo.height - image.width / image.height) > 0.01:
        photo = image
    room_type = json.loads((run_dir / "survey.json").read_text()).get("room_type", "room")
    before = json.loads((run_dir / "labels.json").read_text())["report"]
    taken = {e["id"] for e in json.loads((run_dir / "labels.json").read_text())["labels"]}
    taken |= {p.stem for p in (run_dir / "masks").glob("*.png")}
    crops = run_dir / "debug" / "relook"
    crops.mkdir(parents=True, exist_ok=True)

    flagged, entries = flags(run_dir), []
    for flag in flagged:
        view = view_of(flag["mask"])
        shown = close_up(photo, flag["mask"], view)
        shown.save(crops / f"{flag['segment']}.jpg", quality=88)
        listed = listed_in(run_dir, view)
        prompt = PROMPT.format(room_type=room_type, built_in_kinds=_quoted(BUILT_IN_KINDS), large_kinds=_quoted(LARGE_KINDS),
                               listed="\n".join(f"- {oid} ({kind})" for oid, kind in listed) or "- nothing")
        text, usage = vlm.ask(to_jpeg(shown), prompt, SCHEMA, label=f"missed listing at segment {flag['segment']}")
        ledger.record("relook", usage)
        try:
            found = parse(text)
        except (json.JSONDecodeError, ValueError):
            found = []
        flag["reply"], flag["entries"] = text, []
        vw, vh = view[2] - view[0], view[3] - view[1]
        for e in found:
            ax, ay, bx, by = (float(v) for v in e["bbox_2d"])
            box = [(view[0] + ax / 1000 * vw) * 1000 / image.width, (view[1] + ay / 1000 * vh) * 1000 / image.height,
                   (view[0] + bx / 1000 * vw) * 1000 / image.width, (view[1] + by / 1000 * vh) * 1000 / image.height]
            if box[2] <= box[0] or box[3] <= box[1]:
                continue
            base = re.sub(r"\W+", "_", str(e["id"]).lower()).strip("_") or "object"
            oid, n = base, 2
            while oid in taken:
                oid, n = f"{base}_{n}", n + 1
            taken.add(oid)
            size = [float(v) for v in e.get("size_m", [])][:3]
            kind = str(e.get("kind", "other"))
            entry = {"id": oid, "role": e["role"], "kind": kind if kind in KINDS[e["role"]] else "other", "bbox_2d": box, "description": str(e.get("description", "")),
                     "size_m": size + [0.1] * (3 - len(size)), "on": str(e.get("on", "floor")),
                     "count": int(e.get("count", 1) or 1), "segment": flag["segment"]}
            entries.append(entry)
            flag["entries"].append(oid)

    kept, masks = [], {}
    things = [e for e in entries if not e.get("dropped")]
    if things:
        image_name = comfy.upload_image(to_jpeg(image), f"aomma_{run_dir.name}_relook.jpg")
        inventory = Inventory(objects=[InventoryObject(**{k: e[k] for k in ("id", "bbox_2d", "description", "size_m", "on", "count")})
                                       for e in things])
        got, infos = segment(comfy, image_name, image.size, inventory)
        for e in things:
            mask = got.get(e["id"])
            seg = next(f["mask"] for f in flagged if f["segment"] == e["segment"])
            e["mask_px"] = int(mask.sum()) if mask is not None else 0
            e["box_iou"] = round(float(infos[e["id"]].box_iou), 3) if e["id"] in infos else None
            e["covers_segment"] = round(float((mask & seg).sum() / seg.sum()), 3) if e["mask_px"] else 0.0
            same = next((k for k in kept if e["mask_px"] and (mask & masks[k]).sum() > SAME_THING * min(mask.sum(), masks[k].sum())), None)
            if not e["mask_px"]:
                e["dropped"] = "no mask"
            elif same:
                e["dropped"] = f"same as {same}"
            else:
                kept.append(e["id"])
                masks[e["id"]] = mask
                Image.fromarray(mask).save(run_dir / "masks" / f"{e['id']}.png")
        add_to_surveys(run_dir, [e for e in entries if e["id"] in kept])
    after = label_run(run_dir)

    result = {
        "flags": [{k: v for k, v in f.items() if k != "mask"} for f in flagged],
        "entries": entries,
        "kept": kept,
        "unknown_before_growth": [before["unknown_before_growth"], after["unknown_before_growth"]],
        "unknown": [before["unknown"], after["unknown"]],
    }
    (run_dir / "relook.json").write_text(json.dumps(result, indent=1))
    overlay(image, flagged, entries, masks).save(run_dir / "debug" / "relook.jpg", quality=88)
    return result


def overlay(image: Image.Image, flagged: list[dict], entries: list[dict], masks: dict[str, np.ndarray]) -> Image.Image:
    out = np.asarray(image).astype(float) * 0.55
    tints = [(60, 220, 90), (60, 160, 255), (255, 220, 40), (230, 60, 230), (40, 230, 230)]
    for k, mask in enumerate(masks.values()):
        out[mask] = 0.4 * np.asarray(image)[mask] + 0.6 * np.array(tints[k % len(tints)])
    for f in flagged:
        edge = binary_dilation(f["mask"], iterations=3) & ~f["mask"]
        out[edge] = (255, 140, 0)
    canvas = Image.fromarray(out.astype(np.uint8))
    pen = ImageDraw.Draw(canvas)
    for f in flagged:
        rows, cols = np.nonzero(f["mask"])
        at = (int(cols.min()), int(rows.min()))
        names = [f"{e['id']} ({e['role']}{', ' + e['dropped'] if e.get('dropped') else ''})"
                 for e in entries if e["segment"] == f["segment"]] or ["none"]
        text = f"#{f['segment']}: " + "; ".join(names)
        box = pen.textbbox(at, text)
        pen.rectangle(box, fill=(0, 0, 0))
        pen.text(at, text, fill=(255, 140, 0))
    return canvas


def describe(run_dir: Path, result: dict) -> str:
    ub = result["unknown_before_growth"]
    lines = [f"{run_dir.name}: {len(result['flags'])} flags, {len(result['kept'])} kept; "
             f"unknown before growth {100 * ub[0]:.1f} -> {100 * ub[1]:.1f}%"]
    for e in result["entries"]:
        lines.append(f"  #{e['segment']} {e['id']} ({e['role']})" + (f" covers {e['covers_segment']:.2f}" if "covers_segment" in e else "")
                     + (f" [{e['dropped']}]" if e.get("dropped") else ""))
    return "\n".join(lines)
