"""`aomma ref-large`: which of RoomPlan's objects each pixel of an input shot shows, from the scan alone
(gate 2's reference).

The things gate 2 judges are those stage 2 lists: furniture (`roomplan.FURNITURE_KINDS`), storage standing alone (the
office cabinet, the bedroom dresser; storage in a run is the shell's, `refshell.fixtures`) and appliances. Each is one
of RoomPlan's boxes. A pixel of the LiDAR map is a box's when the shell reference (`aomma ref-shell`) puts it in front
of the shell, or on an appliance, and its point lies in that box, grown by `refshell.BOX_MARGIN_M`. A pixel in two boxes
at once is `SHARED` and not scored: which object it shows can't be told (the office chair's 1 m box, for its back,
takes in the desk's top where the chair is tucked under it).

RoomPlan's boxes are a few cm off and can be larger than the thing (the bed's box 2.02 x 2.19 m, the bed
1.95 x 1.34), so the reference says which object a pixel belongs to, not the object's outline: only pixels in front of
the shell count, and a pillow on the bed is the bed's here. No run, and no segmenter, has a hand in it.

Writes `refs/<scan>/<shot>/large_labels.png` (0 none, `SHARED`, else 1 + the box's index in `large_labels.json`) for
every input shot that passed the scan check (a shot whose pose failed gate 0 too, its judges noting it).
"""

import json
from pathlib import Path

import numpy as np
from PIL import Image

from aomma.capture import REFS, Scan, View, _matrix, to_frame
from aomma.refshell import APPLIANCE, BOX_MARGIN_M, IN_FRONT, fixtures
from aomma.scancheck import usable_shots
from aomma.stages.roomplan import APPLIANCE_KINDS, FURNITURE_KINDS, category


# A pixel in two boxes at once: not scored.
SHARED = 255


def things(room: dict) -> list[dict]:
    """RoomPlan's boxes stage 2 lists: furniture, storage standing alone, appliances."""
    in_runs = {id(element) for element, _ in fixtures(room) if category(element) == "storage"}
    found = []
    for element in room.get("objects", []):
        kind = category(element)
        if kind in FURNITURE_KINDS or kind in APPLIANCE_KINDS or (kind == "storage" and id(element) not in in_runs):
            found.append(element)
    return found


def large_labels(scan: Scan, view: View, shell: np.ndarray) -> tuple[np.ndarray, list[dict]]:
    """The shot's box labels at its LiDAR map's size, from its shell reference `shell`, and the boxes."""
    labels = np.zeros(shell.shape, np.uint8)
    points, pixels = scan.lidar_world(view)
    code = shell[pixels[:, 0], pixels[:, 1]]
    eligible = (code == IN_FRONT) | (code == APPLIANCE)
    boxes = things(scan.room)
    count = np.zeros(len(points), int)
    which = np.zeros(len(points), int)
    for i, box in enumerate(boxes):
        local = to_frame(points, np.linalg.inv(_matrix(box["transform"])))
        inside = (np.abs(local) <= np.asarray(box["dimensions"]) / 2 + BOX_MARGIN_M).all(axis=1) & eligible
        count += inside
        which[inside] = i + 1
    one = count == 1
    labels[pixels[one, 0], pixels[one, 1]] = which[one]
    # In two boxes at once (a chair tucked under a desk, its tall box taking in the desk's top): can't be told.
    labels[pixels[count > 1, 0], pixels[count > 1, 1]] = SHARED
    found = []
    for i, box in enumerate(boxes):
        n = int((labels == i + 1).sum())
        found.append({"value": i + 1, "id": f"{category(box)}_{i}", "category": category(box),
                      "size_m": [round(float(v), 2) for v in box["dimensions"]], "pixels": n})
    return labels, found


def reference_large(scan: Scan, refs: Path = REFS) -> dict[str, list[dict]]:
    """Box labels for every input shot that passed the scan check and has a shell reference; written beside it."""
    usable = usable_shots(scan.name, refs, poses=False)
    made = {}
    for view in scan.shots:
        if view.role != "input" or not view.depth_size or (usable is not None and view.name not in usable):
            continue
        folder = refs / scan.name / view.name
        shell_path = folder / "shell_labels.png"
        if not shell_path.exists():
            raise SystemExit(f"{scan.name} {view.name}: no shell reference; run `aomma ref-shell {scan.folder}` first")
        labels, boxes = large_labels(scan, view, np.asarray(Image.open(shell_path)))
        Image.fromarray(labels).save(folder / "large_labels.png")
        (folder / "large_labels.json").write_text(json.dumps({"boxes": boxes, "box_margin_m": BOX_MARGIN_M}, indent=2))
        made[view.name] = [b for b in boxes if b["pixels"]]
    return made
