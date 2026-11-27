"""Gate 3's hand-marked small things: how much of a shot's real small things a run's small listing finds.

Gate 3's own reference (`judgesmall`) is made by the run's lister and segmenter on the reference shots, so it can say
an item was missed only where the reference found it, and an item a large run item holds passes. One small listing
was found to find 76-85% of a crowded kitchen's real things while gate 3 passed 15 of 16. This is the measure that
sees it.

**The reference** (`refs/<scan>/<shot>/small_marked.json` and a mask per thing in `small_marked/`, written by
the project's marking tool): the candidate things of several small listings of the shot, grouped by mask overlap and
each marked by hand as real (one thing, or several listed as one), not a small item, or unsure (left out). Only things
some listing found are in it, so recall against it is an upper bound.

**The measure**: a real thing is found when one of the run's small items (`small_survey.json` and its masks) and the
thing's mask overlap by more than `small.SAME_THING` of the smaller of the two (the rule the things were grouped by).
Also counted: the run's small items matching a thing marked not small, and those matching nothing marked (new, for a
look by eye).
"""

import json
from pathlib import Path

import numpy as np
from PIL import Image

from aomma.stages.small import SAME_THING


def _mask(path: Path, size: tuple[int, int]) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L").resize(size, Image.NEAREST)) > 127


def recall(run_dir: Path, folder: Path) -> dict | None:
    """The run's small listing against the shot's hand-marked things; None where the shot has none."""
    path = folder / "small_marked.json"
    if not path.exists():
        return None
    marked = json.loads(path.read_text())
    size = tuple(marked["size"])
    things = {t["id"]: (t, _mask(folder / "small_marked" / f"{t['id']}.png", size)) for t in marked["things"]}
    survey = run_dir / "small_survey.json"
    items = {}
    for o in json.loads(survey.read_text())["objects"] if survey.exists() else []:
        p = run_dir / "masks" / f"{o['id']}.png"
        if p.exists() and (m := _mask(p, size)).any():
            items[o["id"]] = m
    hit, new = set(), []
    for oid, m in items.items():
        matched = {tid for tid, (_, u) in things.items() if (m & u).sum() > SAME_THING * min(m.sum(), u.sum())}
        hit |= matched
        if not matched:
            new.append(oid)
    real = [tid for tid, (t, _) in things.items() if t["verdict"] != "not"]
    found = [tid for tid in real if tid in hit]
    return {
        "real": len(real), "found": len(found), "recall": round(len(found) / len(real), 4) if real else None,
        "missed": [things[tid][0]["description"] for tid in real if tid not in hit],
        "not_small_listed": sum(tid in hit for tid, (t, _) in things.items() if t["verdict"] == "not"),
        "new": new,
    }
