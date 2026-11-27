"""The agent: run the whole pipeline on a photo, then review the result and fix what it can.

`run` takes a photo to a finished scene, picking up wherever an earlier
attempt stopped. `review` is the loop that makes the harness more than a
pipeline: a critic looks at each object in the render beside the photo, the
scene checks measure each object against the photo's depth map, its estimated
size and the other models, a planner turns each complaint into one action from
a fixed list, the action is carried out, and its result is checked by outline
and by measurement. Every step is written to the run's
`agent.jsonl`, and the review stops when nothing is wrong, nothing more can
be tried, or its budget is spent.
"""

import json
import math
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image
from pydantic import BaseModel

from aomma import cli
from aomma.batch import BatchProgress
from aomma.config import load_config
from aomma.layout import SceneLayout
from aomma.ledger import Ledger
from aomma.stages.critic import Finding, critique, find_unlisted
from aomma.stages.image import to_jpeg
from aomma.stages.inventory import Inventory, draw_inventory
from aomma.stages.large import LargeSurvey
from aomma.stages.segment import MIN_FILL, box_fills, draw_masks, segment

# What each action does, and how many models it generates.
ACTIONS = {
    "relist": ("look at the entry again and split it into its parts, group them, or redraw its box", 1),
    "repair": ("redraw the object whole and alone, then model it from that picture", 1),
    "flatten": ("build it as a flat panel painted from the photo", 0),
    "regenerate": ("generate its model again with a different seed", 1),
    "repose": ("pose it again holding it to the depth map's distance and its estimated size", 0),
    "pull_forward": ("pose it again and bring it out of the wall towards the camera", 0),
    "square": ("pose it again starting turned onto the heading the photo shows (square to the walls, or its angle)", 0),
}
# For each complaint, the actions worth trying, best first.
PLANS = {
    "missing": ["regenerate"],
    "wrong_object": ["relist"],
    "several_objects": ["relist"],
    "wrong_shape": ["repair", "regenerate"],
    "wrong_side": ["flatten", "regenerate"],
    # Only a partly hidden object's pose is searched freely enough to be held to the depth map.
    "wrong_place": ["repose"],
    # From the scene checks:
    "behind_wall": ["pull_forward"],
    "covers_others": ["regenerate"],
    "wrong_size": ["repose", "regenerate"],
    "collides": ["regenerate"],
}
# A staged run's large items, held to the room's shell while posed: posing one again holds it to the shell
# (`stages.large.hold_to_shell`), which keeps it out of the walls, so that is the fix for "behind a wall"
# where `pull_forward` was. The shell's measures add three complaints.
STAGED_PLANS = PLANS | {
    "behind_wall": ["repose"],
    # Its pose was undone as the wall or its model wrong, or no pose fitted: the walls were right
    # whenever this was measured, so it is the model.
    "left_out": ["repair", "regenerate"],
    "off_support": ["repose"],
    "off_square": ["square", "repose"],
}
# The checks complain about a large item standing this far off its support, or a box-shaped one turned
# this far from the heading the photo shows for it (`hold_to_shell` squares it only within 10 degrees itself).
OFF_SUPPORT_M = 0.05
OFF_SQUARE_DEG = 10.0
# A change is undone if the object's outline fits this much worse and its measured trouble is no better.
REVERT_FIT_LOSS = 0.05
# A change that at least halves an object's measured trouble may cost this much outline fit.
TRADE_FIT_LOSS = 0.10
# The scene checks complain about an object whose visible pixels agree with the depth map less often than
# this (over at least `_DEPTH_MIN_CELLS` of them), whose size is outside this range of its estimated size, or
# that has this share of another model inside it.
DEPTH_AGREE_MIN = 0.5
_DEPTH_MIN_CELLS = 20
SIZE_RANGE = (0.6, 1.6)
COLLIDE_SHARE = 0.25
# The scene checks complain first of all about an object this much of which a wall stands in front of: the
# camera does not see it, so nothing else said of it (by the critic, or the depth check) can be trusted.
BEHIND_WALL = 0.2
# A critic complaint about an object that already fits its outline this well, and that the checks find
# nothing wrong with, is recorded but not acted on: such fixes have cost generations and changed nothing.
ACT_FIT = 0.8
# The scene checks complain, before anything but a wall, about a model that covers, in the render, this
# many times its own pixels of other objects' pixels in the photo: it stands in front of things it should
# not, and every check of those things is spoilt until it is fixed. The garage's 1 m slipper covered 68
# times; models resting a little off their place on another, or a basket over the one inside it, under 1.
COVERS_OTHERS = 2.0
# At most this many unlisted objects are added in a review, largest first.
MAX_ADDED = 3


class Facts(BaseModel):
    """What the planner needs to know about an object to tell which actions apply."""

    on: str
    area_px: int
    flat: bool = False
    repaired: bool = False
    # The review has already posed it held to the depth map and its estimated size, or brought it out of a wall.
    reposed: bool = False
    pulled: bool = False
    # A large item of a staged run, held to the room's shell whenever it is posed.
    staged: bool = False


def applies(action: str, facts: Facts) -> bool:
    if action == "repair":
        return not (facts.flat or facts.repaired)
    if action == "flatten":
        return facts.on == "wall" and not facts.flat
    if action == "regenerate":
        return not facts.flat
    if action == "repose":
        # The other objects' distance comes straight from the depth map already, unless the shell moved them.
        return (facts.repaired or facts.staged) and not (facts.flat or facts.reposed)
    if action == "pull_forward":
        return not (facts.flat or facts.pulled or facts.staged)
    if action == "square":
        return facts.staged and not facts.flat
    return action == "relist"


def measured_findings(
    depth: dict, sizes: dict[str, float], collisions: list[dict], modelled: set[str], walled: dict[str, float] | None = None
) -> list[Finding]:
    """The scene checks' complaints, one at most per object, worst kind first.

    `depth` is the depth check's per-object summary, `sizes` each object's size
    over its estimate, `collisions` the collision check's pairs, `walled` the
    share of each model that a wall stands in front of. A collision is put
    down to the bigger of the two objects: it is the one that swallows the
    other (a closed cabinet over its books), and the smaller is usually right.
    """
    found: dict[str, Finding] = {}
    for object_id, share in (walled or {}).items():
        if object_id in modelled and share >= BEHIND_WALL:
            found[object_id] = Finding(id=object_id, verdict="behind_wall", source="checks", note=f"{share:.0%} of it is behind a wall")
    for object_id, check in depth.items():
        if object_id in modelled and object_id not in found and check.get("over_others", 0) >= COVERS_OTHERS:
            found[object_id] = Finding(
                id=object_id, verdict="covers_others", source="checks",
                note=f"its model covers {check['over_others']:.1f} times its own size of other objects",
            )
    for entry in collisions:
        if entry["share_of_smaller"] < COLLIDE_SHARE:
            continue
        smaller, bigger = sorted(entry["objects"], key=lambda name: entry["volumes"][name])
        if bigger in modelled and bigger not in found:
            found[bigger] = Finding(
                id=bigger, verdict="collides", source="checks",
                note=f"{entry['share_of_smaller']:.0%} of {smaller} is inside it ({entry['litres']} litres)",
            )
    for object_id, ratio in sizes.items():
        if object_id in modelled and object_id not in found and not SIZE_RANGE[0] <= ratio <= SIZE_RANGE[1]:
            found[object_id] = Finding(id=object_id, verdict="wrong_size", source="checks", note=f"{ratio:.2f} times its estimated size")
    for object_id, check in depth.items():
        if object_id in modelled and object_id not in found and check["pixels"] >= _DEPTH_MIN_CELLS and check["agree"] < DEPTH_AGREE_MIN:
            side = "too near" if check["too_near"] >= check["too_far"] else "too far"
            found[object_id] = Finding(
                id=object_id, verdict="wrong_place", source="checks",
                note=f"only {check['agree']:.0%} of its pixels agree with the depth map, most {side}; at {check['ratio']:.2f} of the photo's distance",
            )
    return list(found.values())


def trouble(measured: dict) -> float:
    """How far off the checks find an object, 0 when nothing is: distance beyond 10% out, size beyond
    `SIZE_RANGE`, overlap with another model beyond touching, and the share of it behind a wall, each on a
    log or share scale and added up."""
    total = 0.0
    if measured.get("hidden_by_wall"):
        total += max(0.0, measured["hidden_by_wall"] - 0.02)
    if measured.get("distance_vs_photo"):
        total += max(0.0, abs(math.log(measured["distance_vs_photo"])) - math.log(1.1))
    if measured.get("size_vs_estimate"):
        total += max(0.0, math.log(measured["size_vs_estimate"]) - math.log(SIZE_RANGE[1]), math.log(SIZE_RANGE[0]) - math.log(measured["size_vs_estimate"]))
    if measured.get("inside_other"):
        total += max(0.0, measured["inside_other"] - 0.05)
    # A staged run's large items (see `cli.large_facts`): off its support by tenths of a metre past
    # `OFF_SUPPORT_M`, off square by eighths of a turn past `OFF_SQUARE_DEG`.
    if measured.get("off_support_m") is not None:
        total += max(0.0, abs(measured["off_support_m"]) - OFF_SUPPORT_M) / 0.1
    if measured.get("off_square_deg") is not None:
        total += max(0.0, measured["off_square_deg"] - OFF_SQUARE_DEG) / 45
    return total


def keep_change(before: dict, after: dict) -> bool:
    """Whether a change is better kept: by outline fit, and by the checks (see `trouble`).

    It is kept if the fit holds and the trouble does not grow, or if the
    trouble at least halves for a small loss of fit.
    """
    fit_loss = before["outline_fit"] - after["outline_fit"]
    was, now = trouble(before), trouble(after)
    if fit_loss <= REVERT_FIT_LOSS and now <= was + 0.02:
        return True
    return was > 0 and now <= was / 2 and fit_loss <= TRADE_FIT_LOSS


def _describe(before: dict, after: dict) -> str:
    """What a change did, by the measures that changed."""
    parts = [f"outline fit {before['outline_fit']:.2f} -> {after['outline_fit']:.2f}"]
    for key, name in (
        ("distance_vs_photo", "distance"), ("size_vs_estimate", "size"), ("inside_other", "inside another"),
        ("hidden_by_wall", "behind a wall"), ("off_floor_m", "off the floor"),
        ("off_support_m", "off its support"), ("off_square_deg", "degrees off square"),
    ):
        if before.get(key) != after.get(key) and before.get(key) is not None and after.get(key) is not None:
            parts.append(f"{name} {before[key]:.2f} -> {after[key]:.2f}")
    return ", ".join(parts)


def plan(
    findings: list[Finding], facts: dict[str, Facts], tried: set[tuple[str, str]], generations_left: int,
    plans: dict[str, list[str]] = PLANS,
) -> list[tuple[Finding, str]]:
    """Choose the actions for each complaint: the first that applies, is untried and is affordable.

    An object gets one action, from its first complaint (the critic's, when
    both complain). But when the checks complain too and their action costs
    nothing (a repose), that is tried first and the other kept as a fallback,
    to be carried out only if the free one changed nothing. A model covering
    other objects is served first, since it spoils what is seen of them; then
    bigger objects, since they matter most to the picture and the budget may
    not stretch to all.
    """
    by_object: dict[str, list[Finding]] = {}
    for finding in findings:
        by_object.setdefault(finding.id, []).append(finding)
    steps = []
    def order(object_id: str) -> tuple[bool, int]:
        return (all(f.verdict != "covers_others" for f in by_object[object_id]), -facts[object_id].area_px)

    for object_id in sorted(by_object, key=order):
        chosen = []
        for finding in by_object[object_id]:
            action = next(
                (
                    action for action in plans.get(finding.verdict, [])
                    if applies(action, facts[object_id]) and (object_id, action) not in tried
                    and ACTIONS[action][1] <= generations_left
                ),
                None,
            )
            if action is not None and action not in [a for _, a in chosen]:
                chosen.append((finding, action))
        free = next((step for step in chosen if step[0].source == "checks" and ACTIONS[step[1]][1] == 0), None)
        for step in ([free] if free else []) + chosen[:1]:
            if step not in steps:
                steps.append(step)
                generations_left -= ACTIONS[step[1]][1]
    return steps


def tried_before(log: list[dict]) -> set[tuple[str, str]]:
    """The fixes an earlier review of the run tried. One that failed (no model made, the server
    gone) judged nothing, so a review taken up again may try it again."""
    return {(e["id"], e["action"]) for e in log if e["kind"] == "action" and not e["outcome"].startswith("failed")}


def hand_on(
    finding: Finding, facts: dict[str, Facts], tried: set[tuple[str, str]], generations_left: int,
    queued: list[tuple[Finding, str]], plans: dict[str, list[str]] = PLANS,
) -> list[tuple[Finding, str]]:
    """The next action for a complaint whose fix was undone, planned with what is left once the
    actions still `queued` have had their generations."""
    return plan([finding], facts, tried, generations_left - sum(ACTIONS[action][1] for _, action in queued), plans)


def run(target: Path, config_path: Path, vlm_name: str | None) -> Path:
    """Take a photo, or a run folder left unfinished, all the way to a reviewed scene."""
    if target.is_dir():
        run_dir = target
        source = cli.run_source(run_dir)
    else:
        run_dir = Path("runs") / f"{datetime.now():%Y%m%d-%H%M%S}-{target.stem}"
        source = target
    print(f"run folder: {run_dir}", flush=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    stages = BatchProgress(run_dir, ["perceive", "layout", "models", "review"], name="run", title="Run")

    # Each stage leaves files behind; a stage whose last file exists is done.
    if not (run_dir / "camera.json").exists():
        print("== perceive", flush=True)
        stages.set("perceive", "inventory, masks and depth")
        cli.perceive(source, run_dir, config_path, vlm_name)
    stages.set("perceive", "done")
    if not (run_dir / "layout.json").exists():
        print("== layout", flush=True)
        stages.set("layout", "fitting boxes, walls and floor")
        cli.layout(run_dir)
    stages.set("layout", "done")
    dropped = {oid for oid, entry in cli.overrides(run_dir).items() if entry.get("drop")}
    unmodelled = [
        proxy["id"] for proxy in json.loads((run_dir / "proxies.json").read_text())
        if proxy["id"] not in dropped and not _has_model(run_dir, proxy["id"])
    ]
    if unmodelled:
        print(f"== models for {len(unmodelled)} objects", flush=True)
        stages.set("models", f"generating {len(unmodelled)} models")
        cli.assets(run_dir, unmodelled, config_path)
    stages.set("models", "done")
    print("== review", flush=True)
    stages.set("review", "judging and fixing")
    review(run_dir, config_path, vlm_name)
    stages.set("review", "done")
    return run_dir


def review(run_dir: Path, config_path: Path, vlm_name: str | None, stage: str | None = None) -> None:
    """Judge every modelled object against the photo and act on the complaints, within the config's budget.

    With `stage` "large", only a staged run's large items are judged, before
    anything is placed on them: with the shell's measures among the checks
    (see `_check_large`), the fixes in `STAGED_PLANS`, a budget of its own
    (`large_review_generations`), its own log (review_large.jsonl), no look
    for unlisted objects (that is for the small items), and large.json
    written again at the end.
    """
    config = load_config(config_path)
    vlm = cli.vlm_for(config, vlm_name, run_dir)
    ledger = Ledger(run_dir / "usage.jsonl", config.max_billed_tokens)
    large = stage == "large"
    listed = LargeSurvey.model_validate_json((run_dir / "large_survey.json").read_text()) if large else None
    stage_ids = {o.id for o in listed.objects} if listed else None
    plans = STAGED_PLANS if large else PLANS
    budget = config.large_review_generations if large else config.review_generations
    log_path = run_dir / ("review_large.jsonl" if large else "agent.jsonl")
    log = [json.loads(line) for line in log_path.read_text().splitlines()] if log_path.exists() else []

    def record(**entry) -> None:
        log.append(entry)
        with log_path.open("a") as file:
            file.write(json.dumps(entry) + "\n")

    tried = tried_before(log)
    generations_left = budget - sum(e.get("generated", 0) for e in log)
    first_round = max((e["round"] for e in log), default=0) + 1
    # Later rounds only look again at what the round before changed.
    to_judge: list[str] | None = None
    findings: list[Finding] = []
    changed: list[str] = []

    for number in range(first_round, config.review_rounds + 1):
        inventory = Inventory.model_validate_json((run_dir / "inventory.json").read_text())
        photo, render = Image.open(run_dir / "input.jpg"), Image.open(run_dir / "render.png")
        # Layout leaves out an object whose mask is unusable, though a model made earlier may remain.
        in_scene = {o.name for o in SceneLayout.model_validate_json((run_dir / "layout.json").read_text()).objects}
        modelled = [
            obj for obj in inventory.objects
            if _has_model(run_dir, obj.id) and obj.id in in_scene and (to_judge is None or obj.id in to_judge)
            and (stage_ids is None or obj.id in stage_ids)
        ]
        unlisted = number == 1 and not large
        # Everything this round will do, so the dashboard can show what is still waiting.
        progress = BatchProgress(
            run_dir,
            [f"judge {obj.id}" for obj in modelled] + ["check the scene"] + (["look for unlisted objects"] if unlisted else []),
            name="review", title=f"Review{' of the large items' if large else ''}, round {number}",
        )
        # Every judgement takes its place in the job queue now, so they are in line from the start.
        places = {obj.id: cli.QUEUE.enqueue("vlm", f"judge {obj.id}", run_dir.name) for obj in modelled}
        findings = []
        for obj in modelled:
            progress.set(f"judge {obj.id}", "judging")
            mask = np.asarray(Image.open(run_dir / "masks" / f"{obj.id}.png")) > 0
            try:
                finding, view = critique(photo, render, obj, mask, vlm, ledger, places[obj.id])
            finally:
                cli.QUEUE.cancel(places[obj.id])
            progress.set(f"judge {obj.id}", finding.verdict.replace("_", " "), phase="done")
            view.save(run_dir / "debug" / f"review_{obj.id}.jpg", quality=85)
            record(round=number, kind="finding", id=obj.id, verdict=finding.verdict, note=finding.note, source="critic")
            print(f"[round {number}] {obj.id}: {finding.verdict}" + (f" ({finding.note})" if finding.verdict != "ok" else ""), flush=True)
            if finding.verdict != "ok":
                findings.append(finding)

        # The scene checks: what the depth map, the estimated sizes and the other models say about each object.
        progress.set("check the scene", "depth, size and collisions")
        judged = {obj.id for obj in modelled}
        measured = _check_large(run_dir, inventory, judged) if large else _check_scene(run_dir, inventory, judged)
        for finding in measured:
            record(round=number, kind="finding", id=finding.id, verdict=finding.verdict, note=finding.note, source="checks")
            print(f"[round {number}] {finding.id}: {finding.verdict} by the checks ({finding.note})", flush=True)
        # Both complaints are kept for an object both complain about: the planner chooses between them.
        findings += measured
        progress.set("check the scene", f"{len(measured)} complaints", phase="done")

        # A critic complaint about an object that fits well and that the checks find nothing wrong with is not
        # acted on, nor one about an object behind a wall: what the critic saw of it was the wall.
        confirmed = {finding.id for finding in measured}
        walled = {finding.id for finding in measured if finding.verdict == "behind_wall"}
        acted = []
        for finding in findings:
            fit = _fit(run_dir, finding.id)
            if finding.source == "critic" and finding.id in walled:
                reason = "a wall stands in front of it, so its model cannot be judged until it is brought out"
                record(round=number, kind="skipped", id=finding.id, verdict=finding.verdict, reason=reason)
                print(f"[round {number}] {finding.id}: not acted on: {reason}", flush=True)
            elif finding.source == "critic" and finding.verdict in ("wrong_shape", "wrong_side", "wrong_place") \
                    and fit >= ACT_FIT and finding.id not in confirmed:
                reason = f"fits its outline at {fit:.2f} and the checks find nothing wrong"
                record(round=number, kind="skipped", id=finding.id, verdict=finding.verdict, reason=reason)
                print(f"[round {number}] {finding.id}: not acted on: {reason}", flush=True)
            else:
                acted.append(finding)

        changed = []
        steps = plan(acted, _facts(run_dir, inventory, stage_ids or set()), tried, generations_left, plans)
        for finding, action in steps:
            progress.set(f"fix {finding.id}", f"{action.replace('_', ' ')} queued", phase="waiting")
        while steps:
            finding, action = steps.pop(0)
            # A fallback waits on the free fix before it: if that changed the object, it is not needed.
            if finding.id in changed:
                record(round=number, kind="skipped", id=finding.id, verdict=finding.verdict, reason=f"{action} not needed: an earlier fix helped")
                print(f"[round {number}] {finding.id}: {action} not needed: an earlier fix helped", flush=True)
                continue
            progress.set(f"fix {finding.id}", action.replace("_", " "))
            print(f"[round {number}] {finding.id}: {action} ({ACTIONS[action][0]})", flush=True)
            try:
                outcome, generated, now = _act(run_dir, finding.id, action, config_path, vlm_name)
            except Exception as error:  # one failed fix must not end the review
                outcome, generated, now = f"failed: {error}", 0, []
            tried.add((finding.id, action))
            generations_left -= generated
            changed += now
            record(round=number, kind="action", id=finding.id, action=action, outcome=outcome, generated=generated)
            progress.set(f"fix {finding.id}", outcome, phase="failed" if outcome.startswith(("failed", "undone")) else "done")
            print(f"[round {number}] {finding.id}: {outcome}", flush=True)
            # A fix that was undone or changed nothing hands on to the complaint's next action now: the next
            # round looks only at what changed, so it would never come back to this object.
            if not now and all(f.id != finding.id for f, _ in steps):
                for step in hand_on(finding, _facts(run_dir, inventory, stage_ids or set()), tried, generations_left, steps, plans):
                    progress.set(f"fix {finding.id}", f"{step[1].replace('_', ' ')} queued", phase="waiting")
                    steps.append(step)

        if unlisted:
            progress.set("look for unlisted objects", "looking" if generations_left > 0 else "skipped: no budget left", phase=None if generations_left > 0 else "done")
        if unlisted and generations_left > 0:
            added = _add_unlisted(run_dir, config_path, vlm, ledger, min(MAX_ADDED, generations_left))
            if added:
                generations_left -= len(added)
                changed += added
                record(round=number, kind="added", ids=added, generated=len(added))
                print(f"[round {number}] added objects the inventory had missed: {', '.join(added)}", flush=True)
            progress.set("look for unlisted objects", f"added {', '.join(added)}" if added else "none found", phase="done")

        if not changed:
            break
        to_judge = changed
    unresolved = sorted({f.id for f in findings} - set(changed)) if findings else []
    if listed:
        in_inventory = {o.id for o in Inventory.model_validate_json((run_dir / "inventory.json").read_text()).objects}
        cli._measure_large(run_dir, [o.id for o in listed.objects if o.id in in_inventory], listed)
    print(
        f"review done: {generations_left} of {budget} generations unspent"
        + (f"; still wrong: {', '.join(unresolved)}" if unresolved else ""),
        flush=True,
    )


def _check_scene(run_dir: Path, inventory: Inventory, judged: set[str]) -> list[Finding]:
    """Run the scene checks (walls, depth, size, collisions) and turn what they find about `judged` objects into findings."""
    depth = cli.depth_check(run_dir)["objects"]
    collisions = cli.collisions(run_dir)
    scene = SceneLayout.model_validate_json((run_dir / "layout.json").read_text())
    estimated = {obj.id: max(obj.size_m) for obj in inventory.objects}
    sizes = {
        placement.name: placement.scale / estimated[placement.name]
        for placement in scene.objects
        if placement.name in estimated and (placement.asset.endswith(".usdc") or "_fitted" in placement.asset)
        and not (run_dir / "assets" / placement.name / "flat.json").exists()
    }
    walled = {pose["id"]: pose.get("hidden_by_wall", 0.0) for pose in json.loads((run_dir / "poses.json").read_text())}
    return measured_findings(depth, sizes, collisions, judged, walled)


def _check_large(run_dir: Path, inventory: Inventory, judged: set[str]) -> list[Finding]:
    """The scene checks on a staged run's large items, with what the shell measures added, one complaint per item.

    An item left out of the scene is complained about before anything else
    (nothing else said of it means anything), unless it stands beyond the
    walls, in another room, where nothing is to be fixed. Then the scene
    checks, then an item off its support, then a box-shaped one turned off the
    heading the photo shows for it (square to the walls, or the angle it stands
    at; see `cli.large_facts`): never one whose faces in the photo do not say.
    """
    found = {finding.id: finding for finding in _check_scene(run_dir, inventory, judged)}
    for object_id in sorted(judged):
        facts = cli.large_facts(run_dir, object_id)
        moves = facts["shell_moves"] or {}
        if not facts["placed"]:
            if moves.get("beyond_walls"):
                found.pop(object_id, None)
                continue
            why = "the shell undid its moves, the wall or its model wrong" if moves.get("wall_or_model_wrong") else f"its fit is {facts['outline_fit']:.2f}"
            found[object_id] = Finding(id=object_id, verdict="left_out", source="checks", note=f"left out of the scene: {why}")
        elif object_id in found:
            continue
        elif abs(facts["off_support_m"]) > OFF_SUPPORT_M:
            found[object_id] = Finding(
                id=object_id, verdict="off_support", source="checks", note=f"{facts['off_support_m']:+.2f} m off {facts['on']}",
            )
        elif facts["off_square_deg"] is not None and facts["off_square_deg"] > OFF_SQUARE_DEG:
            found[object_id] = Finding(
                id=object_id, verdict="off_square", source="checks",
                note=f"{facts['off_square_deg']:.0f} deg from "
                + (f"the heading its faces show in the photo ({facts['seen_turn_deg']:+.0f} deg off the walls)"
                   if abs(facts["seen_turn_deg"]) > cli.SQUARE_WITHIN_DEG else "square to the walls")
                + f", {facts['box_shaped']:.2f} box-shaped",
            )
    return list(found.values())


def _measure(run_dir: Path, object_id: str) -> dict:
    """How an object sits in the scene now: `cli.large_facts` for a staged run's large item, otherwise `cli.placement_facts`."""
    survey = run_dir / "large_survey.json"
    if survey.exists() and object_id in {o.id for o in LargeSurvey.model_validate_json(survey.read_text()).objects}:
        return cli.large_facts(run_dir, object_id)
    return cli.placement_facts(run_dir, object_id)


def _has_model(run_dir: Path, object_id: str) -> bool:
    return (run_dir / "assets" / object_id / f"{object_id}.usdc").exists()


def _fit(run_dir: Path, object_id: str) -> float:
    """How well the object's placed model matches its outline in the photo, 0-1; 0 if it has no pose."""
    path = run_dir / "poses.json"
    poses = {pose["id"]: pose for pose in json.loads(path.read_text())} if path.exists() else {}
    return poses[object_id]["iou"] if object_id in poses else 0.0


def _facts(run_dir: Path, inventory: Inventory, staged: set[str] = frozenset()) -> dict[str, Facts]:
    facts = {}
    chosen = cli.overrides(run_dir)
    for obj in inventory.objects:
        folder = run_dir / "assets" / obj.id
        mask = run_dir / "masks" / f"{obj.id}.png"
        facts[obj.id] = Facts(
            on=obj.on,
            area_px=int((np.asarray(Image.open(mask)) > 0).sum()) if mask.exists() else 0,
            flat=(folder / "flat.json").exists(),
            repaired=(folder / "repaired").exists(),
            reposed="pose" in chosen.get(obj.id, {}),
            pulled=chosen.get(obj.id, {}).get("pose", {}).get("pull_forward", False),
            staged=obj.id in staged,
        )
    return facts


def _act(run_dir: Path, object_id: str, action: str, config_path: Path, vlm_name: str | None) -> tuple[str, int, list[str]]:
    """Carry out one action; returns (what happened, models generated, ids of the objects it changed)."""
    if action == "relist":
        new = cli.split(run_dir, [object_id], config_path, vlm_name)
        if not new:
            return "a second look found nothing to change", 0, []
        cli.assets(run_dir, new, config_path)
        return f"now {', '.join(new)}", len(new), new

    before, saved = _measure(run_dir, object_id), cli.overrides(run_dir).get(object_id, {})
    if action in _POSE_TRIES:
        return _repose(run_dir, object_id, before, saved, _POSE_TRIES[action])
    if action == "square":
        if before.get("square_turn_deg") is None:
            return "nothing to square: its model is not box-shaped", 0, []
        # Either quarter turn puts it on that heading; which side faces out is the outline's to choose.
        turn, start = before["square_turn_deg"], (saved.get("pose") or {}).get("turn_deg", 0.0)
        other = turn - math.copysign(90.0, turn)
        tries = tuple(
            ({**(saved.get("pose") or {}), "turn_deg": round(start + by, 1)}, f"turned {by:+.0f} deg, onto the heading the photo shows")
            for by in (turn, other)
        )
        return _repose(run_dir, object_id, before, saved, tries, best=True)

    # Every other action replaces the object's model: keep the old one, and the scene as it stands, in case the new is worse.
    kept = {name: (run_dir / name).read_text() for name in ("layout.json", "poses.json") if (run_dir / name).exists()}
    backup = _set_aside(run_dir, object_id)
    if action == "regenerate":
        cli.set_override(run_dir, object_id, attempt=saved.get("attempt", 0) + 1)
    elif action == "repair":
        cli.set_override(run_dir, object_id, repair=True)
    elif action == "flatten":
        cli.set_override(run_dir, object_id, flat=True)
    cli.assets(run_dir, [object_id], config_path)
    generated = ACTIONS[action][1]
    after = _measure(run_dir, object_id) if _has_model(run_dir, object_id) else None
    complaints = _brought_in_complaints(run_dir, object_id, before, after) if after is not None else []
    if after is not None and not complaints and keep_change(before, after):
        return _describe(before, after), generated, [object_id]

    # Worse, or no model came of it: put the old model back exactly as it was, in its old pose. Posing it
    # again instead could move it, as today's pose settings may differ from those it was posed with.
    produced = _has_model(run_dir, object_id)
    _set_aside(run_dir, object_id)
    if backup is not None:
        backup.rename(run_dir / "assets" / object_id)
    cli.set_override(run_dir, object_id, **{key: saved.get(key) for key in ("attempt", "repair", "flat")})
    for name in ("layout.json", "poses.json"):
        if name in kept:
            (run_dir / name).write_text(kept[name])
        else:
            (run_dir / name).unlink(missing_ok=True)
    cli._compose_and_render(run_dir, SceneLayout.model_validate_json(kept["layout.json"]))
    if not produced:
        return "failed: no new model was produced; the old one is back", 0, []
    return f"undone: {_describe(before, after)}" + (f"; {'; '.join(complaints)}" if complaints else ""), generated, []


def _brought_in_complaints(run_dir: Path, object_id: str, before: dict, after: dict) -> list[str]:
    """Why a large item left out of the scene before a change may not be kept after it.

    By the fit alone, anything placed beats something left out (fit 0), so
    the office desk came in at 0.43 standing in front of the chair, too near,
    turned 29 degrees. Brought in, it must pass the review's checks; still
    left out, the change gained nothing (a regenerate there went from 0.01 to
    0.00 and was kept, as "the fit holds").
    """
    if before.get("placed") is not False:
        return []
    if not after["placed"]:
        return ["still left out"]
    inventory = Inventory.model_validate_json((run_dir / "inventory.json").read_text())
    return [f"brought in, but {finding.note}" for finding in _check_large(run_dir, inventory, {object_id})]


# What the posing actions try, in turn, each with what it is called. "repose": today's defaults (a run may
# have been posed by older code), then the defaults held to the depth map's distance and the estimated size.
# "pull_forward": the model brought out of the wall, then, if that does not help, also stretched down to the
# floor. (A desk pulled out kept its fit, 0.84, and floated 0.32 m; stretched, it stood 0.09 m up at 0.65.)
_POSE_TRIES = {
    "repose": (({}, "with today's pose settings"), ({"depth_weight": 1.0, "size_weight": 1.0}, "held to the depth map and its estimated size")),
    "pull_forward": (({"pull_forward": True}, "brought out of the wall"), ({"pull_forward": True, "reseat": "stretch"}, "brought out of the wall and stretched to the floor")),
}


def _repose(
    run_dir: Path, object_id: str, before: dict, saved: dict, tries: tuple, best: bool = False
) -> tuple[str, int, list[str]]:
    """Pose an object again, keeping the first try that measurably helps (with `best`, the best-fitting of
    all that do); otherwise put its old pose back exactly."""
    kept = {name: (run_dir / name).read_text() for name in ("layout.json", "poses.json")}
    after, helped = before, []
    for settings, how in tries:
        # The settings are kept with the run, so posing the object again later gives the same answer.
        cli.set_override(run_dir, object_id, pose=settings or None)
        cli.pose(run_dir, [object_id])
        after = _measure(run_dir, object_id)
        helps = trouble(after) < trouble(before) - 0.02 or after["outline_fit"] >= before["outline_fit"] + 0.03
        if helps and keep_change(before, after):
            if not best:
                return f"{_describe(before, after)} ({how})", 0, [object_id]
            files = {name: (run_dir / name).read_text() for name in ("layout.json", "poses.json")}
            helped.append((after["outline_fit"], settings, how, after, files))
    if helped:
        _, settings, how, after, files = max(helped, key=lambda entry: entry[0])
        for name, text in files.items():
            (run_dir / name).write_text(text)
        cli.set_override(run_dir, object_id, pose=settings or None)
        cli._compose_and_render(run_dir, SceneLayout.model_validate_json(files["layout.json"]))
        return f"{_describe(before, after)} ({how})", 0, [object_id]
    for name, text in kept.items():
        (run_dir / name).write_text(text)
    cli.set_override(run_dir, object_id, pose=saved.get("pose"))
    cli._compose_and_render(run_dir, SceneLayout.model_validate_json(kept["layout.json"]))
    return f"undone: no try helped (the last: {_describe(before, after)})", 0, []


def _set_aside(run_dir: Path, object_id: str) -> Path | None:
    """Move an object's model folder out of the way, numbered so nothing is overwritten."""
    folder = run_dir / "assets" / object_id
    if not folder.is_dir():
        return None
    (run_dir / "assets_replaced").mkdir(exist_ok=True)
    number = 1
    while (run_dir / "assets_replaced" / f"{object_id}.{number}").exists():
        number += 1
    return Path(shutil.move(folder, run_dir / "assets_replaced" / f"{object_id}.{number}"))


def _add_unlisted(run_dir: Path, config_path: Path, vlm, ledger: Ledger, limit: int) -> list[str]:
    """Add objects the photo shows but the inventory lacks, and give them models; returns their ids.

    A suggestion is taken only if it is big enough to matter, gets a mask
    that fills its box, and that mask is not mostly something already listed.
    """
    config = load_config(config_path)
    image = Image.open(run_dir / "input.jpg")
    inventory = Inventory.model_validate_json((run_dir / "inventory.json").read_text())
    suggested = [obj for obj in find_unlisted(image, inventory, vlm, ledger) if cli._box_share(obj) >= cli.MIN_OBJECT_SHARE]
    if not suggested:
        return []
    comfy = cli.comfy_client(config.comfy.light[0], run_dir)
    image_name = comfy.upload_image(to_jpeg(image), f"aomma_{run_dir.name}.jpg")
    new_masks, new_infos = segment(comfy, image_name, image.size, Inventory(objects=suggested))
    masks = {obj.id: np.asarray(Image.open(run_dir / "masks" / f"{obj.id}.png")) > 0 for obj in inventory.objects}
    listed = np.any(list(masks.values()), axis=0)
    fills = box_fills(Inventory(objects=suggested), new_masks)
    taken = [
        obj for obj in suggested
        if fills[obj.id] >= MIN_FILL and new_masks[obj.id].any() and (new_masks[obj.id] & listed).sum() < 0.5 * new_masks[obj.id].sum()
    ]
    taken = sorted(taken, key=lambda obj: -int(new_masks[obj.id].sum()))[:limit]
    if not taken:
        return []

    infos = json.loads((run_dir / "masks.json").read_text())
    for obj in taken:
        inventory.objects.append(obj)
        masks[obj.id] = new_masks[obj.id]
        Image.fromarray(new_masks[obj.id]).save(run_dir / "masks" / f"{obj.id}.png")
        infos[obj.id] = new_infos[obj.id].model_copy(update={"fill": fills[obj.id]}).model_dump()
    (run_dir / "inventory.json").write_text(inventory.model_dump_json(indent=2))
    (run_dir / "masks.json").write_text(json.dumps(infos, indent=2))
    draw_inventory(image, inventory).save(run_dir / "debug" / "inventory.jpg")
    draw_masks(image, masks).save(run_dir / "debug" / "masks.jpg")
    ids = [obj.id for obj in taken]
    cli.layout(run_dir)
    cli.pose(run_dir, [])
    cli.assets(run_dir, ids, config_path)
    return ids
