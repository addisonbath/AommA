"""The lab: a local model devises and runs experiments on the pipeline, for a supervisor to check.

Questions such as "can this object be generated faster without looking worse?"
take trials of many minutes each, and deciding what to try next needs only a
little thought between them. So that thought is given to the local VLM. It is
handed a question, the settings it may change and a budget of trials. It plans,
runs trials, is shown each result as numbers and a picture, and ends by calling
`finish` with what it found. A trial either generates an object's model again
(`run_trial`, minutes to an hour) or only poses the model it has again
(`pose_trial`, about a minute).

The worker can do nothing else. A trial works on a copy of the run, so the run
itself and the code are never touched. It may generate with any image-to-3D
workflow in the workflows folder, and a setting can only be lowered within what
the server accepts. Everything measured is written down by the harness,
not by the worker: `report.md` puts the harness's table of trials beside the
worker's conclusions, so whoever supervises can check one against the other.
"""

import base64
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image

from aomma import cli
from aomma.backends.comfy import ComfyClient, read_job_labels
from aomma.config import Config, load_config
from aomma.layout import SceneLayout
from aomma.ledger import Ledger
from aomma.stages.critic import close_ups
from aomma.stages.image import to_jpeg
from aomma.stages.inventory import Inventory
from aomma.stages.pose import PoseSettings

DEFAULT_ROOT = Path("lab")

# The settings of an image-to-3D workflow a trial may change, as "node.input", and what each does.
KNOBS = {
    "sample.steps": "sampler steps of the generator's single shape pass",
    "decode.octree_resolution": "grid resolution at which the generated shape is turned into a mesh",
    "shape_sample.steps": "sampler steps of the coarse shape pass",
    "upsample_stage.target_resolution": "resolution of the fine shape pass, the slowest pass for bulky objects",
    "upsample_sample.steps": "sampler steps of the fine shape pass",
    "texture_sample.steps": "sampler steps of the colour pass",
    "remesh.resolution": "grid resolution at which the surface is rebuilt as a clean mesh",
    "decimate.target_face_count": "number of faces in the final mesh",
    "bake_color.texture_size": "side of the baked colour texture, in pixels",
    "bake_normal.resolution": "side of the baked normal map, in pixels",
}
# A setting may be lowered to this share of the workflow's own value, and never raised above it.
_LOWEST_SHARE = 0.25

BRIEF = """\
You run experiments on a pipeline that rebuilds a photo of a room as a 3D scene.
Each object in the photo gets a 3D model, generated from a crop of the photo by
an image-to-3D workflow. The model is then posed: turned, sized and placed so
that its outline matches the object's outline in the photo.

A supervisor has given you the question below. Answer it by running trials.

Every trial works on one object and gives you back these measurements:
- outline_fit: overlap of the placed model's outline with the object in the photo,
  0 to 1, higher is better. A change of less than 0.03 is noise.
- colour_error: mean colour difference from the photo, 0 to 1, lower is better
- hidden_by_wall: the share of the model that a wall of the room stands in front
  of, seen from the camera. A real object is never inside a wall, so this should be 0.
- off_floor_m: height of the model's lowest surface above the floor, in meters
  (a stray spike below it does not count). It should be 0 for a thing that stands
  on the floor, and the height of the support for a thing that stands on furniture.
- longest_side_m: the size the model was given
- distance_vs_photo: how far from the camera the model's visible surface is,
  over where the photo's depth map puts the object. 1 is right; 0.8 means a fifth
  too near, 1.2 a fifth too far.
- size_vs_estimate: the model's size over the size estimated for the object from
  the photo (a rough estimate, so within about 0.7 to 1.4 is plausible).
- inside_other: the largest share of the model's volume (or the other's, if
  smaller) that another model also fills, and inside_which, that model. Real
  objects do not pass through each other, so above about 0.05 is wrong; touching
  does not count.
- a picture with three close-ups: PHOTO, CURRENT (how the run shows the object
  now) and TRIAL (how your trial shows it). Judge shape and texture from it
  yourself. It is taken from the photo's viewpoint, so it cannot show whether the
  model is too far back or inside a wall: the measurements tell you that.
{sections}
OBJECTS YOU MAY TEST
{objects}

RULES
- {budget}
- Before your first trial, write your plan: what you will try, in what order, and
  what each result would tell you.
- Change one thing at a time unless you have a reason not to, and say the reason.
- Report only what the trials showed. If the trials cannot settle the question, say so.
- When you are done, call finish. Give the answer first, then the evidence for it
  trial by trial, then what remains uncertain.

QUESTION
{question}
"""

GENERATION = """
GENERATION TRIALS
The generator the run uses works in passes: a coarse structure pass, a shape pass,
an upsample pass that redoes the shape in fine detail, a colour pass, then meshing
and texture baking. One model takes from 3 minutes to an hour. The upsample and
colour passes take most of that, and bulky objects are the slowest.
Call run_trial with an object, and with the generator to use, the settings to
change, or both. The object is generated again that way, with the same random
seed as its current model, then posed. Besides the measurements above you get
generation_seconds, how long the workflow ran.

GENERATORS YOU MAY USE
A generator is a whole image-to-3D workflow with its own model weights.
{generators}
Leave the generator out to use the one the run uses.

GENERATION SETTINGS YOU MAY CHANGE
{settings}
Any setting you leave out keeps the generator's own value.
"""

POSING = """
POSE TRIALS
Call pose_trial with an object and the pose settings to use. The model the object
already has is posed again; nothing is generated, and it takes about a minute.
The run's poses may have been found by older code, so for each object you test,
first run a pose trial with no settings: that is the baseline to compare with.

POSE SETTINGS YOU MAY CHANGE
- wall_weight: a number from 0 to 3, now 0. While the pose of a partly hidden
  object is searched for, the share of the model hidden by a wall is multiplied by
  this and taken off the candidate's score. 0 ignores the walls. It has no effect
  on objects in plain view, whose distance is measured directly.
- pull_forward: true or false, now false. After the pose is found, a model that a
  wall still hides is moved towards the camera and shrunk in proportion until it
  is clear. Its outline in the photo stays exactly the same, but it may end up
  off the floor.
- reseat: "none", "drop" or "stretch", now "none". Applied last, and only to an
  object that stands on the floor but whose model has ended up above it. "drop"
  lowers the model onto the floor, which moves its outline down in the photo.
  "stretch" keeps the model's top where it is and lengthens it downwards to the
  floor (by at most 1.6 times; it is lowered the rest of the way if that is not
  enough), for a model whose hidden lower part came out too short.
- contact: "lowest" or "trim", now "trim". What a model is stood on: its lowest
  surface ignoring the lowest 1% of it ("trim"), so a stray spike hanging below a
  generated model does not hold it up, or its single lowest point ("lowest").
  Applies to every object that stands on something.
- depth_weight: a number from 0 to 3, now 0. While the pose of a partly hidden
  object is searched for, how far its visible surface is from where the depth map
  puts the object counts against a candidate, multiplied by this.
- size_weight: a number from 0 to 3, now 0. In the same search, how far the
  model's size is from the estimated size counts against a candidate, multiplied
  by this.
- turn_deg: a number from -180 to 180, now 0. The heading the pose search starts
  from is turned by this many degrees about the vertical, and so is the limit on
  how far the model's front may turn from the camera. For a model whose front is
  not where the generator put it, e.g. one modelled from a redrawn picture drawn
  from another angle than the photo.
- fit_box: true or false, now true. Only for an object modelled from a redrawn
  picture: after its pose is found, its model is stretched or squashed (each
  side by at most 1.6 times) onto the box the depth map measured for the object,
  kept only where that fits the outline better than the model's own proportions.
  false keeps the model's own proportions.
"""

RUN_TRIAL = {
    "type": "function",
    "function": {
        "name": "run_trial",
        "description": "Generate one object again with another generator or changed settings, pose it, and measure the result.",
        "parameters": {
            "type": "object",
            "properties": {
                "object_id": {"type": "string", "description": "the object to generate"},
                "generator": {"type": "string", "description": "the generator to use; leave out for the one the run uses"},
                "settings": {"type": "object", "description": 'settings to change, e.g. {"texture_sample.steps": 8}; {} for none'},
                "reason": {"type": "string", "description": "one sentence: what this trial is meant to show"},
            },
            "required": ["object_id", "settings", "reason"],
        },
    },
}
POSE_TRIAL = {
    "type": "function",
    "function": {
        "name": "pose_trial",
        "description": "Pose one object's existing model again with the given pose settings, and measure the result.",
        "parameters": {
            "type": "object",
            "properties": {
                "object_id": {"type": "string", "description": "the object to pose"},
                "settings": {"type": "object", "description": 'pose settings, e.g. {"wall_weight": 1, "pull_forward": true}; {} for none'},
                "reason": {"type": "string", "description": "one sentence: what this trial is meant to show"},
            },
            "required": ["object_id", "settings", "reason"],
        },
    },
}
FINISH = {
    "type": "function",
    "function": {
        "name": "finish",
        "description": "End the experiment and hand your findings to the supervisor.",
        "parameters": {
            "type": "object",
            "properties": {"report": {"type": "string", "description": "the answer, the evidence, and what is uncertain"}},
            "required": ["report"],
        },
    },
}
# The most any of the pose settings' weights may be.
_MAX_WEIGHT = 3.0


def generators(config: Config) -> dict[str, dict]:
    """Every image-to-3D workflow beside the configured one, by file name without its ending; the configured one first."""
    used = config.image_to_3d_workflow
    found = {used.stem: json.loads(used.read_text())}
    for path in sorted(used.parent.glob("*.json")):
        workflow = json.loads(path.read_text())
        # The two nodes every image-to-3D workflow must have (see the config).
        if "input_image" in workflow and "output_model" in workflow:
            found.setdefault(path.stem, workflow)
    return found


def knob_limits(workflow: dict, comfy: ComfyClient) -> dict[str, dict]:
    """The knobs this workflow has, each with its value now and the range a trial may set.

    {name: {"value": now, "lowest": .., "step": .. or None, "base": the server's minimum, "whole": bool}}
    """
    limits = {}
    for name in KNOBS:
        node, key = name.split(".")
        if node not in workflow or key not in workflow[node]["inputs"]:
            continue
        value = workflow[node]["inputs"][key]
        server = comfy.input_limits(workflow[node]["class_type"]).get(key, {})
        base = server.get("min", 0)
        limits[name] = {
            "value": value,
            "lowest": max(base, value * _LOWEST_SHARE),
            "step": server.get("step"),
            "base": base,
            "whole": isinstance(value, int),
        }
    return limits


def check_settings(changes: dict, limits: dict[str, dict]) -> dict:
    """The changes, if every one names a knob and stays in its range; raises ValueError saying what is wrong."""
    checked = {}
    for name, value in changes.items():
        if name not in limits:
            raise ValueError(f"{name} is not a setting you may change; the settings are {', '.join(limits)}")
        limit = limits[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{name} must be a number, not {value!r}")
        if limit["whole"]:
            if value != int(value):
                raise ValueError(f"{name} must be a whole number, not {value}")
            value = int(value)
        if not limit["lowest"] <= value <= limit["value"]:
            raise ValueError(f"{name} must be between {limit['lowest']:g} and {limit['value']:g}, not {value:g}")
        step = limit["step"]
        if step and abs((value - limit["base"]) / step - round((value - limit["base"]) / step)) > 1e-6:
            raise ValueError(f"{name} must be {limit['base']:g} plus a multiple of {step:g}, not {value:g}")
        checked[name] = value
    return checked


def check_pose_settings(changes: dict) -> PoseSettings:
    """The pose settings asked for, if they are ones a trial may use; raises ValueError saying what is wrong."""
    unknown = set(changes) - set(PoseSettings.model_fields)
    if unknown:
        raise ValueError(f"{', '.join(sorted(unknown))}: not a pose setting; the pose settings are {', '.join(PoseSettings.model_fields)}")
    pull, reseat, contact = changes.get("pull_forward", False), changes.get("reseat", "none"), changes.get("contact", "trim")
    weights = {name: changes.get(name, 0.0) for name in ("wall_weight", "depth_weight", "size_weight")}
    for name, weight in weights.items():
        if isinstance(weight, bool) or not isinstance(weight, (int, float)) or not 0 <= weight <= _MAX_WEIGHT:
            raise ValueError(f"{name} must be a number between 0 and {_MAX_WEIGHT:g}, not {weight!r}")
    if not isinstance(pull, bool):
        raise ValueError(f"pull_forward must be true or false, not {pull!r}")
    if reseat not in ("none", "drop", "stretch"):
        raise ValueError(f'reseat must be "none", "drop" or "stretch", not {reseat!r}')
    if contact not in ("lowest", "trim"):
        raise ValueError(f'contact must be "lowest" or "trim", not {contact!r}')
    turn, fit_box = changes.get("turn_deg", 0.0), changes.get("fit_box", True)
    if isinstance(turn, bool) or not isinstance(turn, (int, float)) or not -180 <= turn <= 180:
        raise ValueError(f"turn_deg must be a number between -180 and 180, not {turn!r}")
    if not isinstance(fit_box, bool):
        raise ValueError(f"fit_box must be true or false, not {fit_box!r}")
    return PoseSettings(pull_forward=pull, reseat=reseat, contact=contact, turn_deg=turn, fit_box=fit_box, **weights)


def with_settings(workflow: dict, changes: dict) -> dict:
    """A copy of the workflow with the changed settings."""
    changed = json.loads(json.dumps(workflow))
    for name, value in changes.items():
        node, key = name.split(".")
        changed[node]["inputs"][key] = value
    return changed


# What a trial's copy of a run leaves behind: the models (copied apart), old models and the run's own records.
_LEFT_BEHIND = shutil.ignore_patterns(
    "assets", "assets_replaced", "debug", "batch.json", "progress_*.json", "agent.jsonl", "usage.jsonl"
)
# A model's GLB is only what its USD was made from; posing and rendering read the USD and its textures.
_NOT_READ = shutil.ignore_patterns("*.glb")


def copy_run(run_dir: Path, folder: Path, object_id: str) -> None:
    """Make `folder` a copy of the run in which `object_id` has no model yet.

    The other objects' models are copied too, not linked. The run may still be
    at work while a trial runs, and when it replaces a model it moves the old
    one's folder away first, which would leave a linked trial with nothing to
    read until the new model lands. A copy is the run as it stood when the
    trial began, for as long as the trial lasts.
    """
    shutil.copytree(run_dir, folder, ignore=_LEFT_BEHIND)
    (folder / "debug").mkdir()
    (folder / "assets" / object_id).mkdir(parents=True)
    for entry in (run_dir / "assets").iterdir():
        if entry.name == object_id:
            continue
        if entry.is_dir():
            shutil.copytree(_model_folder(run_dir, entry), folder / "assets" / entry.name, ignore=_NOT_READ)
        else:
            shutil.copy2(entry, folder / "assets" / entry.name)


def _model_folder(run_dir: Path, folder: Path) -> Path:
    """`folder`, or if the run has set its model aside to make a new one, the latest model it set aside.

    While the new model is being made the folder holds none, and the scene
    still refers to it; the one set aside is what the run will put back if the
    new one is worse.
    """
    if (folder / f"{folder.name}.usdc").exists() or (folder / "flat.json").exists():
        return folder
    aside = sorted(
        (old for old in (run_dir / "assets_replaced").glob(f"{folder.name}.*") if (old / f"{folder.name}.usdc").exists()),
        key=lambda old: int(old.suffix[1:]) if old.suffix[1:].isdigit() else 0,
    )
    return aside[-1] if aside else folder


def run_trial(run_dir: Path, folder: Path, object_id: str, workflow: dict, config: Config, seed: int = 0) -> dict:
    """Generate and place one object in a copy of the run, with `workflow`; returns what was measured.

    `seed` 0 uses the random seeds the object's current model was made with;
    any other value moves every seed by that much, for a different draw.
    """
    copy_run(run_dir, folder, object_id)
    if seed:
        attempt = cli.overrides(folder).get(object_id, {}).get("attempt", 0)
        cli.set_override(folder, object_id, attempt=attempt + seed)
    (folder / "workflow.json").write_text(json.dumps(workflow, indent=2))
    trial_config = config.model_copy(update={"image_to_3d_workflow": folder / "workflow.json"})
    comfy = cli.comfy_client(config.comfy.heavy[0], folder, heavy=True)
    cli._generate_asset(folder, object_id, trial_config, comfy)
    cli._place_asset(folder, object_id)
    cli.pose(folder, [object_id])
    model = folder / "assets" / object_id / f"{object_id}.glb"
    return {
        "generation_seconds": round(comfy.last_seconds) if comfy.last_seconds else None,
        **cli.placement_facts(folder, object_id),
        "model_mb": round(model.stat().st_size / 1e6, 1),
    }


def pose_trial(run_dir: Path, folder: Path, object_id: str, settings: PoseSettings) -> dict:
    """Pose one object's model again in a copy of the run, with `settings`; returns what was measured."""
    copy_run(run_dir, folder, object_id)
    shutil.copytree(run_dir / "assets" / object_id, folder / "assets" / object_id, dirs_exist_ok=True)
    cli.pose(folder, [object_id], settings)
    return cli.placement_facts(folder, object_id)


def current_models(run_dir: Path, comfy: ComfyClient) -> dict[str, dict]:
    """Each object of the run that has a generated model of its own, with its description and measurements now."""
    inventory = Inventory.model_validate_json((run_dir / "inventory.json").read_text())
    # Layout leaves out an object whose mask is unusable, though a model made earlier may remain.
    in_scene = {o.name for o in SceneLayout.model_validate_json((run_dir / "layout.json").read_text()).objects}
    posed = {p["id"] for p in json.loads((run_dir / "poses.json").read_text())} & in_scene
    jobs = read_job_labels(cli.JOB_LOG)
    models = {}
    for obj in inventory.objects:
        folder = run_dir / "assets" / obj.id
        if not (folder / f"{obj.id}.glb").exists() or (folder / "flat.json").exists() or obj.id not in posed:
            continue
        # The newest job for this model that the server still remembers.
        ours = [job for job, label in jobs.items() if label == f"3D model {run_dir.name}_{obj.id}"]
        seconds = next((s for s in (comfy.job_seconds(job) for job in reversed(ours)) if s), None)
        models[obj.id] = {
            "description": obj.description,
            "generation_seconds": round(seconds) if seconds else None,
            **cli.placement_facts(run_dir, obj.id),
        }
    return models


def converse(
    chat: Callable[[list[dict]], dict],
    messages: list[dict],
    act: Callable[[str, dict], tuple[dict, bytes | None, bool]],
    max_turns: int,
    record: Callable[..., None],
) -> bool:
    """Let the worker talk and call tools until one ends the experiment; says whether one did.

    `chat` returns the worker's next message. `act` carries out one tool call
    and returns (the result to tell the worker, a JPEG to show it or None,
    whether the experiment is over).
    """
    for _ in range(max_turns):
        message = chat(messages)
        record(kind="worker", text=message.get("content", ""), thinking=message.get("thinking", ""), calls=message.get("tool_calls", []))
        # Its private reasoning is logged but not sent back: it only fills the context.
        messages.append({key: value for key, value in message.items() if key != "thinking"})
        calls = message.get("tool_calls") or []
        if not calls:
            messages.append({"role": "user", "content": "Call a tool: run a trial, or finish to end the experiment."})
            continue
        pictures = []
        for call in calls:
            name, arguments = call["function"]["name"], call["function"].get("arguments") or {}
            if isinstance(arguments, str):
                arguments = json.loads(arguments)
            result, picture, done = act(name, arguments)
            record(kind="result", tool=name, arguments=arguments, result=result)
            if done:
                return True
            messages.append({"role": "tool", "tool_name": name, "content": json.dumps(result)})
            if picture:
                pictures.append((result.get("trial"), picture))
        # A tool's result is text only, so pictures follow as a message of their own.
        for trial, picture in pictures:
            messages.append({
                "role": "user",
                "content": f"Close-ups for trial {trial}: PHOTO, CURRENT, TRIAL.",
                "images": [base64.b64encode(picture).decode()],
            })
    return False


def new_folder(run_dir: Path, root: Path = DEFAULT_ROOT) -> Path:
    """Make the folder for a new experiment on a run, named for the time and the run's photo."""
    folder = root / f"{datetime.now():%Y%m%d-%H%M%S}-{run_dir.name.split('-', 2)[-1]}"
    folder.mkdir(parents=True)
    return folder


# A trial that fails is usually the harness's fault, not the worker's, so the first few failures cost it nothing.
_FREE_FAILURES = 2


def _spent(trials: list[dict], kind: str) -> int:
    """How many trials of `kind` count against the budget: all that ran, less the first few that failed."""
    ran = [trial for trial in trials if trial["kind"] == kind]
    return len(ran) - min(sum("error" in trial for trial in ran), _FREE_FAILURES)


def experiment(
    run_dir: Path,
    question: str,
    config_path: Path,
    vlm_name: str | None,
    max_trials: int,
    max_pose_trials: int = 0,
    folder: Path | None = None,
) -> Path:
    """Have the local model answer `question` about a run by trials; returns the experiment's folder.

    It may generate `max_trials` models and pose models `max_pose_trials`
    times; a kind of trial with no budget is not offered to it at all.
    `folder` is where the experiment keeps its files, a new one by default.
    """
    config = load_config(config_path)
    folder = folder or new_folder(run_dir)
    print(f"experiment folder: {folder}", flush=True)
    vlm = cli.vlm_for(config, vlm_name, folder)
    if not hasattr(vlm, "chat"):
        raise ValueError("the lab's worker must be a local (ollama) model")
    ledger = Ledger(folder / "usage.jsonl", config.max_billed_tokens)
    comfy = cli.comfy_client(config.comfy.heavy[0], folder, heavy=True)
    workflows = generators(config)
    used = config.image_to_3d_workflow.stem
    limits = {name: knob_limits(workflow, comfy) for name, workflow in workflows.items()} if max_trials else {}
    models = current_models(run_dir, comfy)
    photo, current = Image.open(run_dir / "input.jpg"), Image.open(run_dir / "render.png")
    budget = {"run_trial": max_trials, "pose_trial": max_pose_trials}
    tools = [tool for tool in (RUN_TRIAL, POSE_TRIAL) if budget[tool["function"]["name"]]] + [FINISH]
    trials: list[dict] = []
    outcome = {"report": ""}

    def record(**entry) -> None:
        with (folder / "log.jsonl").open("a") as file:
            file.write(json.dumps(entry) + "\n")

    def chat(messages: list[dict]) -> dict:
        message, usage = vlm.chat(messages, tools, label=f"experiment, after {len(trials)} trials")
        ledger.record("lab", usage)
        if message.get("content"):
            print(f"[worker] {message['content']}", flush=True)
        return message

    def act(name: str, arguments: dict) -> tuple[dict, bytes | None, bool]:
        if name == "finish":
            outcome["report"] = str(arguments.get("report", ""))
            return {}, None, True
        if not budget.get(name):
            return {"error": f"there is no tool called {name}"}, None, False
        left = budget[name] - _spent(trials, name)
        if left <= 0:
            return {"error": f"no {name} trials left; use what you have found"}, None, False
        object_id, reason = arguments.get("object_id"), str(arguments.get("reason", ""))
        try:
            if object_id not in models:
                raise ValueError(f"{object_id} is not an object you may test; the objects are {', '.join(models)}")
            if name == "pose_trial":
                generator, changes = used, check_pose_settings(arguments.get("settings") or {}).model_dump()
            else:
                generator = arguments.get("generator") or used
                if generator not in workflows:
                    raise ValueError(f"{generator} is not a generator you may use; the generators are {', '.join(workflows)}")
                changes = check_settings(arguments.get("settings") or {}, limits[generator])
        except ValueError as error:
            # A request that could not be run costs no trial.
            return {"error": str(error)}, None, False
        number = len(trials) + 1
        trial = {"trial": number, "kind": name, "object_id": object_id, "generator": generator, "settings": changes, "reason": reason}
        trials.append(trial)
        print(f"[trial {number}] {name} {object_id} with {generator}, {changes or 'no changes'}: {reason}", flush=True)
        trial_dir = folder / f"{folder.name}-t{number}"
        try:
            if name == "pose_trial":
                trial.update(pose_trial(run_dir, trial_dir, object_id, PoseSettings(**changes)))
            else:
                trial.update(run_trial(run_dir, trial_dir, object_id, with_settings(workflows[generator], changes), config))
            mask = np.asarray(Image.open(run_dir / "masks" / f"{object_id}.png")) > 0
            picture = close_ups([(photo, "PHOTO"), (current, "CURRENT"), (Image.open(trial_dir / "render.png"), "TRIAL")], mask)
            picture.save(folder / f"t{number}_{object_id}.jpg", quality=85)
            trial["picture"] = f"t{number}_{object_id}.jpg"
        except Exception as error:  # a failed trial is a result too, and must not end the experiment
            trial["error"] = str(error)
            picture = None
        if name == "pose_trial":
            # A pose can be found again in a minute, and there may be dozens: keep the numbers and the picture only.
            shutil.rmtree(trial_dir, ignore_errors=True)
        (folder / "trials.json").write_text(json.dumps(trials, indent=2))
        print(f"[trial {number}] {json.dumps({k: v for k, v in trial.items() if k not in ('reason', 'settings')})}", flush=True)
        told = {**{k: v for k, v in trial.items() if k not in ("reason", "picture", "kind")}, f"{name}s_left": left - 1}
        return told, to_jpeg(picture) if picture else None, False

    sections = ""
    if max_trials:
        sections += GENERATION.format(
            generators="\n".join(
                f"- {name}: " + (config.workflow_notes.get(name) or "nothing is recorded about it")
                + (". This is the one the run uses." if name == used else ".")
                for name in workflows
            ),
            settings=_describe_settings(limits),
        )
    if max_pose_trials:
        sections += POSING
    brief = BRIEF.format(
        sections=sections,
        objects="\n".join(
            f"- {object_id}: {model['description']}. Now: "
            + (f"{model['generation_seconds']} s to generate" if model["generation_seconds"] else "generation time not known")
            + "".join(f", {key} {model[key]}" for key in _MEASURES)
            + "."
            for object_id, model in models.items()
        ),
        budget=" ".join(
            text for text, count in (
                (f"You may generate {max_trials} models with run_trial. Each takes minutes to an hour, so choose them with care.", max_trials),
                (f"You may run {max_pose_trials} pose trials with pose_trial.", max_pose_trials),
            ) if count
        ),
        question=question,
    )
    (folder / "brief.txt").write_text(brief)
    messages = [{"role": "user", "content": brief}]
    # Enough turns for a plan, every trial with a thought after it, and a few slips.
    finished = converse(chat, messages, act, 2 * (max_trials + max_pose_trials) + 4, record)
    (folder / "report.md").write_text(_report(question, run_dir, used, models, trials, outcome["report"], finished))
    print(f"-> {folder}/report.md", flush=True)
    return folder


def _describe_settings(limits: dict[str, dict[str, dict]]) -> str:
    """The settings and their ranges as the brief lists them, once for each group of generators that share them."""
    listings: dict[str, list[str]] = {}
    for generator, knobs in limits.items():
        listing = "\n".join(
            f"- {name}: {KNOBS[name]}. Now {limit['value']:g}; may be set from {limit['lowest']:g} to {limit['value']:g}"
            + (f", in steps of {limit['step']:g} from {limit['base']:g}" if limit["step"] else "")
            + (", whole numbers only." if limit["whole"] else ".")
            for name, limit in knobs.items()
        )
        listings.setdefault(listing, []).append(generator)
    if len(listings) == 1:
        return next(iter(listings)) + ("\nEvery generator has these settings, with these values." if len(limits) > 1 else "")
    return "\n".join(f"For {', '.join(names)}:\n{listing}" for listing, names in listings.items())


# What every trial measures about how a model sits in the scene (see `cli.placement_facts`).
_MEASURES = (
    "outline_fit", "colour_error", "hidden_by_wall", "off_floor_m", "longest_side_m", "distance_vs_photo", "size_vs_estimate",
    "inside_other", "inside_which",
)


def _report(
    question: str, run_dir: Path, used: str, models: dict[str, dict], trials: list[dict], findings: str, finished: bool
) -> str:
    """The experiment written up: what the harness measured, then what the worker made of it."""
    lines = [f"# Experiment on {run_dir.name}", "", f"**Question:** {question}", "", "## Trials, as measured by the harness", ""]
    lines += [
        "| Trial | Kind | Object | Generator | Settings | Seconds | Outline fit | Colour error | Hidden by wall | Off floor (m) "
        "| Longest side (m) | Distance vs photo | Size vs estimate | Inside another | Which | Picture |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for object_id in dict.fromkeys(trial["object_id"] for trial in trials):
        model = models[object_id]
        lines.append(
            f"| current | | {object_id} | {used} | as run | {model['generation_seconds'] or 'not known'} | "
            + " | ".join(str(model.get(key, "")) for key in _MEASURES) + " | |"
        )
    for trial in trials:
        kind = "pose" if trial["kind"] == "pose_trial" else "generate"
        changed = ", ".join(f"{name} = {value}" for name, value in trial["settings"].items()) or "none"
        start = f"| {trial['trial']} | {kind} | {trial['object_id']} | {trial['generator']} | {changed} | "
        if "error" in trial:
            lines.append(start + f"failed: {trial['error']} | | | | | | | | | | |")
        else:
            seconds = trial.get("generation_seconds") or ("not known" if kind == "generate" else "")
            lines.append(start + f"{seconds} | " + " | ".join(str(trial.get(key, "")) for key in _MEASURES) + f" | {trial['picture']} |")
    lines += ["", "## The worker's findings", ""]
    lines.append(findings if finished else "The worker did not finish: it ran out of turns before reporting. See log.jsonl.")
    return "\n".join(lines) + "\n"
