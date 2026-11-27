"""Which model does each stage of the pipeline use, how big is it, and what else could do the job.

Nothing here is typed in twice: the models in use are read from the config,
the workflow files and the stages' own constants, and sizes come from the
servers that hold the files. Only the list of models known but not installed
is written by hand.
"""

import json
from pathlib import Path

import httpx

from aomma.config import Config, code_path
from aomma.stages.depth import MOGE_MODEL
from aomma.stages.segment import SAM_CHECKPOINT

_WORKFLOWS = Path("workflows")
_MODEL_SUFFIXES = (".safetensors", ".ckpt", ".pt", ".pth", ".gguf", ".onnx", ".bin", ".sft")

# Plain names for the parts a workflow loads, by the name of the node that loads them.
_ROLES = {
    "unet": "generator", "clip": "text encoder", "vae": "image decoder", "clip vision": "image encoder",
    "matting": "background remover", "shape vae": "shape decoder", "texture vae": "texture decoder",
}
# What happened when a workflow in this repo was tried.
_WORKFLOW_NOTES = {"trellis2_image_to_glb.json": "Tried first in this project: its textures came out smeared."}

# Models that could do a stage's job but are not set up here, with why one might try them.
# Kept short and hand-written; none has been run in this project unless it says so.
_KNOWN = {
    "inventory": [
        ("Other cloud vision models", "Any vision model that returns JSON can be added under `vlms` in the config."),
    ],
    "masks": [
        ("Grounding DINO + SAM 2", "The older pairing: one model turns text into boxes, the other boxes into masks."),
    ],
    "depth": [
        ("Depth Pro", "Metric depth that also estimates the lens angle, for photos without metadata."),
        ("Depth Anything V2", "Sharp relative depth; needs a separate step to get real-world scale."),
    ],
    "repair": [
        ("FLUX.1 Kontext", "A larger image-editing model; slower, often more faithful to the original object."),
        ("Qwen-Image-Edit", "Another instruction-following image editor."),
    ],
    "model": [
        ("Hunyuan3D 2", "Image-to-3D with separate shape and texture stages."),
        ("TripoSR / Stable Fast 3D", "Much faster and much coarser; useful for quick drafts."),
    ],
}


def _comfy_files(url: str) -> dict[str, dict]:
    """Every model file a ComfyUI server holds: name -> {size, folder}."""
    files: dict[str, dict] = {}
    for folder in httpx.get(f"{url}/api/experiment/models", timeout=20).json():
        listing = httpx.get(f"{url}/api/experiment/models/{folder['name']}", timeout=20).json()
        for entry in listing:
            files.setdefault(entry["name"], {"size": entry.get("size"), "folder": folder["name"]})
    return files


def _workflow_files(path: Path) -> list[tuple[str, str]]:
    """The model files a workflow loads, as (node name, file name)."""
    found = []
    for node, spec in json.loads(path.read_text()).items():
        for value in spec.get("inputs", {}).values():
            if isinstance(value, str) and value.endswith(_MODEL_SUFFIXES):
                role = node.replace("_", " ")
                found.append((_ROLES.get(role, role), value))
    return found


def _ollama_models(url: str) -> list[dict]:
    """Models an Ollama server holds, with size and whether each can read images."""
    models = []
    for entry in httpx.get(f"{url}/api/tags", timeout=20).json()["models"]:
        shown = httpx.post(f"{url}/api/show", json={"model": entry["name"]}, timeout=20).json()
        details = shown.get("details", {})
        models.append(
            {
                "name": entry["name"],
                "size": entry["size"],
                "detail": ", ".join(v for v in (details.get("parameter_size"), details.get("quantization_level")) if v),
                "vision": "vision" in shown.get("capabilities", []),
            }
        )
    return models


def pipeline_models(config: Config) -> dict:
    """{"stages": one entry per stage, "problems": servers that could not be asked}.

    A stage says what it does, the models it uses now and the alternatives.
    Each model or alternative is {name, role or note, size in bytes or None,
    where}. A server that cannot be reached leaves sizes empty.
    """
    problems = []
    try:
        files = _comfy_files(config.comfy.all[0])
    except (httpx.HTTPError, ValueError) as error:
        files, problems = {}, [f"ComfyUI at {config.comfy.all[0]}: {error}"]

    def comfy(name: str, role: str) -> dict:
        held = files.get(name)
        return {
            "name": name, "role": role, "size": held["size"] if held else None,
            "where": f"ComfyUI, {held['folder']}" if held else ("ComfyUI" if not files else "not on the ComfyUI server"),
        }

    def siblings(name: str) -> list[dict]:
        """Other files of the same family in the same folder: other sizes or precisions of the model."""
        held = files.get(name)
        family = name.split("_")[0].rstrip("0123456789.")
        return [
            {"name": other, "note": "Installed; another version of the same model.", "size": info["size"], "where": f"ComfyUI, {info['folder']}"}
            for other, info in sorted(files.items())
            if held and other != name and info["folder"] == held["folder"] and other.startswith(family)
        ]

    def known(stage: str) -> list[dict]:
        return [{"name": name, "note": note, "size": None, "where": "not installed"} for name, note in _KNOWN[stage]]

    def workflow_options(current: Path, marker: str) -> list[dict]:
        """Other workflow files in the repo that fill the same slot (they have the same output node)."""
        options = []
        for path in sorted(code_path(_WORKFLOWS).glob("*.json")):
            if path.resolve() == current.resolve() or marker not in json.loads(path.read_text()):
                continue
            needed = _workflow_files(path)
            sizes = [files[name]["size"] for _, name in needed if name in files]
            missing = [name for _, name in needed if name not in files]
            options.append(
                {
                    "name": path.name,
                    "note": (_WORKFLOW_NOTES.get(path.name, "") + " Workflow in this repo; set it in the config to switch. ").lstrip()
                    + (f"Missing on the server: {', '.join(missing)}." if missing and files else "All its files are installed." if files else ""),
                    "size": sum(sizes) if sizes and not missing else None,
                    "where": "workflows/",
                }
            )
        return options

    # The VLM: the configured entry, the other configured entries, and other vision models on the same Ollama.
    active = config.vlms[config.vlm]
    vlm = {"name": active.model, "role": f"config entry `{config.vlm}`", "size": None, "where": "cloud" if active.backend == "anthropic" else "Ollama"}
    vlm_options = [
        {
            "name": entry.model, "note": f"Configured as `{name}`; run with --vlm {name}.", "size": None,
            "where": "cloud, billed per token" if entry.backend == "anthropic" else "Ollama",
        }
        for name, entry in config.vlms.items() if name != config.vlm
    ]
    ollama_url = next((entry.url for entry in config.vlms.values() if entry.backend == "ollama" and entry.url), None)
    if ollama_url:
        try:
            configured = {entry.model for entry in config.vlms.values()}
            for model in _ollama_models(ollama_url):
                if model["name"] == active.model:
                    vlm["size"], vlm["role"] = model["size"], f"{vlm['role']}; {model['detail']}".rstrip("; ")
                elif model["vision"] and model["name"] not in configured:
                    vlm_options.append(
                        {"name": model["name"], "note": f"Installed and reads images; not tried here. {model['detail']}".strip(), "size": model["size"], "where": "Ollama"}
                    )
        except (httpx.HTTPError, ValueError, KeyError) as error:
            problems.append(f"Ollama at {ollama_url}: {error}")

    stages = [
        {
            "key": "inventory", "title": "Inventory",
            "does": "Lists the objects in the photo with boxes, sizes and what rests on what; also splits lumped entries.",
            "models": [vlm], "options": vlm_options + known("inventory"),
        },
        {
            "key": "masks", "title": "Masks",
            "does": "Outlines each object from its box, or from its name when the box fails.",
            "models": [comfy(SAM_CHECKPOINT, "segmenter")], "options": siblings(SAM_CHECKPOINT) + known("masks"),
        },
        {
            "key": "depth", "title": "Depth",
            "does": "Gives every pixel a real-world distance, from which the floor, walls and camera are found.",
            "models": [comfy(MOGE_MODEL, "depth and point map")], "options": siblings(MOGE_MODEL) + known("depth"),
        },
        {
            "key": "repair", "title": "Repair",
            "does": "Redraws a partly hidden object whole on a plain background, and cuts it out again.",
            "models": [comfy(name, role) for role, name in _workflow_files(config.repair_workflow)],
            "options": workflow_options(config.repair_workflow, "output_mask") + known("repair"),
            "workflow": str(config.repair_workflow),
        },
        {
            "key": "model", "title": "Image to 3D",
            "does": "Turns one object's picture into a textured 3D model.",
            "models": [comfy(name, role) for role, name in _workflow_files(config.image_to_3d_workflow)],
            "options": workflow_options(config.image_to_3d_workflow, "output_model") + known("model"),
            "workflow": str(config.image_to_3d_workflow),
        },
    ]
    for stage in stages:
        sizes = [model["size"] for model in stage["models"]]
        stage["total"] = sum(sizes) if all(size is not None for size in sizes) else None
    return {"stages": stages, "problems": problems}
