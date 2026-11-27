"""Asset stage: turn one object's pixels into a textured 3D model."""

import hashlib
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image

from aomma.backends.comfy import ComfyClient
from aomma.stages.image import to_jpeg


def crop_background(workflow_path: Path) -> str:
    """The background an image-to-3D workflow wants its crop on: "black" unless it says otherwise.

    Generators are trained on cut-out objects against one plain colour, and
    not all on the same one. A workflow asks for another by carrying a
    PrimitiveString node named "input_background" with that colour as its value.
    """
    workflow = json.loads(workflow_path.read_text())
    return workflow["input_background"]["inputs"]["value"] if "input_background" in workflow else "black"


def crop_object(
    source: Image.Image, mask: np.ndarray, size: int = 1024, padding: float = 1.1, background: str = "black"
) -> tuple[Image.Image, int]:
    """Cut one object out of the photo: centred, square, on a plain background.

    Returns the crop and the side of the square it covers in `source`, in pixels.

    `source` is the full-resolution photo and `mask` the object's mask at any
    resolution with the same aspect; cropping from the original keeps detail
    that the downscaled working image has lost.
    """
    alpha = Image.fromarray(mask).resize(source.size, Image.Resampling.BILINEAR)
    left, top, right, bottom = alpha.getbbox()
    side = round(max(right - left, bottom - top) * padding)
    cx, cy = (left + right) // 2, (top + bottom) // 2
    box = (cx - side // 2, cy - side // 2, cx - side // 2 + side, cy - side // 2 + side)

    # Composite first so that neighbouring objects inside the square are painted out.
    # The square may hang over the photo's edge, so the crop is pasted onto the background too.
    cutout = Image.composite(source, Image.new("RGB", source.size, background), alpha)
    square = Image.new("RGB", (side, side), background)
    square.paste(cutout, (-box[0], -box[1]))
    return square.resize((size, size), Image.Resampling.LANCZOS), side


def in_object_crop(area: np.ndarray, outline: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """`area`, a boolean array in the photo's frame, in the frame of `crop_object` of an object with this `outline`.

    `size` is the full-resolution photo's. `outline` must have the box of the
    mask the crop was cut with: the mask with its enclosed holes filled does.
    """
    picture = Image.fromarray(area).convert("RGB").resize(size, Image.Resampling.NEAREST)
    square, _ = crop_object(picture, outline, background="black")
    return np.asarray(square.convert("L")) > 127


# Models already made, one GLB per crop, workflow and seed (see `model_key`), shared by every run.
MODEL_CACHE = Path("cache") / "models"


def model_key(workflow_path: Path, crop: Image.Image, fov_deg: float, attempt: int = 0) -> str:
    """What decides a model: the crop's pixels, the workflow as it will be sent, the field of view and the seed shift.

    Two runs with the same key would ask the generator for the same thing, so the
    second can take the first's model. A noise draw (`seed_offset`) shifts
    `attempt`, so it gets keys of its own.
    """
    rgb = crop.convert("RGB")
    digest = hashlib.sha256()
    digest.update(f"{rgb.size}".encode())
    digest.update(rgb.tobytes())
    digest.update(json.dumps(json.loads(workflow_path.read_text()), sort_keys=True).encode())
    digest.update(f"fov={round(fov_deg, 4)} attempt={attempt}".encode())
    return digest.hexdigest()[:32]


def cached_model(key: str, cache: Path = MODEL_CACHE) -> Path | None:
    """The cached model for `key`, or None if there is none or AOMMA_FRESH_MODELS is set (models drawn afresh, still cached)."""
    path = cache / f"{key}.glb"
    return path if path.exists() and not os.environ.get("AOMMA_FRESH_MODELS") else None


def generate_model(
    comfy: ComfyClient,
    workflow_path: Path,
    crop: Image.Image,
    fov_deg: float,
    name: str,
    attempt: int = 0,
    ticket: int | None = None,
    cache: Path | None = MODEL_CACHE,
) -> bytes:
    """`generate_model_uncached`, taking the model from `cache` when the same crop, workflow and seed were modelled before, and keeping each new one there."""
    if cache is None:
        return generate_model_uncached(comfy, workflow_path, crop, fov_deg, name, attempt, ticket)
    key = model_key(workflow_path, crop, fov_deg, attempt)
    if hit := cached_model(key, cache):
        return hit.read_bytes()
    model = generate_model_uncached(comfy, workflow_path, crop, fov_deg, name, attempt, ticket)
    cache.mkdir(parents=True, exist_ok=True)
    part = cache / f".{key}.{os.getpid()}.part"
    part.write_bytes(model)
    part.replace(cache / f"{key}.glb")
    return model


def generate_model_uncached(
    comfy: ComfyClient,
    workflow_path: Path,
    crop: Image.Image,
    fov_deg: float,
    name: str,
    attempt: int = 0,
    ticket: int | None = None,
) -> bytes:
    """Run an image-to-3D workflow on `crop` and return the model as GLB bytes.

    The workflow is any ComfyUI API-format graph with a LoadImage node called
    "input_image" and a 3D save node called "output_model"; swapping the 3D
    generator means pointing at a different workflow file. If it has a
    PrimitiveFloat node called "input_fov", that receives `fov_deg`, the
    crop's horizontal field of view. `attempt` shifts every sampler's seed, so
    a later attempt at the same crop comes out differently. `ticket` is a
    place in the job queue taken earlier for this model.
    """
    workflow = json.loads(workflow_path.read_text())
    workflow["input_image"]["inputs"]["image"] = comfy.upload_image(to_jpeg(crop), f"aomma_{name}.jpg")
    if "input_fov" in workflow:
        workflow["input_fov"]["inputs"]["value"] = fov_deg
    for node in workflow.values():
        for key in ("seed", "noise_seed"):
            if isinstance(node["inputs"].get(key), int):
                node["inputs"][key] += attempt
    outputs = comfy.run(workflow, label=f"3D model {name}", ticket=ticket)
    return comfy.fetch(outputs["output_model"]["3d"][0])
