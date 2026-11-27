from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel

# The checkout this package was loaded from. A round runs its commands from a checkout of its own commit
# (rounds.py), in the project's folder, so the settings and workflows that go with the code are found from here.
CODE_ROOT = Path(__file__).resolve().parents[2]


def code_path(path: Path) -> Path:
    """A path to one of the code's own files (settings, workflows), from this checkout.

    Left as written when this checkout is the folder the command runs in, so records keep the short form.
    """
    if path.is_absolute() or CODE_ROOT == Path.cwd().resolve():
        return path
    return CODE_ROOT / path


DEFAULT_CONFIG = Path("config/default.yaml")
# This machine's own settings (server addresses, chiefly), kept out of git and laid over the defaults. Looked for in
# the folder the command runs in first, then beside the code: a round runs from a checkout of its commit, which holds
# no untracked files.
LOCAL_CONFIG = Path("config/local.yaml")


def _merge(base: dict, over: dict) -> dict:
    """`over` laid on `base`: mappings merged key by key, anything else replaced."""
    out = dict(base)
    for key, value in over.items():
        out[key] = _merge(out[key], value) if isinstance(value, dict) and isinstance(out.get(key), dict) else value
    return out


def local_config_path() -> Path | None:
    for path in (Path.cwd() / LOCAL_CONFIG, CODE_ROOT / LOCAL_CONFIG):
        if path.exists():
            return path
    return None


class VLMConfig(BaseModel):
    backend: Literal["ollama", "anthropic"]
    model: str
    url: str | None = None
    effort: str = "low"


class ComfyServers(BaseModel):
    light: list[str]
    heavy: list[str]

    @property
    def all(self) -> list[str]:
        """Every distinct server, light lane first."""
        return list(dict.fromkeys(self.light + self.heavy))


class Config(BaseModel):
    vlm: str
    vlms: dict[str, VLMConfig]
    comfy: ComfyServers
    min_free_memory_gb: float = 30
    image_to_3d_workflow: Path
    # What is known about each image-to-3D workflow beside that one, by file name
    # without its ending, for experiments that compare them (see lab.py).
    workflow_notes: dict[str, str] = {}
    repair_workflow: Path
    # Redraws a top-down crop of a floor as a texture of it (see stages/floor.py).
    floor_texture_workflow: Path = Path("workflows/flux2_klein_texture.json")
    max_billed_tokens: int = 50000
    image_max_side: int = 1024
    # The review after a run: how many times it looks the scene over, and how
    # many models it may generate in all while fixing what it finds.
    review_rounds: int = 2
    review_generations: int = 6
    # The review of a staged run's large items alone (`aomma review --stage large`), with its own budget.
    large_review_generations: int = 4


def load_config(path: Path = DEFAULT_CONFIG) -> Config:
    """The settings, with the default file and every workflow in it taken from this checkout (`code_path`).

    The default file is overlaid with `config/local.yaml` where one exists (`local_config_path`).
    """
    data = yaml.safe_load(code_path(path).read_text() if path == DEFAULT_CONFIG else path.read_text())
    if path == DEFAULT_CONFIG and (local := local_config_path()):
        data = _merge(data, yaml.safe_load(local.read_text()) or {})
    config = Config.model_validate(data)
    for field in ("image_to_3d_workflow", "repair_workflow", "floor_texture_workflow"):
        setattr(config, field, code_path(getattr(config, field)))
    return config
