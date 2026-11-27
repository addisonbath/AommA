"""Headless Blender jobs: render a USD scene or its depth, convert a GLB model to USD, export a scene for the web."""

import os
import subprocess
import time
from pathlib import Path

from aomma import timings

DEFAULT_BLENDER = "/Applications/Blender.app/Contents/MacOS/Blender"
_SCRIPTS = Path(__file__).parent


def _run(script: str, *args: str, expect: Path) -> None:
    blender = os.environ.get("AOMMA_BLENDER", DEFAULT_BLENDER)
    expect.unlink(missing_ok=True)
    started = time.monotonic()
    result = subprocess.run(
        [blender, "--background", "--factory-startup", "--python", str(_SCRIPTS / script), "--", *args],
        capture_output=True,
        text=True,
    )
    # Blender can exit 0 even when the script raised, so check for the output file.
    failed = result.returncode != 0 or not expect.exists()
    timings.record(
        {
            "kind": "blender",
            "label": f"{script} -> {expect.name}",
            "run": timings.run_of(expect),
            "outcome": "failed" if failed else "done",
            "ran_s": round(time.monotonic() - started, 2),
        }
    )
    if failed:
        tail = "\n".join((result.stdout + result.stderr).splitlines()[-20:])
        raise RuntimeError(f"Blender {script} failed:\n{tail}")


def render(scene: Path, out: Path, width: int, height: int) -> None:
    """Render `scene` through its camera to a PNG."""
    _run("blender_script.py", str(scene.resolve()), str(out.resolve()), str(width), str(height), expect=out)


def render_depth(scene: Path, out: Path, width: int, height: int, objects: Path | None = None) -> None:
    """Render `scene`'s depth through its camera to a .npy array, and which model each pixel shows to `objects` (see depth_script.py)."""
    extra = [str(objects.resolve())] if objects else []
    if objects:
        objects.unlink(missing_ok=True)
    _run("depth_script.py", str(scene.resolve()), str(out.resolve()), str(width), str(height), *extra, expect=out)


def render_depth_views(scene: Path, views: Path, out: Path, last: str) -> None:
    """Render `scene`'s depth, and which model each pixel shows, from each camera listed in `views` into `out`
    (see depth_script.py); `last` is the last camera's name, whose files are written last."""
    out.mkdir(parents=True, exist_ok=True)
    _run("depth_script.py", str(scene.resolve()), "--views", str(views.resolve()), str(out.resolve()), expect=out / f"{last}.npy")


def glb_to_usd(glb: Path, usd: Path) -> None:
    """Convert a GLB model into a normalised USD asset (see convert_script.py)."""
    usd.parent.mkdir(parents=True, exist_ok=True)
    _run("convert_script.py", str(glb.resolve()), str(usd.resolve()), expect=usd)


def usd_to_glb(scene: Path, glb: Path, max_texture: int = 1024) -> None:
    """Export `scene` as one compact GLB file, for viewers that cannot read USD (see export_script.py)."""
    glb.parent.mkdir(parents=True, exist_ok=True)
    _run("export_script.py", str(scene.resolve()), str(glb.resolve()), str(max_texture), expect=glb)
