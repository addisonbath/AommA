"""A run's scene in its scan's world, and rendered from any of the scan's cameras.

A single-shot run is put into the scan's world by its shot's ARKit pose: the run's
camera is the shot's, so the run's world goes to the scan's by the shot's
camera-to-world after the inverse of the run's own. Nothing is scaled: a run whose
depth is too large is too large in the scan's world, and shows so at every other view.
The scene is then rendered (depth, and which model each pixel shows) at any shot's or
keyframe's pose and intrinsics, in one Blender job for all of them.

For scorers: a run's stages never see another view.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from aomma.capture import Scan, View, _manifest
from aomma.layout import SceneLayout
from aomma.render.blender import render_depth_views

# Views are rendered at twice their LiDAR map's size each way, so each LiDAR pixel is the median of four.
LIDAR_FACTOR = 2
# A view without a LiDAR map is rendered at its image's size over this.
IMAGE_DIVISOR = 8


@dataclass
class Rendered:
    """A run's scene seen from one view: depth along the camera's axis (inf where no surface), and which model each
    pixel shows (0 none, else 1 + its index in `names`)."""

    view: View
    depth: np.ndarray
    ids: np.ndarray
    names: list[str]

    def shows(self, name: str) -> np.ndarray:
        return self.ids == self.names.index(name) + 1 if name in self.names else np.zeros(self.ids.shape, bool)


def run_scan(run_dir: Path) -> tuple[Scan, View]:
    """The scan and shot a capture run was made from."""
    manifest = _manifest(run_dir)
    if "scan" not in manifest:
        raise SystemExit(f"{run_dir} was not made from a scan")
    scan = Scan(Path(manifest["scan"]))
    return scan, scan.shot(manifest["shot"])


def run_to_scan(run_dir: Path, shot: View) -> np.ndarray:
    """The 4x4 transform from the run's world (Z up) to its scan's (ARKit's, +y up), by the shot's pose."""
    layout = SceneLayout.model_validate_json((run_dir / "layout.json").read_text())
    return shot.camera_to_world @ np.linalg.inv(layout.camera.matrix)


def render_size(view: View) -> tuple[int, int]:
    if view.depth_size:
        return view.depth_size[0] * LIDAR_FACTOR, view.depth_size[1] * LIDAR_FACTOR
    return view.size[0] // IMAGE_DIVISOR, view.size[1] // IMAGE_DIVISOR


def camera_in_run(view: View, scan_to_run: np.ndarray) -> dict:
    """The view's camera as depth_script.py takes it: placed in the run's world, its intrinsics at the render's size."""
    width, height = render_size(view)
    across, down = width / view.size[0], height / view.size[1]
    return {
        "name": f"{view.kind}_{view.name}",
        "camera_to_world": (scan_to_run @ view.camera_to_world).tolist(),
        "width": width, "height": height,
        "fx": view.k[0, 0] * across, "fy": view.k[1, 1] * down, "cx": view.k[0, 2] * across, "cy": view.k[1, 2] * down,
    }


def render_views(run_dir: Path, views: list[View], out: Path, scene: str = "scene.usda") -> dict[str, Rendered]:
    """Render the run's `scene` (scene.usda, or boxes.usda for gate 4's box layout) from each of `views` (cameras of its
    scan) into `out`; by view name, "<kind>_<name>"."""
    _, shot = run_scan(run_dir)
    scan_to_run = np.linalg.inv(run_to_scan(run_dir, shot))
    cameras = [camera_in_run(view, scan_to_run) for view in views]
    out.mkdir(parents=True, exist_ok=True)
    (out / "views.json").write_text(json.dumps(cameras))
    render_depth_views(run_dir / scene, out / "views.json", out, cameras[-1]["name"])
    rendered = {}
    for view, camera in zip(views, cameras):
        with np.load(out / f"{camera['name']}.npz") as found:
            rendered[camera["name"]] = Rendered(view, np.load(out / f"{camera['name']}.npy"), found["ids"], [str(n) for n in found["names"]])
    return rendered
