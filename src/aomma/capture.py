"""Turn an AommACapture scan into runs: one per input shot, seeded with the photo alone.

What the capture measured for each shot, input or reference, goes to
`refs/<scan>/<shot>/`, out of the runs, for scorers only.
`reference.json` holds it in that shot's camera frame (ARKit axes, as in
`points.npy`: x right, y up, looking down -z): RoomPlan's floor, walls, doors,
windows, openings and objects; the camera's height above the floor; the
intrinsics; and the LiDAR depth (`reference_depth.npy`, metres, with ARKit's
confidence in `reference_conf.npy`). A reference shot gets no run.

A run takes capture data only through `inputs`, and only what its condition allows.
"""

import json
import math
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from functools import cached_property
from pathlib import Path

import numpy as np
from PIL import Image

from aomma.cli import note_source
from aomma.stages.image import _EXIF_IFD, _FOCAL_LENGTH_35MM, _FULL_FRAME_DIAGONAL_MM

# RoomPlan's kinds of surface and object, as keyed in rooms.json.
KINDS = ("floors", "walls", "doors", "windows", "openings", "objects")

REFS = Path("refs")
# ARKit's confidence in a LiDAR depth it is sure of.
LIDAR_SURE = 2
REFERENCE_FILES = ("reference.json", "reference_depth.npy", "reference_conf.npy")

# What each condition may give a run. Only those with
# a loader in `inputs` can be run so far.
CONDITIONS = {
    "photo": (),
    "depth": ("depth", "pose"),
    "depth+roomplan": ("depth", "pose", "roomplan_room"),
    # RoomPlan's walls, floor, doors, windows and openings, but none of its boxes: the built-ins are fitted from the
    # photo and the LiDAR, as a `depth` run's are.
    "depth+walls": ("depth", "pose", "roomplan_walls"),
    "depth+objects": ("depth", "pose", "roomplan_room", "roomplan_objects"),
    "shots": ("shots", "depth", "pose"),
    "shots+roomplan": ("shots", "depth", "pose", "roomplan_room"),
    "dense": ("shots", "depth", "pose", "keyframes"),
}


def _matrix(values: list[float]) -> np.ndarray:
    """A 4x4 transform from ARKit's column-major list."""
    return np.array(values, float).reshape(4, 4).T


# ARKit's world (+y up) turned to the runs' (+z up): x stays, ARKit's -z is +y.
Y_UP_TO_Z_UP = np.array([[1.0, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])


@dataclass
class View:
    """One camera of a scan, a shot or a keyframe, in the scan's world (ARKit's: +y up, metres).

    Camera axes are ARKit's, as in `points.npy`: x right, y up, looking down -z.
    """

    name: str
    kind: str  # "shot" or "keyframe"
    role: str | None  # a shot's "input" or "reference"; None for a keyframe
    label: str
    camera_to_world: np.ndarray
    # The pose of the frame the LiDAR depth came from, which may differ from the image's.
    depth_to_world: np.ndarray
    k: np.ndarray  # 3x3, in pixels of the image
    size: tuple[int, int]  # the image's (width, height)
    depth_size: tuple[int, int] | None  # the LiDAR map's (width, height); None if it has none
    image: Path
    info: dict = field(repr=False)

    @property
    def intrinsics(self) -> dict:
        """fx, fy, cx, cy at the image's width x height, as `inputs` hands them out."""
        return {"fx": self.k[0, 0], "fy": self.k[1, 1], "cx": self.k[0, 2], "cy": self.k[1, 2],
                "width": self.size[0], "height": self.size[1]}

    def project(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """World points (N, 3) as image pixels (N, 2; x across, y down, 0.5 the first pixel's centre) and their depth along the
        camera's axis (N; positive in front)."""
        camera = to_frame(points, np.linalg.inv(self.camera_to_world))
        depth = -camera[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            pixels = np.c_[self.k[0, 0] * camera[:, 0] / depth + self.k[0, 2], -self.k[1, 1] * camera[:, 1] / depth + self.k[1, 2]]
        return pixels, depth

    def distance_to(self, other: "View") -> tuple[float, float]:
        """How far this camera is from `other`: metres between them, and degrees between the ways they look."""
        apart = float(np.linalg.norm(self.camera_to_world[:3, 3] - other.camera_to_world[:3, 3]))
        cos = float(np.clip(-self.camera_to_world[:3, 2] @ -other.camera_to_world[:3, 2], -1, 1))
        return apart, math.degrees(math.acos(cos))


def to_frame(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """(..., 3) points moved by a 4x4 transform."""
    return points @ transform[:3, :3].T + transform[:3, 3]


class Scan:
    """A whole AommACapture scan in its world frame: the shots with their roles, the keyframes, RoomPlan's room and the
    LiDAR mesh, each read when first asked for (scan-contract.md in AommACapture).

    For scorers and checks, never for stages: a run gets capture data only through `inputs`.
    """

    def __init__(self, folder: Path):
        self.folder = Path(folder)
        self.name = self.folder.name

    @cached_property
    def capture(self) -> dict:
        path = self.folder / "capture.json"
        return json.loads(path.read_text()) if path.exists() else {}

    @cached_property
    def shots(self) -> list[View]:
        """Every shot; one from before the app wrote roles is an input."""
        return [self._view(s, "shot", self.folder / "shots") for s in json.loads((self.folder / "shots" / "shots.json").read_text())]

    @cached_property
    def keyframes(self) -> list[View]:
        path = self.folder / "frames" / "frames.json"
        return [self._view(f, "keyframe", path.parent) for f in json.loads(path.read_text())] if path.exists() else []

    def shot(self, name: str) -> View:
        [found] = [s for s in self.shots if s.name == name]
        return found

    @cached_property
    def room(self) -> dict:
        """RoomPlan's room, as rooms.json has it; a scan of several rooms is refused."""
        rooms = json.loads((self.folder / "rooms.json").read_text())
        if len(rooms) != 1:
            raise SystemExit(f"{self.folder} has {len(rooms)} rooms; one is handled")
        return rooms[0]

    @cached_property
    def diagnostics(self) -> dict:
        path = self.folder / "diagnostics.json"
        return json.loads(path.read_text()) if path.exists() else {}

    @cached_property
    def mesh(self) -> np.ndarray:
        """The LiDAR mesh's vertices (N, 3), in the scan's world."""
        from pxr import Usd, UsdGeom

        stage = Usd.Stage.Open(str(self.folder / "mesh.usda"))
        chunks = []
        for prim in stage.Traverse():
            if prim.IsA(UsdGeom.Mesh):
                vertices = np.array(UsdGeom.Mesh(prim).GetPointsAttr().Get())
                matrix = np.array(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()))
                chunks.append(vertices @ matrix[:3, :3] + matrix[3, :3])
        return np.concatenate(chunks) if chunks else np.zeros((0, 3))

    @staticmethod
    def _view(entry: dict, kind: str, folder: Path) -> View:
        camera_to_world = _matrix(entry["transform"])
        return View(
            name=entry["name"], kind=kind, role=entry.get("role", "input") if kind == "shot" else None,
            label=entry.get("label", ""), camera_to_world=camera_to_world,
            depth_to_world=_matrix(entry["depthTransform"]) if "depthTransform" in entry else camera_to_world,
            k=np.array(entry["intrinsics"], float).reshape(3, 3).T, size=tuple(entry["imageSize"]),
            depth_size=tuple(entry["depthSize"]) if entry.get("depthSize") else None,
            image=folder / f"{entry['name']}.jpg", info=entry,
        )

    def depth(self, view: View) -> tuple[np.ndarray, np.ndarray]:
        """The view's LiDAR depth (metres, along the camera's axis) and ARKit's confidence in it (0-2), at the map's own size."""
        if view.depth_size is None:
            raise ValueError(f"{self.name} {view.kind} {view.name} has no depth")
        width, height = view.depth_size
        stem = view.image.with_suffix("")
        depth = np.fromfile(f"{stem}.depth", "<f4").reshape(height, width)
        conf = np.fromfile(f"{stem}.conf", "u1").reshape(height, width)
        return depth, conf

    def lidar_world(self, view: View, confidence: int = LIDAR_SURE) -> tuple[np.ndarray, np.ndarray]:
        """The view's LiDAR as world points (N, 3), each from its pixel's centre and moved by the depth frame's pose, and
        which pixels they are (N, 2: row, column). Only those of ARKit's `confidence` or more (by default, those it is sure of)."""
        depth, conf = self.depth(view)
        keep = np.isfinite(depth) & (depth > 0) & (conf >= confidence)
        rows, cols = np.nonzero(keep)
        width, height = view.size
        x = ((cols + 0.5) * width / depth.shape[1] - view.k[0, 2]) / view.k[0, 0]
        y = -((rows + 0.5) * height / depth.shape[0] - view.k[1, 2]) / view.k[1, 1]
        d = depth[keep]
        return to_frame(np.stack([x * d, y * d, -d], -1), view.depth_to_world), np.c_[rows, cols]

    def depth_at(self, view: View, pixels: np.ndarray, sure: bool = True) -> np.ndarray:
        """The LiDAR's depth at image `pixels` (N, 2, as `View.project` gives them), interpolated between the map's pixel
        centres; NaN outside the image, and with `sure`, where any of the four it is taken from is unsure."""
        depth, conf = self.depth(view)
        rows, cols = depth.shape
        x = pixels[:, 0] * cols / view.size[0] - 0.5
        y = pixels[:, 1] * rows / view.size[1] - 0.5
        inside = np.isfinite(x) & np.isfinite(y) & (x >= 0) & (y >= 0) & (x <= cols - 1) & (y <= rows - 1)
        out = np.full(len(pixels), np.nan)
        x, y = x[inside], y[inside]
        x0, y0 = np.minimum(x.astype(int), cols - 2), np.minimum(y.astype(int), rows - 2)
        fx, fy = x - x0, y - y0
        corners = [(y0, x0), (y0, x0 + 1), (y0 + 1, x0), (y0 + 1, x0 + 1)]
        weights = [(1 - fx) * (1 - fy), fx * (1 - fy), (1 - fx) * fy, fx * fy]
        value = sum(w * depth[r, c] for w, (r, c) in zip(weights, corners))
        if sure:
            unsure = np.any([conf[r, c] != LIDAR_SURE for r, c in corners], axis=0)
            value = np.where(unsure, np.nan, value)
        out[inside] = value
        return out

    def upsampled_depth(self, view: View, size: tuple[int, int] | None = None, sure: bool = True) -> np.ndarray:
        """The LiDAR's depth as a (height, width) map at `size` (the image's by default), interpolated as `depth_at` does."""
        width, height = size or view.size
        v, u = np.mgrid[0:height, 0:width]
        pixels = np.c_[(u.ravel() + 0.5) * view.size[0] / width, (v.ravel() + 0.5) * view.size[1] / height]
        return self.depth_at(view, pixels, sure).reshape(height, width)


def _in_camera(element: dict, world_to_camera: np.ndarray) -> dict:
    """A RoomPlan surface or object moved into the camera frame, transform written row-major."""
    transform = world_to_camera @ _matrix(element["transform"])
    out = {
        "id": element["identifier"],
        "category": next(iter(element["category"])),
        "attributes": element.get("attributes", {}),
        "parent": element.get("parentIdentifier"),
        "confidence": next(iter(element["confidence"])),
        "dimensions": element["dimensions"],
        "transform": transform.round(5).tolist(),
    }
    if element.get("polygonCorners"):
        corners = np.c_[np.array(element["polygonCorners"], float), np.ones(len(element["polygonCorners"]))]
        out["polygon"] = (transform @ corners.T).T[:, :3].round(4).tolist()
    return out


def focal_35mm(fx: float, width: int, height: int) -> float:
    """The 35mm-equivalent focal length of a camera with focal length `fx` pixels on a `width` x `height` image."""
    return fx * _FULL_FRAME_DIAGONAL_MM / math.hypot(width, height)


def reference(shot: dict, room: dict) -> dict:
    """What the capture measured for one shot, in its camera frame."""
    camera_to_world = _matrix(shot["transform"])
    world_to_camera = np.linalg.inv(camera_to_world)
    width, height = shot["imageSize"]
    k = np.array(shot["intrinsics"], float).reshape(3, 3).T
    floor_y = _matrix(room["floors"][0]["transform"])[1, 3] if room["floors"] else None
    depth_width, depth_height = shot["depthSize"]
    return {
        "shot": shot["name"],
        "label": shot.get("label", ""),
        "role": shot.get("role", "input"),
        "high_resolution": shot.get("highResolution"),
        "depth_gap_s": round(shot["timestamp"] - shot.get("depthTimestamp", shot["timestamp"]), 4),
        "intrinsics": {
            "width": width, "height": height,
            "fx": k[0, 0], "fy": k[1, 1], "cx": k[0, 2], "cy": k[1, 2],
            "fov_x_deg": math.degrees(2 * math.atan(width / (2 * k[0, 0]))),
            "focal_35mm": focal_35mm(k[0, 0], width, height),
        },
        "camera_height_m": None if floor_y is None else camera_to_world[1, 3] - floor_y,
        # World up (ARKit +Y) as seen from the camera; the floor's normal.
        "up": world_to_camera[:3, 1].round(5).tolist(),
        "camera_to_world": camera_to_world.round(6).tolist(),
        "depth": {"file": "reference_depth.npy", "confidence": "reference_conf.npy",
                  "width": depth_width, "height": depth_height,
                  "scale": [depth_width / width, depth_height / height]},
        **{kind: [_in_camera(e, world_to_camera) for e in room.get(kind, [])] for kind in KINDS},
    }


def _room_name(scan: Path) -> str:
    """"Office" from "Scan-Office-1795028251"; else the folder's name."""
    parts = scan.name.split("-")
    return "-".join(parts[1:-1]) if len(parts) > 2 and parts[0] == "Scan" and parts[-1].isdigit() else scan.name


def import_capture(scan: Path, runs: Path = Path("runs"), refs: Path = REFS, seed_offset: int = 0) -> list[Path]:
    """Make a run folder for each input shot in `scan`, holding the photo and run.json, and every shot's reference under `refs`.

    A `seed_offset` other than 0 moves every model's seeds in the runs, for a second draw of the same shots.

    Shots from scans older than the shot roles count as input shots. The scan is checked first (`scancheck`):
    a scan that fails it gets no runs, and nor does an input shot that fails it. A shot whose pose failed gate 0 still
    gets its run (a photo run is given no pose); it is only noted, since its scores against the references are not to be
    trusted.
    """
    from aomma.scancheck import check_scan, describe, usable
    from aomma.scanapi import scan_manifest

    try:
        scan_manifest(scan)
    except ValueError as error:
        raise SystemExit(f"{scan.name}: {error}; no runs made") from None
    whole = Scan(scan)
    shots = [s.info for s in whole.shots]
    room = whole.room
    check = check_scan(whole, refs)
    print("\n".join(describe(check)))
    if not usable(check):
        raise SystemExit(f"{scan.name} failed its check (refs/{scan.name}/scan_check.json): no runs made")
    from aomma.scancheck import pose_failures

    pose_off = pose_failures(scan.name, refs)
    stamp = f"{datetime.now():%Y%m%d-%H%M%S}"
    made = []
    for shot in shots:
        ref = reference(shot, room)
        ref["scan"] = str(scan)
        ref_dir = refs / scan.name / shot["name"]
        ref_dir.mkdir(parents=True, exist_ok=True)
        depth, conf = whole.depth(whole.shot(shot["name"]))
        np.save(ref_dir / "reference_depth.npy", depth)
        np.save(ref_dir / "reference_conf.npy", conf)
        (ref_dir / "reference.json").write_text(json.dumps(ref, indent=2))
        if ref["role"] != "input" or not check["shots"][shot["name"]]["ok"]:
            continue
        if shot["name"] in pose_off:
            print(f"{scan.name} {shot['name']} pose off (gate 0; its scores are not to be trusted): {pose_off[shot['name']]}", flush=True)

        name = "-".join(filter(None, [_room_name(scan), shot["name"], shot.get("label", "").replace(" ", "_")]))
        run_dir = runs / f"{stamp}-{name}"
        run_dir.mkdir(parents=True)
        # A phone photo carries its 35mm-equivalent focal length, a whole number of mm. Stills from
        # before the app wrote it get it added; later ones are copied as they are.
        still = scan / "shots" / f"{shot['name']}.jpg"
        photo = Image.open(still)
        exif = photo.getexif()
        if _FOCAL_LENGTH_35MM in exif.get_ifd(_EXIF_IFD):
            shutil.copyfile(still, run_dir / "photo.jpg")
        else:
            exif.get_ifd(_EXIF_IFD)[_FOCAL_LENGTH_35MM] = round(ref["intrinsics"]["focal_35mm"])
            photo.save(run_dir / "photo.jpg", quality=95, exif=exif)
        note_source(run_dir, run_dir / "photo.jpg", scan=str(scan), shot=shot["name"], condition="photo",
                    **({"seed_offset": seed_offset} if seed_offset else {}))
        made.append(run_dir)
    return made


def _manifest(run_dir: Path) -> dict:
    path = run_dir / "run.json"
    return json.loads(path.read_text()) if path.exists() else {}


def reference_dir(run_dir: Path, refs: Path = REFS) -> Path | None:
    """Where the references for `run_dir`'s shot are: under `refs`, or beside the run for one made before they moved; None if it has none."""
    manifest = _manifest(run_dir)
    if "scan" in manifest and (refs / Path(manifest["scan"]).name / manifest["shot"] / "reference.json").exists():
        return refs / Path(manifest["scan"]).name / manifest["shot"]
    return run_dir if (run_dir / "reference.json").exists() else None


def inputs(run_dir: Path) -> dict:
    """The capture data `run_dir`'s condition lets it use: nothing for a photo-only run.

    `depth`: the shot's LiDAR depth (metres) and ARKit confidence, at the depth map's size.
    `pose`: `camera_to_world` (4x4, ARKit's world, +y up) and `intrinsics` (fx, fy, cx, cy at the photo's width x height).
    `roomplan_room`: `roomplan`, RoomPlan's room (ARKit's world) without the furniture's boxes (`roomplan.given_room`).
    `roomplan_walls`: `roomplan`, RoomPlan's room without any of its boxes (`roomplan.walls_only`).
    `roomplan_objects`: `roomplan_objects`, RoomPlan's furniture boxes (`roomplan.FURNITURE_KINDS`), as its elements.
    """
    manifest = _manifest(run_dir)
    condition = manifest.get("condition", "photo")
    if condition not in CONDITIONS:
        raise SystemExit(f"{run_dir}: unknown condition {condition!r}")
    allowed = CONDITIONS[condition]
    if not allowed:
        return {}
    missing = set(allowed) - {"depth", "pose", "roomplan_room", "roomplan_walls", "roomplan_objects"}
    if missing:
        raise SystemExit(f"{run_dir}: condition {condition!r} needs {', '.join(sorted(missing))}, which is not built yet")
    scan = Scan(Path(manifest["scan"]))
    shot = scan.shot(manifest["shot"])
    found = {}
    if "depth" in allowed:
        found["depth"], found["conf"] = scan.depth(shot)
    if "pose" in allowed:
        found["camera_to_world"] = shot.camera_to_world
        found["intrinsics"] = shot.intrinsics
    if "roomplan_room" in allowed:
        from aomma.stages.roomplan import given_room

        found["roomplan"] = given_room(scan.room)
    if "roomplan_walls" in allowed:
        from aomma.stages.roomplan import walls_only

        found["roomplan"] = walls_only(scan.room)
    if "roomplan_objects" in allowed:
        from aomma.stages.roomplan import FURNITURE_KINDS, category

        found["roomplan_objects"] = [o for o in scan.room.get("objects", []) if category(o) in FURNITURE_KINDS]
    return found


def move_references(run_dirs: list[Path], refs: Path = REFS, apply: bool = False) -> list[str]:
    """Move references kept beside runs (from before `refs`) to `refs/<scan>/<shot>/`, noting the scan and shot in each run.json.

    Copies of one shot's run hold the same references; a copy whose differ from what `refs`
    already has is left as it is and reported. Says what it did, or with `apply` off, would do.
    """
    said = []
    for run_dir in run_dirs:
        if not (run_dir / "reference.json").exists():
            continue
        ref = json.loads((run_dir / "reference.json").read_text())
        target = refs / Path(ref["scan"]).name / ref["shot"]
        if (target / "reference.json").exists():
            same = all((run_dir / f).read_bytes() == (target / f).read_bytes() for f in REFERENCE_FILES)
            if not same:
                said.append(f"{run_dir}: differs from {target}; left as it is")
                continue
            said.append(f"{run_dir}: same as {target}; removed from the run")
        else:
            said.append(f"{run_dir} -> {target}")
        if not apply:
            continue
        target.mkdir(parents=True, exist_ok=True)
        for f in REFERENCE_FILES:
            if not (target / f).exists():
                shutil.copyfile(run_dir / f, target / f)
        manifest = _manifest(run_dir)
        manifest.setdefault("scan", ref["scan"])
        manifest.setdefault("shot", ref["shot"])
        manifest.setdefault("condition", "photo")
        (run_dir / "run.json").write_text(json.dumps(manifest))
        for f in REFERENCE_FILES:
            (run_dir / f).unlink()
    return said
