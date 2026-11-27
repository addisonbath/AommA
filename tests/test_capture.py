import json
import re
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from aomma import scancheck
from aomma.capture import Scan, import_capture, inputs, move_references, reference_dir

# The rules themselves, kept before `board_checked` swaps them out.
MARKER_SCALE = scancheck.marker_scale
from aomma.cli import _save_points
from aomma.stages.image import load_working_image

# A camera 1.5 m above the floor at the world origin, turned 90 degrees about up so that its -z (ARKit's view) is world +x.
# Column-major, as ARKit writes it: one row here per column.
TURN = np.array([[0.0, 0, 1, 0], [0, 1, 0, 0], [-1, 0, 0, 0], [0, 1.5, 0, 1]])
WIDTH, HEIGHT, FX = 300, 400, 280.0


@pytest.fixture(autouse=True)
def board_checked(monkeypatch):
    """The made-up scans have no board to find: their LiDAR is taken as checked against it."""
    monkeypatch.setattr(scancheck, "marker_scale", lambda marker, views, measured=None: {"status": "within tolerance", "ok": True})


def element(category: str, transform: np.ndarray, dimensions: list[float], **extra) -> dict:
    return {
        "identifier": category, "category": {category: {}}, "confidence": {"high": {}},
        "parentIdentifier": None, "dimensions": dimensions,
        "transform": transform.T.ravel().tolist(), "polygonCorners": [], **extra,
    }


def scan(tmp_path, roles=(None,)):
    """A scan with one shot per entry of `roles`: "input", "reference", or None for a shot from before the app wrote roles."""
    folder = tmp_path / "Scan-Office-123"
    (folder / "shots").mkdir(parents=True)
    intrinsics = [FX, 0, 0, 0, FX, 0, WIDTH / 2, HEIGHT / 2, 1]
    shots = []
    for i, role in enumerate(roles):
        name = f"{i:03d}"
        Image.new("RGB", (WIDTH, HEIGHT), "orange").save(folder / "shots" / f"{name}.jpg")
        np.full((8, 6), 2.0 + i, "<f4").tofile(folder / "shots" / f"{name}.depth")
        np.full((8, 6), 2, "u1").tofile(folder / "shots" / f"{name}.conf")
        shot = {
            "name": name, "label": "desk corner" if i == 0 else "", "timestamp": 10.0, "depthTimestamp": 10.0, "highResolution": True,
            "transform": TURN.ravel().tolist(), "intrinsics": intrinsics, "imageSize": [WIDTH, HEIGHT], "depthSize": [6, 8],
        }
        if role:
            shot["role"] = role
        shots.append(shot)
    (folder / "shots" / "shots.json").write_text(json.dumps(shots))
    # A wall 2 m in front of the camera, at x = 2 facing -x.
    wall = np.eye(4)
    wall[:3, :3] = [[0, 0, -1], [0, 1, 0], [1, 0, 0]]
    wall[:3, 3] = [2, 1.2, 0]
    floor = np.eye(4)
    floor[:3, :3] = [[1, 0, 0], [0, 0, -1], [0, 1, 0]]
    room = {kind: [] for kind in ("doors", "windows", "openings", "objects")}
    room["walls"] = [element("wall", wall, [4.0, 2.4, 0])]
    room["floors"] = [element("floor", floor, [4.0, 4.0, 0], polygonCorners=[[-2, -2, 0], [2, -2, 0], [2, 2, 0], [-2, 2, 0]])]
    (folder / "rooms.json").write_text(json.dumps([room]))
    return folder


def test_import_seeds_a_run_with_the_photo_and_keeps_the_capture_out_of_it(tmp_path):
    [run_dir] = import_capture(scan(tmp_path), tmp_path / "runs", tmp_path / "refs")
    assert run_dir.name.endswith("-Office-000-desk_corner")
    # `code/` holds the patch of any uncommitted changes the run was started with.
    assert {p.name for p in run_dir.iterdir()} - {"code"} == {"photo.jpg", "run.json"}
    manifest = json.loads((run_dir / "run.json").read_text())
    assert (Path(manifest["scan"]).name, manifest["shot"], manifest["condition"]) == ("Scan-Office-123", "000", "photo")
    assert "commit" in manifest and "diff" in manifest
    refs = reference_dir(run_dir, tmp_path / "refs")
    assert refs == tmp_path / "refs" / "Scan-Office-123" / "000"
    ref = json.loads((refs / "reference.json").read_text())
    assert ref["role"] == "input"
    assert abs(ref["camera_height_m"] - 1.5) < 1e-6
    assert np.allclose(ref["up"], [0, 1, 0])
    # In the camera frame the wall is straight ahead, 2 m away, facing the camera.
    wall = np.array(ref["walls"][0]["transform"])
    assert np.allclose(wall[:3, 3], [0, 1.2 - 1.5, -2])
    assert np.allclose(wall[:3, 2], [0, 0, 1])
    assert np.allclose(np.array(ref["floors"][0]["polygon"])[:, 1], -1.5)
    assert np.load(refs / "reference_depth.npy").shape == (8, 6)


def test_seed_offset_noted_only_when_given(tmp_path):
    [plain] = import_capture(scan(tmp_path), tmp_path / "runs", tmp_path / "refs")
    assert "seed_offset" not in json.loads((plain / "run.json").read_text())
    [moved] = import_capture(scan(tmp_path / "again"), tmp_path / "runs-again", tmp_path / "refs", seed_offset=100)
    assert json.loads((moved / "run.json").read_text())["seed_offset"] == 100


def test_photo_focal_length_gives_the_shots_field_of_view(tmp_path):
    [run_dir] = import_capture(scan(tmp_path), tmp_path / "runs", tmp_path / "refs")
    ref = json.loads((reference_dir(run_dir, tmp_path / "refs") / "reference.json").read_text())
    _, intrinsics = load_working_image(run_dir / "photo.jpg", 1024)
    assert intrinsics.source == "exif"
    # The EXIF focal length is a whole number of mm, so the field of view is near, not exact.
    assert abs(intrinsics.fov_x_deg - ref["intrinsics"]["fov_x_deg"]) < 1.5


def test_a_still_with_its_own_focal_length_is_copied_as_it_is(tmp_path):
    folder = scan(tmp_path)
    still = folder / "shots" / "000.jpg"
    photo = Image.open(still)
    exif = photo.getexif()
    exif.get_ifd(0x8769)[0xA405] = 26
    photo.save(still, exif=exif)
    [run_dir] = import_capture(folder, tmp_path / "runs", tmp_path / "refs")
    assert (run_dir / "photo.jpg").read_bytes() == still.read_bytes()


def test_a_reference_shot_gets_no_run_but_keeps_its_reference(tmp_path):
    runs = import_capture(scan(tmp_path, ("input", "reference")), tmp_path / "runs", tmp_path / "refs")
    assert [json.loads((r / "run.json").read_text())["shot"] for r in runs] == ["000"]
    ref = json.loads((tmp_path / "refs" / "Scan-Office-123" / "001" / "reference.json").read_text())
    assert ref["role"] == "reference"


def test_a_photo_run_is_given_no_capture_data_and_a_depth_run_only_its_shots(tmp_path):
    folder = scan(tmp_path, ("input", "input"))
    run_dir = import_capture(folder, tmp_path / "runs", tmp_path / "refs")[1]
    assert inputs(run_dir) == {}
    manifest = json.loads((run_dir / "run.json").read_text())
    (run_dir / "run.json").write_text(json.dumps(manifest | {"condition": "depth"}))
    given = inputs(run_dir)
    assert set(given) == {"depth", "conf", "camera_to_world", "intrinsics"}
    assert np.all(given["depth"] == 3.0)  # shot 001's, not 000's
    assert np.allclose(given["camera_to_world"][:3, 3], [0, 1.5, 0])


def test_a_depth_run_has_its_points_fitted_to_the_lidar_and_says_so(tmp_path):
    folder = scan(tmp_path, ("input",))
    # A LiDAR map big enough to fit: a flat wall 2.5 m away.
    shots = json.loads((folder / "shots" / "shots.json").read_text())
    shots[0]["depthSize"] = [30, 40]
    (folder / "shots" / "shots.json").write_text(json.dumps(shots))
    np.full((40, 30), 2.5, "<f4").tofile(folder / "shots" / "000.depth")
    np.full((40, 30), 2, "u1").tofile(folder / "shots" / "000.conf")
    [run_dir] = import_capture(folder, tmp_path / "runs", tmp_path / "refs")
    moge = np.zeros((HEIGHT, WIDTH, 3), np.float32)
    moge[..., 2] = -np.linspace(1.5, 2.5, HEIGHT)[:, None]  # MoGe's, too near and tilted

    _save_points(run_dir, moge)  # a photo run: MoGe's as they are
    assert np.array_equal(np.load(run_dir / "points.npy"), moge)
    assert json.loads((run_dir / "provenance.json").read_text())["points.npy"]["made_by"] == "moge"

    manifest = json.loads((run_dir / "run.json").read_text())
    (run_dir / "run.json").write_text(json.dumps(manifest | {"condition": "depth"}))
    _save_points(run_dir, moge)
    assert np.allclose(np.load(run_dir / "points.npy")[..., 2], -2.5, atol=0.01)
    assert np.array_equal(np.load(run_dir / "points_moge.npy"), moge)
    made = json.loads((run_dir / "provenance.json").read_text())
    assert made["points.npy"]["made_by"] == "lidar_aligned(moge)" and made["points_moge.npy"]["made_by"] == "moge"


def test_references_beside_old_runs_move_to_refs_once(tmp_path):
    [run_dir] = import_capture(scan(tmp_path), tmp_path / "runs", tmp_path / "refs")
    refs = tmp_path / "refs" / "Scan-Office-123" / "000"
    # As runs were made before the move: references beside the run, run.json with only the source; and a copy.
    for f in refs.iterdir():
        (run_dir / f.name).write_bytes(f.read_bytes())
    (run_dir / "run.json").write_text(json.dumps({"source": "photo.jpg"}))
    copy = tmp_path / "runs" / "copy"
    copy.mkdir()
    for f in run_dir.iterdir():
        if f.is_file():
            (copy / f.name).write_bytes(f.read_bytes())
    for f in refs.iterdir():
        f.unlink()
    refs.rmdir()
    assert reference_dir(run_dir, tmp_path / "refs") == run_dir

    said = move_references([run_dir, copy], tmp_path / "refs")
    assert len(said) == 2 and (run_dir / "reference.json").exists()  # only said, not done
    move_references([run_dir, copy], tmp_path / "refs", apply=True)
    for r in (run_dir, copy):
        assert not (r / "reference.json").exists()
        assert reference_dir(r, tmp_path / "refs") == refs
        assert json.loads((r / "run.json").read_text())["condition"] == "photo"


def test_no_stage_reads_the_references_or_the_scan():
    """A tripwire, not a lock: stages get capture data only through `capture.inputs`."""
    for path in (Path(__file__).parents[1] / "src" / "aomma" / "stages").glob("*.py"):
        text = path.read_text()
        assert not re.search(r"reference_(depth|conf)|reference\.json|[\"'/](refs|scans)/", text), path
        for imported in re.findall(r"from aomma\.capture import ([\w, ]+)", text):
            assert {name.strip() for name in imported.split(",")} <= {"inputs"}, path


MARKER = {"type": "charuco", "dictionary": "DICT_5X5_100", "squaresX": 5, "squaresY": 7,
          "squareMillimetres": 35.0, "markerMillimetres": 26.0, "measuredSquareMillimetres": 34.8}


def board_shots(*ratios):
    return {f"{i:03d}": {"role": "reference", "board": {"lidar_over_board": r} if r else None} for i, r in enumerate(ratios)}


def test_a_scan_is_used_only_with_its_lidar_checked_against_the_board():
    rules = MARKER_SCALE
    assert not rules(None, {})["ok"]  # a scan from before the protocol
    assert not rules(MARKER, board_shots(None))["ok"]  # the board in no reference shot
    assert rules(MARKER, board_shots(1.004, 0.998))["ok"]
    assert not rules(MARKER, board_shots(1.03, 1.02))["ok"]  # measured, and 2% off
    # Off, but the square typed is the designed one, which may never have been measured: noted, not refused...
    assert rules(MARKER | {"measuredSquareMillimetres": 35.0}, board_shots(1.03))["ok"]
    # ...unless it is known to have been measured.
    assert not rules(MARKER | {"measuredSquareMillimetres": 35.0}, board_shots(1.03), measured=True)["ok"]
    # Keyframes that show the board count too.
    keyframes = {"keyframe 000001": {"role": "keyframe", "board": {"lidar_over_board": 1.002}}}
    assert rules(MARKER, board_shots(1.03) | keyframes | {"k2": {"role": "keyframe", "board": {"lidar_over_board": 1.0}}})["ok"]


def test_a_failed_scan_is_used_only_when_the_user_said_so_and_says_it_failed(tmp_path, monkeypatch):
    folder = scan(tmp_path, ("input",))
    monkeypatch.setattr(scancheck, "marker_scale", lambda marker, views, measured=None: {"status": "off by +1.2%", "ok": False})
    try:
        import_capture(folder, tmp_path / "runs", tmp_path / "refs")
        raise AssertionError("a failed scan made runs")
    except SystemExit:
        pass
    (tmp_path / "refs" / "Scan-Office-123" / "use_anyway.json").write_text(
        json.dumps({"problems": ["marker scale"], "why": "provisional", "decided": "test"}))
    assert len(import_capture(folder, tmp_path / "runs", tmp_path / "refs")) == 1
    check = json.loads((tmp_path / "refs" / "Scan-Office-123" / "scan_check.json").read_text())
    assert not check["ok"] and check["used_anyway"]["why"] == "provisional"


def test_an_input_shot_showing_the_board_gets_no_run(tmp_path, monkeypatch):
    folder = scan(tmp_path, ("input", "input"))
    monkeypatch.setattr(scancheck, "board", lambda marker: object())
    monkeypatch.setattr(scancheck, "find_board", lambda s, view, charuco: {"corners": 20, "distance_m": 1.0, "lidar_over_board": None} if view.name == "001" else None)
    (folder / "capture.json").write_text(json.dumps({"marker": MARKER}))
    runs = import_capture(folder, tmp_path / "runs", tmp_path / "refs")
    assert [json.loads((r / "run.json").read_text())["shot"] for r in runs] == ["000"]
    check = json.loads((tmp_path / "refs" / "Scan-Office-123" / "scan_check.json").read_text())
    assert check["shots"]["001"]["problems"] == ["the marker board is in it, and a run would see it"]


def test_a_scan_projects_its_lidar_into_the_world_and_back(tmp_path):
    whole = Scan(scan(tmp_path, ("input",)))
    shot = whole.shot("000")
    points, pixels = whole.lidar_world(shot)
    # The camera looks along world +x from 1.5 m up; the depth is 2 m.
    assert np.allclose(points[:, 0], 2.0)
    projected, depth = shot.project(points)
    assert np.allclose(depth, 2.0)
    assert np.allclose(whole.depth_at(shot, projected), 2.0)
    assert np.allclose(projected[:, 0], (pixels[:, 1] + 0.5) * WIDTH / 6)


def test_roomplans_furniture_is_given_only_to_depth_objects(tmp_path):
    folder = scan(tmp_path, ("input",))
    rooms = json.loads((folder / "rooms.json").read_text())
    rooms[0]["objects"] = [element(kind, np.eye(4), [1.0, 1.0, 1.0]) for kind in ("bed", "storage")]
    (folder / "rooms.json").write_text(json.dumps(rooms))
    [run_dir] = import_capture(folder, tmp_path / "runs", tmp_path / "refs")
    manifest = json.loads((run_dir / "run.json").read_text())
    (run_dir / "run.json").write_text(json.dumps(manifest | {"condition": "depth+roomplan"}))
    given = inputs(run_dir)
    assert "roomplan_objects" not in given and [next(iter(o["category"])) for o in given["roomplan"]["objects"]] == ["storage"]
    (run_dir / "run.json").write_text(json.dumps(manifest | {"condition": "depth+objects"}))
    given = inputs(run_dir)
    assert [next(iter(o["category"])) for o in given["roomplan_objects"]] == ["bed"]
    assert [next(iter(o["category"])) for o in given["roomplan"]["objects"]] == ["storage"]
    # `depth+walls`: the room's walls and floor, none of its boxes.
    (run_dir / "run.json").write_text(json.dumps(manifest | {"condition": "depth+walls"}))
    given = inputs(run_dir)
    assert "roomplan_objects" not in given and given["roomplan"]["objects"] == []
    assert given["roomplan"]["walls"] == rooms[0]["walls"]
