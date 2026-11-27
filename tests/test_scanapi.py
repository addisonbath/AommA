import json
import os
import zipfile
from pathlib import Path

import numpy as np
import pytest

from aomma import scanapi
from aomma.capture import Y_UP_TO_Z_UP
from aomma.dashboard import save_scan
from aomma.layout import Camera, Placement, SceneLayout
from aomma.usd.assets import write_proxy_box


def _scan(scans, name, manifest=None, created="2026-11-20T01:08:20Z", shot_to_world=np.eye(4)):
    folder = scans / name
    (folder / "shots").mkdir(parents=True)
    (folder / "capture.json").write_text(json.dumps({"created": created}))
    if manifest is not None:
        (folder / "manifest.json").write_text(json.dumps(manifest))
    shot = {"name": "005", "transform": shot_to_world.T.flatten().tolist(), "intrinsics": [1000, 0, 0, 0, 1000, 0, 480, 320, 1],
            "imageSize": [960, 640], "depthSize": []}
    (folder / "shots" / "shots.json").write_text(json.dumps([shot]))
    return folder


def _run(runs, name, scan, layout=None):
    run_dir = runs / name
    run_dir.mkdir(parents=True)
    (run_dir / "run.json").write_text(json.dumps({"scan": f"scans/{scan}", "shot": "005"}))
    if layout is not None:
        (run_dir / "layout.json").write_text(layout.model_dump_json())
        (run_dir / "scene.usda").write_text("#usda 1.0\n")
    return run_dir


def _zip(path, files):
    with zipfile.ZipFile(path, "w") as zipped:
        for name, text in files.items():
            zipped.writestr(name, text)
    return path


def test_a_scan_without_a_manifest_reads_as_0_3_from_ios_and_an_unknown_major_is_refused(tmp_path):
    old = _scan(tmp_path, "Scan-Old-1")
    assert scanapi.scan_manifest(old)["contractVersion"] == "0.3"
    newer = _scan(tmp_path, "Scan-New-2", {"contractVersion": "0.9", "unknownField": 1})
    assert scanapi.scan_manifest(newer)["contractVersion"] == "0.9"
    future = _scan(tmp_path, "Scan-Future-3", {"contractVersion": "1.0"})
    with pytest.raises(ValueError):
        scanapi.scan_manifest(future)

    archive = _zip(tmp_path / "future.zip", {"Scan-Future-4/capture.json": "{}", "Scan-Future-4/manifest.json": '{"contractVersion": "1.0"}'})
    with pytest.raises(ValueError):
        save_scan(archive, tmp_path / "scans")
    assert not [p for p in (tmp_path / "scans").iterdir()]


def test_scans_are_listed_newest_first_with_their_platform_and_newest_run(tmp_path):
    scans, runs = tmp_path / "scans", tmp_path / "runs"
    _scan(scans, "Scan-Office-1", created="2026-11-20T01:00:00Z")
    _scan(scans, "Scan-Room-2", {"contractVersion": "0.4", "createdAt": "2026-11-25T09:00:00Z", "producer": {"platform": "visionOS"}})
    _scan(scans, "Scan-Future-3", {"contractVersion": "1.0"})
    _run(runs, "20261121-100000-Office-005-photo", "Scan-Office-1")
    _run(runs, "20261122-100000-Office-005-photo", "Scan-Office-1")
    _run(runs, "20261123-100000-Other-005-photo", "Scan-Other-9")
    assert scanapi.list_scans(scans, runs) == [
        {"scan": "Scan-Room-2", "created": "2026-11-25T09:00:00Z", "platform": "visionOS", "latestRun": None},
        {"scan": "Scan-Office-1", "created": "2026-11-20T01:00:00Z", "platform": "iOS", "latestRun": "20261122-100000-Office-005-photo"},
    ]


def test_a_result_puts_the_runs_objects_in_the_scans_world_with_provenance(tmp_path):
    from pxr import Usd, UsdGeom

    scans, runs = tmp_path / "scans", tmp_path / "runs"
    # The shot stands 2 m along the scan's +x, 1.5 m up, looking down the scan's -z.
    shot_to_world = np.eye(4)
    shot_to_world[:3, 3] = (2.0, 1.5, 0.0)
    _scan(scans, "Scan-Office-1", shot_to_world=shot_to_world)
    # The run's camera stands 1.5 m over its origin, level, looking down the run's +y: the shot's pose in its Z-up world.
    camera = Y_UP_TO_Z_UP.copy()
    camera[:3, 3] = (0.0, 0.0, 1.5)
    layout = SceneLayout(camera=Camera(camera_to_world=tuple(camera.flatten())), objects=[
        Placement(name="chair_1", asset="./assets/chair_1.usda", translate=(0.0, 3.0, 0.0), rotate_xyz_deg=(0, 0, 90)),
        Placement(name="lamp_1", asset="./assets/lamp_1.usda", left_out=True),
    ])
    run_dir = _run(runs, "20261122-100000-Office-005-photo", "Scan-Office-1", layout)
    write_proxy_box(run_dir / "assets" / "chair_1.usda", (0.5, 0.5, 1.0))
    _run(runs, "20261123-100000-Office-005-photo", "Scan-Office-1")  # newer, but no scene yet

    folder = scanapi.result_for("Scan-Office-1", runs, tmp_path)
    assert folder == run_dir / "result"
    result = json.loads((folder / "result.json").read_text())
    assert result == {
        "scan": "Scan-Office-1", "runId": run_dir.name, "contractVersion": scanapi.CONTRACT_VERSION, "scene": "scene.usdz",
        "objects": [{"objectId": "chair_1", "primPath": "/AommA/Objects/chair_1", "label": "chair"}],
    }

    layer = Usd.Stage.Open(str(run_dir / "aomma.usda"))
    assert UsdGeom.GetStageUpAxis(layer) == UsdGeom.Tokens.y
    assert layer.GetRootLayer().customLayerData["aomma:runId"] == run_dir.name
    chair = layer.GetPrimAtPath("/AommA/Objects/chair_1")
    assert chair.GetCustomData()["aomma:objectId"] == "chair_1"
    assert chair.GetCustomData()["aomma:source"] == "Scan-Office-1"
    assert UsdGeom.Xformable(chair).GetXformOpOrderAttr().Get() == ["xformOp:transform"]
    # 3 m ahead of the camera on the floor: 3 m down the scan's -z from the shot, at the floor 1.5 m below it.
    box = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default"]).ComputeWorldBound(chair).ComputeAlignedRange()
    centre = (np.array(box.GetMin()) + np.array(box.GetMax())) / 2
    assert np.allclose(centre, (2.0, 0.5, -3.0), atol=1e-6)
    # The box stands 1 m tall along the scan's +y.
    assert np.isclose(box.GetMax()[1] - box.GetMin()[1], 1.0)

    packed = Usd.Stage.Open(str(folder / "scene.usdz"))
    assert packed.GetPrimAtPath("/AommA/Objects/chair_1").IsValid()
    assert not packed.GetPrimAtPath("/AommA/Objects/lamp_1").IsValid()

    # Asked again with nothing changed, the result is not written again.
    written = (folder / "result.json").stat().st_mtime_ns
    scanapi.result_for("Scan-Office-1", runs, tmp_path)
    assert (folder / "result.json").stat().st_mtime_ns == written

    assert scanapi.result_for("Scan-Room-2", runs, tmp_path) is None


def test_a_scans_runs_with_a_scene_are_listed_and_any_one_can_be_chosen(tmp_path):
    scans, runs = tmp_path / "scans", tmp_path / "runs"
    _scan(scans, "Scan-Office-1")
    _scan(scans, "Scan-Room-2")
    camera = Y_UP_TO_Z_UP.copy()
    camera[:3, 3] = (0.0, 0.0, 1.5)

    def layout(*names):
        return SceneLayout(camera=Camera(camera_to_world=tuple(camera.flatten())), objects=[
            Placement(name=name, asset=f"./assets/{name}.usda", left_out=name.startswith("gone")) for name in names])

    older = _run(runs, "20261122-100000-Office-005-photo", "Scan-Office-1", layout("chair_1", "desk_1", "gone_1"))
    newer = _run(runs, "20261123-100000-Office-006-photo", "Scan-Office-1", layout("wall_1"))
    _run(runs, "20261124-100000-Office-005-photo", "Scan-Office-1")  # newest, but no scene
    other = _run(runs, "20261123-110000-Room-005-photo", "Scan-Room-2", layout("bed_1"))
    for run_dir, names in ((older, ("chair_1", "desk_1")), (newer, ("wall_1",)), (other, ("bed_1",))):
        for name in names:
            write_proxy_box(run_dir / "assets" / f"{name}.usda", (0.5, 0.5, 1.0))

    listed = scanapi.list_runs("Scan-Office-1", runs)
    assert [(r["run"], r["shot"], r["objects"]) for r in listed] == [(newer.name, "005", 1), (older.name, "005", 2)]
    assert listed[1]["created"].startswith("2026-11-22T10:00:00")

    # The newest is the default; the older one can be chosen.
    assert scanapi.result_for("Scan-Office-1", runs, tmp_path) == newer / "result"
    chosen = scanapi.run_result("Scan-Office-1", older.name, runs, tmp_path)
    assert chosen == older / "result"
    result = json.loads((chosen / "result.json").read_text())
    assert result["runId"] == older.name and [o["objectId"] for o in result["objects"]] == ["chair_1", "desk_1"]
    # Not a run of this scan, no scene, or outside runs/: none.
    assert scanapi.run_result("Scan-Office-1", other.name, runs, tmp_path) is None
    assert scanapi.run_result("Scan-Office-1", "20261124-100000-Office-005-photo", runs, tmp_path) is None
    assert scanapi.run_result("Scan-Office-1", "..", runs, tmp_path) is None

    # A result written under an older contract is written again.
    stale = chosen / "result.json"
    stale.write_text(json.dumps({**result, "contractVersion": "0.4"}))
    scanapi.run_result("Scan-Office-1", older.name, runs, tmp_path)
    assert json.loads(stale.read_text())["contractVersion"] == scanapi.CONTRACT_VERSION


def test_a_review_is_kept_under_its_scan_and_never_replaces_one(tmp_path):
    reviews = tmp_path / "reviews"
    review = {"contractVersion": "0.4", "scan": "Scan-Office-1", "runId": "r", "objects": [], "note": ""}
    archive = _zip(tmp_path / "review.zip", {
        "Scan-Office-1-1795000000/review.json": json.dumps(review),
        "Scan-Office-1-1795000000/edits.usda": "#usda 1.0\n",
        "__MACOSX/Scan-Office-1-1795000000/._review.json": "",
    })
    assert scanapi.save_review(archive, "Scan-Office-1", reviews) == "Scan-Office-1-1795000000"
    kept = reviews / "Scan-Office-1" / "Scan-Office-1-1795000000"
    assert sorted(p.name for p in kept.iterdir()) == ["edits.usda", "review.json"]
    assert scanapi.save_review(archive, "Scan-Office-1", reviews).startswith("Scan-Office-1-1795000000-")

    with pytest.raises(ValueError, match="sent to"):
        scanapi.save_review(archive, "Scan-Other-2", reviews)
    future = _zip(tmp_path / "future.zip", {"review.json": json.dumps({**review, "contractVersion": "1.0"})})
    with pytest.raises(ValueError, match="contract version"):
        scanapi.save_review(future, "Scan-Office-1", reviews)
    empty = _zip(tmp_path / "empty.zip", {"edits.usda": ""})
    with pytest.raises(ValueError, match="review.json"):
        scanapi.save_review(empty, "Scan-Office-1", reviews)
    # Only the two kept reviews remain: the unpacking folders are gone.
    assert len(list((reviews / "Scan-Office-1").iterdir())) == 2


LIVE_OUT = os.environ.get("AOMMA_LIVE_OUT")


@pytest.mark.skipif(not LIVE_OUT, reason="needs AOMMA_LIVE_OUT from AommACapture's LiveAommATests")
def test_a_review_from_the_headset_client_composes_over_the_runs_layer():
    """The review AommACapture's own client sent (its LiveAommATests, run against this dashboard) lands under
    reviews/<scan>/, and its edits.usda, over the run's aomma.usda, moves, deletes and approves what it says."""
    from pxr import Gf, Usd, UsdGeom

    sent = json.loads((Path(LIVE_OUT) / "sent.json").read_text())
    folder = Path("reviews") / sent["scan"] / sent["review"]
    review = json.loads((folder / "review.json").read_text())
    assert review["scan"] == sent["scan"] and review["runId"] == sent["runId"]
    scanapi.check_version(review["contractVersion"], "the review")

    layer = Path("runs") / sent["runId"] / "aomma.usda"
    stage = Usd.Stage.CreateInMemory()
    stage.GetRootLayer().subLayerPaths = [str(folder / "edits.usda"), str(layer)]  # the edits are stronger
    original = Usd.Stage.Open(str(layer))

    moved = stage.GetPrimAtPath(sent["moved"])
    assert moved.IsValid()
    # The review's matrix is column-major (simd); USD's rows are simd's columns, so it reads row by row as is.
    expected = Gf.Matrix4d(*sent["movedMatrix"])
    got = UsdGeom.Xformable(moved).GetLocalTransformation()
    assert np.allclose(np.array(got), np.array(expected), atol=1e-6)
    assert not np.allclose(np.array(got), np.array(UsdGeom.Xformable(original.GetPrimAtPath(sent["moved"])).GetLocalTransformation()))
    assert not stage.GetPrimAtPath(sent["deleted"]).IsActive()
    assert stage.GetPrimAtPath(sent["approved"]).GetCustomData().get("aomma:approved") is True
    # Untouched objects keep the run's transform and provenance.
    untouched = [p for p in original.GetPrimAtPath(scanapi.OBJECTS).GetChildren()
                 if str(p.GetPath()) not in (sent["moved"], sent["deleted"], sent["approved"])]
    for prim in untouched:
        composed = stage.GetPrimAtPath(prim.GetPath())
        assert composed.IsActive()
        assert composed.GetCustomData().get("aomma:runId") == sent["runId"]
        assert np.allclose(np.array(UsdGeom.Xformable(composed).GetLocalTransformation()),
                           np.array(UsdGeom.Xformable(prim).GetLocalTransformation()))
