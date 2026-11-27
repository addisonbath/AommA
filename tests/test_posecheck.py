import numpy as np
import pytest

from aomma.posecheck import _inside_polygon, robust_shift


def _layer(rng, offset, n, spread=0.01):
    return offset + rng.normal(0, spread, n)


def test_the_shift_lays_the_walls_and_floor_on_their_surfaces_past_the_furniture_in_front():
    rng = np.random.default_rng(0)
    # The shot sits 5 cm off along x: the wall facing +x reads -5 cm, the one facing +z and the floor read 0.
    normals = np.array([[1.0, 0, 0], [0, 0, 1.0], [0, 1.0, 0]])
    wall_x = np.r_[_layer(rng, -0.05, 2000), _layer(rng, 0.23, 2500)]  # a dresser's front 28 cm out of it, more points
    wall_z = _layer(rng, 0.0, 3000)
    floor = _layer(rng, 0.0, 8000)
    shift, weights = robust_shift([wall_x, wall_z, floor], normals)
    assert abs(shift[0] - 0.05) < 0.005 and abs(shift[1]) < 0.005 and abs(shift[2]) < 0.005
    assert weights[0][2000:].max() == 0  # the dresser no longer counts


def test_a_direction_no_surface_faces_is_left_alone():
    rng = np.random.default_rng(1)
    normals = np.array([[0, 1.0, 0]])
    shift, _ = robust_shift([_layer(rng, 0.03, 5000)], normals)
    assert abs(shift[1] + 0.03) < 0.005 and shift[0] == 0 and shift[2] == 0


def test_points_inside_a_floor_polygon():
    square = [[0, 0, 0], [2, 0, 0], [2, 2, 0], [0, 2, 0]]
    xy = np.array([[1, 1], [3, 1], [-0.1, 0.5], [1.9, 1.9]])
    assert _inside_polygon(xy, square).tolist() == [True, False, False, True]


def test_the_bar_fails_a_shot_too_far_from_the_references_or_too_little_seen_by_them():
    from aomma.posecheck import verdict

    def refs(error, views):
        return {"views": [{}] * views, "ratio": 1.03, "error_m": error}

    assert verdict(refs(0.037, 4))["ok"]
    assert not verdict(refs(0.057, 3))["ok"] and "5.7 cm" in verdict(refs(0.057, 3))["why"]
    assert not verdict(refs(0.01, 1))["ok"] and "can't be judged" in verdict(refs(0.01, 1))["why"]


def _kitchen_refs(tmp_path):
    """A scan's references with three shots: 005's pose passed, 007's failed, 009's was never judged."""
    import json

    refs = tmp_path / "refs"
    scan = "Scan-KitchenScan-1795219255"
    (refs / scan).mkdir(parents=True)
    (refs / scan / "scan_check.json").write_text(json.dumps({"shots": {n: {"ok": True} for n in ("005", "007", "009")}}))
    (refs / scan / "pose_check.json").write_text(json.dumps({"shots": {
        "005": {"verdict": {"ok": True, "why": None}},
        "007": {"verdict": {"ok": False, "why": "its LiDAR lies 6.8 cm from the reference shots'"}},
        "009": {"verdict": None},  # measured before the bar: not judged, so not left out
    }}))
    return refs, scan


def test_a_shot_whose_pose_failed_is_left_out_of_the_usable_shots(tmp_path):
    from aomma import scancheck

    refs, scan = _kitchen_refs(tmp_path)
    assert scancheck.usable_shots(scan, refs) == {"005", "009"}
    assert scancheck.usable_shots(scan, refs, poses=False) == {"005", "007", "009"}
    assert scancheck.pose_failures(scan, refs) == {"007": "its LiDAR lies 6.8 cm from the reference shots'"}


@pytest.mark.lab
def test_a_shot_whose_pose_failed_is_left_out_of_the_bank(tmp_path, monkeypatch):
    import json
    import sys

    from aomma import scancheck

    refs, scan = _kitchen_refs(tmp_path)
    sys.path.insert(0, "lab")
    import condition_copies

    monkeypatch.setattr(scancheck, "REFS", refs)
    monkeypatch.setattr(scancheck.pose_failures, "__defaults__", (refs,))
    runs = []
    for shot in ("005", "007"):
        run = tmp_path / f"run-{shot}"
        run.mkdir()
        (run / "run.json").write_text(json.dumps({"scan": f"scans/{scan}", "shot": shot}))
        runs.append(str(run))
    bank = tmp_path / "bank.toml"
    bank.write_text(f'[scans.{scan}.draws]\nA = {json.dumps(runs)}\n')
    assert [p.name for p in condition_copies.from_bank("KitchenScan", bank)] == ["run-005"]
