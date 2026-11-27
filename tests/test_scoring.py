import numpy as np
import pytest

from aomma.refitems import box, merge
from aomma.scoring import compare_boxes, depth_stats, heading_error_deg


def slab(centre, length, width, height, heading_deg, count=4000, seed=0):
    """Points filling an upright box in ARKit's world (+y up), its length along `heading_deg` about +y from +x."""
    rng = np.random.default_rng(seed)
    a, c = rng.uniform(-length / 2, length / 2, count), rng.uniform(-width / 2, width / 2, count)
    turn = np.radians(heading_deg)
    return np.c_[centre[0] + a * np.cos(turn) + c * np.sin(turn), rng.uniform(0, height, count),
                 centre[1] - a * np.sin(turn) + c * np.cos(turn)]


def test_a_box_finds_the_centre_heading_and_size_of_its_points():
    for heading in (0, 30, 120, 170):
        found = box(slab((1.5, -2.0), 2.0, 0.6, 0.8, heading))
        assert np.allclose(found["centre"], [1.5, 0.4, -2.0], atol=0.02)
        assert abs(found["heading_deg"] - heading) <= 1.5
        # The 2nd to 98th percentiles: 4% short.
        assert np.allclose(found["size"], [1.92, 0.78, 0.576], atol=0.03)


def test_sightings_of_one_item_merge_and_a_shot_away_from_the_rest_is_left_out():
    bed = [{"shot": s, "id": "bed", "kind": "bed", "description": "", "points": slab((0, 0), 2, 1.5, 0.5, 0, seed=i)}
           for i, s in enumerate(("000", "001", "002"))]
    bed.append({"shot": "003", "id": "bed", "kind": "bed", "description": "", "points": slab((2.0, 0), 2, 1.5, 0.5, 0)})
    chair = {"shot": "000", "id": "chair", "kind": "armchair", "description": "", "points": slab((0, 3), 0.6, 0.6, 0.9, 0)}
    other_chair = {"shot": "001", "id": "seat", "kind": "chair", "description": "", "points": slab((0.1, 3), 0.6, 0.6, 0.9, 0)}
    items = {i["id"]: i for i in merge(bed + [chair, other_chair])}
    # 003's bed only touches the others' at one edge: it shares no surface with them, so it is its own item.
    assert items["bed"]["shots"].keys() == {"000", "001", "002"}
    assert items["chair"]["shots"] == {"000": "chair", "001": "seat"}  # a chair and an armchair, one item


def test_a_large_item_seen_face_by_face_is_one_item():
    # BedroomScan's bed: each shot sees the top and its own end or side, so the shots' centres lie up to 1 m apart (two
    # items and a shot dropped under the old centre rule), but they share the top.
    def seen(top_from, top_to, face):
        top = slab(((top_from + top_to) / 2, 0), top_to - top_from, 1.5, 0.02, 0, count=3000) + [0, 0.5, 0]
        return np.concatenate([top, face])
    views = {
        "000": seen(-1.0, 1.0, slab((1.0, 0), 0.02, 1.5, 0.5, 0, count=1500)),  # the foot end
        "001": seen(-1.0, 0.2, slab((-0.4, 0.75), 1.2, 0.02, 0.5, 0, count=1500)),  # the left side's near half
        "003": seen(-1.0, -0.2, slab((-1.0, 0), 0.02, 1.5, 0.5, 0, count=1500)),  # the head end
        "004": seen(0.0, 1.0, slab((0.5, -0.75), 1.0, 0.02, 0.5, 0, count=1500)),  # the right side's far half
    }
    [bed] = merge([{"shot": s, "id": "bed", "kind": "bed", "description": "", "points": p} for s, p in views.items()])
    assert bed["shots"].keys() == views.keys() and not bed["flagged"]
    assert np.allclose(bed["centre"][::2], [0, 0], atol=0.1)


def test_a_thin_item_seen_from_both_sides_is_joined_through_a_view_of_its_edge():
    # OfficeScan's guitar: its front and back faces lie 10 cm apart and share nothing; a shot from the side sees both.
    front = slab((0, 0.05), 0.4, 0.01, 1.0, 0, count=2000)
    back = slab((0, -0.05), 0.4, 0.01, 1.0, 0, count=2000, seed=1)
    side = np.concatenate([slab((0, 0.05), 0.4, 0.01, 1.0, 0, count=1000, seed=2), slab((0, -0.05), 0.4, 0.01, 1.0, 0, count=1000, seed=3)])
    found = merge([{"shot": s, "id": "guitar", "kind": "other", "description": "", "points": p}
                   for s, p in (("000", front), ("002", back), ("003", side))])
    assert [sorted(i["shots"]) for i in found] == [["000", "002", "003"]] and not found[0]["flagged"]


def test_a_mask_of_something_else_in_one_of_three_shots_is_dropped_and_flagged():
    shots = [{"shot": s, "id": "desk", "kind": "desk", "description": "", "points": slab((0, 0), 1.2, 0.6, 0.75, 0, seed=i)}
             for i, s in enumerate(("000", "001"))]
    # The third mask is mostly the cabinet beside the desk, with a little of the desk: the desk's points lie on it, so
    # it joins, but few of its own lie on the desk's.
    shots.append({"shot": "002", "id": "desk", "kind": "desk", "description": "",
                  "points": np.concatenate([slab((0, 0), 1.2, 0.6, 0.75, 0, count=300, seed=5), slab((1.2, 0), 0.6, 0.6, 1.8, 0, count=5000)])})
    [desk] = merge(shots)
    assert desk["left_out"] == ["002"] and desk["flagged"]
    assert np.allclose(desk["centre"][::2], [0, 0], atol=0.05)


def test_headings_fold_for_a_square_footprint():
    long = {"size": [2.0, 0.5, 0.9], "heading_deg": 10.0, "centre": [0, 0, 0]}
    assert heading_error_deg(long | {"heading_deg": 100.0}, long) == 90.0
    square = {"size": [1.0, 0.5, 0.95], "heading_deg": 10.0, "centre": [0, 0, 0]}
    assert heading_error_deg(square | {"heading_deg": 100.0}, square) == 0.0
    compared = compare_boxes(long | {"centre": [0.3, 0.1, 0.4]}, long)
    assert compared["plan_m"] == 0.5 and compared["size_ratio"] == [1.0, 1.0, 1.0]


def test_depth_stats_count_a_missing_surface_apart_from_a_wrong_one():
    lidar = np.full((4, 4), 2.0)
    scene = np.full((4, 4), 2.05)
    scene[0] = np.inf  # no surface
    scene[1] = 2.5  # too far
    found = depth_stats(scene, lidar, np.ones((4, 4), bool))
    assert (found["agree"], found["too_far"], found["missing"], found["too_near"]) == (0.5, 0.25, 0.25, 0.0)


def test_a_model_covering_two_items_is_matched_to_one():
    from aomma.scoring import match_items

    # OfficeScan 006 after review: the redrawn sofa also covers the stool beside it.
    overlaps = {"green_sofa": {"green_sofa": 1.6, "plant_shelf": 0.1}, "stool": {"green_sofa": 0.3},
                "shelf": {"plant_shelf": 0.9}, "pouf": {"plant_shelf": 0.02}}
    assert match_items(overlaps) == {"green_sofa": "green_sofa", "shelf": "plant_shelf"}


def test_a_generated_model_is_scored_by_its_surface_not_its_proxy(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace

    import aomma.cli
    import aomma.scoring as scoring

    # OfficeScan 005: the chair has a model and a proxy, the built-in shelf only a proxy.
    surface = np.array([[0.0, 0.0, 0.0], [0.5, 0.9, 0.5]])
    monkeypatch.setattr(aomma.cli, "_placed_models", lambda run_dir, scene, samples: {"chair": surface})
    scene = SimpleNamespace(objects=[SimpleNamespace(name=n, left_out=False) for n in ("chair", "shelf")])
    monkeypatch.setattr(scoring.SceneLayout, "model_validate_json", staticmethod(lambda text: scene))
    (tmp_path / "layout.json").write_text("{}")
    proxy = {"size": [1, 1, 1], "translate": [5, 5, 0], "yaw_deg": 0}
    (tmp_path / "proxies.json").write_text(json.dumps([{"id": "chair", **proxy}, {"id": "shelf", **proxy}]))
    found = scoring.model_points(tmp_path, np.eye(4))
    assert len(found["chair"]) == 2 and len(found["shelf"]) == 8


def test_the_box_layout_scores_each_item_by_its_gate_4_box(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace

    import aomma.cli
    import aomma.scoring as scoring

    # A layout-only run: the chair's gate 4 box, not its proxy or any model; the built-in shelf by its proxy; the
    # stool's box is left out of the layout, so it isn't scored.
    monkeypatch.setattr(aomma.cli, "_placed_models", lambda *a: pytest.fail("the box layout has no models"))
    scene = SimpleNamespace(objects=[SimpleNamespace(name=n, left_out=n == "stool") for n in ("chair", "shelf", "stool")])
    read = []
    monkeypatch.setattr(scoring.SceneLayout, "model_validate_json", staticmethod(lambda text: read.append(text) or scene))
    (tmp_path / "layout.json").write_text("shell")
    (tmp_path / "boxes_layout.json").write_text("boxes")
    box = {"size": [2, 2, 1], "translate": [1, 0, 0], "yaw_deg": 90}
    (tmp_path / "boxes.json").write_text(json.dumps({"boxes": [{"proxy": {"id": "chair", **box}}, {"proxy": {"id": "stool", **box}}]}))
    proxy = {"size": [1, 1, 1], "translate": [5, 5, 0], "yaw_deg": 0}
    (tmp_path / "proxies.json").write_text(json.dumps([{"id": "chair", **proxy}, {"id": "shelf", **proxy}]))
    found = scoring.model_points(tmp_path, np.eye(4), scene="boxes")
    assert read == ["boxes"] and set(found) == {"chair", "shelf"}
    assert np.allclose(found["chair"].min(axis=0), [0, -1, 0]) and np.allclose(found["chair"].max(axis=0), [2, 1, 1])
    assert np.allclose(found["shelf"].min(axis=0), [4.5, 4.5, 0])


def test_a_box_of_one_face_is_partial_and_a_whole_one_is_not():
    from aomma.refitems import partial

    # OfficeScan's cabinet: 0.11 m deep where it was listed 0.45, against the wall.
    assert partial([0.74, 0.47, 0.11], [1.0, 0.45, 0.8])
    # Its amplifier, small but whole: 0.43 of its listed depth at the least.
    assert not partial([0.204, 0.199, 0.128], [0.4, 0.3, 0.4])
    # A box's sides are matched by length, not by order: the chair as listed, the other way round.
    assert not partial([0.605, 0.899, 0.553], [0.65, 0.65, 0.9])


def test_roomplan_boxes_pair_with_the_nearest_model_one_to_one():
    from aomma.scoring import match_to_roomplan

    found = [{"category": "chair", "centre": [0.0, 0.0, 0.0]}, {"category": "table", "centre": [0.3, 0.0, 0.0]},
             {"category": "sofa", "centre": [5.0, 0.0, 0.0]}]
    boxes = {"desk": {"centre": [0.35, 0.4, 0.1]}, "office_chair": {"centre": [0.1, 0.5, 0.0]}}
    paired = [(f["category"], name) for f, name in match_to_roomplan(found, boxes)]
    # The table takes the desk (0.11 m), the chair the office chair (0.10 m); the sofa has no model near.
    assert paired == [("chair", "office_chair"), ("table", "desk"), ("sofa", None)]


def test_runs_given_roomplan_are_not_judged_by_it():
    from aomma.scoring import took_roomplan

    assert not took_roomplan("photo") and not took_roomplan("depth") and not took_roomplan("shots")
    assert not took_roomplan("depth+walls")  # given RoomPlan's walls, never its boxes
    assert took_roomplan("depth+roomplan") and took_roomplan("depth+objects") and took_roomplan("shots+roomplan")


def test_the_summary_names_the_model_of_an_item_whose_reference_is_a_face():
    from aomma.scoring import summary

    rows = [
        {"measure": "item_box", "item": "cabinet", "kind": "cabinet", "model": "cabinet", "partial": True, "flagged": False},
        {"measure": "outline", "item": "cabinet", "iou": 0.8},
        {"measure": "item_box", "item": "stool", "kind": "bench", "model": None, "partial": False, "flagged": False},
    ]
    lines = summary(rows)
    assert lines == ["  cabinet (cabinet) as cabinet: the reference is a face of it, not scored by its box, outline 0.80",
                     "  stool (bench): no model"]
