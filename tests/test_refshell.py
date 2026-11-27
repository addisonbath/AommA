import numpy as np

from aomma.judgeshell import run_on_grid
from aomma.refshell import APPLIANCE, FIXTURE, deepest_layer, fit_surface, fixtures


def test_the_wall_is_the_deepest_layer_not_the_furniture_in_front_or_a_surface_seen_edge_on():
    rng = np.random.default_rng(0)
    wall = rng.normal(-0.10, 0.008, 1500)  # 10 cm behind RoomPlan's plane, as KitchenScan's wall_9
    cabinets = rng.normal(0.20, 0.008, 6000)  # more points, in front of it
    assert abs(deepest_layer(np.r_[wall, cabinets]) + 0.10) < 0.005
    assert deepest_layer(rng.uniform(-0.25, 0.25, 3000)) is None  # spread evenly: no layer


def test_a_wall_seen_turned_a_little_is_followed_and_a_floor_stays_level():
    rng = np.random.default_rng(1)
    x = rng.uniform(-2, 2, 4000)
    d = 0.02 + 0.03 * x + rng.normal(0, 0.005, len(x))  # 1.7 degrees off RoomPlan's
    xy = np.c_[x, rng.uniform(0, 2.4, len(x))]
    c, a = fit_surface(d, xy, deepest_layer(d) or 0.0, turns=True)
    assert abs(a - 0.03) < 0.003 and abs(c - 0.02) < 0.005
    _, level = fit_surface(d, xy, 0.02, turns=False)
    assert level == 0.0


def _box(category, x, z, w=0.6, d=0.6, h=0.9):
    transform = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, x, h / 2, z, 1]
    return {"category": {category: {}}, "dimensions": [w, h, d], "transform": transform}


def test_storage_counts_as_a_fixture_only_in_a_run_and_appliances_get_their_own_code():
    room = {"objects": [
        _box("storage", 0.0, 0.0), _box("storage", 0.62, 0.0),  # two cabinets side by side: a run
        _box("storage", 4.0, 3.0),  # a dresser on its own: furniture
        _box("oven", 1.24, 0.0),
    ]}
    found = [(round(e["transform"][12], 2), code) for e, code in fixtures(room)]
    assert found == [(0.0, FIXTURE), (0.62, FIXTURE), (1.24, APPLIANCE)]


def test_the_run_is_read_at_the_lidar_size_by_the_label_most_common_in_each_block():
    labels = np.zeros((8, 8), np.uint8)
    labels[:4, :4] = 3
    labels[0, 0] = 5  # one stray pixel in the block
    labels[4:, 4:] = 7
    grid = run_on_grid(labels, (2, 2))
    assert grid.tolist() == [[3, 0], [0, 7]]


def test_gate_one_passes_a_run_on_every_bar_it_can_judge_and_names_what_failed():
    from aomma.judgeshell import verdict

    def run(wall, ceiling_false, things_of, floor_false=0.0):
        return {"iou": {"floor": {"iou": 0.97, "reference": 5000, "false": floor_false},
                        "wall": {"iou": wall, "reference": 9000}, "ceiling": {"false": ceiling_false}},
                "shell_on_things": {"share": 0.5, "of": things_of}, "fixtures": {"builtin": {"share": None, "of": 0}}}

    assert verdict(run(0.90, 0.0, 137))["ok"]  # 71% of 137 pixels in front: too few to judge
    bad = verdict(run(0.70, 0.113, 3000, floor_false=0.006))
    assert not bad["ok"]
    assert [k for k, v in bad["checks"].items() if v["ok"] is False] == [
        "wall_iou", "false_ceiling", "false_floor", "shell_on_things"]


def test_gate_one_fails_floor_where_no_floor_is_in_view():
    from aomma.judgeshell import verdict

    # KitchenScan 011: no floor in the reference, 0.59% of the scored pixels labelled floor (on the backsplash).
    r = {"iou": {"floor": {"iou": 0.0, "reference": 0, "run": 210, "false": 0.0059}, "wall": {"iou": 0.94, "reference": 6937}},
         "shell_on_things": {"share": 0.0, "of": 600}, "fixtures": {"builtin": {"share": 0.97, "of": 9000}}}
    checks = verdict(r)["checks"]
    assert checks["floor_iou"]["ok"] is None  # no floor to judge its IoU
    assert checks["false_floor"]["ok"] is False


def test_gate_two_judges_furniture_lone_storage_and_appliances_but_not_cabinet_runs():
    from aomma.refthings import things

    room = {"objects": [
        _box("storage", 0.0, 0.0), _box("storage", 0.62, 0.0),  # a run of cabinets: the shell's
        _box("storage", 4.0, 3.0),  # a dresser on its own
        _box("refrigerator", 1.5, 2.0),
        _box("bed", 3.0, 0.0),
        _box("sink", 0.31, 0.0),
    ]}
    found = [(next(iter(e["category"])), e["transform"][12]) for e in things(room)]
    assert found == [("storage", 4.0), ("refrigerator", 1.5), ("bed", 3.0)]


def test_small_reference_box_holds_the_points_it_was_made_from():
    import numpy as np

    from aomma.judgesmall import _inside
    from aomma.refitems import box

    rng = np.random.default_rng(0)
    turn = np.radians(30)
    local = rng.uniform([-0.2, 0.0, -0.05], [0.2, 0.3, 0.05], (2000, 3))
    points = np.c_[local[:, 0] * np.cos(turn) + local[:, 2] * np.sin(turn) + 1.0, local[:, 1] + 0.9,
                   -local[:, 0] * np.sin(turn) + local[:, 2] * np.cos(turn) - 2.0]
    item = box(points)
    assert _inside(points, item, 0.04).mean() > 0.97
    assert not _inside(points + [0.0, 0.0, 0.5], item, 0.04).any()


def test_gate3_verdict_counts_a_large_holder_as_listed():
    from aomma.judgesmall import verdict

    ref = lambda item, share, large, pixels=50: {"item": item, "share": share, "large_item": "bed" if large else None,
                                                 "large_share": large, "pixels": pixels}
    ok = {"unknown": 0.02, "reference_items": {"quilt": ref(None, 0, 0.99), "tiny": ref(None, 0, 0, 12), "cup": ref("cup", 0.9, 0)}}
    assert verdict(ok)["ok"]
    bad = {"unknown": 0.22, "reference_items": {"pan": ref(None, 0, 0.1)}}
    assert verdict(bad)["failed"] == {"unknown": 0.22, "missed": ["pan"]}


def test_world_box_turns_a_run_box_into_the_scan_world():
    import numpy as np

    from aomma.judgeboxes import world_box

    # Run world Z up -> scan world +y up: (x, y, z) -> (x, z, -y), then moved by (1, 0, 2).
    to_world = np.eye(4)
    to_world[:3, :3] = [[1, 0, 0], [0, 0, 1], [0, -1, 0]]
    to_world[:3, 3] = [1.0, 0.0, 2.0]
    proxy = {"size": [0.4, 1.0, 0.8], "translate": [0.5, 0.0, 0.0], "yaw_deg": 0.0}
    found = world_box(proxy, to_world)
    assert found["centre"] == [1.5, 0.4, 2.0]
    # Its longer side is its own y, which lies along the scan's -z: a heading of 90 degrees.
    assert found["size"] == [1.0, 0.8, 0.4]
    assert found["heading_deg"] == 90.0


def test_gate4_verdict_judges_large_items_and_heading_only_where_oblong():
    from aomma.judgeboxes import verdict

    row = lambda **k: {"class": "large", "centre_m": 0.1, "plan_m": 0.1, "height_m": 0.0, "size_ratio": [1.0, 1.0, 1.0],
                       "heading_deg": 5.0, "oblong": True} | k
    r = {"items": {"chair": row(heading_deg=39.0, oblong=False), "desk": row(), "pillow": row(**{"class": "small", "plan_m": 0.9}),
                   "bed": row(plan_m=0.48, size_ratio=[1.05, 1.51, 1.3])}}
    assert verdict(r)["failed"] == {"bed": ["plan 0.48 m", "size x1.05/1.51/1.3"]}


def test_gate5_model_fails_by_gate4_bars():
    from aomma.judgemodels import _fails

    item = {"size": [1.0, 0.8, 0.5]}
    good = {"plan_m": 0.1, "height_m": 0.05, "size_ratio": [1.0, 1.1, 0.9], "heading_deg": 5.0}
    assert _fails(good, item) == []
    assert _fails(good | {"plan_m": 0.4, "heading_deg": 30.0}, item) == ["plan 0.40 m", "heading 30 deg"]
