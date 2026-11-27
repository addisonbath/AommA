import json

import numpy as np
from PIL import Image

from aomma.labels import RANK, SETTLED_LAYER, grow, label_run, layer_map, relations, resolve

# Photos the size of a run's (1024 x 768) would be slow; a quarter of that keeps the windows a dozen pixels and more.
H, W = 240, 320


def _points(depth, f=None):
    """A camera-frame point map from each pixel's depth along the view axis, through a pinhole."""
    h, w = depth.shape
    f = f or w
    v, u = np.mgrid[0:h, 0:w].astype(float)
    return np.stack([(u - w / 2) / f * depth, (v - h / 2) / f * depth, depth], axis=-1)


def _box(rows, cols):
    m = np.zeros((H, W), bool)
    m[rows, cols] = True
    return m


def _scene():
    """A wall 3 m away, a nearer panel across the bottom third at 2 m, and a box 1.5 m away in the middle."""
    depth = np.full((H, W), 3.0)
    depth[160:, :] = 2.0
    depth[80:160, 120:200] = 1.5
    return depth


def test_an_overlap_goes_to_the_claimant_whose_surface_its_point_lies_on():
    wall = _box(slice(0, 180), slice(None))  # runs 20 rows onto the panel, and over the whole box
    panel = _box(slice(160, H), slice(None))
    box = _box(slice(80, 160), slice(120, 200))
    labels, how, _ = resolve([wall, panel, box], [RANK["wall"], RANK["floor"], RANK["large"]], _points(_scene()))
    # The rows the wall's mask took stand on the panel's plane: the panel's, though the wall ranks higher.
    assert (labels[160:180, :] == 2).all()
    assert (labels[80:160, 120:200] == 3).all()
    assert (labels[:80, :] == 1).all()
    assert how["by_rank"] == 0 and how["by_depth"] == 20 * W + 80 * 80


def test_rank_decides_only_where_depth_cannot():
    counter = _box(slice(40, 200), slice(None))
    kettle = _box(slice(80, 140), slice(100, 160))  # wholly inside the counter's mask, on the counter's plane
    labels, how, _ = resolve([counter, kettle], [RANK["builtin"], RANK["small"]], _points(np.full((H, W), 2.0)))
    assert (labels[80:140, 100:160] == 2).all()
    assert how["by_rank"] == 60 * 60 and how["by_depth"] == 0


def test_a_small_item_lying_flat_on_a_surface_is_a_layer_and_one_standing_off_it_is_not():
    """A fridge door 2 m away, facing the camera (a surface facing sideways, not up). A paper on it lies on its plane;
    a magnet box stands 1.5 cm off it, under `SAME_DEPTH_M` (so depth alone cannot decide) but well past the door's own
    spread (none, on exact depth). The door keeps the paper's pixels, the paper a layer; the magnet box keeps its own."""
    depth = np.full((H, W), 2.0)
    depth[100:140, 200:240] -= 0.015
    door = _box(slice(20, 220), slice(40, 280))
    paper = _box(slice(60, 120), slice(60, 140))
    magnet = _box(slice(100, 140), slice(200, 240))
    masks, ranks = [door, paper, magnet], [RANK["large"], RANK["small"], RANK["small"]]
    labels, how, settled = resolve(masks, ranks, _points(depth), [False, True, True])
    assert (labels[paper] == 1).all() and (settled[paper] == SETTLED_LAYER).all()
    assert (labels[magnet] == 3).all()
    assert how["layer"] == 60 * 80
    layers = layer_map(masks, ranks, settled)
    assert (layers[paper] == 2).all() and (layers[~paper] == 0).all()


def test_a_layer_on_several_surfaces_goes_to_the_most_specific():
    """OfficeScan 006: a blind flat in its window, in its wall, all on one plane 3 m away. The blind is a layer, and the
    window, not the wall, keeps its pixels."""
    wall = _box(slice(0, 200), slice(None))
    window = _box(slice(40, 160), slice(60, 200))
    blind = _box(slice(40, 120), slice(60, 200))
    masks, ranks = [wall, window, blind], [RANK["wall"], RANK["opening"], RANK["small"]]
    labels, how, settled = resolve(masks, ranks, _points(np.full((H, W), 3.0)), [False, False, True])
    assert (settled[blind] == SETTLED_LAYER).all()
    assert (labels[blind] == 2).all()


def test_only_layer_classes_lie_on_a_surface():
    """A mirror (a large item) flat on a wall stays the mirror's: only the masks marked as layers give way."""
    wall = _box(slice(0, 200), slice(None))
    mirror = _box(slice(40, 120), slice(100, 200))
    labels, how, _ = resolve([wall, mirror], [RANK["wall"], RANK["large"]], _points(np.full((H, W), 3.0)), [False, False])
    assert (labels[mirror] == 2).all() and how["layer"] == 0


def test_relations_tell_an_item_on_its_support_from_one_in_front_of_it():
    """A counter top running away from the camera (depth grows up the photo); a mug standing on it, its foot at the
    counter's depth where they meet; a jar 1 m nearer the camera, in front of the counter, not on it."""
    depth = np.full((H, W), 3.0)
    v = np.arange(H, dtype=float)[:, None]
    depth[:] = 1.0 + (H - v) / H * 2.0
    mug_rows, jar_rows = slice(120, 160), slice(60, 100)
    depth[mug_rows, 40:80] = depth[159, 40]  # an upright face at the depth of its foot
    depth[jar_rows, 200:240] = depth[99, 200] - 1.0
    labels = np.ones((H, W), np.int32)
    labels[mug_rows, 40:80] = 2
    labels[jar_rows, 200:240] = 3
    layers = np.zeros((H, W), np.int32)
    found = relations(labels, layers, ["counter", "mug", "jar"], {"mug": "counter", "jar": "counter", "pen": "desk"},
                      _points(depth))
    assert found["mug"]["relation"] == "touching"
    assert found["jar"]["relation"] == "not touching"
    assert "pen" not in found


def test_pixels_without_depth_fall_to_rank():
    points = _points(_scene())
    points[80:160, 120:200] = np.nan
    wall = _box(slice(0, 160), slice(None))
    box = _box(slice(80, 160), slice(120, 200))
    labels, how, _ = resolve([wall, box], [RANK["wall"], RANK["large"]], points)
    assert (labels[80:160, 120:200] == 2).all()
    assert how["by_rank"] == 80 * 80


def test_a_point_in_front_of_a_wall_seen_edge_on_is_not_the_wall():
    """BedroomScan 006's lampshade: a wall seen nearly edge-on (the left half of the photo, running away from the
    camera) and a round shade standing 9 cm in front of it along each ray, its mask wholly inside the wall's. Square to
    the wall, the shade's edge is 1-3 cm off it, nearer than to the plane through the shade's own curved points, so the
    wall would take it; along the ray it stands 9 cm in front of the wall, which bounds the room: the shade's."""
    f = float(W)
    v, u = np.mgrid[0:H, 0:W].astype(float)
    a = (u - W / 2) / f
    with np.errstate(divide="ignore"):
        wall_depth = np.where(a < 0, -0.5 / np.minimum(a, -1e-9), 10.0)  # the wall at x = -0.5 m
    sees_wall = a < -0.05
    depth = np.where(sees_wall, wall_depth, 10.0)
    shade = _box(slice(60, 180), slice(30, 130))
    # A round shade: 9 cm in front of the wall along the ray at its edges, bulging towards the camera in its middle.
    bulge = 0.4 * (1 - ((u - 80) / 50) ** 2)
    depth = np.where(shade, wall_depth - 0.09 - np.clip(bulge, 0, None), depth)
    wall = sees_wall
    points = _points(depth, f)
    labels, _, _ = resolve([wall, shade], [RANK["wall"], RANK["small"]], points)
    assert (labels[shade] == 2).all()
    assert (labels[wall & ~shade] == 1).all()


def test_a_ceiling_mask_run_down_a_wall_gives_the_wall_back():
    # Looking at a wall 3 m off, the ceiling 1 m above the camera: the top rows see the ceiling, the rest the wall, at
    # about the same distance where they meet.
    f = 160.0
    v, _ = np.mgrid[0:H, 0:W].astype(float)
    dy = (v - H / 2) / f
    with np.errstate(divide="ignore"):
        t_ceiling = np.where(dy < 0, -1.0 / dy, np.inf)
    sees_ceiling = t_ceiling < 3.0
    points = _points(np.minimum(t_ceiling, 3.0), f)
    ceiling = _box(slice(0, 120), slice(None))  # down to the middle of the photo, over the wall's top rows
    wall = ~sees_ceiling
    labels, _, _ = resolve([ceiling, wall], [RANK["ceiling"], RANK["wall"]], points)
    assert (labels[wall & ceiling] == 2).mean() > 0.99
    assert (labels[sees_ceiling] == 1).all()


def test_label_run_counts_unknown_and_names_each_mask(tmp_path):
    run = tmp_path / "run"
    (run / "masks").mkdir(parents=True)
    Image.new("RGB", (W, H), (128, 128, 128)).save(run / "input.jpg")
    np.save(run / "points.npy", _points(_scene()).astype(np.float32))
    (run / "survey.json").write_text(json.dumps({"walls": [{"id": "back_wall"}], "openings": [], "built_ins": []}))
    (run / "large_survey.json").write_text(json.dumps({"objects": [{"id": "dresser"}]}))
    (run / "floors.json").write_text(json.dumps([{"id": "carpet", "kind": "floor"}, {"id": "rug", "kind": "covering"}]))
    for name, mask in {
        "back_wall": _box(slice(0, 160), slice(None)),
        "dresser": _box(slice(80, 160), slice(120, 200)),
        "floor_carpet": _box(slice(160, H), slice(0, 240)),  # the last 80 columns of floor left unmasked
        "lamp": _box(slice(0, 20), slice(0, 20)),  # in no list
    }.items():
        Image.fromarray(mask).save(run / "masks" / f"{name}.png")
    Image.fromarray(_box(slice(160, H), slice(0, 200))).save(run / "floor.png")  # the same floor: no overlap

    report = label_run(run)

    # The 80 unmasked columns of floor lie on the carpet's plane: grown into, none left unknown.
    assert abs(report["unknown_before_growth"] - (80 * 80) / (H * W)) < 1e-3
    assert report["unknown"] == 0 and abs(report["grown"] - (80 * 80) / (H * W)) < 1e-3
    assert report["within_bar"] is True
    assert report["unnamed"] == ["lamp"]
    assert report["classes"]["floor"] == round(80 * W / (H * W), 4)  # its 240 columns and the 80 grown into
    saved = json.loads((run / "labels.json").read_text())
    by_id = {e["id"]: e for e in saved["labels"]}
    assert by_id["dresser"]["class"] == "large" and by_id["floor_carpet"]["class"] == "floor"
    assert by_id["floor_carpet"]["grown_px"] == 80 * 80 and by_id["floor_carpet"]["kept_px"] == 80 * 240
    assert by_id["back_wall"]["lost"] > 0  # the dresser's pixels, and the lamp's (same plane, ranked above)
    labels = np.asarray(Image.open(run / "labels.png"))
    assert labels[120, 160] == by_id["dresser"]["value"]
    assert (run / "debug" / "labels.jpg").exists()


def test_each_wall_pixel_goes_to_the_survey_wall_of_the_fitted_wall_it_lies_on(tmp_path):
    """OfficeScan 005: the survey listed a patch of the left wall as `back_wall`, and `left_wall`'s mask ran past the
    corner onto the right wall. The shell's fitted walls (masks/wall_surface/) give the patch to `left_wall` and the
    run past the corner to `right_wall`; a wall pixel on no fitted wall stays with its mask."""
    run = tmp_path / "run"
    (run / "masks" / "wall_surface").mkdir(parents=True)
    Image.new("RGB", (W, H), (128, 128, 128)).save(run / "input.jpg")
    np.save(run / "points.npy", _points(np.full((H, W), 3.0)).astype(np.float32))
    (run / "survey.json").write_text(json.dumps(
        {"walls": [{"id": "left_wall"}, {"id": "back_wall"}, {"id": "right_wall"}], "openings": [], "built_ins": []}))
    for name, mask in {
        "left_wall": _box(slice(0, 200), slice(0, 200)),  # 40 columns past the corner at 160
        "back_wall": _box(slice(120, 190), slice(60, 120)),  # a patch of the left wall
        "right_wall": _box(slice(0, 200), slice(200, W)),
    }.items():
        Image.fromarray(mask).save(run / "masks" / f"{name}.png")
    Image.fromarray(_box(slice(0, 190), slice(0, 160))).save(run / "masks" / "wall_surface" / "left_wall.png")
    Image.fromarray(_box(slice(0, 190), slice(160, W))).save(run / "masks" / "wall_surface" / "right_wall.png")

    label_run(run)

    by_id = {e["id"]: e["value"] for e in json.loads((run / "labels.json").read_text())["labels"]}
    labels = np.asarray(Image.open(run / "labels.png"))
    assert (labels[:190, :160] == by_id["left_wall"]).all()
    assert (labels[:190, 160:] == by_id["right_wall"]).all()
    assert (labels == by_id["back_wall"]).sum() == 0
    # Rows 190-200 lie on no fitted wall: each mask keeps its own there.
    assert (labels[195, 170:200] == by_id["left_wall"]).all() and (labels[195, 200:] == by_id["right_wall"]).all()


def test_unclaimed_pixels_on_a_neighbours_surface_are_grown_into_and_others_left():
    """A wall 3 m away with an unmasked band along its foot (on the wall) and an unmasked box 1.5 m away standing in
    it (not on the wall): the band goes to the wall, the box stays unknown."""
    depth = np.full((H, W), 3.0)
    depth[150:220, 130:190] = 1.5
    labels = np.zeros((H, W), np.int32)
    labels[:140] = 1  # the wall's mask stops 100 rows short of the frame's foot
    grown, how = grow(labels, [RANK["wall"]], _points(depth))
    assert (grown[140:, :120] == 1).all() and (grown[140:, 200:] == 1).all()
    assert (grown[150:220, 130:190] == 0).all()
    assert how.sum() == (grown == 1).sum() - 140 * W


def test_growth_goes_to_the_claimant_whose_plane_the_pixel_lies_on():
    """A wall at 3 m and a panel at 2 m, each masked short of an unclaimed band between them that lies on the wall:
    the band is the wall's, though it touches both."""
    depth = np.full((H, W), 3.0)
    depth[170:, :] = 2.0
    labels = np.zeros((H, W), np.int32)
    labels[:120] = 1
    labels[170:] = 2
    grown, _ = grow(labels, [RANK["wall"], RANK["large"]], _points(depth))
    assert (grown[120:170] == 1).all()
