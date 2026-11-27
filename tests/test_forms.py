import numpy as np

from aomma.layout import Camera
from aomma.stages.forms import BOX, SHEET, SHEET_LIFT_M, built_on, decide_forms, fit_flat, sheet_pixels
from aomma.stages.proxy import Opening, Proxy

H, W = 120, 160


def _scene():
    """Looking along +y at a wall 3 m off (world y = 3), the camera 1.5 m up: a world point map, and the camera."""
    camera = Camera.looking_at((0.0, 0.0, 1.5), (0.0, 3.0, 1.5), width=W, height=H, focal_length_mm=36.0)
    f = camera.focal_px
    v, u = np.mgrid[0:H, 0:W].astype(float) + 0.5
    x, z = (u - W / 2) / f * 3.0, 1.5 - (v - H / 2) / f * 3.0
    return np.stack([x, np.full((H, W), 3.0), z], axis=-1), camera


def _labels(entries):
    return [{"value": i + 1, "id": oid, "class": kind, "kept_px": kept, "layer_px": layer}
            for i, (oid, kind, kept, layer) in enumerate(entries)]


def test_an_item_mostly_a_layer_on_a_surface_is_a_sheet_and_one_standing_off_it_a_box():
    points, camera = _scene()
    label_map = np.ones((H, W), np.int32)  # the wall
    layer_map = np.zeros((H, W), np.int32)
    layer_map[20:60, 20:60] = 2  # the poster: every pixel a layer on the wall
    label_map[80:110, 100:130] = 3  # a box 1 m nearer
    points[80:110, 100:130, 1] = 2.0
    labels = _labels([("wall", "wall", int((label_map == 1).sum()), 0), ("poster", "small", 0, 1600),
                      ("box", "small", 900, 0)])
    forms = decide_forms(label_map, layer_map, labels, points, camera.matrix[:3, 3])
    assert forms["poster"].form == SHEET and forms["poster"].surface == "wall"
    assert abs(abs(forms["poster"].normal[1]) - 1) < 1e-6
    assert forms["box"].form == BOX and abs(forms["box"].standoff_m - 1.0) < 0.01


def _paper_on(surface_kind, surface_id="door"):
    points, camera = _scene()
    label_map = np.ones((H, W), np.int32)
    layer_map = np.zeros((H, W), np.int32)
    layer_map[20:60, 30:70] = 2
    labels = _labels([(surface_id, surface_kind, int((label_map == 1).sum()), 0), ("paper", "small", 0, 1600)])
    labels[0]["class"] = surface_kind
    return decide_forms(label_map, layer_map, labels, points, camera.matrix[:3, 3])["paper"], layer_map == 2, camera


def test_a_sheet_covers_its_pixels_on_its_measured_plane_where_nothing_is_built():
    form, pixels, camera = _paper_on("large")
    flat = fit_flat(form, pixels, camera)
    corners = np.array(flat.corners)
    assert np.allclose(corners[:, 1], 3.0 - SHEET_LIFT_M, atol=1e-6)  # on the wall, toward the camera
    assert flat.built_on is None and abs(flat.off_measured_m - SHEET_LIFT_M) < 1e-6
    outline = np.array(flat.outline_px)
    assert outline[:, 0].min() <= 30.01 and outline[:, 0].max() >= 69.99
    assert outline[:, 1].min() <= 20.01 and outline[:, 1].max() >= 59.99


def test_a_sheet_lies_on_its_box_face_not_the_measured_plane():
    """The door built as a box turned 5 degrees, its front 4 cm proud of the points: the sheet lies on that front
    (every corner at the same distance off it), not on the plane measured from the points, so the box cannot hide it."""
    form, pixels, camera = _paper_on("large")
    box = Proxy(id="door", size=(1.0, 0.6, 2.5), translate=(0.0, 3.26, 0.0), yaw_deg=5.0, color=(0.5, 0.5, 0.5))
    assert built_on(form, {"door": box}, []) is box
    flat = fit_flat(form, pixels, camera, box)
    yaw = np.radians(5.0)
    front_normal = np.array([np.sin(yaw), -np.cos(yaw), 0.0])  # the box's -y face, toward the camera
    face_point = np.array(box.translate) + front_normal * 0.3
    off = (np.array(flat.corners) - face_point) @ front_normal
    assert np.allclose(off, SHEET_LIFT_M, atol=1e-6)
    assert np.allclose(flat.normal, front_normal, atol=1e-6) and flat.built_on == "door"
    outline = np.array(flat.outline_px)  # it still covers its pixels as the photo shows them
    assert outline[:, 0].min() <= 30.01 and outline[:, 0].max() >= 69.99


def _wall(wid, y, openings=()):
    return Proxy(id=wid, size=(6.0, 0.05, 2.4), translate=(0.0, y, 0.0), yaw_deg=0.0, color=(0.5, 0.5, 0.5),
                 openings=[Opening(id=o, ends=((-0.5, y), (0.5, y)), top=2.0, bottom=1.0, window=True) for o in openings])


def test_a_sheet_on_a_window_lies_on_the_room_face_of_the_wall_holding_it():
    form, pixels, camera = _paper_on("opening", "window")
    walls = [_wall("wall_1", 5.0), _wall("wall_2", 3.02, ["window"])]
    wall = built_on(form, {}, walls)
    assert wall.id == "wall_2"
    flat = fit_flat(form, pixels, camera, wall)
    assert np.allclose(np.array(flat.corners)[:, 1], 3.02 - 0.025 - SHEET_LIFT_M, atol=1e-6)


def test_a_sheet_on_a_wall_label_lies_on_the_nearest_built_wall():
    form, _, _ = _paper_on("wall", "back_wall")
    walls = [_wall("far", 4.0), _wall("near", 3.04)]
    assert built_on(form, {}, walls).id == "near"
    form.surface_class = "floor"
    assert built_on(form, {}, walls) is None


def test_a_sheet_is_cut_out_along_its_mask_over_its_surface_but_not_over_what_stands_in_front():
    """A blind on its window (1): the label map gives it a band, the window the rest of its mask by depth, a lamp (3)
    in front a corner. Its picture is its whole mask but the lamp's corner."""
    label_map = np.ones((H, W), np.int32)
    layer_map = np.zeros((H, W), np.int32)
    mask = np.zeros((H, W), bool)
    mask[20:80, 30:90] = True
    label_map[40:50, 30:90] = 2
    label_map[60:80, 70:90] = 3
    pixels = sheet_pixels(mask, label_map, layer_map, 2, 1)
    expected = mask & (label_map != 3)
    assert (pixels == expected).all()
    assert (sheet_pixels(None, label_map, layer_map, 2, 1) == (label_map == 2)).all()
