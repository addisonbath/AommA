import numpy as np

from aomma.stages.inventory import InventoryObject
from aomma.stages.wallpoints import box_mask, scene_points, wall_mask


def _entry(oid, box):
    return InventoryObject(id=oid, bbox_2d=box, description="", size_m=[1, 1, 1], on="floor")


def _room(h=240, w=320):
    """Looking level at a wall 3 m off (camera frame: y up, looking down -z), the floor 1.4 m below."""
    f = 200.0
    v, u = np.mgrid[0:h, 0:w].astype(float)
    ray = np.stack([(u - w / 2) / f, -(v - h / 2) / f, -np.ones((h, w))], axis=-1)
    with np.errstate(divide="ignore"):
        t_floor = np.where(ray[..., 1] < 0, -1.4 / ray[..., 1], np.inf)
    return ray * np.minimum(t_floor, 3.0)[..., None]


def test_a_walls_points_lie_on_its_bare_surface_not_on_what_is_listed_in_front_of_it():
    h, w = 240, 320
    points = _room(h, w)
    wall = _entry("back_wall", [0, 0, 1000, 700])
    cabinets = _entry("wall_cabinets", [0, 0, 600, 400])  # the left part of the wall's box
    found = scene_points([wall], [cabinets], points, np.array([0.0, 1.0, 0.0]), (w, h))
    picked = found["back_wall"]
    assert 1 <= len(picked) <= 3
    listed = box_mask(cabinets.bbox_2d, (w, h))
    floor_rows = points[..., 1] < -1.3  # the floor, which faces up
    for x, y in picked:
        assert not listed[y, x] and not floor_rows[y, x]
    # Spread apart.
    if len(picked) > 1:
        gaps = [np.hypot(a[0] - b[0], a[1] - b[1]) for i, a in enumerate(picked) for b in picked[i + 1:]]
        assert min(gaps) >= 60 / 1024 * w - 1


def test_a_wall_whose_box_holds_only_listed_things_gets_no_points():
    h, w = 240, 320
    found = scene_points([_entry("wall", [100, 100, 400, 400])], [_entry("tall_cabinet", [50, 50, 450, 450])],
                         _room(h, w), np.array([0.0, 1.0, 0.0]), (w, h))
    assert found["wall"] == []


def test_the_wall_is_what_its_points_found_plus_its_box_outside_the_listed_things():
    size = (100, 100)
    from_box = np.zeros((100, 100), bool)
    from_box[:, :] = True  # the box's mask took the cabinets too
    from_points = np.zeros((100, 100), bool)
    from_points[50:60, 0:30] = True  # a backsplash band, under the cabinets' box
    cabinets = _entry("cabinets", [0, 0, 500, 600])
    mask = wall_mask(from_points, from_box, [cabinets], size)
    assert not mask[10, 10]  # in the cabinets' box, and not found by the points
    assert mask[55, 10]  # in the cabinets' box, but found by the points
    assert mask[10, 80]  # outside every listed box, found by the box


def test_a_wall_with_no_points_keeps_its_boxs_mask_whole():
    from_box = np.ones((100, 100), bool)
    assert wall_mask(None, from_box, [_entry("cabinets", [0, 0, 500, 600])], (100, 100)).all()
