"""The shell from RoomPlan (stages/roomplan.py): walls facing the room, openings, the storage rule, snapping to the LiDAR."""

import numpy as np

from aomma.stages import roomplan as rp
from aomma.stages.world import camera_over_floor


def element(centre, along_deg, dims, kind, ident=None, parent=None):
    """A RoomPlan element in ARKit's world (+y up): its x axis turned `along_deg` about +y from world x."""
    a = np.radians(along_deg)
    x = np.array([np.cos(a), 0.0, -np.sin(a)])
    y = np.array([0.0, 1.0, 0.0])
    m = np.eye(4)
    m[:3, 0], m[:3, 1], m[:3, 2], m[:3, 3] = x, y, np.cross(x, y), centre
    found = {"transform": m.T.flatten().tolist(), "dimensions": list(dims), "category": {kind: {}}}
    return found | ({"identifier": ident} if ident else {}) | ({"parentIdentifier": parent} if parent else {})


def square_room(floor=-1.5, height=2.5):
    """A 4 x 4 m room centred on the origin, a door in the wall at z = -2, the floor outline given."""
    mid = floor + height / 2
    walls = [
        element([0, mid, -2], 0, [4, height, 0], "wall", "w_back"),
        element([0, mid, 2], 0, [4, height, 0], "wall", "w_front"),
        element([-2, mid, 0], 90, [4, height, 0], "wall", "w_left"),
        element([2, mid, 0], 90, [4, height, 0], "wall", "w_right"),
    ]
    floor_el = element([0, floor, 0], 0, [4, 4, 0], "floor")
    # The floor's polygon in its own (x, y) plane; its transform turns that plane flat.
    m = np.eye(4)
    m[:3, 0], m[:3, 1], m[:3, 2], m[:3, 3] = [1, 0, 0], [0, 0, -1], [0, 1, 0], [0, floor, 0]
    floor_el["transform"] = m.T.flatten().tolist()
    floor_el["polygonCorners"] = [[-2, -2, 0], [2, -2, 0], [2, 2, 0], [-2, 2, 0]]
    door = element([0.5, floor + 1.0, -2], 0, [0.8, 2.0, 0], "door", parent="w_back")
    return {"walls": walls, "floors": [floor_el], "doors": [door], "windows": [], "openings": [], "objects": []}


def level_camera(height=1.5):
    """An ARKit camera at (0, height - 1.5, 0) looking down -z, level, and the run's camera for it."""
    pose = np.eye(4)
    pose[1, 3] = -1.5 + height
    run = camera_over_floor(np.linalg.inv(pose)[:3, 1], height)
    return pose, run, run @ np.linalg.inv(pose)


def test_the_camera_height_is_over_roomplans_floor():
    pose, _, _ = level_camera(1.4)
    assert abs(rp.camera_height(pose, square_room()) - 1.4) < 1e-9


def test_walls_face_the_room_with_their_face_on_roomplans_plane():
    room = square_room()
    _, run, s2r = level_camera()
    eye = run[:3, 3]
    walls = rp.walls_from_room(room, s2r, eye)
    sides = rp.room_sides(walls, room, s2r, eye)
    assert len(walls) == 4
    for wall in walls:
        normal = sides[wall.id]
        face = np.array(wall.translate[:2]) + normal * wall.size[1] / 2
        # Every face is 2 m from the room's centre (the camera), and its normal points back at it.
        assert abs(np.linalg.norm(face) - 2.0) < 1e-6
        assert normal @ (eye[:2] - face) > 0
        assert wall.translate[2] == 0.0 and abs(wall.size[2] - 2.5) < 1e-9


def test_a_door_is_cut_through_its_parent_wall_from_the_floor():
    _, run, s2r = level_camera()
    walls = rp.walls_from_room(square_room(), s2r, run[:3, 3])
    with_door = [w for w in walls if w.openings]
    assert len(with_door) == 1
    door = with_door[0].openings[0]
    assert door.bottom == 0.0 and abs(door.top - 2.0) < 1e-6 and not door.window
    assert abs(np.linalg.norm(np.subtract(*door.ends)) - 0.8) < 1e-6


def boxes_of(*specs):
    """(category, box) pairs from (category, centre x, centre z, width, depth, height) at floor -1.5, in a run world
    that is ARKit's turned Z up (the level camera's)."""
    _, _, s2r = level_camera()
    room = {"objects": [element([x, -1.5 + h / 2, z], 0, [w, h, d], kind) for kind, x, z, w, d, h in specs]}
    return rp.boxes_from_room(room, s2r)


def test_storage_alone_is_furniture_and_in_a_run_is_built_in():
    boxes = boxes_of(("storage", 0, -1.7, 1.4, 0.5, 0.91), ("storage", 1.5, -1.7, 0.6, 0.6, 0.9), ("storage", 2.1, -1.7, 0.6, 0.6, 0.9))
    assert rp.is_built_in(0, boxes, {}) is None  # a dresser, alone against the wall, counter-high or not
    assert rp.is_built_in(1, boxes, {}) and rp.is_built_in(2, boxes, {})


def test_storage_alone_is_built_in_where_the_survey_lists_a_built_in_in_it():
    boxes = boxes_of(("storage", 0, 0, 2.9, 1.5, 0.91))
    island = boxes[0][1]
    inside = np.array([[island.translate[0], island.translate[1], 0.5]] * 10)
    assert rp.is_built_in(0, boxes, {"kitchen_island": inside}) == "the survey's kitchen_island"


def test_a_seen_appliance_is_left_to_stage_2_and_an_unseen_one_kept():
    boxes = boxes_of(("oven", 0, -1.6, 0.8, 0.7, 0.93))
    assert rp.is_built_in(0, boxes, {}, seen=set()) is not None
    assert rp.is_built_in(0, boxes, {}, seen={boxes[0][1].id}) is None


def test_in_box_takes_height_so_a_hood_is_not_in_the_oven_below_it():
    _, oven = boxes_of(("oven", 0, -1.6, 0.8, 0.7, 0.93))[0]
    assert rp.in_box(oven, (oven.translate[0], oven.translate[1], 0.5))
    assert not rp.in_box(oven, (oven.translate[0], oven.translate[1], 1.8))


def test_a_wall_moves_onto_the_shots_lidar_within_its_limit():
    rng = np.random.default_rng(0)
    along, normal = np.array([1.0, 0.0]), np.array([0.0, 1.0])
    s, z = rng.uniform(0.5, 3.5, 500), rng.uniform(0.5, 2.0, 500)
    points = np.c_[s, np.full(500, 0.03), z]
    shift, count = rp.snap_offset(points, np.zeros(3), along, normal, 4.0, 2.5)
    assert count == 500 and abs(shift - 0.03) < 1e-9
    few = rp.snap_offset(points[:50], np.zeros(3), along, normal, 4.0, 2.5)
    assert few == (0.0, 50)


def test_the_furniture_is_not_given():
    room = {"objects": [element([0, 0, 0], 0, [1, 1, 1], kind) for kind in ("bed", "storage", "sofa", "oven")]}
    assert [rp.category(o) for o in rp.given_room(room)["objects"]] == ["storage", "oven"]


def furniture_boxes(*specs):
    """RoomPlan furniture boxes, as `boxes_of` makes fixtures."""
    _, _, s2r = level_camera()
    room = {"objects": [element([x, -1.5 + h / 2, z], 0, [w, h, d], kind) for kind, x, z, w, d, h in specs]}
    return rp.boxes_from_room(room, s2r, rp.FURNITURE_KINDS)


def points_in(box, count=500, seed=0, grow=0.0):
    """Points filling a box (run world), grown by `grow` on every side in plan."""
    rng = np.random.default_rng(seed)
    yaw = np.radians(box.yaw_deg)
    a = rng.uniform(-box.size[0] / 2 - grow, box.size[0] / 2 + grow, count)
    c = rng.uniform(-box.size[1] / 2 - grow, box.size[1] / 2 + grow, count)
    z = rng.uniform(box.translate[2], box.translate[2] + box.size[2], count)
    return np.c_[box.translate[0] + a * np.cos(yaw) - c * np.sin(yaw), box.translate[1] + a * np.sin(yaw) + c * np.cos(yaw), z]


def test_an_item_takes_the_box_of_its_kind_its_points_lie_in():
    boxes = furniture_boxes(("chair", 0, -1.0, 0.5, 0.5, 0.9), ("table", 0.0, -1.0, 1.2, 0.7, 0.75), ("bed", 1.0, 0.5, 1.6, 2.0, 0.5))
    chair, table, bed = (b for _, b in boxes)
    found = rp.match_objects({"office_chair": ("chair", points_in(chair)), "desk": ("desk", points_in(table, seed=1)),
                              "rug": ("other", points_in(bed, seed=2))}, boxes)
    taken = {f["item"]: f for f in found if f["item"]}
    # The chair stands in the table's box too: kinds keep each item to its own; the rug's kind has no box.
    assert taken["office_chair"]["box"] == chair.id and taken["desk"]["box"] == table.id
    assert "rug" not in taken
    assert [f["box"] for f in found if f["item"] is None] == [bed.id]
    assert all(f["refused"] is None for f in taken.values())


def test_a_box_the_items_own_points_reach_past_is_refused():
    [(_, bed)] = furniture_boxes(("bed", 0, -0.5, 1.4, 2.0, 0.5))
    # The item's points reach 0.3 m past the box all round (RoomPlan's box too small): most still lie in it, grown.
    found = rp.match_objects({"bed": ("bed", points_in(bed, 4000, grow=0.12))}, [("bed", bed)])
    assert found[0]["item"] == "bed" and found[0]["refused"] is None
    found = rp.match_objects({"bed": ("bed", points_in(bed, 4000, grow=0.4))}, [("bed", bed)])
    assert found[0]["refused"] is not None


def test_an_item_straddling_two_boxes_takes_neither_unless_most_lies_in_one():
    boxes = furniture_boxes(("sofa", -0.6, -1.0, 1.0, 0.8, 0.8), ("sofa", 0.6, -1.0, 1.0, 0.8, 0.8))
    left, right = boxes[0][1], boxes[1][1]
    both = np.concatenate([points_in(left, 500), points_in(right, 500, seed=1)])
    found = rp.match_objects({"sofa": ("sofa", both)}, boxes)
    assert all(f["item"] is None for f in found)


def test_furniture_is_given_to_depth_objects_only():
    from aomma.capture import CONDITIONS

    assert "roomplan_objects" in CONDITIONS["depth+objects"]
    assert "roomplan_objects" not in CONDITIONS["depth+roomplan"]


def test_only_boxes_the_shot_sees_count_as_unlisted():
    boxes = furniture_boxes(("chair", 0, -1.0, 0.5, 0.5, 0.9), ("sofa", 2.0, 1.0, 1.8, 0.9, 0.8), ("bed", -2.0, 1.0, 1.6, 2.0, 0.5))
    chair, sofa, bed = (b for _, b in boxes)
    found = rp.match_objects({"chair": ("chair", points_in(chair))}, boxes)
    # The shot's LiDAR holds the chair and the sofa's front; the bed is out of the shot.
    lidar = np.concatenate([points_in(chair, seed=3), points_in(sofa, rp.MIN_SEEN_POINTS, seed=4)])
    assert rp.unlisted_seen(found, boxes, lidar) == ([sofa.id], [bed.id])
