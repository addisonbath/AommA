import json

import numpy as np

from aomma.stages.large import (
    hold_to_shell, off_square_deg, parse_large, seen_turn_deg, square_to_built_in, square_turn_deg,
)
from aomma.stages.pose import Pose, rotation_z
from aomma.stages.proxy import Proxy


def box_points(size=(1.0, 0.5, 0.8), count=4000) -> np.ndarray:
    """Points over a box's surface, bottom-centre at the origin, as an asset's samples are."""
    rng = np.random.default_rng(0)
    points = rng.uniform(-0.5, 0.5, (count, 3))
    face = rng.integers(0, 3, count)
    points[np.arange(count), face] = np.sign(points[np.arange(count), face]) * 0.5
    return (points + (0.0, 0.0, 0.5)) * size


def box_facets() -> tuple[np.ndarray, ...]:
    normals = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]], dtype=float)
    return normals, np.ones(6), np.zeros((6, 3))


# A wall along x at y = 2, facing the room (towards -y), 4 m long and 2.4 m high.
BACK_WALL = (np.array([-2.0, 2.0, 0.0]), np.array([4.0, 0.0, 0.0]), np.array([0.0, -1.0, 0.0]), 2.4)


def pose(translate, yaw=0.0) -> Pose:
    return Pose(id="sofa", translate=translate, rotate_xyz_deg=(0.0, 0.0, yaw), scale=1.0, iou=0.5, color_error=0.1)


def test_parse_large_keeps_ids_apart_from_the_shell_and_floors_unknown_supports():
    reply = json.dumps({"objects": [
        {"id": "Counter", "kind": "chair", "bbox_2d": [0, 0, 10, 10], "description": "", "size_m": [0.4, 0.4, 0.9], "on": "counter"},
        {"id": "stool", "kind": "chair", "bbox_2d": [0, 0, 10, 10], "description": "", "size_m": [0.4, 0.4, 0.6], "on": "table"},
    ]})
    listed = parse_large(reply, {"counter", "left_wall"})
    assert [o.id for o in listed.objects] == ["counter_2", "stool"]
    assert [o.on for o in listed.objects] == ["counter", "floor"]
    assert listed.kinds == {"counter_2": "chair", "stool": "chair"}



def test_parse_large_leaves_out_an_entry_whose_box_is_a_number_short():
    # As qwen replied for one capture shot: the desk's box lost its bottom, leaving a trailing comma.
    reply = """{"objects": [
        {"id": "chair", "kind": "chair", "bbox_2d": [0, 628, 392, 998], "description": "", "size_m": [0.6, 0.6, 1.1], "on": "floor"},
        {"id": "desk", "kind": "desk", "bbox_2d": [0, 592, 302,], "description": "", "size_m": [1.2, 0.6, 0.75], "on": "floor"},
    ]}"""
    assert [o.id for o in parse_large(reply, set()).objects] == ["chair"]


def test_a_model_past_a_wall_is_brought_out_and_stood_on_the_floor():
    points = box_points()
    # Centred 1.9 m out, 0.25 m deep: its back is 0.15 m past the wall. It floats 0.1 m.
    held, moves = hold_to_shell(pose((0.0, 1.9, 0.1)), points, box_facets(), [BACK_WALL], 0.0)
    world = points + np.array(held.translate)
    assert abs(world[:, 2].min()) < 0.01 and moves.settled_m < 0
    assert world[:, 1].max() <= 2.0 + 0.031
    assert moves.cleared_m > 0.1


def test_a_model_nearly_against_a_wall_is_backed_onto_it_and_squared():
    points = box_points()
    held, moves = hold_to_shell(pose((0.0, 1.65, 0.0), yaw=6.0), points, box_facets(), [BACK_WALL], 0.0)
    assert abs(moves.turned_deg + 6.0) < 0.5
    assert off_square_deg(box_facets(), rotation_z(held.rotate_xyz_deg[2]), [BACK_WALL]) < 0.5
    world = points @ rotation_z(held.rotate_xyz_deg[2]).T + np.array(held.translate)
    assert abs(world[:, 1].max() - 2.0) < 0.01 and moves.backed_m > 0


def test_off_square_is_left_unmeasured_for_a_round_model():
    # Upright faces evenly round a circle agree on no direction.
    turns = np.radians(np.arange(0, 360, 10))
    normals = np.stack([np.cos(turns), np.sin(turns), np.zeros_like(turns)], axis=1)
    round_facets = (normals, np.ones(len(turns)), np.zeros((len(turns), 3)))
    assert off_square_deg(round_facets, rotation_z(0.0), [BACK_WALL]) is None
    assert off_square_deg(box_facets(), rotation_z(30.0), [BACK_WALL]) == 30.0
    assert square_turn_deg(box_facets(), rotation_z(30.0), [BACK_WALL]) == -30.0


def seen_front(yaw: float, size: int = 80) -> tuple[np.ndarray, np.ndarray]:
    """World points per pixel of an upright face turned `yaw` deg from the back wall, and its mask."""
    rows, cols = np.mgrid[0:size, 0:size] * 0.01
    along = np.array([np.cos(np.radians(yaw)), np.sin(np.radians(yaw)), 0.0])
    world = cols[..., None] * along + (2.0 - rows)[..., None] * np.array([0.0, 0.0, 1.0]) + (0.0, 1.0, 0.0)
    return world, np.ones((size, size), dtype=bool)


def test_the_photo_says_whether_an_item_stands_square_to_the_walls():
    assert abs(seen_turn_deg(*seen_front(3.0), [BACK_WALL]) - 3.0) < 0.5
    assert abs(seen_turn_deg(*seen_front(-35.0), [BACK_WALL]) + 35.0) < 0.5
    # Too little of it shows to say.
    world, mask = seen_front(-35.0, size=20)
    assert seen_turn_deg(world, mask, [BACK_WALL]) is None


def test_a_model_is_judged_against_the_heading_the_photo_shows():
    # A model at 44 deg, an item seen at 35: 9 deg off what the photo shows, 44 off the walls.
    assert square_turn_deg(box_facets(), rotation_z(44.0), [BACK_WALL], seen_turn=35.0) == -9.0
    assert square_turn_deg(box_facets(), rotation_z(44.0), [BACK_WALL]) == -44.0


def test_a_model_the_photo_shows_at_an_angle_is_not_squared():
    points = box_points()
    held, moves = hold_to_shell(pose((0.0, 1.0, 0.0), yaw=6.0), points, box_facets(), [BACK_WALL], 0.0, square=False)
    assert moves.turned_deg == 0.0 and abs(held.rotate_xyz_deg[2] - 6.0) < 1e-6


def test_a_model_well_off_square_or_far_from_walls_is_left_so():
    points = box_points()
    held, moves = hold_to_shell(pose((0.0, 0.5, 0.0), yaw=25.0), points, box_facets(), [BACK_WALL], 0.0)
    assert moves.turned_deg == 0.0 and moves.backed_m == 0.0 and moves.cleared_m == 0.0
    assert held.translate == (0.0, 0.5, 0.0)


def test_a_model_far_past_a_wall_whose_item_is_inside_is_marked_to_be_posed_again():
    points = box_points()
    # Centred 2.6 m out, 0.25 m deep: its back is 0.85 m past the wall, as a floor contact behind a hidden bottom
    # leaves it. The depth map puts the item itself 0.5 m in front of the wall.
    seen = np.array([[x, 1.5, z] for x in (-0.4, 0.0, 0.4) for z in (0.2, 0.6)])
    held, moves = hold_to_shell(pose((0.0, 2.6, 0.1)), points, box_facets(), [BACK_WALL], 0.0, seen=seen)
    assert moves.far_past and not moves.beyond_walls and moves.cleared_m == 0.0
    assert held.translate == (0.0, 2.6, 0.0) and held.scale == 1.0


def test_a_model_whose_item_stands_past_the_walls_is_beyond_them():
    points = box_points()
    # Seen through a doorway, 4 m past the wall, and so are its depth points.
    seen = np.array([[x, 6.0, z] for x in (-0.4, 0.0, 0.4) for z in (0.2, 0.6)])
    held, moves = hold_to_shell(pose((0.0, 6.0, 0.0)), points, box_facets(), [BACK_WALL], 0.0, seen=seen)
    assert moves.beyond_walls and not moves.far_past and moves.cleared_m == 0.0
    assert held.translate == (0.0, 6.0, 0.0)


def test_without_the_items_points_a_model_far_past_a_wall_is_pushed_out():
    points = box_points()
    held, moves = hold_to_shell(pose((0.0, 2.6, 0.0)), points, box_facets(), [BACK_WALL], 0.0)
    assert not moves.far_past and moves.cleared_m > 0.8
    assert (points + np.array(held.translate))[:, 1].max() <= 2.0 + 0.031


def test_what_stands_on_any_built_in_is_left_for_the_small_items():
    reply = json.dumps({"objects": [
        {"id": "toaster_oven", "kind": "appliance", "bbox_2d": [0, 0, 10, 10], "description": "", "size_m": [0.5, 0.4, 0.3], "on": "back_counter"},
        {"id": "basket", "kind": "other", "bbox_2d": [0, 0, 10, 10], "description": "", "size_m": [0.5, 0.4, 0.3], "on": "wall_cabinets"},
        {"id": "refrigerator", "kind": "appliance", "bbox_2d": [0, 0, 10, 10], "description": "", "size_m": [0.9, 0.8, 1.8], "on": "floor"},
    ]})
    listed = parse_large(reply, {"back_counter", "wall_cabinets"}, {"back_counter", "wall_cabinets"})
    assert [o.id for o in listed.objects] == ["refrigerator"]
    assert listed.kinds == {"refrigerator": "appliance"}


# The camera at 1.5 m looks along +y at a fridge whose front stands at y = 1.5, 0.9 m wide from x = -0.5, under a
# cabinet 0.35 deep whose back is against a wall at y = 2.2. The photo's frame cuts the fridge at x = 0.3.
CABINET = Proxy(id="cabinet", size=(0.35, 0.9, 0.5), translate=(-0.05, 2.025, 1.85), yaw_deg=90.0, color=(1.0, 1.0, 1.0))
FRIDGE_BOX = Proxy(id="fridge", size=(0.7, 0.8, 1.75), translate=(-0.1, 1.85, 0.0), yaw_deg=80.0, color=(0.5, 0.5, 0.5))
EYE = np.array([0.0, 0.0, 1.5])


def fridge_front(right_x=0.3, columns=40, rows=20) -> tuple[np.ndarray, np.ndarray]:
    """World points per pixel over the fridge's front, and its mask; it runs to the frame's right edge if `right_x` is 0.3."""
    world = np.full((rows, columns, 3), np.nan)
    mask = np.zeros((rows, columns), dtype=bool)
    xs = np.linspace(-0.5, right_x, columns - 5)
    zs = np.linspace(0.0, 1.75, rows)
    world[:, 5:, 0], world[:, 5:, 1], world[:, 5:, 2] = xs[None, :], 1.5, zs[:, None]
    mask[:, 5:] = True
    if right_x != 0.3:
        mask[:, -1] = False
    return world, mask


def test_an_item_under_a_built_in_is_squared_to_it_backed_to_its_back_and_widened_past_the_frame():
    world, mask = fridge_front()
    squared = square_to_built_in(FRIDGE_BOX, world, mask, [CABINET], (0.9, 0.7, 1.8), EYE)
    assert squared is not None
    box, under = squared
    assert under == "cabinet"
    assert box.yaw_deg == 90.0
    # Front where its points are, back where the cabinet's is; the width cut off by the frame is the listed one.
    assert np.allclose(box.size[:2], (0.7, 0.9), atol=0.03)
    assert np.allclose(box.translate, (-0.05, 1.85, 0.0), atol=0.02)
    assert box.size[2] == 1.75


def test_a_side_the_frame_does_not_cut_keeps_its_measured_width():
    world, mask = fridge_front(right_x=0.2)
    box, _ = square_to_built_in(FRIDGE_BOX, world, mask, [CABINET], (0.9, 0.7, 1.8), EYE)
    assert abs(box.size[1] - 0.7) < 0.05


def test_an_item_with_no_built_in_over_it_is_left_as_measured():
    world, mask = fridge_front()
    high = CABINET.model_copy(update={"translate": (-0.05, 2.025, 2.6)})
    beside = CABINET.model_copy(update={"translate": (1.5, 2.025, 1.85)})
    assert square_to_built_in(FRIDGE_BOX, world, mask, [high, beside], (0.9, 0.7, 1.8), EYE) is None


def test_only_a_large_share_of_holes_in_something_that_holds_things_counts_as_its_contents():
    from aomma.stages.large import contents_holes

    shelf = np.zeros((40, 60), dtype=bool)
    shelf[5:35, 5:55] = True
    shelf[10:30, 10:50] = False  # an open front full of books, 0.53 of the outline
    holes = contents_holes(shelf, "shelving")
    assert holes is not None and holes.sum() == 20 * 40 and not (holes & shelf).any()
    # A chair's open back is a real opening, and a few small holes are not contents.
    assert contents_holes(shelf, "chair") is None
    stool = np.zeros((40, 60), dtype=bool)
    stool[5:35, 5:55] = True
    stool[15:18, 20:30] = False  # a handle, 0.02
    assert contents_holes(stool, "shelving") is None
    # Cut by the photo's right edge, the shelf's open front is closed by the edge.
    cut = shelf[:, :50].copy()
    holes = contents_holes(cut, "shelving")
    assert holes is not None and holes[10:30, 10:50].all() and not holes[:5].any() and not holes[35:].any()


def test_an_area_is_framed_as_the_object_crop_frames_the_object():
    from PIL import Image

    from aomma.stages.asset import crop_object, in_object_crop

    mask = np.zeros((100, 200), dtype=bool)
    mask[20:80, 40:160] = True
    hole = np.zeros_like(mask)
    hole[40:60, 80:120] = True
    mask &= ~hole
    outline = mask | hole
    # The photo at twice the working size; the hole painted white in it is where the framed area says.
    photo = Image.fromarray(np.where(np.kron(hole, np.ones((2, 2), dtype=bool))[..., None], 255, 0).astype(np.uint8).repeat(3, axis=2))
    square, _ = crop_object(photo, outline, background="black")
    framed = in_object_crop(hole, outline, photo.size)
    painted = np.asarray(square.convert("L")) > 127
    assert framed.sum() > 0 and (framed & painted).sum() / (framed | painted).sum() > 0.95


def test_a_mask_stopping_well_short_of_its_box_has_its_bottom_hidden():
    from aomma.stages.large import below_mask, bottom_short

    desk = np.zeros((100, 200), dtype=bool)
    desk[20:60, 0:80] = True  # the top and drawers; the box reaches the leg's foot at row 80
    box = [0.0, 200.0, 500.0, 800.0]
    assert abs(bottom_short(desk, box) - 0.33) < 0.01
    below = below_mask(desk, box)
    assert below.sum() == 20 * 100 and below[60:80, 0:100].all()
    sofa = np.zeros((100, 200), dtype=bool)
    sofa[20:79, 0:100] = True
    assert bottom_short(sofa, box) < 0.02


def test_only_an_item_the_side_edge_cuts_well_short_of_its_listed_size_is_cut_by_the_frame():
    from aomma.stages.large import cut_by_frame

    at_edge = np.zeros((100, 200), dtype=bool)
    at_edge[20:80, 0:60] = True
    inside = np.roll(at_edge, 10, axis=1)
    listed = (1.5, 0.7, 0.75)
    assert cut_by_frame(at_edge, (0.7, 0.95, 1.03), listed)
    assert not cut_by_frame(inside, (0.7, 0.95, 1.03), listed)
    # The fridge: on the edge, but its box is 0.80 of its listed size.
    assert not cut_by_frame(at_edge, (0.72, 0.6, 1.8), (0.9, 0.7, 1.8))


def test_gate2_verdict_counts_small_items_toward_completeness():
    from aomma.judgelarge import verdict

    box = lambda item, share, small, pixels=500: {"item": item, "share": share, "pixels": pixels,
                                                  "shares": {"large": share, "builtin": 0, "small": small, "unknown": 0, "shell": 0}}
    item = lambda on_shell, pixels=500: {"pixels": pixels, "boxes": [], "on_shell": on_shell, "in_other_box": 0, "outside_boxes": 0}
    passing = {"boxes": {"bed_1": box("bed", 0.31, 0.69)}, "items": {"bed": item(0.0)}, "missed": []}
    assert verdict(passing)["ok"]
    failing = {"boxes": {"bed_1": box("bed", 0.5, 0.2), "chair_0": box(None, 0, 0.97), "tiny": box(None, 0, 0, 60)},
               "items": {"bed": item(0.0), "stove": item(0.32), "small": item(0.9, 60)}, "missed": ["chair_0", "tiny"]}
    assert verdict(failing)["failed"] == {"missed": ["chair_0"], "complete": {"bed_1": 0.7}, "on_shell": {"stove": 0.32}}


def test_gate2_box_held_by_one_small_item_is_listed():
    from aomma.judgelarge import verdict

    r = {"boxes": {"chair_0": {"item": "wooden_box", "small_item": True, "share": 0.97, "pixels": 222,
                               "shares": {"large": 0, "builtin": 0, "small": 0.97, "unknown": 0, "shell": 0.03}}},
         "items": {}, "missed": []}
    assert verdict(r)["ok"]
