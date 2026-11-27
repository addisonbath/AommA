import numpy as np

from aomma.stages.proxy import Proxy
from aomma.stages.room import wall_face
from aomma.stages.square import Footprint, box_error_m, chamfered_corners, line_up, room_grid_deg, square_built_in, squareness

EYE = np.array([0.0, 0.0, 1.5])
# A wall along x at y = 3, facing the room (towards -y).
BACK_WALL = (np.array([-3.0, 3.0, 0.0]), np.array([6.0, 0.0, 0.0]), np.array([0.0, -1.0, 0.0]), 2.4)


def box(id="cabinet", size=(1.0, 0.6, 0.9), translate=(0.0, 2.7, 0.0), yaw=0.0) -> Proxy:
    return Proxy(id=id, size=size, translate=translate, yaw_deg=yaw, color=(1.0, 1.0, 1.0))


def front_points(x=(-0.5, 0.5), y=2.4, count=400) -> np.ndarray:
    """Depth points over a cabinet front along x at `y`: all the camera sees of it."""
    rng = np.random.default_rng(0)
    return np.column_stack([rng.uniform(*x, count), np.full(count, y), rng.uniform(0.0, 0.9, count)])


def test_squareness_tells_a_box_from_a_cylinder_and_finds_its_heading():
    turn = np.radians(20)
    sides = np.array([[np.cos(turn + k * np.pi / 2), np.sin(turn + k * np.pi / 2), 0.0] for k in range(4)])
    heading, share = squareness(sides, np.ones(4))
    assert abs(heading - 20) < 1e-6 and share > 0.99
    around = np.linspace(0, 2 * np.pi, 36, endpoint=False)
    _, share = squareness(np.column_stack([np.cos(around), np.sin(around), np.zeros(36)]), np.ones(36))
    assert share < 0.05


def test_the_grid_is_the_longest_walls_direction_on_a_quarter_turn():
    walls = [box("wall_1", (4.0, 0.05, 2.4), yaw=-15.9), box("wall_2", (2.0, 0.05, 2.4), yaw=74.1)]
    assert abs(room_grid_deg(walls) - -15.9) < 1e-9
    assert room_grid_deg([]) is None


def test_a_built_in_off_the_grid_is_turned_onto_it_refitted_and_backed_to_the_wall():
    crooked = box(yaw=12.0, translate=(0.0, 2.65, 0.0))
    squared = square_built_in(crooked, front_points(), 0.0, [BACK_WALL], 0.6, EYE)
    assert squared is not None
    assert squared.yaw_deg == 0.0
    # Front where its points are, back on the wall, 1 m wide.
    assert np.allclose(squared.size[:2], (0.96, 0.6), atol=0.03)
    assert np.allclose(squared.translate[:2], (0.0, 2.7), atol=0.02)
    assert squared.size[2] == crooked.size[2]


def test_a_built_in_square_already_or_far_off_the_grid_is_left():
    assert square_built_in(box(yaw=0.3), front_points(), 0.0, [BACK_WALL], 0.6, EYE) is None
    assert square_built_in(box(yaw=40.0), front_points(), 0.0, [BACK_WALL], 0.6, EYE) is None


def test_box_error_is_how_far_its_points_lie_from_its_surface():
    points = front_points()
    assert box_error_m(box(), points) < 0.01
    assert abs(box_error_m(box(translate=(0.0, 2.8, 0.0)), points) - 0.1) < 0.01


def item(id, centre, size=(0.7, 0.7), heading=0.0, z=(0.0, 1.0), fixed=False, steadiness=0.8) -> Footprint:
    return Footprint(id=id, heading_deg=heading, centre=np.array(centre), size=np.array(size), z=z, fixed=fixed, steadiness=steadiness)


def test_neighbours_are_turned_to_the_steadier_one_and_lined_up_side_by_side():
    # A washer and a dryer side by side, the dryer a little turned, 5 cm back and 8 cm apart.
    washer = item("washer", (0.0, 2.6), steadiness=0.81)
    dryer = item("dryer", (0.78, 2.65), heading=4.0, steadiness=0.7)
    placed = {f.id: f for f in line_up([washer, dryer], EYE)}
    assert placed["washer"].turn_deg == 0 and not placed["washer"].shift.any()
    assert abs(placed["dryer"].turn_deg + 4.0) < 1e-9
    # Its front brought level with the washer's, its side against the washer's.
    assert np.allclose(placed["dryer"].shift, (-0.08, -0.05), atol=1e-6)


def test_a_built_in_stays_put_and_round_or_far_things_are_not_lined_up():
    cabinets = item("cabinets", (0.0, 2.7), size=(1.0, 0.6), z=(0.0, 0.9), fixed=True)
    fridge = item("fridge", (0.9, 2.62), heading=3.0, z=(0.0, 1.8))
    stool = item("stool", (0.0, 1.0), size=(0.3, 0.3), z=(0.0, 0.3))
    placed = {f.id: f for f in line_up([cabinets, fridge, stool], EYE)}
    assert not placed["cabinets"].shift.any() and placed["cabinets"].turn_deg == 0
    assert abs(placed["fridge"].turn_deg + 3.0) < 1e-9
    assert placed["stool"].turn_deg == 0 and not placed["stool"].shift.any()


def test_neighbours_far_apart_in_depth_keep_their_fronts():
    near = item("near", (0.0, 2.0))
    far = item("far", (0.75, 2.5))
    placed = {f.id: f for f in line_up([near, far], EYE)}
    # Their fronts differ by 0.5 m, more than lining up allows; only the side gap is closed.
    assert abs(placed["far"].shift[1]) < 1e-9


def test_a_built_in_over_an_item_gives_it_its_heading_but_does_not_move_it():
    cabinet = item("cabinet", (0.0, 2.75), size=(0.9, 0.35), heading=2.0, z=(1.87, 2.37), fixed=True)
    fridge = item("fridge", (0.05, 2.6), size=(0.9, 0.7), z=(0.0, 1.8))
    placed = {f.id: f for f in line_up([cabinet, fridge], EYE)}
    assert abs(placed["fridge"].turn_deg - 2.0) < 1e-9
    assert not placed["fridge"].shift.any()


def test_a_corner_cabinets_45_degree_front_becomes_its_outline():
    # A run 1 m wide backed to the wall at y = 3, its straight front at y = 2.4 from x = -0.1; from x = -0.5 to
    # -0.1 its front runs at 45 degrees out to y = 2.0, a corner cabinet filling the inside corner of two runs.
    rng = np.random.default_rng(1)
    xs = rng.uniform(-0.5, 0.5, 3000)
    ys = np.where(xs < -0.1, 2.4 - (-0.1 - xs), 2.4)
    points = np.column_stack([xs, ys, rng.uniform(0.0, 0.9, 3000)])
    crooked = box(yaw=14.0, translate=(0.0, 2.5, 0.0))
    squared = square_built_in(crooked, points, 0.0, [BACK_WALL], 0.6, EYE)
    assert squared is not None and squared.footprint is not None
    assert squared.yaw_deg == 0.0
    ring = np.array(squared.footprint) + np.array(squared.translate[:2])
    # Its back on the wall, its straight front where the points are, and the corner reaching out to y = 2.0.
    assert abs(ring[:, 1].max() - 3.0) < 0.03
    assert abs(ring[:, 1].min() - 2.0) < 0.04
    # One box over it all stands its straight front 0.4 m out from where the drawers are.
    plain = box(size=(1.0, 1.0, 0.9), translate=(0.0, 2.5, 0.0))
    assert box_error_m(squared, points) < 0.02 and box_error_m(plain, points) > 0.1


def test_a_straight_front_gets_no_corner():
    squared = square_built_in(box(yaw=12.0), front_points(), 0.0, [BACK_WALL], 0.6, EYE)
    assert squared.footprint is None


def test_a_corner_cut_at_45_degrees_becomes_a_wall_between_its_neighbours():
    # A back wall along y = 3 and a left wall along x = 0, their corner cut from (0, 2.6) to (0.4, 3).
    eye = np.array([1.5, 0.5, 1.5])
    back = box("wall_1", (3.0, 0.05, 2.4), (1.5, 3.025, 0.0))
    left = box("wall_2", (2.2, 0.05, 2.4), (-0.025, 1.4, 0.0), yaw=90.0)
    rng = np.random.default_rng(2)
    along = rng.uniform(0, 1, 3000)
    points = np.concatenate([
        np.column_stack([0.4 + 2.6 * along[:1000], np.full(1000, 3.0)]),
        np.column_stack([np.zeros(1000), 0.3 + 2.3 * along[1000:2000]]),
        np.column_stack([0.4 * along[2000:], 2.6 + 0.4 * along[2000:]]),
    ])
    points = np.column_stack([points, rng.uniform(0.2, 2.4, 3000)])
    walls, added = chamfered_corners([back, left], points, eye)
    assert len(added) == 1
    corner, across, normal = wall_face(added[0], eye)
    ends = sorted([tuple(np.round(corner[:2], 2)), tuple(np.round((corner + across)[:2], 2))])
    assert np.allclose(ends, [(0.0, 2.6), (0.4, 3.0)], atol=0.03)
    assert normal[:2] @ np.array([1.0, -1.0]) > 0.99 * np.sqrt(2)
    # The back wall now starts at the cut, the left wall ends at it.
    faces = {w.id: wall_face(w, eye) for w in walls}
    assert abs(min(faces["wall_1"][0][0], (faces["wall_1"][0] + faces["wall_1"][1])[0]) - 0.4) < 0.03
    assert abs(max(faces["wall_2"][0][1], (faces["wall_2"][0] + faces["wall_2"][1])[1]) - 2.6) < 0.03


def test_a_square_corner_gets_no_wall_across_it():
    eye = np.array([1.5, 0.5, 1.5])
    back = box("wall_1", (3.0, 0.05, 2.4), (1.5, 3.025, 0.0))
    left = box("wall_2", (3.0, 0.05, 2.4), (-0.025, 1.5, 0.0), yaw=90.0)
    rng = np.random.default_rng(3)
    along = rng.uniform(0, 1, 2000)
    points = np.concatenate([np.column_stack([3 * along[:1000], np.full(1000, 3.0)]), np.column_stack([np.zeros(1000), 3 * along[1000:]])])
    points = np.column_stack([points, rng.uniform(0.2, 2.4, 2000)])
    assert chamfered_corners([back, left], points, eye)[1] == []


def test_a_corner_cabinet_reaches_back_to_the_45_degree_wall_behind_it():
    # The corner cabinet's diagonal front from (-0.5, 2.0) to (-0.1, 2.4); a 45-degree wall 0.8 m behind it, and
    # the back wall at y = 3 behind the straight run.
    rng = np.random.default_rng(1)
    xs = rng.uniform(-0.5, 0.5, 3000)
    points = np.column_stack([xs, np.where(xs < -0.1, 2.4 - (-0.1 - xs), 2.4), rng.uniform(0.0, 0.9, 3000)])
    back_wall = (np.array([-3.0, 3.0, 0.0]), np.array([6.0, 0.0, 0.0]), np.array([0.0, -1.0, 0.0]), 2.4)
    shift = 0.8 / np.sqrt(2)
    start = np.array([-0.5 - shift - 0.5, 2.0 + shift - 0.5, 0.0])
    diagonal = (start, np.array([1.5, 1.5, 0.0]), np.array([1.0, -1.0, 0.0]) / np.sqrt(2), 2.4)
    squared = square_built_in(box(yaw=14.0, translate=(0.0, 2.5, 0.0)), points, 0.0, [back_wall, diagonal], 0.6, EYE)
    ring = np.array(squared.footprint) + np.array(squared.translate[:2])
    assert len(ring) == 6
    # Every back corner stands on one of the two walls.
    on_back = np.abs(ring[:, 1] - 3.0) < 0.02
    on_diagonal = np.abs((ring - start[:2]) @ diagonal[2][:2]) < 0.02
    assert (on_back | on_diagonal).sum() >= 3 and on_diagonal.sum() >= 2
