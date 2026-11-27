import numpy as np

from aomma.stages.ceiling import build_ceiling, ceiling_height
from aomma.stages.proxy import Proxy

EYE = np.array([0.0, 0.0, 1.5])
BACK = Proxy(id="wall_1", size=(6.0, 0.05, 2.4), translate=(0.0, 3.025, 0.0), yaw_deg=0.0, color=(0.5, 0.5, 0.5))


def ceiling_points(height=2.45, count=2000, seed=0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.column_stack([rng.uniform(-1.5, 1.5, count), rng.uniform(1.0, 2.9, count), height + rng.normal(0, 0.02, count)])


def test_the_ceiling_is_the_level_most_of_its_points_lie_at():
    points = ceiling_points()
    # The mask's edge smears into the wall and the cabinets below.
    smeared = np.column_stack([np.zeros(400), np.full(400, 2.9), np.linspace(1.6, 2.4, 400)])
    height, on = ceiling_height(np.concatenate([points, smeared]))
    assert abs(height - 2.45) < 0.01
    assert on[:2000].mean() > 0.99 and on[2000:].mean() < 0.2


def test_too_little_ceiling_or_one_at_no_believable_height_is_none():
    assert ceiling_height(ceiling_points(count=100)) is None
    assert ceiling_height(ceiling_points(height=1.2)) is None


def test_the_ceiling_stops_at_the_walls_and_takes_the_photos_colour():
    points = ceiling_points()
    ceiling = build_ceiling(points, np.full((len(points), 3), 200.0), [BACK], EYE, points[:, :2], 100_000)
    assert ceiling is not None and ceiling.bounded_by == ["wall_1"]
    assert max(y for _, y in ceiling.polygon) <= 3.05 + 1e-6
    assert abs(ceiling.color[0] - (200 / 255) ** 2.2) < 1e-6


def test_a_ceiling_seen_only_past_the_walls_is_another_rooms():
    # The hallway's ceiling through a doorway in the back wall, 3.1 m up, beyond it.
    rng = np.random.default_rng(4)
    hallway = np.column_stack([rng.uniform(-0.5, 0.5, 2000), rng.uniform(3.2, 4.5, 2000), np.full(2000, 3.1)])
    floor = ceiling_points()[:, :2]
    assert build_ceiling(hallway, np.full((2000, 3), 200.0), [BACK], EYE, floor, 100_000) is None


def test_a_sliver_of_ceiling_is_left_out():
    points = ceiling_points(count=800)
    assert build_ceiling(points, np.full((800, 3), 200.0), [BACK], EYE, points[:, :2], 786_432) is None


def test_only_the_ceiling_masks_pixels_that_face_down_from_above_the_eye_are_kept():
    import numpy as np

    from aomma.stages.ceiling import ceiling_pixels

    # Looking level at a wall 3 m off; the ceiling 1.2 m above the camera, the floor 1.4 m below (camera frame: y up,
    # looking down -z).
    h, w, f = 120, 160, 80.0
    v, u = np.mgrid[0:h, 0:w].astype(float)
    ray = np.stack([(u - w / 2) / f, -(v - h / 2) / f, -np.ones((h, w))], axis=-1)
    with np.errstate(divide="ignore"):
        t_ceiling = np.where(ray[..., 1] > 0, 1.2 / ray[..., 1], np.inf)
        t_floor = np.where(ray[..., 1] < 0, -1.4 / ray[..., 1], np.inf)
    t = np.minimum(np.minimum(t_ceiling, t_floor), 3.0)
    points = ray * t[..., None]
    mask = np.ones((h, w), bool)  # the segmenter's "ceiling" over the whole photo, walls and floor too
    kept = ceiling_pixels(mask, points, np.array([0.0, 1.0, 0.0]))
    ceiling = (t_ceiling < 3.0)[1:-1, 1:-1]
    inner = kept[1:-1, 1:-1]
    assert inner[ceiling].mean() > 0.95  # the ceiling kept, but where it meets the wall
    assert not inner[~ceiling].any()  # nothing of the wall or the floor
