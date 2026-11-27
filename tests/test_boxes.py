import numpy as np

from aomma.stages.boxes import stand_on
from aomma.stages.proxy import Proxy


def _box(id, size, translate):
    return Proxy(id=id, size=size, translate=translate, yaw_deg=0.0, color=(0.5, 0.5, 0.5))


def _surface(height, centre=(0.0, 0.0), half=0.3, n=400):
    xy = np.random.default_rng(0).uniform(-half, half, (n, 2)) + centre
    return np.column_stack([xy, np.full(n, height)])


DESK = _box("desk", (1.2, 0.6, 0.87), (0.0, 0.0, 0.0))  # its top is the hutch's, at 0.87


def test_item_on_a_lower_level_stands_on_it():
    laptop = _box("laptop", (0.3, 0.25, 0.02), (0.0, 0.0, 0.76))
    stood, how, moved = stand_on(laptop, DESK, np.vstack([_surface(0.77), _surface(0.87, (0.5, 0.0), 0.05)]))
    assert how == "level" and abs(stood.translate[2] - 0.77) < 0.005 and moved > 0


def test_item_nearer_the_top_stands_on_the_top():
    clock = _box("clock", (0.1, 0.1, 0.1), (0.0, 0.0, 0.88))
    stood, how, _ = stand_on(clock, DESK, _surface(0.77))
    assert how == "top" and stood.translate[2] == DESK.translate[2] + DESK.size[2]


def test_no_points_stands_on_the_top():
    laptop = _box("laptop", (0.3, 0.25, 0.02), (0.0, 0.0, 0.76))
    assert stand_on(laptop, DESK)[1] == "top"


def test_points_far_from_the_item_are_not_its_level():
    laptop = _box("laptop", (0.3, 0.25, 0.02), (0.0, 0.0, 0.80))
    assert stand_on(laptop, DESK, _surface(0.77, (3.0, 0.0), 0.1))[1] == "top"


def test_item_below_the_middle_is_inside():
    box = _box("box", (0.2, 0.2, 0.1), (0.0, 0.0, 0.2))
    assert stand_on(box, DESK, _surface(0.21))[1:] == ("inside", 0.0)
