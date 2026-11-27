import numpy as np
from PIL import Image

from aomma.stages.repair import in_crop, is_light, where_it_can_be


def test_redrawn_object_may_lie_behind_its_occluders():
    shape = (200, 200)
    counter = np.zeros(shape, dtype=bool)
    counter[80:100, 20:140] = True
    mug = np.zeros(shape, dtype=bool)
    mug[60:90, 50:70] = True  # stands on the counter, hiding part of it

    region = where_it_can_be(counter, [mug])

    assert region[85, 60]  # behind the mug
    assert region[65, 60]  # where the mug is
    assert region[102, 30]  # a little past the visible edge
    assert not region[150, 60]  # the cabinets below the counter
    assert not region[30, 60]  # well above anything


def test_in_crop_follows_the_context_crop():
    mask = np.zeros((100, 200), dtype=bool)
    mask[40:60, 50:150] = True
    square = in_crop(mask, mask, 64)
    assert square.shape == (64, 64)
    assert square[32, 32] and not square[5, 32]


def test_light_object_is_told_from_dark():
    area = np.zeros((10, 10), dtype=bool)
    area[2:8, 2:8] = True
    assert is_light(Image.new("RGB", (10, 10), (240, 240, 235)), area)
    assert not is_light(Image.new("RGB", (10, 10), (40, 30, 30)), area)


def test_a_floor_item_the_frame_cuts_is_carried_below_it():
    from PIL import Image

    from aomma.stages.large import cut_below_frame
    from aomma.stages.repair import extend_below

    mask = np.zeros((100, 80), dtype=bool)
    mask[40:100, 20:60] = True  # a table top running off the bottom edge
    box = np.array([[15, 30], [65, 30], [15, 180], [65, 180]], dtype=float)  # its box's foot 80 rows below the frame
    assert cut_below_frame(mask, box) == 80
    assert cut_below_frame(mask, box - [0, 85]) == 0  # its box ends within the frame's reach
    off_edge = mask.copy()
    off_edge[95:] = False
    assert cut_below_frame(off_edge, box) == 0  # its mask stops above the edge: not cut by it

    photo = Image.new("RGB", (160, 200), (200, 180, 150))
    taller, grown, area = extend_below(photo, mask, 80, box)
    assert taller.size == (160, 360) and taller.getpixel((80, 300)) == (0, 0, 0)
    assert grown.shape == (180, 80) and grown[:100].sum() == mask.sum() and not grown[100:].any()
    assert not area[:100].any() and area[150, 40] and not area[150, 5]
