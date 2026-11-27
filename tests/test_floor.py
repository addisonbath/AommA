import json

import numpy as np

from aomma.stages.floor import (
    assign_floor, clean_square, clip_polygon, covering_rectangle, even_light, grain, make_tileable, match_colour, parse_floor_survey,
    room_polygon, take_floor_survey,
)
from aomma.stages.proxy import Proxy


def wall(oid: str, x: float, y: float, yaw: float, length: float = 4.0) -> Proxy:
    return Proxy(id=oid, size=(length, 0.1, 2.4), translate=(x, y, 0.0), yaw_deg=yaw, color=(1, 1, 1))


def test_floor_survey_puts_kinds_and_materials_by_unique_id():
    reply = {"floors": [
        {"id": "Carpet", "kind": "floor", "bbox_2d": [0, 500, 1000, 1000], "material": "beige carpet"},
        {"id": "carpet", "kind": "covering", "bbox_2d": [100, 700, 400, 900], "material": "grey rug"},
    ]}
    survey = parse_floor_survey(f"```json\n{json.dumps(reply)}\n```")
    assert [f.id for f in survey.floors] == ["carpet", "carpet_2"]
    assert survey.kinds == {"carpet": "floor", "carpet_2": "covering"}
    assert survey.floors[1].description == "grey rug"


def test_floor_survey_listed_as_bare_names_is_asked_again(tmp_path):
    from PIL import Image

    from aomma.ledger import Ledger

    good = {"floors": [{"id": "boards", "kind": "floor", "bbox_2d": [0, 600, 1000, 1000], "material": "oak boards"}]}
    replies = [json.dumps({"floors": ["wood floor"]}), json.dumps(good)]
    prompts = []

    class Model:
        def ask(self, image, prompt, schema, label=""):
            prompts.append(prompt)
            from aomma.backends.vlm import Usage
            return replies.pop(0), Usage(model="m", input_tokens=1, output_tokens=1, seconds=0.0, billed=False)

    survey, raw = take_floor_survey(Image.new("RGB", (8, 8)), Model(), Ledger(tmp_path / "usage.jsonl", 1000))
    assert [f.id for f in survey.floors] == ["boards"]
    assert "could not be read" in prompts[1] and "wood floor" in raw


def test_a_covering_takes_its_pixels_from_the_floor_and_the_floor_keeps_the_rest():
    floor_pixels = np.ones((10, 10), dtype=bool)
    floor_pixels[0] = False
    rug = np.zeros((10, 10), dtype=bool)
    rug[5:8, 2:6] = True
    # The floor's own mask misses some of the floor; it keeps every floor pixel no covering takes.
    boards = np.zeros((10, 10), dtype=bool)
    boards[:, :5] = True
    labels, ids = assign_floor(floor_pixels, {"rug": rug, "boards": boards}, {"rug": "covering", "boards": "floor"})
    assert ids == ["boards", "rug"]
    assert (labels[0] == -1).all()
    assert (labels[rug] == 1).all()
    assert (labels[floor_pixels & ~rug] == 0).all()


def test_clipping_a_square_by_a_half_plane():
    square = np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0]])
    kept = clip_polygon(square, np.array([1.0, 0.0]), np.array([-1.0, 0.0]))
    assert np.allclose(sorted(map(tuple, kept)), [(0, 0), (0, 2), (1, 0), (1, 2)])


def test_the_floor_stops_under_the_walls_unless_a_wall_would_cut_away_seen_floor():
    eye = np.array([0.0, 0.0, 1.5])
    # A wall 3 m ahead, facing the camera, and one 1 m ahead that the photo shows floor beyond (a recess).
    back, recess = wall("back", 0.0, 3.0, 0.0), wall("recess", 0.0, 1.0, 0.0, length=0.5)
    seen = np.array([[x, y] for x in np.linspace(-1, 1, 5) for y in np.linspace(0.5, 2.9, 10)])
    polygon, used = room_polygon([back, recess], eye, seen)
    assert used == ["back"]
    # The cut is along the back face, 3.05 m out, so the floor reaches under the 10 cm wall.
    assert np.isclose(polygon[:, 1].max(), 3.05)
    assert np.isclose(polygon[:, 1].min(), -8.0)


def test_a_rug_is_laid_square_to_the_longest_wall():
    turn = np.radians(30.0)
    local = np.array([[x, y] for x in np.linspace(-1, 1, 41) for y in np.linspace(-0.6, 0.6, 25)])
    points = local @ np.array([[np.cos(turn), np.sin(turn)], [-np.sin(turn), np.cos(turn)]])
    corners, heading = covering_rectangle(points, [0.0])
    assert heading == 0.0
    assert len(corners) == 4
    # With no wall, it is the smallest rectangle around the points.
    assert covering_rectangle(points, [])[1] == 30.0


def test_a_tileable_texture_has_no_seam_where_it_wraps():
    rng = np.random.default_rng(0)
    # A smooth texture whose opposite edges differ a lot: a ramp from dark to light.
    texture = np.repeat(np.repeat(np.linspace(0, 250, 64)[None, :, None], 64, axis=0), 3, axis=2).astype(np.uint8)
    texture = np.clip(texture + rng.integers(0, 4, texture.shape), 0, 255).astype(np.uint8)
    tiled = make_tileable(texture).astype(float)
    wrap = np.abs(tiled[:, 0] - tiled[:, -1]).mean()
    step = np.abs(np.diff(tiled, axis=1)).mean()
    assert np.abs(texture[:, 0].astype(float) - texture[:, -1]).mean() > 200
    assert wrap < 3 * step


def test_matching_colour_moves_the_median_and_keeps_the_pattern():
    texture = np.zeros((4, 4, 3), dtype=np.uint8)
    texture[::2] = 100
    texture[1::2] = 140
    matched = match_colour(texture, np.array([0.2, 0.2, 0.2]))
    median = np.median((matched / 255.0) ** 2.2)
    assert abs(median - 0.2) < 0.02
    assert matched[1, 0, 0] > matched[0, 0, 0]


def test_the_clean_square_is_found_among_the_nearer_cells_first():
    clean = np.zeros((60, 60), dtype=bool)
    clean[:, :] = True
    clean[10:20, 10:20] = False
    distance = np.tile(np.arange(60, dtype=float)[:, None], (1, 60))
    row, col, side = clean_square(clean, distance, cell=0.01)
    # The nearer half is rows 0-29 (0-30 by the median), where the hole leaves at most 20 cells clear below it,
    # and 30 is the limit to the nearer half's depth.
    assert row + side <= 31
    assert clean[row : row + side, col : col + side].all()
    assert side >= 25
    assert clean_square(np.zeros((5, 5), dtype=bool), np.zeros((5, 5)), 0.01) is None


def test_grain_runs_along_the_boards_and_carpet_has_none():
    rows, cols = np.mgrid[0:80, 0:80]
    # Boards 8 cells wide, running across, each a different tone.
    boards = np.repeat(((rows // 8) % 3 * 60 + 60)[..., None], 3, axis=2).astype(np.uint8)
    clean = np.ones((80, 80), dtype=bool)
    lines, coherence = grain(boards, clean)
    assert coherence > 0.9
    assert min(lines, 180 - lines) < 2
    carpet = np.random.default_rng(0).integers(80, 120, (80, 80, 3)).astype(np.uint8)
    assert grain(carpet, clean)[1] < 0.15


def test_evening_the_light_takes_out_a_ramp_and_keeps_the_pattern():
    # Lit from the right, with stripes 4 rows wide.
    ramp = np.linspace(60, 200, 128)[None, :]
    stripes = np.where((np.arange(128) // 4) % 2, 1.15, 0.85)[:, None]
    texture = np.repeat(np.clip(ramp * stripes, 0, 255)[..., None], 3, axis=2).astype(np.uint8)
    evened = even_light(texture).astype(float)
    assert abs(evened[:, :16].mean() - evened[:, -16:].mean()) < 0.25 * abs(texture[:, :16].mean() - texture[:, -16:].astype(float).mean())
    assert evened[4:8].mean() > evened[0:4].mean()
