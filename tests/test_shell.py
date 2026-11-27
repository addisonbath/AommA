import json

import numpy as np

from aomma.stages.proxy import Proxy
from aomma.stages.shell import floor_from_heights, parse_survey, scale_from_heights, wall_agreement


def survey_reply(**lists) -> str:
    return json.dumps({"indoor": True, "room_type": "kitchen", "walls": [], "built_ins": [], "openings": [], **lists})


ENTRY = {"bbox_2d": [0, 100, 10, 110], "description": "", "size_m": [1, 1, 1], "on": "floor"}


def test_survey_ids_are_unique_across_walls_built_ins_and_openings():
    reply = survey_reply(
        walls=[{"id": "Left Wall", "bbox_2d": [0, 0, 500, 1000], "description": "white"}],
        built_ins=[{"id": "Cabinets", "kind": "base_cabinets", **ENTRY}, {"id": "counter", "kind": "counter", **ENTRY, "on": "Cabinets"}],
        openings=[{"id": "cabinets", "kind": "window", **ENTRY, "on": "wall"}],
    )
    survey = parse_survey(f"```json\n{reply}\n```")
    assert [o.id for o in survey.walls] == ["left_wall"]
    assert [o.id for o in survey.built_ins] == ["cabinets", "counter"]
    assert survey.built_ins[1].on == "cabinets"
    assert [o.id for o in survey.openings] == ["cabinets_2"]
    assert survey.kinds == {"cabinets": "base_cabinets", "counter": "counter", "cabinets_2": "window"}


def test_scale_comes_from_standard_tops_and_is_refused_when_the_camera_would_be_unlikely():
    survey = parse_survey(survey_reply(
        built_ins=[{"id": "counter", "kind": "counter", **ENTRY}, {"id": "shelf", "kind": "shelving", **ENTRY}],
        openings=[{"id": "door", "kind": "door", **ENTRY}, {"id": "cut", "kind": "door", **{**ENTRY, "bbox_2d": [0, 0, 10, 900]}}],
    ))

    def box(oid: str, top: float) -> Proxy:
        return Proxy(id=oid, size=(1.0, 1.0, top), translate=(0.0, 0.0, 0.0), yaw_deg=0.0, color=(1, 1, 1))

    # Measured 10% short: the counter and the door agree; shelving has no standard height, and the
    # photo's upper edge cuts off the second door's top.
    proxies = [box("counter", 0.91 / 1.1), box("shelf", 3.0), box("door", 2.03 / 1.1), box("cut", 0.5)]
    scale, used, _ = scale_from_heights(survey, proxies, camera_height_m=1.3)
    assert abs(scale - 1.1) < 1e-9 and used == ["counter", "door"]
    # The same scale would put a camera measured 1.8 m up at 1.98 m.
    assert scale_from_heights(survey, proxies, camera_height_m=1.8)[:2] == (1.0, [])
    assert scale_from_heights(survey, [box("shelf", 3.0)], camera_height_m=1.3)[:2] == (1.0, [])


def test_a_floor_out_of_view_is_placed_from_standard_tops():
    survey = parse_survey(survey_reply(
        built_ins=[{"id": "counter", "kind": "counter", **ENTRY}, {"id": "island", "kind": "island", **ENTRY},
                   {"id": "shelf", "kind": "shelving", **ENTRY}],
    ))

    def box(oid: str, top: float) -> Proxy:
        return Proxy(id=oid, size=(1.0, 1.0, 0.04), translate=(0.0, 0.0, top - 0.04), yaw_deg=0.0, color=(1, 1, 1))

    # The floor taken was the counter's top, the camera 0.6 m over it: the island's top 2 cm above the counter's.
    proxies = [box("counter", 0.0), box("island", 0.02), box("shelf", 1.2)]
    drop, used = floor_from_heights(survey, proxies, camera_height_m=0.6)
    assert abs(drop - 0.90) < 1e-9 and used == ["counter", "island"]
    # Nothing of a standard height, or a camera that would end up 2.5 m up: no placing.
    assert floor_from_heights(survey, [box("shelf", 1.2)], camera_height_m=0.6) is None
    assert floor_from_heights(survey, proxies, camera_height_m=1.6) is None


def test_wall_agreement_tells_a_wall_in_front_from_one_behind():
    eye = np.array([0.0, 0.0, 1.5])
    # A wall 3 m ahead, facing the camera, 4 m wide and 2.4 m tall.
    wall = (np.array([-2.0, 3.0, 0.0]), np.array([4.0, 0.0, 0.0]), np.array([0.0, -1.0, 0.0]), 2.4)
    on_wall = np.array([[0.0, 3.0, 1.0], [0.5, 3.05, 2.0]])
    beyond = np.array([[0.0, 4.0, 1.0]])  # a doorway: the wall stands in front of it
    short = np.array([[0.0, 2.0, 1.0]])  # a cabinet fitted as wall would leave the real wall behind it
    outside = np.array([[3.0, 3.0, 1.0]])  # past the wall's end
    found = wall_agreement(np.concatenate([on_wall, beyond, short, outside]), [wall], eye)
    assert found == {"agree": 0.4, "too_near": 0.2, "too_far": 0.2, "missed": 0.2, "points": 5}
    assert wall_agreement(np.zeros((0, 3)), [wall], eye) is None


def test_a_wall_showing_only_its_cabinets_stands_against_their_back():
    from aomma.stages.room import wall_behind, wall_face

    # Wall cabinets 2 m wide and 0.3 m deep, their front 2 m from a camera looking along +y.
    cabinets = Proxy(id="cabinets", size=(2.0, 0.3, 0.7), translate=(0.0, 2.15, 1.4), yaw_deg=0.0, color=(1.0, 1.0, 1.0))
    eye = np.array([0.0, 0.0, 1.5])
    wall = wall_behind(cabinets, eye)
    origin, along, normal = wall_face(wall, eye)
    assert np.isclose(origin[1], 2.3) and np.allclose(normal, [0, -1, 0])
    assert np.isclose(np.linalg.norm(along), 2.0) and wall.translate[2] == 0.0
