import numpy as np
from PIL import Image

from aomma.stages.faces import as_inventory, from_faces, list_faces, parse_faces, to_photo, view_of, within_box
from aomma.stages.inventory import InventoryObject


def test_a_face_box_in_the_close_up_maps_back_to_the_whole_photo():
    image = Image.new("RGB", (1000, 500))
    view = view_of(image, (400, 400, 600, 600))  # a 200 x 100 px box, grown 1.5 times
    assert view == (350, 175, 650, 325)
    np.testing.assert_allclose(to_photo(view, (0, 0, 1000, 1000), image.size), (350, 350, 650, 650))
    np.testing.assert_allclose(to_photo(view, (500, 500, 1000, 1000), image.size), (500, 500, 650, 650))
    # A box past the close-up's edge is kept within it.
    np.testing.assert_allclose(to_photo(view, (-50, 0, 1200, 1000), image.size), (350, 350, 650, 650))


def test_the_built_in_takes_its_faces_each_kept_within_its_own_box():
    entry = InventoryObject(id="base_cabinets", bbox_2d=(0, 500, 1000, 1000), description="white cabinets",
                            size_m=(2.0, 0.6, 0.9), on="floor")
    faces = as_inventory(entry, [])
    assert faces.objects == []
    from aomma.stages.faces import Face

    faces = as_inventory(entry, [Face(id="top", bbox_2d=(0, 500, 1000, 600), description="counter top"),
                                 Face(id="front", bbox_2d=(0, 600, 1000, 1000), description="doors")])
    assert [f.id for f in faces.objects] == ["base_cabinets__top", "base_cabinets__front"]
    old = np.zeros((10, 10), bool)
    old[5, :] = True  # the top only
    leaky = np.ones((10, 10), bool)  # the front's mask leaks over the whole photo
    together = from_faces(old, faces, {"base_cabinets__front": leaky})
    assert together[6:, :].all() and not together[:5, :].any()
    assert within_box(leaky, (0, 0, 500, 500)).sum() == 25


def test_faces_are_parsed_from_a_fenced_or_bare_reply_and_bad_boxes_dropped():
    assert parse_faces('```json\n{"faces": [{"id": "top", "bbox_2d": [0, 0, 10, 10], "description": "x"}]}\n```')[0]["id"] == "top"
    assert len(parse_faces('[{"id": "a", "bbox_2d": [0, 0, 10, 10]}, {"id": "b", "bbox_2d": [0, 0]}]')) == 1
    assert parse_faces("no faces") == []


def test_listed_faces_get_unique_ids_and_whole_photo_boxes():
    class FakeVLM:
        def ask(self, image, prompt, schema, label=""):
            assert "white cabinets" in prompt
            return ('{"faces": [{"id": "Front Doors", "bbox_2d": [0, 0, 500, 1000], "description": "doors"},'
                    ' {"id": "Front Doors", "bbox_2d": [500, 0, 1000, 1000], "description": "drawers"},'
                    ' {"id": "empty", "bbox_2d": [10, 10, 10, 20], "description": "none"}]}', {})

    class FakeLedger:
        def check_budget(self):
            pass

        def record(self, kind, usage):
            pass

    entry = InventoryObject(id="base_cabinets", bbox_2d=(250, 250, 750, 750), description="white cabinets",
                            size_m=(2.0, 0.6, 0.9), on="floor")
    faces, _ = list_faces(Image.new("RGB", (1000, 1000)), entry, FakeVLM(), FakeLedger())
    assert [f.id for f in faces] == ["front_doors", "front_doors_2"]
    np.testing.assert_allclose(faces[0].bbox_2d, (125, 125, 500, 875))
