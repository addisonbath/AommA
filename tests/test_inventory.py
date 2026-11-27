from aomma.stages.inventory import parse_inventory


def test_parse_tolerates_fences_bare_lists_and_bad_ids():
    reply = """```json
[
  {"id": "Photo Frame", "bbox_2d": [1, 2, 3, 4], "description": "a", "size_m": [0.2, 0.02, 0.25], "on": "shelf"},
  {"id": "photo-frame", "bbox_2d": [5, 6, 7, 8], "description": "b", "size_m": [0.2, 0.02, 0.25], "on": "shelf"},
  {"id": "shelf", "bbox_2d": [0, 0, 9, 9], "description": "c", "size_m": [1.5, 0.4, 0.8], "on": "floor"}
]
```"""
    inventory = parse_inventory(reply)
    assert [o.id for o in inventory.objects] == ["photo_frame", "photo_frame_2", "shelf"]
    assert inventory.objects[0].on == "shelf"
    assert inventory.objects[2].on == "floor"

