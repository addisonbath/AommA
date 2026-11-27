import json

import numpy as np

from aomma.stages.inventory import Inventory, InventoryObject
from aomma.stages.occlusion import find_occlusions


def _obj(oid, on="floor"):
    return InventoryObject(id=oid, bbox_2d=(0, 0, 1, 1), description="", size_m=(1, 1, 1), on=on)


def test_nearer_touching_object_and_resting_object_are_occluders():
    shape = (60, 100)
    masks = {name: np.zeros(shape, dtype=bool) for name in ("table", "chair", "vase", "far_lamp")}
    masks["table"][20:50, 20:60] = True
    masks["chair"][30:60, 60:80] = True  # touches the table's right edge, nearer
    masks["vase"][10:20, 30:40] = True  # touches the table's top edge, rests on it
    masks["far_lamp"][0:15, 85:95] = True  # touches nothing

    depth = np.full(shape, 5.0)
    depth[masks["table"]] = 3.0
    depth[masks["chair"]] = 2.0
    depth[masks["vase"]] = 3.0  # same distance: only "rests on" makes it an occluder
    points = np.zeros((*shape, 3))
    points[..., 2] = -depth

    inventory = Inventory(objects=[_obj("table"), _obj("chair"), _obj("vase", on="table"), _obj("far_lamp")])
    found = find_occlusions(masks, points, inventory)

    assert sorted(found["table"].occluders) == ["chair", "vase"]
    assert 0.2 < found["table"].hidden_edge < 0.5
    assert found["chair"].occluders == [] and found["chair"].hidden_edge == 0.0
    assert found["far_lamp"].occluders == []


def test_listed_small_items_with_masks_count_for_large_items(tmp_path):
    from PIL import Image

    from aomma import cli

    assert cli._listed_small(tmp_path) == []  # before gate 3: none
    item = {"bbox_2d": [0, 0, 1, 1], "description": "", "size_m": [0.1, 0.1, 0.1], "on": "round_table"}
    (tmp_path / "small_survey.json").write_text(json.dumps({"objects": [
        {"id": "cup", **item}, {"id": "tray", **item}, {"id": "no_mask", **item},
    ]}))
    (tmp_path / "masks").mkdir()
    for oid in ("cup", "tray"):
        Image.fromarray(np.zeros((4, 4), dtype=np.uint8)).save(tmp_path / "masks" / f"{oid}.png")
    listed = cli._listed_small(tmp_path, exclude={"tray"})
    assert [o.id for o in listed] == ["cup"] and listed[0].on == "round_table"


def test_thin_small_items_join_the_scene_as_boxes(tmp_path, monkeypatch):
    from aomma import cli
    from aomma.layout import Camera, SceneLayout

    item = {"bbox_2d": [0, 0, 1, 1], "description": "", "size_m": [0.3, 0.3, 0.02], "on": "table"}
    tray = {"id": "tray", "size": [0.34, 0.35, 0.02], "translate": [0.0, 0.5, 0.7], "yaw_deg": 30.0, "color": [0.1, 0.1, 0.1]}
    (tmp_path / "small_survey.json").write_text(json.dumps({"objects": [{"id": "tray", **item}]}))
    (tmp_path / "boxes.json").write_text(json.dumps({"boxes": [{"kind": "small", "proxy": tray}]}))
    (tmp_path / "inventory.json").write_text(json.dumps({"objects": []}))
    (tmp_path / "proxies.json").write_text("[]")
    (tmp_path / "assets").mkdir()
    scene = SceneLayout(camera=Camera(camera_to_world=tuple(np.eye(4).ravel())), objects=[])
    (tmp_path / "layout.json").write_text(scene.model_dump_json())
    monkeypatch.setattr(cli, "_load_points", lambda _: np.zeros((4, 4, 3)))
    monkeypatch.setattr(cli, "_compose_and_render", lambda *_: None)
    monkeypatch.setattr(cli, "assets", lambda *_: (_ for _ in ()).throw(AssertionError("a thin item is not modelled")))

    cli.small_models(tmp_path, [], tmp_path / "config.toml")

    placed = SceneLayout.model_validate_json((tmp_path / "layout.json").read_text()).objects
    assert [(o.name, o.asset) for o in placed] == [("tray", "./assets/tray.usda")]
    assert (tmp_path / "assets" / "tray.usda").exists()
    assert [p["id"] for p in json.loads((tmp_path / "proxies.json").read_text())] == ["tray"]
