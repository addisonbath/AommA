from pxr import Usd, UsdGeom

from aomma.layout import Camera, Placement, SceneLayout
from aomma.usd.assets import write_proxy_box
from aomma.usd.compose import CAMERA_PATH, compose_scene


def test_scene_references_assets_and_places_them(tmp_path):
    write_proxy_box(tmp_path / "assets" / "box.usda", (1.0, 2.0, 3.0))
    layout = SceneLayout(
        camera=Camera.looking_at((0.0, -5.0, 1.0), (0.0, 0.0, 1.0)),
        objects=[Placement(name="box", asset="./assets/box.usda", translate=(1.0, 0.0, 0.5))],
    )
    compose_scene(layout, tmp_path / "scene.usda")

    stage = Usd.Stage.Open(str(tmp_path / "scene.usda"))
    assert UsdGeom.GetStageUpAxis(stage) == UsdGeom.Tokens.z
    assert stage.GetPrimAtPath(CAMERA_PATH).IsA(UsdGeom.Camera)

    # The mesh comes through the reference, moved by the placement.
    geom = stage.GetPrimAtPath("/World/Objects/box/Geom")
    assert geom.IsA(UsdGeom.Mesh)
    bbox = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    box = bbox.ComputeWorldBound(geom).ComputeAlignedRange()
    assert tuple(box.GetMin()) == (0.5, -1.0, 0.5)
    assert tuple(box.GetMax()) == (1.5, 1.0, 3.5)

    # Camera sits where the layout says and looks along +Y.
    cam = UsdGeom.Camera(stage.GetPrimAtPath(CAMERA_PATH)).GetCamera()
    assert tuple(cam.transform.ExtractTranslation()) == (0.0, -5.0, 1.0)
    forward = cam.transform.TransformDir((0, 0, -1))
    assert abs(forward[1] - 1.0) < 1e-9
