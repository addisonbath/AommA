"""Compose a SceneLayout into a scene.usda that references the asset files."""

from pathlib import Path

from pxr import Gf, UsdGeom, UsdLux

from aomma.layout import Camera, SceneLayout
from aomma.usd.assets import new_stage

CAMERA_PATH = "/World/Camera"


def compose_scene(layout: SceneLayout, path: Path) -> None:
    stage = new_stage(path)
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())

    for obj in layout.objects:
        if obj.left_out:
            continue
        xform = UsdGeom.Xform.Define(stage, f"/World/Objects/{obj.name}")
        # The geometry stays in the asset file; the scene only holds the pose.
        xform.GetPrim().GetReferences().AddReference(obj.asset)
        api = UsdGeom.XformCommonAPI(xform)
        api.SetTranslate(Gf.Vec3d(*obj.translate))
        api.SetRotate(Gf.Vec3f(*obj.rotate_xyz_deg), UsdGeom.XformCommonAPI.RotationOrderXYZ)
        api.SetScale(Gf.Vec3f(obj.scale))

    _add_camera(stage, layout.camera)

    sun = UsdLux.DistantLight.Define(stage, "/World/Sun")
    # Weak: just enough to hint at form on untextured proxy boxes.
    sun.CreateIntensityAttr(0.5)
    UsdGeom.XformCommonAPI(sun).SetRotate(Gf.Vec3f(40.0, 0.0, 30.0))

    stage.GetRootLayer().Save()


def _add_camera(stage, camera: Camera) -> None:
    cam = UsdGeom.Camera.Define(stage, CAMERA_PATH)
    cam.CreateFocalLengthAttr(camera.focal_length_mm)
    cam.CreateHorizontalApertureAttr(camera.sensor_width_mm)
    cam.CreateVerticalApertureAttr(camera.sensor_width_mm * camera.height / camera.width)
    cam.CreateClippingRangeAttr(Gf.Vec2f(0.01, 1000.0))
    # USD matrices act on row vectors, so pass the transpose of ours.
    cam.AddTransformOp().Set(Gf.Matrix4d(*camera.matrix.T.flatten().tolist()))
