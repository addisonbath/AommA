"""Runs inside Blender (not the project venv): render a USD scene's depth from its camera and save it as a .npy.

The saved array is (height, width), top row first, in scene units, with
anything the camera sees no surface on (the empty world) set to infinity.
With <objects.npz>, it also saves which model each pixel shows: "ids", a
(height, width) array of 1-based indices into "names", 0 where no named
model is seen (the room, flat surfaces).

With --views, it renders the scene from each camera a JSON file lists instead of its
own, saving <out folder>/<name>.npy and <name>.npz (the ids) for each. A camera is
`camera_to_world` (4x4, row-major, in the scene's frame, looking down its -z), `width`
and `height` (the render's), `fx`, `fy`, `cx`, `cy` (in pixels of that size).

Usage: blender --background --factory-startup --python depth_script.py -- \
           <scene.usda> <out.npy> <width> <height> [<objects.npz>]
       blender --background --factory-startup --python depth_script.py -- \
           <scene.usda> --views <views.json> <out folder>
"""

import json

import sys
import tempfile
from pathlib import Path

import bpy
import numpy as np

from mathutils import Matrix

arguments = sys.argv[sys.argv.index("--") + 1 :]
if arguments[1] == "--views":
    scene_path, _, views_path, out_folder = arguments
    views = json.loads(Path(views_path).read_text())
    objects_path = [True]
else:
    scene_path, out_path, width, height, *objects_path = arguments
    views = [{"name": None, "width": int(width), "height": int(height)}]

bpy.ops.wm.read_factory_settings(use_empty=True)
# Blender folds an Xform holding a single mesh into a bare mesh named after the mesh ("Geom"), so a box, a built-in's
# proxy or a one-mesh wall would have no name to tell it by. With --views (`aomma score`) every placed prim keeps its
# Xform, so each pixel can be told by its object; the photo's own depth check (one camera) is left as it was.
bpy.ops.wm.usd_import(filepath=scene_path, merge_parent_xform=not arguments[1] == "--views")

scene = bpy.context.scene
scene.camera = next(o for o in scene.objects if o.type == "CAMERA")
# One sample of Cycles is enough for depth, which needs no light, and its depth is exact.
scene.render.engine = "CYCLES"
scene.cycles.samples = 1
scene.cycles.use_denoising = False
scene.render.resolution_percentage = 100
scene.view_layers[0].use_pass_z = True

# A generated model comes in as its meshes under one parent named after the object. Without --views, walls, the floor,
# boxes and flat art come in as bare meshes, with no name to tell them by.
names = []
for obj in scene.objects:
    if obj.type == "MESH" and obj.parent is not None:
        root = obj.parent
        while root.parent is not None:
            root = root.parent
        if root.name not in names:
            names.append(root.name)
        obj.pass_index = names.index(root.name) + 1
if objects_path:
    scene.view_layers[0].use_pass_object_index = True

# The depth pass goes to a 32-bit EXR through the compositor, which is read back here.
scene.use_nodes = True
tree = scene.node_tree
tree.nodes.clear()
layers = tree.nodes.new("CompositorNodeRLayers")
output = tree.nodes.new("CompositorNodeOutputFile")
folder = Path(tempfile.mkdtemp())
output.base_path = str(folder)
output.format.file_format = "OPEN_EXR"
output.format.color_depth = "32"
output.file_slots[0].path = "depth"
tree.links.new(layers.outputs["Depth"], output.inputs[0])
if objects_path:
    output.file_slots.new("objects")
    tree.links.new(layers.outputs["IndexOB"], output.inputs[1])


def aim(camera, view: dict) -> None:
    """Put the scene's camera where `view` says, with its intrinsics: Blender's shift is in units of the larger side."""
    width, height = view["width"], view["height"]
    larger = max(width, height)
    camera.matrix_world = Matrix(view["camera_to_world"])
    data = camera.data
    data.type = "PERSP"
    data.sensor_fit = "AUTO"
    data.sensor_width = 36.0
    data.lens = view["fx"] * 36.0 / larger
    data.shift_x = (width / 2 - view["cx"]) / larger
    data.shift_y = (view["cy"] - height / 2) / larger
    data.clip_start, data.clip_end = 0.01, 1000.0


def read(prefix: str, width: int, height: int) -> np.ndarray:
    image = bpy.data.images.load(str(next(folder.glob(f"{prefix}*.exr"))))
    pixels = np.array(image.pixels[:], dtype=np.float32).reshape(height, width, -1)[..., 0]
    bpy.data.images.remove(image)
    return np.flipud(pixels)  # Blender's rows start at the bottom


for view in views:
    width, height = view["width"], view["height"]
    scene.render.resolution_x, scene.render.resolution_y = width, height
    if view["name"] is not None:
        aim(scene.camera, view)
        out_path = str(Path(out_folder) / f"{view['name']}.npy")
        objects_path = [str(Path(out_folder) / f"{view['name']}.npz")]
    for old in folder.glob("*.exr"):
        old.unlink()
    bpy.ops.render.render(write_still=False)
    if objects_path:
        np.savez(objects_path[0], ids=np.rint(read("objects", width, height)).astype(np.int16), names=np.array(names))
    depth = read("depth", width, height).astype(np.float64)
    depth[depth > 1e9] = np.inf
    np.save(out_path, depth)
