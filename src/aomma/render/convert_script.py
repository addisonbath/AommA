"""Runs inside Blender: convert a GLB model into a normalised USD asset.

The asset follows the project conventions: Z-up, meters, default prim
`/Asset`, pivot at bottom-centre. Its longest side is scaled to 1 m, so a
placement's `scale` is simply the object's real longest dimension.

Usage: blender --background --factory-startup --python convert_script.py -- \
           <model.glb> <asset.usdc>
"""

import sys

import bpy
from mathutils import Matrix, Vector

glb_path, usd_path = sys.argv[sys.argv.index("--") + 1 :]

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=glb_path)

meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]
bpy.ops.object.select_all(action="DESELECT")
for mesh in meshes:
    mesh.select_set(True)
bpy.context.view_layer.objects.active = meshes[0]
if len(meshes) > 1:
    bpy.ops.object.join()
model = bpy.context.view_layer.objects.active

# Bake the importer's Y-up-to-Z-up rotation (and any node transforms) into the vertices.
bpy.ops.object.parent_clear(type="CLEAR_KEEP_TRANSFORM")
bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
for other in [o for o in bpy.context.scene.objects if o is not model]:
    bpy.data.objects.remove(other)

corners = [Vector(c) for c in model.bound_box]
low = Vector(min(c[i] for c in corners) for i in range(3))
high = Vector(max(c[i] for c in corners) for i in range(3))
pivot = Vector(((low.x + high.x) / 2, (low.y + high.y) / 2, low.z))
scale = 1.0 / max(high - low)
model.data.transform(Matrix.Scale(scale, 4) @ Matrix.Translation(-pivot))
model.name = "Geom"

bpy.ops.wm.usd_export(
    filepath=usd_path,
    root_prim_path="/Asset",
    export_materials=True,
    generate_preview_surface=True,
    export_textures_mode="NEW",
    # A model made again in the same folder must not keep the last one's images.
    overwrite_textures=True,
    relative_paths=True,
)
