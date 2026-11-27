"""Runs inside Blender (not the project venv): convert a USD scene to a GLB a browser can show.

Usage: blender --background --factory-startup --python export_script.py -- \
           <scene.usda> <out.glb> <max texture side>
"""

import sys

import bpy

scene_path, out_path, max_side = sys.argv[sys.argv.index("--") + 1 :]

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.wm.usd_import(filepath=scene_path)

# Generated models carry several large textures each; a scene of them would be
# hundreds of megabytes. Shrink them to what a phone can load.
for image in bpy.data.images:
    longest = max(image.size)
    if longest > int(max_side):
        factor = int(max_side) / longest
        image.scale(max(int(image.size[0] * factor), 1), max(int(image.size[1] * factor), 1))

bpy.ops.export_scene.gltf(
    filepath=out_path,
    export_format="GLB",
    export_image_format="JPEG",
    export_image_quality=80,
    export_cameras=True,
    export_draco_mesh_compression_enable=True,
)
