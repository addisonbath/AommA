"""Runs inside Blender (not the project venv): import a USD scene and render it.

Usage: blender --background --factory-startup --python blender_script.py -- \
           <scene.usda> <out.png> <width> <height>
"""

import sys

import bpy

scene_path, out_path, width, height = sys.argv[sys.argv.index("--") + 1 :]

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.wm.usd_import(filepath=scene_path)

scene = bpy.context.scene
scene.camera = next(o for o in scene.objects if o.type == "CAMERA")

# Uniform white light from all directions, so surfaces show their own colours.
# Textures taken from a photo already contain its light and shadow; adding
# strong directional light on top would shade everything twice.
world = bpy.data.worlds.new("World")
world.use_nodes = True
world.node_tree.nodes["Background"].inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
scene.world = world

# EEVEE's identifier differs between Blender versions.
for engine in ("BLENDER_EEVEE_NEXT", "BLENDER_EEVEE"):
    try:
        scene.render.engine = engine
        break
    except TypeError:
        continue

# "Standard" shows colours as authored; the default filmic look desaturates them.
scene.view_settings.view_transform = "Standard"
scene.render.resolution_x = int(width)
scene.render.resolution_y = int(height)
scene.render.resolution_percentage = 100
scene.render.image_settings.file_format = "PNG"
scene.render.filepath = out_path
bpy.ops.render.render(write_still=True)
