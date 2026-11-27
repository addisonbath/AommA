import argparse
import functools
import hashlib
import queue
import re
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import json

import numpy as np
from PIL import Image

from aomma import background
from aomma.backends.comfy import ComfyClient
from aomma.backends.vlm import make_vlm
from aomma.batch import BatchProgress
from aomma.config import CODE_ROOT, DEFAULT_CONFIG, Config, load_config
from aomma.dashboard import experiment_status, newest_experiment, open_actions, record_action, record_review, serve, unreviewed
from aomma.jobqueue import JobQueue
from aomma.layout import Camera, Placement, SceneLayout, Vec3
from aomma.ledger import Ledger
from aomma.monitor import snapshot, watch
from aomma.render.blender import glb_to_usd, render
from aomma.stages.asset import MODEL_CACHE, cached_model, crop_background, crop_object, generate_model, in_object_crop, model_key
from aomma.stages.depth import LIDAR_SURE, draw_depth, estimate_points, fit_to_lidar, lidar_pixels, lidar_points
from aomma.stages.flat import FlatArt, fit_flat_art, flat_art_texture, frame_normal_map, is_flat_art
from aomma.stages.image import Intrinsics, load_source, load_working_image, to_jpeg
from aomma.stages.inventory import SPLIT_UP_TO, Inventory, InventoryObject, split_entry
from aomma.stages.occlusion import find_occlusions
from aomma.stages.pose import (
    _MAX_STRETCH, FLOOR, Pose, PoseSettings, Wall, behind_walls, euler_xyz_deg, fit_to_box, hidden_by_walls, in_box, lowest,
    matrix_from_euler_xyz_deg,
    outline_fit, placed_points, pull_forward, reseat_on_floor, rotation_z, solve_occluded_pose, solve_pose, view_rotation,
)
from aomma.stages.score import project
from aomma.stages.large import (
    BOTTOM_SHORT, CUT_SIZE_WEIGHT, OFF_DEPTH_RATIO, REPOSE_DEPTH_WEIGHT, REPOSE_SIZE_WEIGHT, REPOSE_TURN_WITHIN_DEG, REPOSE_WALL_WEIGHT,
    SQUARED_MAX_STRETCH, UNDO_FIT_DROP, LargeSurvey, SQUARE_WITHIN_DEG, ShellMoves, below_mask, bottom_short, contents_holes, contents_id,
    cut_below_frame,
    cut_by_frame, distance_ratio, hold_to_shell, seen_turn_deg, settle_on_support, share_past_walls, square_to_built_in, square_turn_deg, take_large,
)
from aomma.stages.square import (
    BUILT_IN_KEEP_M, ITEM_KEEP_FIT_DROP, SQUARE_SHARE, Footprint, box_error_m, chamfered_corners, line_up, room_grid_deg, square_built_in, squareness,
)
from aomma.stages.ceiling import THICKNESS_M as CEILING_THICKNESS_M, Ceiling, build_ceiling
from aomma.stages.floor import Drawer, FloorSurvey, draw_texture, floor_regions, take_floor_survey
from aomma.stages.shell import Survey, floor_from_heights, scale_from_heights, take_survey, wall_agreement
from aomma.stages.proxy import Proxy, depth_scale_correction, exclusive_masks, fit_proxies
from aomma.stages.repair import (
    REPAIR_THRESHOLD, context_crop, crop_fov_deg, extend_below, fill_contents, in_crop, is_light, repair_object, where_it_can_be,
)
from aomma.stages.room import (
    BASEBOARD_THICKNESS_M, WALL_REACH_M, built_in_ends, cut_openings, find_baseboard, fit_walls, glaze, merge_walls, opening_spans, reach_frame, room_baseboards, sight_face, surface_texture, tidy_walls, wall_behind,
    wall_face, wall_foot_pixels, wall_surface_offset, wall_surface_pixels,
)
from aomma.stages.score import BUILT_IN_EDGE, LARGE_EDGE, SMALL_EDGE, draw_wireframes, score_layout, silhouette
from aomma.stages.support import rest_on
from aomma.stages.world import camera_over_floor, facing, fit_camera_to_world, fit_floor, surface_normals, to_world
from aomma.stages.inventory import draw_inventory, take_inventory
from aomma.stages.segment import (
    MIN_BOX_IOU, MIN_FILL, MIN_USABLE_BOX_IOU, MaskInfo, box_fills, describe_mask, draw_masks, judge_second_look, segment, segment_floor,
    segment_surface,
)
from aomma.usd.assets import write_floor, write_picture_slab, write_proxy_box, write_reshaped_asset, write_textured_quad, write_wall
from aomma.usd.compose import compose_scene
from aomma.usd.sample import sample_surface, surface_facets

# Labels of submitted ComfyUI jobs, shared by all runs so `aomma jobs` can show them.
JOB_LOG = Path("runs") / "comfy_jobs.jsonl"
# The one queue every job waits in, whichever command and run it comes from.
QUEUE = JobQueue()
# The longest a job may run once started, in seconds. A bulky object's model has taken
# 55 minutes and a bulkier one was still going at 60, so a model gets three hours.
QUICK_JOB_LIMIT_S, MODEL_LIMIT_S = 3600, 3 * 3600


def comfy_client(url: str, run_dir: Path, heavy: bool = False) -> ComfyClient:
    """A client for one ComfyUI server whose jobs wait their turn in the queue."""
    return ComfyClient(
        url, JOB_LOG, timeout_s=MODEL_LIMIT_S if heavy else QUICK_JOB_LIMIT_S,
        queue=QUEUE, kind="3d" if heavy else "light", run=run_dir.name,
    )


def vlm_for(config: Config, name: str | None, run_dir: Path):
    """The configured VLM (or the one named), its questions waiting their turn in the queue; sampled with the run's
    seed when run.json names one (a noise draw, `backends.vlm.noise_sampling`)."""
    manifest = run_dir / "run.json"
    sampling = json.loads(manifest.read_text()).get("vlm_sampling") if manifest.exists() else None
    return make_vlm(config.vlms[name or config.vlm], QUEUE, run_dir.name, sampling)

# name -> (size in meters, colour, translate, rotation about Z in degrees)
_DEMO_OBJECTS = {
    "ground": ((4.0, 4.0, 0.02), (0.45, 0.45, 0.45), (0.0, 0.0, -0.02), 0.0),
    "table": ((1.2, 0.6, 0.75), (0.55, 0.35, 0.2), (0.0, 0.0, 0.0), 0.0),
    "mug": ((0.08, 0.08, 0.10), (0.8, 0.2, 0.2), (-0.3, 0.05, 0.75), 0.0),
    "lamp": ((0.15, 0.15, 0.40), (0.9, 0.8, 0.3), (0.35, -0.05, 0.75), 30.0),
}


def demo(run_dir: Path) -> None:
    """Build and render a hand-authored proxy scene (milestone M0)."""
    objects = []
    for name, (size, color, translate, yaw) in _DEMO_OBJECTS.items():
        write_proxy_box(run_dir / "assets" / f"{name}.usda", size, color)
        objects.append(
            Placement(
                name=name,
                asset=f"./assets/{name}.usda",
                translate=translate,
                rotate_xyz_deg=(0.0, 0.0, yaw),
            )
        )
    layout = SceneLayout(
        camera=Camera.looking_at((2.2, -2.4, 1.5), (0.0, 0.0, 0.6)),
        objects=objects,
    )

    (run_dir / "layout.json").write_text(layout.model_dump_json(indent=2))
    compose_scene(layout, run_dir / "scene.usda")
    render(run_dir / "scene.usda", run_dir / "render.png", layout.camera.width, layout.camera.height)
    print(f"Wrote {run_dir}/scene.usda and render.png")


def perceive(image_path: Path, run_dir: Path, config_path: Path, vlm_name: str | None) -> None:
    """Run the perception stages on a photo: inventory, masks, depth, camera."""
    config = load_config(config_path)
    vlm = vlm_for(config, vlm_name, run_dir)
    (run_dir / "debug").mkdir(parents=True, exist_ok=True)
    ledger = Ledger(run_dir / "usage.jsonl", config.max_billed_tokens)

    image, intrinsics = load_working_image(image_path, config.image_max_side)
    image.save(run_dir / "input.jpg", quality=90)
    note_source(run_dir, image_path)

    inventory, raw = take_inventory(image, vlm, ledger)
    (run_dir / "debug" / "inventory_raw.txt").write_text(raw)
    # A few things listed as one entry get an entry each, if each is big enough to
    # be worth a model; a large packed mass, or a huddle of tiny things, stays whole.
    for lumped in [obj.id for obj in inventory.objects if 1 < obj.count <= SPLIT_UP_TO and _box_share(obj) / obj.count >= MIN_OBJECT_SHARE]:
        split_entry(image, inventory, lumped, vlm, ledger)
    (run_dir / "inventory.json").write_text(inventory.model_dump_json(indent=2))
    draw_inventory(image, inventory).save(run_dir / "debug" / "inventory.jpg")

    comfy = comfy_client(config.comfy.light[0], run_dir)
    image_name = comfy.upload_image(to_jpeg(image), f"aomma_{run_dir.name}.jpg")

    masks, infos = segment(comfy, image_name, image.size, inventory)
    # A box the masks leave mostly uncovered holds something they missed: look at it again, once.
    doubtful = [oid for oid, fill in box_fills(inventory, masks).items() if fill < MIN_FILL]
    if doubtful:
        _relist(image, inventory, doubtful, vlm, ledger, comfy, image_name, masks, infos)
        (run_dir / "inventory.json").write_text(inventory.model_dump_json(indent=2))
        draw_inventory(image, inventory).save(run_dir / "debug" / "inventory.jpg")
    for oid, fill in box_fills(inventory, masks).items():
        infos[oid].fill = fill
    (run_dir / "masks").mkdir(exist_ok=True)
    for oid, mask in masks.items():
        Image.fromarray(mask).save(run_dir / "masks" / f"{oid}.png")
    (run_dir / "masks.json").write_text(
        json.dumps({oid: info.model_dump() for oid, info in infos.items()}, indent=2)
    )
    draw_masks(image, masks).save(run_dir / "debug" / "masks.jpg")
    Image.fromarray(segment_floor(comfy, image_name, image.size)).save(run_dir / "floor.png")

    # A focal length from EXIF is trusted; otherwise MoGe estimates the FOV.
    known_fov = intrinsics.fov_x_deg if intrinsics.source == "exif" else None
    points, fov = estimate_points(comfy, image_name, image.size, known_fov)
    if known_fov is None:
        intrinsics = intrinsics.with_fov_x(fov, "moge")
    points = _save_points(run_dir, points)
    (run_dir / "camera.json").write_text(intrinsics.model_dump_json(indent=2))
    draw_depth(points).save(run_dir / "debug" / "depth.jpg")

    for entry in ledger.entries():
        print(
            f"{entry['stage']}: {entry['model']} {entry['input_tokens']} in / "
            f"{entry['output_tokens']} out, {entry['seconds']}s"
            + ("" if entry["billed"] else " (local, not billed)")
        )
    weak = [oid for oid, info in infos.items() if info.box_iou < MIN_BOX_IOU or info.fill < MIN_FILL]
    print(f"{len(inventory.objects)} objects, {len(weak)} with a doubtful mask: {', '.join(weak) or '-'}")
    print(f"camera: {intrinsics.fov_x_deg:.1f} deg horizontal FOV ({intrinsics.source}) -> {run_dir}")


# An object must cover at least this share of the photo to be worth an entry of its own.
MIN_OBJECT_SHARE = 0.002


def _box_share(obj) -> float:
    """Share of the photo an inventory entry's box covers."""
    x1, y1, x2, y2 = obj.bbox_2d
    return max(x2 - x1, 0) * max(y2 - y1, 0) / 1e6


def overrides(run_dir: Path) -> dict[str, dict]:
    """Per-object departures from the pipeline's own choices, set by the review (see agent.py).

    {object id: {"attempt": n, "repair": true, "flat": true, "drop": true,
                 "pose": {pose setting: value}}}
    """
    path = run_dir / "overrides.json"
    return json.loads(path.read_text()) if path.exists() else {}


def set_override(run_dir: Path, object_id: str, **changes) -> None:
    """Record overrides for one object; a value of None removes that override."""
    saved = overrides(run_dir)
    entry = saved.setdefault(object_id, {})
    for key, value in changes.items():
        if value is None:
            entry.pop(key, None)
        else:
            entry[key] = value
    (run_dir / "overrides.json").write_text(json.dumps(saved, indent=2))


def _relist(
    image: Image.Image,
    inventory: Inventory,
    object_ids: list[str],
    vlm,
    ledger: Ledger,
    comfy: ComfyClient,
    image_name: str,
    masks: dict[str, np.ndarray],
    infos: dict[str, MaskInfo],
) -> dict[str, tuple[str, list[str]]]:
    """Look again at each of `object_ids` and split it, group it, or leave it alone.

    The VLM lists again what the entry covers, from a close-up, and each thing
    it lists is segmented. Then:
    - split: every part's mask fills its own box, so each becomes an entry;
    - group: the parts do not separate cleanly, so the entry stays one object
      whose mask is everything found in its box;
    - keep: neither covers the box better than before, so nothing changes.

    `inventory`, `masks` and `infos` are updated in place. Returns, for each
    entry, what was done and the ids now standing for it.
    """
    outcome = {}
    for object_id in object_ids:
        entry = next(obj for obj in inventory.objects if obj.id == object_id)
        # Try the new listing on a copy, so a worse one can be dropped.
        relisted = inventory.model_copy(deep=True)
        parts = split_entry(image, relisted, object_id, vlm, ledger)
        part_masks, part_infos = segment(comfy, image_name, image.size, Inventory(objects=parts))
        done, together = judge_second_look(inventory, masks, object_id, relisted, part_masks)
        if done in ("split", "redrawn"):
            inventory.objects = relisted.objects
            masks.pop(object_id)
            infos.pop(object_id)
            masks.update(part_masks)
            infos.update(part_infos)
            outcome[object_id] = (done, [part.id for part in parts])
            continue
        if done == "grouped":
            masks[object_id] = together
            infos[object_id] = describe_mask(entry.bbox_2d, together, infos[object_id].prompt)
            entry.count = max(entry.count, len(parts))
        outcome[object_id] = (done, [object_id])
    return outcome


def _compose_and_render(run_dir: Path, scene: SceneLayout) -> None:
    """Write scene.usda, render it, and save the photo-vs-render comparison."""
    camera = scene.camera
    compose_scene(scene, run_dir / "scene.usda")
    render(run_dir / "scene.usda", run_dir / "render.png", camera.width, camera.height)
    compare = Image.new("RGB", (camera.width * 2, camera.height))
    compare.paste(Image.open(run_dir / "input.jpg"), (0, 0))
    compare.paste(Image.open(run_dir / "render.png").convert("RGB"), (camera.width, 0))
    compare.save(run_dir / "debug" / "compare.jpg")


def _save_points(run_dir: Path, points: np.ndarray) -> np.ndarray:
    """Save MoGe's point map as points.npy, fitted to the shot's LiDAR when the run's condition gives it one; returns what was saved.

    The fitted run keeps MoGe's own as points_moge.npy, and provenance.json says which points.npy is.
    """
    # Imported here because it reads this module's run-folder helpers.
    from aomma.capture import inputs

    given = inputs(run_dir)
    if "depth" not in given:
        np.save(run_dir / "points.npy", points)
        note_made(run_dir, "points.npy", "moge")
        return points
    np.save(run_dir / "points_moge.npy", points)
    note_made(run_dir, "points_moge.npy", "moge")
    points, fit = fit_to_lidar(points, given["depth"], given["conf"])
    np.save(run_dir / "points.npy", points)
    note_made(run_dir, "points.npy", "lidar_aligned(moge)", inputs=["points_moge.npy", "depth", "conf"], **fit)
    print(f"depth fitted to the LiDAR: x{fit['scale']:.3f} {fit['shift_m']:+.3f} m, median error {fit['median_error_m'] * 100:.1f} cm "
          f"over {fit['pixels']} of {fit['of']} sure pixels", flush=True)
    return points


def _metric(run_dir: Path) -> bool:
    """Whether the run's depth is measured (its condition gives it the LiDAR), so it is not to be rescaled."""
    from aomma.capture import inputs

    return "depth" in inputs(run_dir)


def _known_floor(run_dir: Path, floor: np.ndarray, points: np.ndarray) -> dict:
    """What the run's condition says of its floor, as `fit_floor` takes it: the world's up in the camera
    frame, from the pose; and with the LiDAR, its sure points on the `floor` mask and all of them. Nothing for a photo run.
    Only the mask's pixels whose surfaces face up in `points`, the photo's point map, count (`stages.world.facing`):
    KitchenScan 011's mask was its backsplash, whose LiDAR points put the camera 0.25 m off the floor."""
    from aomma.capture import inputs

    given = inputs(run_dir)
    if "camera_to_world" not in given:
        return {}
    known = {"up": np.linalg.inv(given["camera_to_world"])[:3, 1]}
    if "depth" in given:
        floor = floor & facing(surface_normals(points), known["up"] / np.linalg.norm(known["up"]))
        sure = given["conf"] == LIDAR_SURE
        v, u = lidar_pixels(given["depth"].shape, floor.shape)
        points = lidar_points(given["depth"], given["intrinsics"])
        known["measured"] = (points[sure & floor[v][:, u]], points[sure])
    return known


def _load_points(run_dir: Path) -> np.ndarray:
    """The camera-frame point map, with the layout's depth-scale correction if there is one."""
    points = np.load(run_dir / "points.npy")
    world = run_dir / "world.json"
    return points * json.loads(world.read_text())["depth_scale"] if world.exists() else points


def _usable_masks(run_dir: Path, inventory: Inventory) -> tuple[dict[str, np.ndarray], list[str]]:
    """Every object's mask, and which were emptied because they lie nowhere near the object's box.

    Such a mask is not of the object, so the object is left out of the scene,
    and its scraps are left to the objects they belong to.
    """
    masks = {o.id: np.asarray(Image.open(run_dir / "masks" / f"{o.id}.png")) > 0 for o in inventory.objects}
    infos = json.loads((run_dir / "masks.json").read_text())
    unusable = [oid for oid in masks if oid in infos and infos[oid]["box_iou"] < MIN_USABLE_BOX_IOU]
    for oid in unusable:
        masks[oid] = np.zeros_like(masks[oid])
    return masks, unusable


# A pixel shows the floor if its point is within this distance of the floor plane, in meters.
_ON_FLOOR_M = 0.03


def _build_room(
    run_dir: Path,
    image: Image.Image,
    points_world: np.ndarray,
    camera: Camera,
    walls: list[Proxy],
    free: np.ndarray,
    wall_pixels: np.ndarray,
    floor_survey: FloorSurvey | None = None,
    floor_masks: dict[str, np.ndarray] | None = None,
    draw: Drawer | None = None,
    wall_masks: dict[str, np.ndarray] | None = None,
) -> list[Placement]:
    """The floor and the `walls` as placements; the walls are also saved to walls.json, the floor to floors.json.

    Walls are not painted from the photo, which would paint in the shadows and
    the edges of what stands against them: those things are placed in front.
    Each wall's colour is the median of the `wall_pixels` that show it. The
    floor is the `free` pixels (no object or built-in) whose points lie on the
    floor plane: a segmenter asked for "floor" took the bedroom's rug in the
    next room, through a doorway, for all of it. With a `floor_survey` and its
    `floor_masks`, the floor's materials are told apart and each drawn as a
    texture with `draw` (see `stages.floor`); without, it is one colour.

    With the survey's `wall_masks`, the pixels at each wall's foot (`wall_foot_pixels`) are written to
    masks/wall_foot/<wall>.png for the survey wall whose mask holds most of the fitted wall's own pixels: the
    segmenter's wall mask stops at the baseboard, and the label map adds these to it.
    """
    camera_position = camera.matrix[:3, 3]
    assets_dir = run_dir / "assets"
    assets_dir.mkdir(exist_ok=True)
    objects: list[Placement] = []

    with np.errstate(invalid="ignore"):
        floor_pixels = free & (np.abs(points_world[..., 2]) < _ON_FLOOR_M)
    regions = floor_regions(run_dir, image, points_world, camera, walls, floor_pixels, floor_survey, floor_masks or {}, draw)
    for region in regions:
        # Placed untransformed: the slab's corners are in the world frame.
        name = "floor" if region.kind == "floor" else f"floor_{region.id}"
        write_floor(
            assets_dir / f"{name}.usda", region.polygon, region.top_m, 0.02 if region.kind == "floor" else region.top_m,
            region.color, Path(region.texture) if region.texture else None, region.tile_m or 1.0, region.angle_deg,
        )
        objects.append(Placement(name=name, asset=f"./assets/{name}.usda"))
    (run_dir / "floors.json").write_text(json.dumps([region.model_dump() for region in regions], indent=2))

    ceiling = _ceiling(run_dir, image, points_world, camera, walls)
    if ceiling is not None:
        # The walls stand from the floor to the ceiling.
        walls = [wall.model_copy(update={"size": (wall.size[0], wall.size[1], ceiling.height_m)}) for wall in walls]
        objects.append(Placement(name="ceiling", asset="./assets/ceiling.usda"))

    colors, sightings = {}, {}
    for wall in walls:
        origin, across, _ = wall_face(wall, camera_position)
        # The face stands behind the wall's layer of points, so the whole layer counts towards its colour.
        _, colors[wall.id] = surface_texture(
            image, points_world, camera, origin, across, np.array([0.0, 0.0, wall.size[2]]), allowed=wall_pixels, reach=WALL_REACH_M,
        )
        sightings[wall.id] = find_baseboard(
            image, points_world, wall, camera_position, free & ~floor_pixels, wall_pixels, colors[wall.id]
        )
    baseboards = room_baseboards(sightings)
    if wall_masks is not None:
        _save_wall_feet(run_dir, points_world, walls, camera_position, free, wall_pixels, wall_masks)
    saved = []
    for wall in walls:
        baseboard = baseboards[wall.id]
        color = colors[wall.id]
        wall = glaze(wall, image, points_world, camera)
        for opening in wall.openings:
            if opening.window:
                frame = f"a frame {opening.frame_width * 100:.0f} cm wide, its edge {opening.frame_out * 100:+.0f} cm out" if opening.frame else "no frame"
                print(f"{wall.id}: {opening.id} glazed, {frame}", flush=True)
        _, _, normal = wall_face(wall, camera_position)
        yaw = np.radians(wall.yaw_deg)
        room_side = 1.0 if normal @ np.array([-np.sin(yaw), np.cos(yaw), 0.0]) > 0 else -1.0
        write_wall(
            assets_dir / f"{wall.id}.usda", wall.size, tuple(float(c) for c in color),
            (baseboard.height, BASEBOARD_THICKNESS_M, baseboard.color) if baseboard else None, room_side, opening_spans(wall),
        )
        saved.append(wall.model_dump() | {"baseboard": baseboard.model_dump() if baseboard else None})
        if baseboard:
            seen = sightings[wall.id].baseboard
            print(f"{wall.id}: baseboard {baseboard.height * 100:.0f} cm high" + ("" if seen == baseboard else ", as seen best on another wall"), flush=True)
        objects.append(
            Placement(
                name=wall.id,
                asset=f"./assets/{wall.id}.usda",
                translate=wall.translate,
                rotate_xyz_deg=(0.0, 0.0, wall.yaw_deg),
            )
        )
    (run_dir / "walls.json").write_text(json.dumps(saved, indent=2))
    return objects


def _save_wall_feet(
    run_dir: Path, points_world: np.ndarray, walls: list[Proxy], eye: np.ndarray, free: np.ndarray, wall_pixels: np.ndarray,
    wall_masks: dict[str, np.ndarray],
) -> None:
    """Write masks/wall_foot/<survey wall>.png: the pixels at each fitted wall's foot (`wall_foot_pixels`), given to the
    survey wall whose mask holds most of that fitted wall's own pixels (those on its surface).

    Also masks/wall_surface/<survey wall>.png: the wall pixels that lie on one fitted wall's surface and clearly off
    every other's (`SAME_DEPTH_M` further), given to the same survey wall. The survey's masks follow the paint, not the
    walls: OfficeScan 005's `back_wall` was a patch of the same wall as `left_wall`, and `left_wall` ran round the
    corner onto the right wall, while the shell fitted two walls. The label map gives each such pixel to the survey wall
    of the fitted wall it lies on. Where two walls meet, or a wall steps, a pixel near both is left to its own mask."""
    from aomma.labels import SAME_DEPTH_M

    folders = {"foot": run_dir / "masks" / "wall_foot", "surface": run_dir / "masks" / "wall_surface"}
    for folder in folders.values():
        shutil.rmtree(folder, ignore_errors=True)
    offsets = [wall_surface_offset(points_world, wall, eye, wall_pixels, wall_pixels) for wall in walls]
    shown: dict[str, dict[str, np.ndarray]] = {"foot": {}, "surface": {}}
    for i, wall in enumerate(walls):
        pixels = wall_foot_pixels(points_world, wall, eye, free, wall_pixels)
        own = wall_surface_pixels(points_world, wall, eye, wall_pixels, wall_pixels)
        owner = max(wall_masks, key=lambda oid: int((wall_masks[oid] & own).sum()), default=None)
        if owner is None or not (wall_masks[owner] & own).any():
            continue
        others = np.min([o for j, o in enumerate(offsets) if j != i], axis=0) if len(walls) > 1 else np.inf
        clearly = own & (offsets[i] + SAME_DEPTH_M <= others)
        for kind, found in (("foot", pixels), ("surface", clearly)):
            if found.any():
                shown[kind][owner] = shown[kind].get(owner, np.zeros_like(found)) | found
    for kind, by_wall in shown.items():
        if by_wall:
            folders[kind].mkdir(parents=True)
        for oid, pixels in by_wall.items():
            Image.fromarray(pixels).save(folders[kind] / f"{oid}.png")
            print(f"{oid}: {kind} {int(pixels.sum())} px", flush=True)


def layout(run_dir: Path) -> None:
    """Build, render and score a proxy-box scene from a perceived run (milestone M2)."""
    image = Image.open(run_dir / "input.jpg")
    inventory = Inventory.model_validate_json((run_dir / "inventory.json").read_text())
    intrinsics = Intrinsics.model_validate_json((run_dir / "camera.json").read_text())
    points = np.load(run_dir / "points.npy")
    floor = np.asarray(Image.open(run_dir / "floor.png")) > 0
    masks, unusable = _usable_masks(run_dir, inventory)

    def fit(points: np.ndarray) -> tuple[np.ndarray, list]:
        camera_to_world = fit_camera_to_world(points, floor, np.any(list(masks.values()), axis=0))
        # Fit each box to the object's own pixels, but score it against the whole
        # mask: a shelf's box should cover the shelf including what stands on it.
        proxies = fit_proxies(
            image, to_world(points, camera_to_world), exclusive_masks(masks), inventory, camera_to_world[:3, 3]
        )
        return camera_to_world, proxies

    # Fit once to measure the objects, correct the depth map's overall scale
    # from their expected sizes, then fit again at the corrected scale.
    depth_scale = depth_scale_correction(fit(points)[1], inventory)
    (run_dir / "world.json").write_text(json.dumps({"depth_scale": depth_scale}))
    camera_to_world, proxies = fit(_load_points(run_dir))
    camera = Camera(
        camera_to_world=tuple(camera_to_world.flatten()),
        focal_length_mm=intrinsics.focal_length_mm,
        sensor_width_mm=intrinsics.sensor_width_mm,
        width=intrinsics.width,
        height=intrinsics.height,
    )

    points_world = to_world(_load_points(run_dir), camera_to_world)
    camera_position = camera_to_world[:3, 3]

    # Walls: whatever is neither an object nor the floor, well clear of the ground.
    occupied = np.any(list(masks.values()), axis=0)
    with np.errstate(invalid="ignore"):
        background = ~occupied & ~floor & (points_world[..., 2] > 0.15)
        shown = points_world[occupied & np.isfinite(points_world).all(axis=-1)]
    walls = fit_walls(image, points_world, background, camera_position, in_front=shown)
    objects = _build_room(run_dir, image, points_world, camera, walls, ~occupied, ~occupied & ~floor)
    # The photo shows every object, so no wall may hide one; what still is hidden is seen through an opening.
    faces = [(*wall_face(wall, camera_position), wall.size[2]) for wall in walls]
    for object_id, mask in masks.items():
        seen = points_world[mask & np.isfinite(points_world).all(axis=-1)]
        if len(seen) and (share := hidden_by_walls(seen, faces, camera_position)) >= 0.2:
            print(f"{object_id}: {share:.0%} of what the photo shows of it is behind a wall (seen through an opening?)", flush=True)

    dropped = {oid for oid, entry in overrides(run_dir).items() if entry.get("drop")}
    for proxy in proxies:
        if proxy.id in dropped:
            continue  # the review judged that nothing real is there
        write_proxy_box(run_dir / "assets" / f"{proxy.id}.usda", proxy.size, proxy.color)
        placement = Placement(
            name=proxy.id,
            asset=f"./assets/{proxy.id}.usda",
            translate=proxy.translate,
            rotate_xyz_deg=(0.0, 0.0, proxy.yaw_deg),
        )
        # Keep a generated model if there is one; `aomma pose` then refines its placement.
        if (run_dir / "assets" / proxy.id / f"{proxy.id}.usdc").exists():
            placement.asset = f"./assets/{proxy.id}/{proxy.id}.usdc"
            placement.scale = max(proxy.size)
        objects.append(placement)
    scene = SceneLayout(camera=camera, objects=objects)

    occlusions = find_occlusions(exclusive_masks(masks), _load_points(run_dir), inventory)
    (run_dir / "occlusion.json").write_text(
        json.dumps({oid: found.model_dump() for oid, found in occlusions.items()}, indent=2)
    )
    scores = score_layout(proxies, masks, camera)
    (run_dir / "proxies.json").write_text(json.dumps([p.model_dump() for p in proxies], indent=2))
    (run_dir / "layout.json").write_text(scene.model_dump_json(indent=2))
    (run_dir / "scores.json").write_text(json.dumps(scores, indent=2))
    draw_wireframes(image, proxies, camera).save(run_dir / "debug" / "layout.jpg")

    _compose_and_render(run_dir, scene)

    skipped = [o.id for o in inventory.objects if o.id not in scores and o.id not in unusable]
    print(f"depth scale x{depth_scale:.2f}, camera {camera_to_world[2, 3]:.2f} m above the floor")
    print(f"{len(proxies)} proxies, mean silhouette IoU {np.mean(list(scores.values())):.2f}, {len(walls)} walls")
    if skipped:
        print(f"skipped (no 3D points): {', '.join(skipped)}")
    if unusable:
        print(f"skipped (mask nowhere near its box): {', '.join(unusable)}")
    print(f"-> {run_dir}/scene.usda, debug/layout.jpg, debug/compare.jpg")



def shell(run_dir: Path, config_path: Path, vlm_name: str | None) -> None:
    """The first stage of a staged run: survey the photo, then build the room alone (floor, walls, built-ins) and measure it.

    A run surveyed before is only built again, from the same survey, masks
    and depth, so a change to the building can be compared on equal terms.
    """
    if not (run_dir / "camera.json").exists():
        if not _survey_room(run_dir, config_path, vlm_name):
            return
    config = load_config(config_path)
    if not (run_dir / "floor_survey.json").exists():
        _survey_floor(run_dir, config, vlm_name)
    if not (run_dir / "faces.json").exists():
        _built_in_faces(run_dir, config, vlm_name)
    comfy = comfy_client(config.comfy.light[0], run_dir)

    def draw(crop: Image.Image, material: str, name: str, grained: bool) -> Image.Image:
        return draw_texture(comfy, config.floor_texture_workflow, crop, material, f"{run_dir.name}_{name}", grained)

    _build_shell(run_dir, draw, comfy)


def large(run_dir: Path, config_path: Path, vlm_name: str | None) -> None:
    """The second stage of a staged run: list, model and place the furniture and appliances inside the shell.

    Builds the shell first if the run has none. Writes large_survey.json and
    the items' masks; rebuilds the shell (never rescaled from the items' estimated sizes, which were too often
    wrong); adds the items to inventory.json, proxies.json
    and the layout; generates and poses their models, held to the shell (see
    `stages.large`); and writes large.json, the stage's measures. Models and
    the survey already made are reused, so a run left part-way picks up.
    """
    if not (run_dir / "shell.json").exists():
        shell(run_dir, config_path, vlm_name)
        if not (run_dir / "shell.json").exists():
            return  # not indoors
    config = load_config(config_path)
    image = Image.open(run_dir / "input.jpg")
    survey = Survey.model_validate_json((run_dir / "survey.json").read_text())
    if not (run_dir / "large_survey.json").exists():
        _survey_large(run_dir, image, survey, config, vlm_name)
    listed = LargeSurvey.model_validate_json((run_dir / "large_survey.json").read_text())
    usable = [o for o in listed.objects if listed.box_iou.get(o.id, 0) >= MIN_USABLE_BOX_IOU]
    furniture = Inventory(objects=usable)
    masks = {o.id: np.asarray(Image.open(run_dir / "masks" / f"{o.id}.png")) > 0 for o in usable}

    comfy = comfy_client(config.comfy.light[0], run_dir)

    def draw(crop: Image.Image, material: str, name: str, grained: bool) -> Image.Image:
        return draw_texture(comfy, config.floor_texture_workflow, crop, material, f"{run_dir.name}_{name}", grained)

    # Rebuilt, so that a run carried on starts again from the shell's inventory and layout.
    print("== shell, rebuilt", flush=True)
    _build_shell(run_dir, draw, comfy)

    # The items' boxes, in the shell's frame, each fitted to the pixels no built-in or smaller item claims.
    scene = SceneLayout.model_validate_json((run_dir / "layout.json").read_text())
    built = Inventory.model_validate_json((run_dir / "inventory.json").read_text())
    built_masks = {o.id: np.asarray(Image.open(run_dir / "masks" / f"{o.id}.png")) > 0 for o in built.objects}
    points_world = to_world(_load_points(run_dir), scene.camera.matrix)
    owned = exclusive_masks(built_masks | masks)
    proxies = fit_proxies(image, points_world, {oid: owned[oid] for oid in masks}, furniture, scene.camera.matrix[:3, 3])
    proxies = _roomplan_priors(run_dir, proxies, points_world, owned, listed.kinds)
    fitted = {p.id for p in proxies}
    inventory = Inventory(objects=built.objects + [o for o in usable if o.id in fitted])
    (run_dir / "inventory.json").write_text(inventory.model_dump_json(indent=2))
    saved = json.loads((run_dir / "proxies.json").read_text())
    (run_dir / "proxies.json").write_text(json.dumps(saved + [p.model_dump() for p in proxies], indent=2))
    for proxy in proxies:
        write_proxy_box(run_dir / "assets" / f"{proxy.id}.usda", proxy.size, proxy.color)
        scene.objects.append(Placement(
            name=proxy.id, asset=f"./assets/{proxy.id}.usda", translate=proxy.translate, rotate_xyz_deg=(0.0, 0.0, proxy.yaw_deg),
        ))
    (run_dir / "layout.json").write_text(scene.model_dump_json(indent=2))
    # Small items already listed (a staged run lists them before any model) stand on and in front of these: their
    # pixels are cut out of the large items' masks, so they count as occluders, or a table is modelled with holes
    # where its cups stood (20261124-153236-Coffee).
    small_listed = _listed_small(run_dir, exclude={o.id for o in inventory.objects})
    small_masks = {o.id: np.asarray(Image.open(run_dir / "masks" / f"{o.id}.png").convert("L")) > 127 for o in small_listed}
    occlusions = find_occlusions(
        exclusive_masks(built_masks | masks | small_masks), _load_points(run_dir), Inventory(objects=inventory.objects + small_listed),
    )
    occlusions = {oid: found for oid, found in occlusions.items() if oid not in small_masks}
    # What stands inside an item (a shelf's books) is stage 3's: until then its pixels are judged as an occluder's.
    for oid in sorted(fitted):
        holes = contents_holes(masks[oid], listed.kinds.get(oid))
        if holes is not None:
            Image.fromarray(holes).save(run_dir / "masks" / f"{contents_id(oid)}.png")
            occlusions[oid].occluders.append(contents_id(oid))
            print(f"{oid}: {holes.sum() / (holes.sum() + masks[oid].sum()):.0%} of its outline holds things, left to stage 3", flush=True)
    (run_dir / "occlusion.json").write_text(json.dumps({oid: found.model_dump() for oid, found in occlusions.items()}, indent=2))
    _compose_and_render(run_dir, scene)
    unfitted = [o.id for o in usable if o.id not in fitted]
    dropped = [o.id for o in listed.objects if o not in usable]
    print(
        f"{len(proxies)} large items: {', '.join(sorted(fitted)) or '-'}"
        + (f"; no 3D points: {', '.join(unfitted)}" if unfitted else "")
        + (f"; mask nowhere near its box: {', '.join(dropped)}" if dropped else ""),
        flush=True,
    )

    print("== models", flush=True)
    assets(run_dir, sorted(fitted), config_path)
    _measure_large(run_dir, sorted(fitted), listed)


# How far in plan the outline may move an item from RoomPlan's box (`_hold_to_shell` with `prior`): the margin the
# match allows (`roomplan.IN_BOX_M`).
PRIOR_SHIFT_M = 0.10


def _roomplan_priors(run_dir: Path, proxies: list[Proxy], points_world: np.ndarray, masks: dict[str, np.ndarray], kinds: dict[str, str]) -> list[Proxy]:
    """For a run given RoomPlan's object boxes (`depth+objects`), each item's proxy replaced by the box it matches
    (`roomplan.match_objects`): RoomPlan's furniture and the fixtures the shell left to stage 2. Records the matches,
    the refusals and the boxes no listed item took in roomplan.json (`priors`; `objects_unlisted`, those the shot sees,
    and `objects_out_of_view`)."""
    from aomma.capture import inputs
    from aomma.stages import roomplan as rp

    given = inputs(run_dir)
    if "roomplan_objects" not in given:
        return proxies
    record = json.loads((run_dir / "roomplan.json").read_text())
    scan_to_run = np.array(record["scan_to_run"])
    boxes = rp.boxes_from_room({"objects": given["roomplan_objects"]}, scan_to_run, rp.FURNITURE_KINDS)
    boxes += [b for b in rp.boxes_from_room(given["roomplan"], scan_to_run) if b[1].id in record.get("left_to_stage_2", [])]
    valid = np.isfinite(points_world).all(axis=-1)
    items = {p.id: (kinds.get(p.id), points_world[masks[p.id] & valid]) for p in proxies if p.id in masks}
    found = rp.match_objects(items, boxes)
    by_box = {box.id: box for _, box in boxes}
    priors = {f["item"]: f for f in found if f["item"] is not None}
    out = []
    for proxy in proxies:
        match = priors.get(proxy.id)
        if match is None:
            print(f"{proxy.id} ({kinds.get(proxy.id)}): no RoomPlan box, its own", flush=True)
            out.append(proxy)
        elif match["refused"]:
            print(f"{proxy.id}: RoomPlan's {match['box']} refused, its points reach past it ({match['refused']}); its own", flush=True)
            out.append(proxy)
        else:
            box = by_box[match["box"]]
            print(f"{proxy.id}: RoomPlan's {match['box']}, {box.size[0]:.2f} x {box.size[1]:.2f} x {box.size[2]:.2f} m "
                  f"({match['share']:.0%} of its points in it; its own {proxy.size[0]:.2f} x {proxy.size[1]:.2f} x {proxy.size[2]:.2f})", flush=True)
            out.append(box.model_copy(update={"id": proxy.id, "color": proxy.color}))
    unlisted, out_of_view = rp.unlisted_seen(found, boxes, points_world[valid])
    if unlisted:
        print(f"RoomPlan's boxes the shot sees and no listed item took: {', '.join(unlisted)}", flush=True)
    own = {p.id: p.model_dump() for p in proxies}
    record |= {
        "priors": {oid: match | {"own": own[oid]} for oid, match in priors.items()},
        "objects_unlisted": unlisted,
        "objects_out_of_view": out_of_view,
    }
    (run_dir / "roomplan.json").write_text(json.dumps(record, indent=2))
    return out


def roomplan_priors(run_dir: Path) -> dict[str, dict]:
    """The items whose proxy is RoomPlan's box, with their match (its `own` proxy kept; see `_roomplan_priors`)."""
    path = run_dir / "roomplan.json"
    if not path.exists():
        return {}
    return {oid: match for oid, match in json.loads(path.read_text()).get("priors", {}).items() if not match["refused"]}


def _listed_small(run_dir: Path, exclude: set[str] = frozenset()) -> list[InventoryObject]:
    """The small items small_survey.json lists that have a mask, leaving out `exclude`; none before gate 3."""
    path = run_dir / "small_survey.json"
    if not path.exists():
        return []
    listed = [InventoryObject(**o) for o in json.loads(path.read_text())["objects"]]
    return [o for o in listed if o.id not in exclude and (run_dir / "masks" / f"{o.id}.png").exists()]


def small(run_dir: Path, config_path: Path, vlm_name: str | None) -> None:
    """Gate 3: list and segment the small items of a run whose shell and large items are made; no model is made.

    Writes small_survey.json, masks/<id>.png, debug/small_masks.jpg and debug/small/ (see `stages.small`).
    """
    from aomma.stages.small import survey_small

    if not (run_dir / "survey.json").exists():
        raise SystemExit(f"{run_dir}: no shell yet; run `aomma shell {run_dir}` first")
    config = load_config(config_path)
    ledger = Ledger(run_dir / "usage.jsonl", config.max_billed_tokens)
    result = survey_small(run_dir, vlm_for(config, vlm_name, run_dir), ledger, comfy_client(config.comfy.light[0], run_dir))
    print(f"small items: {len(result.objects)} kept over {len(result.supports)} supports and the room"
          + (f"; {len(result.merged)} found twice, kept once" if result.merged else ""), flush=True)


# `aomma staged`'s steps, in order, as staged.json records them done.
STAGED_STEPS = ("shell", "large-survey", "shell-again", "small", "labels", "relook", "boxes")


def staged(run_dir: Path, config_path: Path, vlm_name: str | None) -> None:
    """A photo through the layout-first gates' build steps, as a round runs them (lab/fresh_runs.toml), no models:
    the shell; the large items' listing and masks (the first half of `large`); the shell again with those masks;
    the small items; the labels; the relook; the box layout. Gate 5 (models) is left to `aomma small-models` and
    the like once gate 4 holds.

    Each step done is written to staged.json, so a run stopped part-way picks up where it left off.
    """
    from aomma.labels import describe, label_run
    from aomma.stages.relook import describe as describe_relook, predict, relook

    record = run_dir / "staged.json"
    done = json.loads(record.read_text())["done"] if record.exists() else []
    config = load_config(config_path)
    for step in STAGED_STEPS:
        if step in done:
            continue
        print(f"== {step}", flush=True)
        if step in ("shell", "shell-again"):
            shell(run_dir, config_path, vlm_name)
            if not (run_dir / "shell.json").exists():
                print(f"{run_dir}: no shell built (not indoors?); stopping", flush=True)
                return
        elif step == "large-survey":
            if not (run_dir / "large_survey.json").exists():
                survey = Survey.model_validate_json((run_dir / "survey.json").read_text())
                _survey_large(run_dir, Image.open(run_dir / "input.jpg"), survey, config, vlm_name)
        elif step == "small":
            small(run_dir, config_path, vlm_name)
        elif step == "labels":
            print(describe(run_dir, label_run(run_dir)), flush=True)
        elif step == "relook":
            predict([run_dir])
            ledger = Ledger(run_dir / "usage.jsonl", config.max_billed_tokens)
            result = relook(run_dir, vlm_for(config, vlm_name, run_dir), ledger, comfy_client(config.comfy.light[0], run_dir))
            print(describe_relook(run_dir, result), flush=True)
        elif step == "boxes":
            box_layout(run_dir)
        done.append(step)
        record.write_text(json.dumps({"done": done}))
    print(f"-> {run_dir}: shell, large and small items, labels and box layout (no models); see debug/labels.jpg", flush=True)


def small_models(run_dir: Path, object_ids: list[str], config_path: Path) -> None:
    """Gate 5 for small items: generate and pose a model for each small item the box layout built as a box
    (`aomma boxes`), through the same path as the large items' (`assets`). An item thinner than `labels.SAME_DEPTH_M`
    on any side stays a box: the depth can't show more of its shape than the box does. Each item joins the run's
    inventory, its gate 4 box as its proxy (placed in the layout), and its occlusions are found among every mask.
    With `object_ids`, only those."""
    from aomma.labels import SAME_DEPTH_M

    if not (run_dir / "boxes.json").exists():
        raise SystemExit(f"{run_dir}: no box layout yet; run `aomma boxes {run_dir}` first")
    small = {o["id"]: InventoryObject(**o) for o in json.loads((run_dir / "small_survey.json").read_text())["objects"]}
    boxes = {b["proxy"]["id"]: b for b in json.loads((run_dir / "boxes.json").read_text())["boxes"] if b["kind"] == "small"}
    chosen = [oid for oid, b in boxes.items() if oid in small and (not object_ids or oid in object_ids)
              and min(b["proxy"]["size"]) >= SAME_DEPTH_M]
    thin = sorted(oid for oid, b in boxes.items() if oid in small and (not object_ids or oid in object_ids)
                  and min(b["proxy"]["size"]) < SAME_DEPTH_M)
    print(f"small items to model: {len(chosen)}" + (f"; thinner than {SAME_DEPTH_M * 100:.0f} cm, kept as boxes: {', '.join(thin)}" if thin else ""), flush=True)
    # A thin item joins the scene as its box; it was left out altogether before (a tray, say).
    kept = chosen + thin
    if not kept:
        return
    inventory = Inventory.model_validate_json((run_dir / "inventory.json").read_text())
    known = {o.id for o in inventory.objects}
    inventory.objects += [small[oid] for oid in kept if oid not in known]
    (run_dir / "inventory.json").write_text(inventory.model_dump_json(indent=2))
    proxies = json.loads((run_dir / "proxies.json").read_text())
    have = {p["id"] for p in proxies}
    proxies += [boxes[oid]["proxy"] for oid in kept if oid not in have]
    (run_dir / "proxies.json").write_text(json.dumps(proxies, indent=2))
    scene = SceneLayout.model_validate_json((run_dir / "layout.json").read_text())
    placed = {o.name for o in scene.objects}
    for oid in kept:
        if oid in placed:
            continue
        proxy = Proxy(**boxes[oid]["proxy"])
        write_proxy_box(run_dir / "assets" / f"{oid}.usda", proxy.size, proxy.color)
        scene.objects.append(Placement(name=oid, asset=f"./assets/{oid}.usda", translate=proxy.translate,
                                       rotate_xyz_deg=(0.0, 0.0, proxy.yaw_deg)))
    (run_dir / "layout.json").write_text(scene.model_dump_json(indent=2))
    masks = {o.id: np.asarray(Image.open(run_dir / "masks" / f"{o.id}.png").convert("L")) > 127
             for o in inventory.objects if (run_dir / "masks" / f"{o.id}.png").exists()}
    found = find_occlusions(exclusive_masks(masks), _load_points(run_dir), Inventory(objects=[o for o in inventory.objects if o.id in masks]))
    saved = json.loads((run_dir / "occlusion.json").read_text()) if (run_dir / "occlusion.json").exists() else {}
    saved |= {oid: found[oid].model_dump() for oid in chosen if oid in found}
    (run_dir / "occlusion.json").write_text(json.dumps(saved, indent=2))
    if not chosen:
        _compose_and_render(run_dir, scene)
        return
    assets(run_dir, chosen, config_path)


def draw_box_layout(image: Image.Image, built_ins: list[Proxy], boxes: list, flats: list, camera: Camera) -> Image.Image:
    """The whole box layout over the photo: every built-in's and item's box edges, coloured by what it is (built-ins
    green, large items blue, small items yellow), and each sheet's outline (in violet)."""
    from PIL import ImageDraw

    colors = {p.id: BUILT_IN_EDGE for p in built_ins} | {b.proxy.id: LARGE_EDGE if b.kind == "large" else SMALL_EDGE for b in boxes}
    drawn = draw_wireframes(image, built_ins + [b.proxy for b in boxes], camera, colors).convert("RGB")
    pen = ImageDraw.Draw(drawn)
    for flat in flats:
        pen.polygon([tuple(xy) for xy in flat.outline_px], outline=(190, 80, 220), width=2)
    return drawn


def box_layout(run_dir: Path) -> None:
    """Gate 4: every large and small item as a box in the shell, stood on its support (see `stages.boxes`); no model.

    Each small item is first given its form (`stages.forms`): a sheet (more of it a layer on a surface than not) is
    painted from the photo onto that surface, and gets no box.

    Takes the label map (`aomma labels`) and the shell's layout; leaves layout.json and scene.usda as the shell made
    them. Writes forms.json, assets/flats/<id>.usda and .png, boxes.json, assets/<id>.usda, boxes_layout.json,
    boxes.usda, boxes.png, debug/boxes.jpg (the built-ins' and the items' box edges over the photo, the flats' outlines),
    debug/boxes_compare.jpg (photo beside render).
    """
    from aomma.stages.boxes import fit_boxes
    from aomma.stages.forms import BOX, SHEET, built_on, decide_forms, fit_flat, flat_texture, sheet_pixels
    from aomma.stages.inventory import InventoryObject
    from aomma.usd.assets import write_painted_sheet

    if not (run_dir / "labels.json").exists():
        raise SystemExit(f"{run_dir}: no label map yet; run `aomma labels {run_dir}` first")
    image = Image.open(run_dir / "input.jpg").convert("RGB")
    scene = SceneLayout.model_validate_json((run_dir / "layout.json").read_text())
    built_ins = [Proxy(**p) for p in json.loads((run_dir / "proxies.json").read_text())]
    objects = {
        o["id"]: InventoryObject(**o)
        for name in ("large_survey.json", "small_survey.json") if (run_dir / name).exists()
        for o in json.loads((run_dir / name).read_text())["objects"]
    }
    labels = json.loads((run_dir / "labels.json").read_text())["labels"]
    label_map = np.asarray(Image.open(run_dir / "labels.png"))
    points_world = to_world(_load_points(run_dir), scene.camera.matrix)
    layers_path = run_dir / "debug" / "labels_layers.png"
    layer_map = np.asarray(Image.open(layers_path)) if layers_path.exists() else np.zeros_like(label_map)
    eye = scene.camera.matrix[:3, 3]
    forms = decide_forms(label_map, layer_map, labels, points_world, eye)
    try:
        source = load_source(run_source(run_dir))
    except (FileNotFoundError, KeyError):
        source = image
    value = {e["id"]: e["value"] for e in labels}
    sheet_ids = {f.id for f in forms.values() if f.form == SHEET}
    # Each large item's own mask, as the segmenter made it (`stages.boxes.fit_boxes`).
    large_ids = {e["id"] for e in labels if e["class"] == "large"}
    own_masks = {oid: np.asarray(Image.open(run_dir / "masks" / f"{oid}.png").convert("L").resize(image.size, Image.NEAREST)) > 127
                 for oid in large_ids if (run_dir / "masks" / f"{oid}.png").exists()}
    found, skipped = fit_boxes(image, points_world, label_map, labels, objects, built_ins, eye, sheet_ids, own_masks)
    # What a sheet's surface may be built as: the built-ins' and the items' boxes just fitted, and the walls.
    built = {p.id: p for p in built_ins} | {b.proxy.id: b.proxy for b in found}
    walls_path = run_dir / "walls.json"
    walls = [Proxy(**w) for w in json.loads(walls_path.read_text())] if walls_path.exists() else []
    flats, flat_placements, unbuilt = [], [], []
    flat_dir = run_dir / "assets" / "flats"
    for form in forms.values():
        if form.form != SHEET:
            continue
        mask_path = run_dir / "masks" / f"{form.id}.png"
        mask = None
        if mask_path.exists():
            mask_image = Image.open(mask_path).convert("L")
            mask = np.asarray(mask_image.resize(label_map.shape[::-1], Image.NEAREST) if mask_image.size != label_map.shape[::-1]
                              else mask_image) > 127
        pixels = sheet_pixels(mask, label_map, layer_map, value[form.id], value.get(form.surface))
        flat = fit_flat(form, pixels, scene.camera, built_on(form, built, walls))
        if flat is None:
            unbuilt.append(form.id)
            continue
        flats.append(flat)
        flat_dir.mkdir(parents=True, exist_ok=True)
        flat_texture(source, pixels, flat, scene.camera).save(flat_dir / f"{form.id}.png")
        pivot = np.array(flat.corners).mean(axis=0)
        write_painted_sheet(flat_dir / f"{form.id}.usda", [tuple(np.array(c) - pivot) for c in flat.corners],
                            flat_dir / f"{form.id}.png")
        flat_placements.append(Placement(name=form.id, asset=f"./assets/flats/{form.id}.usda",
                                         translate=tuple(float(x) for x in pivot)))
    (run_dir / "forms.json").write_text(json.dumps({"forms": [f.model_dump() for f in forms.values()],
                                                    "flats": [f.model_dump() for f in flats], "unbuilt": unbuilt},
                                                   indent=2))

    # The shell's own placements, without any item an earlier stage put in the layout.
    placed = {e["id"] for e in labels if e["class"] in ("large", "small")}
    shell_placements = [o for o in scene.objects if o.name not in placed]
    placements = []
    for box in found:
        proxy = box.proxy
        write_proxy_box(run_dir / "assets" / f"{proxy.id}.usda", proxy.size, proxy.color)
        placements.append(Placement(
            name=proxy.id, asset=f"./assets/{proxy.id}.usda", translate=proxy.translate, rotate_xyz_deg=(0.0, 0.0, proxy.yaw_deg),
        ))
    layout = SceneLayout(camera=scene.camera, objects=shell_placements + placements + flat_placements)
    (run_dir / "boxes.json").write_text(json.dumps({"boxes": [b.model_dump() for b in found], "no_points": skipped}, indent=2))
    (run_dir / "boxes_layout.json").write_text(layout.model_dump_json(indent=2))
    draw_box_layout(image, built_ins, found, flats, scene.camera).save(run_dir / "debug" / "boxes.jpg", quality=88)
    compose_scene(layout, run_dir / "boxes.usda")
    camera = scene.camera
    render(run_dir / "boxes.usda", run_dir / "boxes.png", camera.width, camera.height)
    compare = Image.new("RGB", (camera.width * 2, camera.height))
    compare.paste(image, (0, 0))
    compare.paste(Image.open(run_dir / "boxes.png").convert("RGB"), (camera.width, 0))
    compare.save(run_dir / "debug" / "boxes_compare.jpg", quality=88)

    counts = {k: sum(b.kind == k for b in found) for k in ("large", "small")}
    stood = {s: sum(b.stood == s for b in found) for s in ("floor", "top", "level", "inside", "own")}
    shapes = {k: sum(f.form == k for f in forms.values()) for k in (SHEET, BOX)}
    print(f"small items' forms: {shapes[SHEET]} sheets, {shapes[BOX]} boxes"
          + (f" (sheets: {', '.join(f.id for f in flats)})" if flats else "")
          + (f"; no sheet fits its plane: {', '.join(unbuilt)}" if unbuilt else ""), flush=True)
    print(f"boxes: {counts['large']} large, {counts['small']} small; on the floor {stood['floor']}, on a support's top "
          f"{stood['top']}, on a lower level of one {stood['level']}, inside a support {stood['inside']}, as measured {stood['own']}"
          + (f"; no 3D points: {', '.join(skipped)}" if skipped else ""), flush=True)
    print(f"-> {run_dir}/boxes.json, boxes.usda, debug/boxes.jpg, debug/boxes_compare.jpg", flush=True)


def _survey_large(run_dir: Path, image: Image.Image, survey: Survey, config: Config, vlm_name: str | None) -> None:
    """List the large items given the shell's survey, and segment each from its box; writes large_survey.json and masks/<id>.png."""
    vlm = vlm_for(config, vlm_name, run_dir)
    ledger = Ledger(run_dir / "usage.jsonl", config.max_billed_tokens)
    taken = {o.id for o in survey.walls + survey.built_ins + survey.openings}
    listed, raw = take_large(image, survey.room_type, survey.built_ins, taken, vlm, ledger)
    (run_dir / "debug" / "large_survey_raw.txt").write_text(raw)
    print(f"large items: {', '.join(f'{o.id} ({listed.kinds[o.id]}, on {o.on})' for o in listed.objects) or '-'}", flush=True)
    if listed.objects:
        draw_inventory(image, Inventory(objects=listed.objects)).save(run_dir / "debug" / "large_inventory.jpg")
        comfy = comfy_client(config.comfy.light[0], run_dir)
        image_name = comfy.upload_image(to_jpeg(image), f"aomma_{run_dir.name}.jpg")
        masks, infos = segment(comfy, image_name, image.size, Inventory(objects=listed.objects))
        for oid, fill in box_fills(Inventory(objects=listed.objects), masks).items():
            infos[oid].fill = fill
        for oid, mask in masks.items():
            Image.fromarray(mask).save(run_dir / "masks" / f"{oid}.png")
        saved = json.loads((run_dir / "masks.json").read_text())
        (run_dir / "masks.json").write_text(json.dumps(saved | {oid: info.model_dump() for oid, info in infos.items()}, indent=2))
        listed.box_iou = {oid: info.box_iou for oid, info in infos.items()}
        draw_masks(image, masks).save(run_dir / "debug" / "large_masks.jpg")
    (run_dir / "large_survey.json").write_text(listed.model_dump_json(indent=2))


def _measure_large(run_dir: Path, object_ids: list[str], listed: LargeSurvey) -> dict:
    """Stage 2's measures, per item and for the run; writes large.json.

    Per item: its outline fit, the share a wall hides, how far its lowest
    surface is off its support, how far it is turned from square to the
    walls, and what holding it to the shell moved. For the run: the depth
    check of the whole scene, and the shell's own measures, which stage 2
    must not make worse.
    """
    scene = SceneLayout.model_validate_json((run_dir / "layout.json").read_text())
    placed = {o.name: o for o in scene.objects}
    poses = {p["id"]: p for p in json.loads((run_dir / "poses.json").read_text())} if (run_dir / "poses.json").exists() else {}
    items = {}
    for object_id in object_ids:
        placement = placed.get(object_id)
        if placement is None or object_id not in poses or not (run_dir / placement.asset).exists() or placement.asset.endswith(f"{object_id}.usda"):
            items[object_id] = {"kind": listed.kinds.get(object_id), "placed": False}
            continue
        facts = large_facts(run_dir, object_id)
        items[object_id] = {
            "kind": listed.kinds.get(object_id),
            **{key: facts[key] for key in (
                "on", "placed", "outline_fit", "hidden_by_wall", "off_support_m", "box_shaped", "seen_turn_deg", "off_square_deg",
                "longest_side_m", "size_vs_estimate", "inside_other", "inside_which", "shell_moves",
            )},
        }
    checked = depth_check(run_dir)
    measures = {
        "items": items,
        "world": json.loads((run_dir / "world.json").read_text()),
        "shell": json.loads((run_dir / "shell.json").read_text()),
        "depth_check": {"agree": checked["agree"], "room": checked["room"], "objects": checked["objects"]},
    }
    (run_dir / "large.json").write_text(json.dumps(measures, indent=2))
    for object_id, item in items.items():
        if item["placed"]:
            print(
                f"  {object_id:24s} fit {item['outline_fit']:.2f}, {item['hidden_by_wall']:.0%} behind a wall, "
                f"{item['off_support_m']:+.2f} m off its support, "
                + (f"{item['off_square_deg']} deg off square" if item["off_square_deg"] is not None
                   else "not box-shaped" if item["seen_turn_deg"] is not None else "square unknown from the photo")
                + (f" (stands {item['seen_turn_deg']:+.0f} deg off the walls)" if abs(item["seen_turn_deg"] or 0) > SQUARE_WITHIN_DEG else "")
            )
        else:
            print(f"  {object_id:24s} not placed")
    print(f"-> {run_dir}/large.json, debug/compare.jpg")
    return measures


def large_facts(run_dir: Path, object_id: str) -> dict:
    """`placement_facts` for a large item of a staged run, with what the shell adds.

    `off_support_m` is how far its lowest surface is off what it stands on (the
    floor, or its built-in's top), `box_shaped` how box-shaped its model is,
    `seen_turn_deg` how far the item's own faces in the photo turn from the walls
    (see `stages.large.seen_turn_deg`; None where they do not say), and
    `off_square_deg` how far its model is turned from the heading the photo
    shows: square to the walls for an item within `SQUARE_WITHIN_DEG` of them,
    its seen heading for one that stands at an angle. None for a round model
    (see `stages.large.square_turn_deg`), or where the photo does not say.
    `square_turn_deg` is the turn onto that heading, `placed` whether it stands in the scene, and
    `shell_moves` what holding it to the shell moved, or marked it as.
    """
    scene = SceneLayout.model_validate_json((run_dir / "layout.json").read_text())
    placement = next(o for o in scene.objects if o.name == object_id)
    solved = {p["id"]: p for p in json.loads((run_dir / "poses.json").read_text())}[object_id]
    proxies = {p["id"]: p for p in json.loads((run_dir / "proxies.json").read_text())}
    on = {o.id: o.on for o in LargeSurvey.model_validate_json((run_dir / "large_survey.json").read_text()).objects}[object_id]
    support = proxies[on]["translate"][2] + proxies[on]["size"][2] if on in proxies else 0.0
    facts = placement_facts(run_dir, object_id)
    rotation = matrix_from_euler_xyz_deg(placement.rotate_xyz_deg)
    facets = surface_facets(run_dir / placement.asset)
    walls = _wall_faces(run_dir, scene.camera)
    mask = np.asarray(Image.open(run_dir / "masks" / f"{object_id}.png")) > 0
    seen_turn = seen_turn_deg(to_world(np.load(run_dir / "points.npy"), scene.camera.matrix), mask, walls)
    turn = None if seen_turn is None else square_turn_deg(
        facets, rotation, walls, 0.0 if abs(seen_turn) <= SQUARE_WITHIN_DEG else seen_turn
    )
    return facts | {
        "on": on,
        "placed": not placement.left_out,
        "off_support_m": round(facts["off_floor_m"] - support, 3),
        "box_shaped": round(squareness(facets[0] @ rotation.T, facets[1])[1], 2),
        "seen_turn_deg": seen_turn,
        "off_square_deg": None if turn is None else abs(turn),
        "square_turn_deg": turn,
        "shell_moves": solved.get("shell_moves"),
    }


def _survey_floor(run_dir: Path, config: Config, vlm_name: str | None) -> None:
    """Survey the floor's materials and segment each from its box; writes floor_survey.json and masks/floor_<id>.png.

    Its own question, after the room's survey, so that a run surveyed before
    floors were told apart gets one without being surveyed again.
    """
    vlm = vlm_for(config, vlm_name, run_dir)
    ledger = Ledger(run_dir / "usage.jsonl", config.max_billed_tokens)
    image = Image.open(run_dir / "input.jpg")
    survey, raw = take_floor_survey(image, vlm, ledger)
    (run_dir / "debug" / "floor_survey_raw.txt").write_text(raw)
    print(f"floor: {', '.join(f'{f.id} ({survey.kinds[f.id]}: {f.description})' for f in survey.floors) or '-'}", flush=True)
    if survey.floors:
        comfy = comfy_client(config.comfy.light[0], run_dir)
        image_name = comfy.upload_image(to_jpeg(image), f"aomma_{run_dir.name}.jpg")
        masks, infos = segment(comfy, image_name, image.size, Inventory(objects=survey.floors))
        for oid, mask in masks.items():
            Image.fromarray(mask).save(run_dir / "masks" / f"floor_{oid}.png")
        survey.box_iou = {oid: info.box_iou for oid, info in infos.items()}
    (run_dir / "floor_survey.json").write_text(survey.model_dump_json(indent=2))


def _built_in_faces(run_dir: Path, config: Config, vlm_name: str | None) -> None:
    """Look again, face by face, at each built-in whose mask covers less than half its box (`stages.faces`); keep the
    faces if the built-in's mask and its faces together cover more of its box.

    The masks as first segmented are kept in masks/from_box/ and every look starts from there. Writes faces.json (for
    each built-in looked at: its faces, its box's cover before and after, and whether the faces were taken) and
    masks/faces/<built-in>__<face>.png, and updates the built-in's mask and its entry in masks.json.
    """
    from aomma.stages.faces import as_inventory, from_faces, list_faces

    survey = Survey.model_validate_json((run_dir / "survey.json").read_text())
    infos = {oid: MaskInfo.model_validate(info) for oid, info in json.loads((run_dir / "masks.json").read_text()).items()}
    kept = run_dir / "masks" / "from_box"
    everything = Inventory(objects=survey.walls + survey.built_ins + survey.openings)

    def first(oid: str) -> np.ndarray:
        path = kept / f"{oid}.png" if (kept / f"{oid}.png").exists() else run_dir / "masks" / f"{oid}.png"
        return np.asarray(Image.open(path)) > 0

    masks = {o.id: first(o.id) for o in everything.objects if o.id in infos and (run_dir / "masks" / f"{o.id}.png").exists()}
    fills = box_fills(everything, masks)
    doubtful = [o for o in survey.built_ins if o.id in masks and infos[o.id].box_iou >= MIN_USABLE_BOX_IOU
                and fills[o.id] < MIN_FILL]
    found: dict[str, dict] = {}
    if doubtful:
        vlm = vlm_for(config, vlm_name, run_dir)
        ledger = Ledger(run_dir / "usage.jsonl", config.max_billed_tokens)
        image = Image.open(run_dir / "input.jpg").convert("RGB")
        comfy = comfy_client(config.comfy.light[0], run_dir)
        image_name = comfy.upload_image(to_jpeg(image), f"aomma_{run_dir.name}.jpg")
        (run_dir / "masks" / "faces").mkdir(parents=True, exist_ok=True)
        kept.mkdir(parents=True, exist_ok=True)
        for entry in doubtful:
            faces, raw = list_faces(image, entry, vlm, ledger)
            record = {"faces": [f.model_dump() for f in faces], "fill_before": fills[entry.id], "fill_after": fills[entry.id],
                      "taken": False, "raw": raw}
            if faces:
                listed = as_inventory(entry, faces)
                face_masks, _ = segment(comfy, image_name, image.size, listed)
                for oid, mask in face_masks.items():
                    Image.fromarray(mask).save(run_dir / "masks" / "faces" / f"{oid}.png")
                together = from_faces(masks[entry.id], listed, face_masks)
                after = box_fills(everything, masks | {entry.id: together})[entry.id]
                record["fill_after"] = after
                if after > fills[entry.id]:
                    if not (kept / f"{entry.id}.png").exists():
                        Image.fromarray(masks[entry.id]).save(kept / f"{entry.id}.png")
                    masks[entry.id] = together
                    Image.fromarray(together).save(run_dir / "masks" / f"{entry.id}.png")
                    infos[entry.id] = describe_mask(entry.bbox_2d, together, "faces")
                    record["taken"] = True
            found[entry.id] = record
            print(f"{entry.id}: {len(faces)} faces ({', '.join(f.id for f in faces) or '-'}); box covered "
                  f"{record['fill_before']:.2f} -> {record['fill_after']:.2f}" + ("" if record["taken"] else ", not taken"), flush=True)
        for oid, fill in box_fills(everything, masks).items():
            if oid in infos:
                infos[oid].fill = fill
        (run_dir / "masks.json").write_text(json.dumps({oid: info.model_dump() for oid, info in infos.items()}, indent=2))
    (run_dir / "faces.json").write_text(json.dumps(found, indent=2))


def _survey_room(run_dir: Path, config_path: Path, vlm_name: str | None) -> bool:
    """Survey the photo and segment and measure what the room is built of; False if it is not indoors.

    Writes survey.json, masks/ and masks.json (each wall, built-in and
    opening), floor.png, ceiling.png, points.npy and camera.json; masks/wall_text.png if no
    wall could be segmented from its box.
    """
    config = load_config(config_path)
    vlm = vlm_for(config, vlm_name, run_dir)
    (run_dir / "debug").mkdir(parents=True, exist_ok=True)
    ledger = Ledger(run_dir / "usage.jsonl", config.max_billed_tokens)
    image, intrinsics = load_working_image(run_source(run_dir), config.image_max_side)
    image.save(run_dir / "input.jpg", quality=90)

    survey, raw = take_survey(image, vlm, ledger)
    (run_dir / "debug" / "survey_raw.txt").write_text(raw)
    (run_dir / "survey.json").write_text(survey.model_dump_json(indent=2))
    print(
        f"{'indoors' if survey.indoor else 'outdoors'}, {survey.room_type}; "
        f"walls: {', '.join(o.id for o in survey.walls) or '-'}; built in: {', '.join(o.id for o in survey.built_ins) or '-'}; "
        f"openings: {', '.join(o.id for o in survey.openings) or '-'}",
        flush=True,
    )
    if not survey.indoor:
        print("not indoors: a staged run builds rooms only, so it stops here")
        return False
    everything = Inventory(objects=survey.walls + survey.built_ins + survey.openings)
    draw_inventory(image, everything).save(run_dir / "debug" / "inventory.jpg")

    comfy = comfy_client(config.comfy.light[0], run_dir)
    image_name = comfy.upload_image(to_jpeg(image), f"aomma_{run_dir.name}.jpg")
    # Each wall is segmented from its own box, as objects are: one text prompt for "wall" found only the most striking.
    masks, infos = segment(comfy, image_name, image.size, everything) if everything.objects else ({}, {})
    for oid, fill in box_fills(everything, masks).items():
        infos[oid].fill = fill
    (run_dir / "masks").mkdir(exist_ok=True)
    for oid, mask in masks.items():
        Image.fromarray(mask).save(run_dir / "masks" / f"{oid}.png")
    (run_dir / "masks.json").write_text(json.dumps({oid: info.model_dump() for oid, info in infos.items()}, indent=2))
    floor = segment_floor(comfy, image_name, image.size)
    Image.fromarray(floor).save(run_dir / "floor.png")
    Image.fromarray(segment_surface(comfy, image_name, image.size, "ceiling")).save(run_dir / "ceiling.png")
    usable = {oid for oid, info in infos.items() if info.box_iou >= MIN_USABLE_BOX_IOU}
    if not any(o.id in usable for o in survey.walls):
        print("no wall found from its box: falling back to one text prompt for \"wall\"", flush=True)
        Image.fromarray(segment_surface(comfy, image_name, image.size, "wall")).save(run_dir / "masks" / "wall_text.png")

    known_fov = intrinsics.fov_x_deg if intrinsics.source == "exif" else None
    points, fov = estimate_points(comfy, image_name, image.size, known_fov)
    if known_fov is None:
        intrinsics = intrinsics.with_fov_x(fov, "moge")
    points = _save_points(run_dir, points)
    (run_dir / "camera.json").write_text(intrinsics.model_dump_json(indent=2))
    draw_depth(points).save(run_dir / "debug" / "depth.jpg")
    return True


# A wall segmented from its own box need only be this tall: a backsplash may be all that shows of it.
_OWN_WALL_MIN_HEIGHT_M = 0.25
# A wall whose mask is less than this share its own (not a built-in's, an opening's or floor) is placed behind its built-ins, or
# without them is not fitted.
_OWN_WALL_MIN_SHARE = 0.1
# A ceiling pixel stands at least this high; OfficeScan 007's ceiling is 2.56 m up. Starting value.
_CEILING_MIN_M = 2.0


def _clean_ceiling(run_dir: Path, up: np.ndarray) -> None:
    """Keep only the segmenter's ceiling pixels that can be a ceiling (`stages.ceiling.ceiling_pixels`): ceiling.png is
    rewritten from ceiling_raw.png, which keeps the segmenter's mask as it came. `up` is the world's up in the camera
    frame."""
    from aomma.stages.ceiling import ceiling_pixels

    path, raw = run_dir / "ceiling.png", run_dir / "ceiling_raw.png"
    if not path.exists():
        return
    if not raw.exists():
        shutil.copyfile(path, raw)
    mask = np.asarray(Image.open(raw)) > 0
    kept = ceiling_pixels(mask, _load_points(run_dir), up)
    Image.fromarray(kept).save(path)
    if mask.any():
        print(f"ceiling: kept {kept.sum() / mask.sum():.0%} of the segmenter's mask (the rest faces sideways or is below the eye)", flush=True)


def _walls_from_points(run_dir: Path, up: np.ndarray, comfy: ComfyClient) -> None:
    """Segment each wall again from points on its bare surface, chosen from the scene (`stages.wallpoints`), and
    write its mask as what the points found plus what its box found outside every listed thing's box. The box-prompted
    masks are kept in masks/from_box/ and always start from there, so a rebuild gives the same walls. The listed things
    are the survey's built-ins and openings and, once stage 2 has listed them, the large items. `up` is the world's up
    in the camera frame."""
    from aomma.stages.wallpoints import scene_points, wall_mask
    from aomma.stages.segment import segment_points

    survey = Survey.model_validate_json((run_dir / "survey.json").read_text())
    things = survey.built_ins + survey.openings
    if (run_dir / "large_survey.json").exists():
        things += LargeSurvey.model_validate_json((run_dir / "large_survey.json").read_text()).objects
    image = Image.open(run_dir / "input.jpg").convert("RGB")
    size = image.size
    kept = run_dir / "masks" / "from_box"
    kept.mkdir(parents=True, exist_ok=True)
    walls = [w for w in survey.walls if (kept / f"{w.id}.png").exists() or (run_dir / "masks" / f"{w.id}.png").exists()]
    for wall in walls:
        if not (kept / f"{wall.id}.png").exists():
            shutil.copyfile(run_dir / "masks" / f"{wall.id}.png", kept / f"{wall.id}.png")
    if not walls:
        return
    chosen = scene_points(walls, things, _load_points(run_dir), up, size)
    boxes = {w.id: tuple(v * s / 1000 for v, s in zip(w.bbox_2d, (size[0], size[1], size[0], size[1]))) for w in walls}
    image_name = comfy.upload_image(to_jpeg(image), f"aomma_{run_dir.name}.jpg")
    found = segment_points(comfy, image_name, size, chosen, boxes)
    infos = json.loads((run_dir / "masks.json").read_text())
    for wall in walls:
        from_box = np.asarray(Image.open(kept / f"{wall.id}.png")) > 0
        Image.fromarray(wall_mask(found.get(wall.id), from_box, things, size)).save(run_dir / "masks" / f"{wall.id}.png")
        if wall.id in infos:
            infos[wall.id]["points"] = [list(p) for p in chosen[wall.id]]
    (run_dir / "masks.json").write_text(json.dumps(infos, indent=2))
    print("walls from points on their bare surface: " + ", ".join(f"{w.id} {len(chosen[w.id])}" for w in walls), flush=True)


def _build_shell(run_dir: Path, draw: Drawer | None = None, comfy: ComfyClient | None = None) -> None:
    """Build the room from a surveyed run: floor, walls and built-ins; then measure it.

    The depth map is scaled from the standard heights of counters and doors; with none of them, or when its
    depth is fitted to the LiDAR, it is left as measured.

    Writes wall.png, inventory.json (the built-ins, the run's first objects),
    world.json, walls.json, proxies.json, layout.json and the render; and
    shell.json, which says how well the walls stand where the photo shows wall
    (see `wall_agreement`), how much of each built-in a wall hides, and the
    depth check of the whole shell.

    A run given RoomPlan (`depth+roomplan`) takes its floor, walls and fixtures from it instead
    (`_build_shell_roomplan`). With `comfy`, the walls are segmented again from points on their bare surface once the
    room's up is known (`_walls_from_points`).
    """
    from aomma.capture import inputs

    if "roomplan" in inputs(run_dir):
        return _build_shell_roomplan(run_dir, draw, comfy)
    image = Image.open(run_dir / "input.jpg")
    intrinsics = Intrinsics.model_validate_json((run_dir / "camera.json").read_text())
    survey = Survey.model_validate_json((run_dir / "survey.json").read_text())
    infos = json.loads((run_dir / "masks.json").read_text())
    floor = np.asarray(Image.open(run_dir / "floor.png")) > 0
    points = np.load(run_dir / "points.npy")

    # A mask nowhere near its box is not of the thing: it neither builds nor scales anything.
    usable = {
        oid: np.asarray(Image.open(run_dir / "masks" / f"{oid}.png")) > 0
        for oid, info in infos.items() if info["box_iou"] >= MIN_USABLE_BOX_IOU
    }
    wall_masks = {o.id: usable[o.id] for o in survey.walls if o.id in usable}
    if not wall_masks:
        wall_masks = {"wall": np.asarray(Image.open(run_dir / "masks" / "wall_text.png")) > 0}
    # A wall's box takes in what hangs on it, so its mask does too: every built-in is built, and its pixels
    # are not the wall's. (The kitchen's wall cabinets were once dropped as "mostly wall".)
    built_masks = {o.id: usable[o.id] for o in survey.built_ins if o.id in usable}
    built = Inventory(objects=[o for o in survey.built_ins if o.id in built_masks])
    (run_dir / "inventory.json").write_text(built.model_dump_json(indent=2))

    measurable = built_masks | {o.id: usable[o.id] for o in survey.openings if o.id in usable}
    structure = np.any(list(measurable.values()), axis=0) if measurable else np.zeros(image.size[::-1], dtype=bool)
    scaled = Inventory(objects=[o for o in survey.built_ins + survey.openings if o.id in measurable])

    # With the capture's pose the floor's tilt is given, and with its LiDAR the floor's height is measured.
    known = _known_floor(run_dir, floor, points)

    def fit(points: np.ndarray) -> tuple[np.ndarray, list[Proxy], list[str]]:
        """The camera over the floor, the built-ins' and openings' boxes, and what placed the floor if it is out of view."""
        found = fit_floor(points, floor, structure, **known)
        camera_to_world = found.camera_to_world

        def boxes(camera_to_world: np.ndarray) -> list[Proxy]:
            world = to_world(points, camera_to_world)
            return fit_proxies(image, world, exclusive_masks(measurable), scaled, camera_to_world[:3, 3]) if measurable else []

        proxies = boxes(camera_to_world)
        # A floor out of view is a guess, often a counter's top: the tops of counters, whose heights vary little,
        # say where it is instead.
        if not found.seen and (placed := floor_from_heights(survey, proxies, float(camera_to_world[2, 3]))) is not None:
            drop, used = placed
            camera_to_world = camera_over_floor(camera_to_world[2, :3], float(camera_to_world[2, 3]) + drop)
            return camera_to_world, boxes(camera_to_world), used
        return camera_to_world, proxies, []

    # The depth map's overall scale comes from the tops of counters and doors, whose heights vary little;
    # with none of them, it is left as measured. A run given the LiDAR has its depth measured already.
    raw_camera, measured, floor_by = fit(points)
    if floor_by:
        print(f"floor: out of view, placed from the tops of {', '.join(floor_by)}", flush=True)
    if _metric(run_dir):
        depth_scale, scaled_by, how = 1.0, [], "none: the depth is fitted to the LiDAR"
    elif floor_by:
        # Those tops are at their standard heights by construction, so they say nothing of the scale.
        depth_scale, scaled_by, how = 1.0, [], "the floor placed from the tops of standard height: left as measured"
    else:
        depth_scale, scaled_by, how = scale_from_heights(survey, measured, float(raw_camera[2, 3]))
    (run_dir / "world.json").write_text(json.dumps({"depth_scale": depth_scale, "scaled_by": scaled_by, "how": how, "floor_by": floor_by}))
    camera_to_world, _, _ = fit(_load_points(run_dir))
    camera = Camera(
        camera_to_world=tuple(camera_to_world.flatten()),
        focal_length_mm=intrinsics.focal_length_mm,
        sensor_width_mm=intrinsics.sensor_width_mm,
        width=intrinsics.width,
        height=intrinsics.height,
    )
    points_world = to_world(_load_points(run_dir), camera_to_world)
    eye = camera_to_world[:3, 3]
    valid = np.isfinite(points_world).all(axis=-1)
    # The world's Z in the camera frame is the camera-to-world's third row.
    _clean_ceiling(run_dir, camera_to_world[2, :3])
    if comfy is not None:
        _walls_from_points(run_dir, camera_to_world[2, :3], comfy)
        wall_masks = {oid: np.asarray(Image.open(run_dir / "masks" / f"{oid}.png")) > 0 for oid in wall_masks
                      if (run_dir / "masks" / f"{oid}.png").exists()} or wall_masks

    if (run_dir / "ceiling.png").exists():
        # A wall's box reaches the ceiling, and so does its mask: a fifth of the kitchen's back wall was ceiling.
        # The ceiling's mask is only taken where it stands high: OfficeScan 005's took in the yellow walls
        # (0.3-1.9 m up), and cut 93% of the poster wall's mask away, leaving the notch beside it to be fitted.
        with np.errstate(invalid="ignore"):
            ceiling_pixels = (np.asarray(Image.open(run_dir / "ceiling.png")) > 0) & (points_world[..., 2] > _CEILING_MIN_M)
        wall_masks = {oid: mask & ~ceiling_pixels for oid, mask in wall_masks.items()}
    # A survey wall lying wholly between the wall behind a built-in and that built-in's front is the built-in's end
    # panel (KitchenScan 010's tall cabinet): not built as a wall; the label map gives its
    # pixels to the built-in.
    from aomma.labels import SAME_DEPTH_M

    ends = built_in_ends(wall_masks, built_masks, points_world, SAME_DEPTH_M) if survey.walls else {}
    if len(ends) < len(wall_masks):
        for oid, built_id in ends.items():
            print(f"{oid}: the end of {built_id}, not a wall (it lies between the wall behind {built_id} and its front)", flush=True)
        wall_masks = {oid: mask for oid, mask in wall_masks.items() if oid not in ends}
    else:
        ends = {}
    wall = np.any(list(wall_masks.values()), axis=0)
    Image.fromarray(wall).save(run_dir / "wall.png")
    ceiling = {"CEILING": ceiling_pixels} if (run_dir / "ceiling.png").exists() else {}
    draw_masks(image, built_masks | {o.id: usable[o.id] for o in survey.openings if o.id in usable} | {"FLOOR": floor} | ceiling | {
        f"WALL {oid}": mask for oid, mask in wall_masks.items()
    }).save(run_dir / "debug" / "masks.jpg")

    # Built-ins: a box each, fitted before the walls, since a wall they cover entirely stands behind them.
    proxies = fit_proxies(image, points_world, exclusive_masks(built_masks), built, eye) if built_masks else []

    # Walls: one fitted to each wall's own pixels, set back behind the built-ins in front of it, then made
    # to agree with each other, longest first. The text prompt's one mask may hold several walls.
    built_area = np.any(list(built_masks.values()), axis=0) if built_masks else np.zeros_like(structure)
    # What an opening shows stands behind its wall (a window's glass and blinds, what is past a doorway): OfficeScan
    # 005's window-side wall, its mask 88% window, was fitted 9 cm behind the wall it is part of.
    opening_area = np.any([usable[o.id] for o in survey.openings if o.id in usable], axis=0) if any(
        o.id in usable for o in survey.openings) else np.zeros_like(structure)
    # Walls of one paint segment as one: the office's back-wall mask took in all of the left wall, and its
    # one fitted wall was the left wall's. The smaller mask keeps the pixels both claim.
    walls = []
    for oid, mask in exclusive_masks(wall_masks).items():
        with np.errstate(invalid="ignore"):
            background = mask & ~built_area & ~opening_area & ~floor & (points_world[..., 2] > 0.15)
        # A wall that shows little but what is built against it stands behind the largest of those: the
        # kitchen's wall over the fridge kept a sliver above its cabinets, fitted as a slab seen edge on.
        covering = [p for p in proxies if (built_masks[p.id] & mask).sum() > 0.5 * built_masks[p.id].sum()]
        if survey.walls and covering and background.sum() < _OWN_WALL_MIN_SHARE * mask.sum():
            largest = max(covering, key=lambda p: int((built_masks[p.id] & mask).sum()))
            print(f"{oid}: shows little but {largest.id}, so placed behind it", flush=True)
            walls.append(wall_behind(largest, eye))
            continue
        # A wall that shows little but an opening says nothing of where it stands: Office 002's back wall was its
        # window but for 578 pixels of frame, which fitted a 0.6 m wall across the corner.
        if survey.walls and background.sum() < _OWN_WALL_MIN_SHARE * mask.sum():
            print(f"{oid}: shows little but its openings, so not fitted", flush=True)
            continue
        walls += fit_walls(
            image, points_world, background, eye, in_front=points_world[built_area & valid],
            **({"max_walls": 1, "min_height": _OWN_WALL_MIN_HEIGHT_M} if survey.walls else {}), tidy=False,
        )
    # Two survey walls on one line (one wall named twice) are one.
    walls = sorted(merge_walls(walls), key=lambda w: -w.size[0])
    walls = tidy_walls([w.model_copy(update={"id": f"wall_{n}"}) for n, w in enumerate(walls, start=1)])
    # Squared up (see `stages.square`): a wall cutting a corner at 45 degrees where the wall pixels show one,
    # then the built-ins turned onto the walls' grid, backed to their wall, a corner cabinet given its outline.
    with np.errstate(invalid="ignore"):
        seen = points_world[wall & ~built_area & ~floor & valid]
    walls, added = chamfered_corners(walls, seen[seen[:, 2] > 0.15], eye)
    for corner_wall in added:
        print(f"{corner_wall.id}: a wall cutting the corner, {corner_wall.size[0]:.2f} m long", flush=True)
    # A wall the photo shows running off its side edge carries on past the frame.
    reached = reach_frame(walls, wall & ~built_area & ~floor, points_world, camera)
    for before, after in zip(walls, reached):
        if after.size[0] != before.size[0]:
            print(f"{after.id}: carried past the photo's edge, {before.size[0]:.2f} -> {after.size[0]:.2f} m", flush=True)
    walls = reached
    # A door, doorway or window is cut through its wall, which carries on past it: the photo shows it as an opening.
    openings = {
        o.id: (usable[o.id], o.size_m[0], survey.kinds.get(o.id) == "window")
        for o in survey.openings if survey.kinds.get(o.id) in ("door", "doorway", "window") and o.id in usable
    }
    cut = cut_openings(walls, openings, wall & ~built_area & ~floor, points_world, camera)
    for before, after in zip(walls, cut):
        for opening in after.openings[len(before.openings):]:
            width = float(np.linalg.norm(np.subtract(*opening.ends)))
            print(f"{after.id}: {opening.id} cut through it, {width:.2f} m wide, {opening.bottom:.2f}-{opening.top:.2f} m high; "
                  f"{before.size[0]:.2f} -> {after.size[0]:.2f} m long", flush=True)
    walls = cut
    grid = room_grid_deg(walls)
    if grid is not None and proxies:
        squared = {proxy.id: proxy for proxy in proxies}
        _square_built_ins(squared, survey, built_masks, points_world, [(*wall_face(w, eye), w.size[2]) for w in walls], grid, eye)
        proxies = list(squared.values())
    floor_survey = FloorSurvey.model_validate_json((run_dir / "floor_survey.json").read_text()) if (run_dir / "floor_survey.json").exists() else None
    floor_masks = {
        f.id: np.asarray(Image.open(run_dir / "masks" / f"floor_{f.id}.png")) > 0
        for f in (floor_survey.floors if floor_survey else []) if floor_survey.box_iou.get(f.id, 0) >= MIN_USABLE_BOX_IOU
    }
    objects = _build_room(
        run_dir, image, points_world, camera, walls, ~built_area, wall & ~built_area & ~floor, floor_survey, floor_masks, draw,
        wall_masks,
    )

    # Built-ins are of their typical colour, not painted from the photo: what stands on a counter
    # is placed on it later, not painted into it.
    photo = np.asarray(image.convert("RGB"))
    for proxy in proxies:
        color = tuple(float(c) for c in (np.median(photo[built_masks[proxy.id]], axis=0) / 255) ** 2.2)
        write_proxy_box(run_dir / "assets" / f"{proxy.id}.usda", proxy.size, color, proxy.footprint)
        objects.append(Placement(
            name=proxy.id, asset=f"./assets/{proxy.id}.usda", translate=proxy.translate, rotate_xyz_deg=(0.0, 0.0, proxy.yaw_deg),
        ))
    scene = SceneLayout(camera=camera, objects=objects)
    (run_dir / "proxies.json").write_text(json.dumps([p.model_dump() for p in proxies], indent=2))
    (run_dir / "layout.json").write_text(scene.model_dump_json(indent=2))
    draw_wireframes(image, proxies, camera, BUILT_IN_EDGE).save(run_dir / "debug" / "layout.jpg")
    _compose_and_render(run_dir, scene)

    faces = _wall_faces(run_dir, camera)
    agreement = wall_agreement(points_world[wall & ~built_area & valid], faces, eye)
    hidden = {
        oid: round(hidden_by_walls(points_world[mask & valid], faces, eye), 3)
        for oid, mask in built_masks.items() if (mask & valid).any()
    }
    print(f"depth scale x{depth_scale:.2f} ({how}), camera {camera_to_world[2, 3]:.2f} m above the floor")
    print(f"{len(walls)} walls, {len(proxies)} built-ins", flush=True)
    if agreement:
        print(
            f"wall pixels: {agreement['agree']:.0%} agree with a wall, {agreement['too_near']:.0%} have a wall in front, "
            f"{agreement['too_far']:.0%} a wall behind, {agreement['missed']:.0%} no wall at all"
        )
    for oid, share in hidden.items():
        if share >= 0.05:
            print(f"{oid}: {share:.0%} of it is behind a wall")
    checked = depth_check(run_dir)
    (run_dir / "shell.json").write_text(json.dumps(
        {"walls": len(walls), "built_ins": len(proxies), "built_in_ends": ends, "depth_scale": depth_scale, "wall_pixels": agreement,
         "built_ins_behind_walls": hidden, "depth_check": {"agree": checked["agree"], "room": checked["room"], "objects": checked["objects"]}},
        indent=2,
    ))
    print(f"-> {run_dir}/shell.json, debug/compare.jpg, debug/masks.jpg")


def _build_shell_roomplan(run_dir: Path, draw: Drawer | None = None, comfy: ComfyClient | None = None) -> None:
    """Build the room from RoomPlan, for a run given it (`depth+roomplan`; see `stages.roomplan`), writing what
    `_build_shell` writes.

    The camera's height is RoomPlan's, put right by the shot's LiDAR where it sees the floor; the walls, with their
    doors, windows and openings, are RoomPlan's, each moved onto the shot's LiDAR where it sees it; the built-ins are
    RoomPlan's fixtures and the storage `roomplan.is_built_in` keeps, each named after the survey's built-in in it.
    A survey built-in RoomPlan has no box for (a range hood) is fitted from the photo as `_build_shell` does. The
    survey still gives the walls' pixels, the floor and its materials, and the built-ins' colours.
    """
    from aomma.capture import inputs
    from aomma.stages import roomplan as rp

    given = inputs(run_dir)
    room, shot_to_world = given["roomplan"], given["camera_to_world"]
    # The masks are put right first, from the room's up (the pose's), before anything reads them.
    _clean_ceiling(run_dir, np.linalg.inv(shot_to_world)[:3, 1])
    if comfy is not None:
        _walls_from_points(run_dir, np.linalg.inv(shot_to_world)[:3, 1], comfy)
    image = Image.open(run_dir / "input.jpg")
    intrinsics = Intrinsics.model_validate_json((run_dir / "camera.json").read_text())
    survey = Survey.model_validate_json((run_dir / "survey.json").read_text())
    infos = json.loads((run_dir / "masks.json").read_text())
    floor = np.asarray(Image.open(run_dir / "floor.png")) > 0
    usable = {
        oid: np.asarray(Image.open(run_dir / "masks" / f"{oid}.png")) > 0
        for oid, info in infos.items() if info["box_iou"] >= MIN_USABLE_BOX_IOU
    }
    wall_masks = {o.id: usable[o.id] for o in survey.walls if o.id in usable}
    if not wall_masks:
        wall_masks = {"wall": np.asarray(Image.open(run_dir / "masks" / "wall_text.png")) > 0}
    built_masks = {o.id: usable[o.id] for o in survey.built_ins if o.id in usable}
    built_area = np.any(list(built_masks.values()), axis=0) if built_masks else np.zeros_like(floor)

    # The run's world: Z up, the floor at 0, the camera over the origin, at RoomPlan's height above its floor, put
    # right by the shot's own LiDAR on the floor where it sees enough of it.
    up = np.linalg.inv(shot_to_world)[:3, 1]
    height = rp.camera_height(shot_to_world, room)
    lidar = lidar_points(given["depth"], given["intrinsics"])[given["conf"] == LIDAR_SURE]
    on_floor = to_world(lidar, camera_over_floor(up, height))[:, 2]
    on_floor = on_floor[np.abs(on_floor) < rp.SNAP_BAND_M / 2]
    floor_shift = float(np.clip(np.median(on_floor), -rp.MAX_SNAP_M, rp.MAX_SNAP_M)) if len(on_floor) >= rp.MIN_SNAP_POINTS else 0.0
    camera_to_world = camera_over_floor(up, height - floor_shift)
    scan_to_run = camera_to_world @ np.linalg.inv(shot_to_world)
    (run_dir / "world.json").write_text(json.dumps({"depth_scale": 1.0, "scaled_by": [], "how": "none: the depth is fitted to the LiDAR; the room is RoomPlan's"}))
    camera = Camera(
        camera_to_world=tuple(camera_to_world.flatten()), focal_length_mm=intrinsics.focal_length_mm,
        sensor_width_mm=intrinsics.sensor_width_mm, width=intrinsics.width, height=intrinsics.height,
    )
    eye = camera_to_world[:3, 3]
    print(f"floor: RoomPlan's, camera {height:.2f} m above it; the shot's LiDAR on the floor ({len(on_floor)} points) "
          + (f"moves it {floor_shift * 100:+.1f} cm" if floor_shift else "does not move it"), flush=True)
    points_world = to_world(_load_points(run_dir), camera_to_world)
    valid = np.isfinite(points_world).all(axis=-1)
    lidar_world = to_world(lidar, camera_to_world)

    # Walls: RoomPlan's, each moved onto the shot's LiDAR where it sees enough of it.
    walls = rp.walls_from_room(room, scan_to_run, eye)
    walls, snaps = rp.snap_walls(walls, lidar_world, rp.room_sides(walls, room, scan_to_run, eye))
    for wall in walls:
        shift, count = snaps[wall.id]
        opened = ", ".join(o.id for o in wall.openings)
        seen = f"moved {shift * 100:+.1f} cm onto the shot's LiDAR ({count} points)" if shift else (
            f"as RoomPlan has it ({count} LiDAR points on it)" if count else "out of the shot")
        print(f"{wall.id}: {wall.size[0]:.2f} m, {seen}" + (f"; {opened}" if opened else ""), flush=True)

    # Built-ins: RoomPlan's fixtures, storage by `is_built_in`, each named after the survey's built-in in it (the
    # one with the most pixels); others in the same box are its aliases, which stage 2 may name as a support.
    survey_points = {oid: points_world[mask & valid] for oid, mask in exclusive_masks(built_masks).items()}
    boxes = rp.boxes_from_room(room, scan_to_run)
    seen = {box.id for _, box in boxes if rp.in_box_points(box, lidar_world).sum() >= rp.MIN_SEEN_POINTS}
    kept, left = [], []
    for index, (kind, box) in enumerate(boxes):
        why = rp.is_built_in(index, boxes, survey_points, seen)
        if why is None:
            left.append(box.id)
            print(f"{box.id}: " + (f"{kind} in the shot, left to stage 2" if kind in rp.APPLIANCE_KINDS else f"{kind} standing alone, left to stage 2"), flush=True)
        else:
            kept.append((box, why))
    holders: dict[str, list[str]] = {}
    for oid, points in survey_points.items():
        if not len(points):
            continue
        middle = np.median(points, axis=0)
        holding = [box for box, _ in kept if rp.in_box(box, middle)]
        if holding:
            nearest = min(holding, key=lambda box: float(np.linalg.norm(np.subtract(box.translate[:2], middle[:2]))))
            holders.setdefault(nearest.id, []).append(oid)
    photo = np.asarray(image.convert("RGB"))
    proxies, aliases, roomplan_ids = [], [], set()
    for box, why in kept:
        names = sorted(holders.get(box.id, []), key=lambda oid: -int(built_masks[oid].sum()))
        name = names[0] if names else box.id
        if names:
            color = tuple(float(c) for c in (np.median(photo[built_masks[name]], axis=0) / 255) ** 2.2)
        else:
            # Its colour where the photo's points stand in it, if enough do.
            within = rp.in_box_points(box, points_world) & valid
            color = tuple(float(c) for c in (np.median(photo[within], axis=0) / 255) ** 2.2) if within.sum() >= 100 else box.color
        proxy = box.model_copy(update={"id": name, "color": color})
        proxies.append(proxy)
        roomplan_ids.add(name)
        aliases += [proxy.model_copy(update={"id": other}) for other in names[1:]]
        print(f"{name}: RoomPlan's {box.id}, {box.size[0]:.2f} x {box.size[1]:.2f} x {box.size[2]:.2f} m ({why})"
              + (f"; also {', '.join(names[1:])}" if names[1:] else ""), flush=True)
    # A survey built-in RoomPlan has no box for is fitted from the photo, as `_build_shell` fits every one.
    held = {oid for names in holders.values() for oid in names}
    unheld = {oid: mask for oid, mask in built_masks.items() if oid not in held}
    if unheld:
        fitted = fit_proxies(image, points_world, exclusive_masks(unheld), Inventory(objects=[o for o in survey.built_ins if o.id in unheld]), eye)
        for proxy in fitted:
            print(f"{proxy.id}: not in RoomPlan, fitted from the photo, {proxy.size[0]:.2f} x {proxy.size[1]:.2f} x {proxy.size[2]:.2f} m", flush=True)
        proxies += fitted
    # The inventory holds the survey's built-ins that stand in the scene (they have masks); RoomPlan's own boxes
    # are in proxies.json and the layout only.
    in_scene = {p.id for p in proxies} | {p.id for p in aliases}
    built = Inventory(objects=[o for o in survey.built_ins if o.id in built_masks and o.id in in_scene])
    (run_dir / "inventory.json").write_text(built.model_dump_json(indent=2))

    wall = np.any(list(wall_masks.values()), axis=0)
    Image.fromarray(wall).save(run_dir / "wall.png")
    draw_masks(image, built_masks | {o.id: usable[o.id] for o in survey.openings if o.id in usable} | {"FLOOR": floor} | {
        f"WALL {oid}": mask for oid, mask in wall_masks.items()
    }).save(run_dir / "debug" / "masks.jpg")
    floor_survey = FloorSurvey.model_validate_json((run_dir / "floor_survey.json").read_text()) if (run_dir / "floor_survey.json").exists() else None
    floor_masks = {
        f.id: np.asarray(Image.open(run_dir / "masks" / f"floor_{f.id}.png")) > 0
        for f in (floor_survey.floors if floor_survey else []) if floor_survey.box_iou.get(f.id, 0) >= MIN_USABLE_BOX_IOU
    }
    objects = _build_room(
        run_dir, image, points_world, camera, walls, ~built_area, wall & ~built_area & ~floor, floor_survey, floor_masks, draw,
        wall_masks,
    )
    for proxy in proxies:
        write_proxy_box(run_dir / "assets" / f"{proxy.id}.usda", proxy.size, proxy.color, proxy.footprint)
        objects.append(Placement(
            name=proxy.id, asset=f"./assets/{proxy.id}.usda", translate=proxy.translate, rotate_xyz_deg=(0.0, 0.0, proxy.yaw_deg),
        ))
    scene = SceneLayout(camera=camera, objects=objects)
    (run_dir / "proxies.json").write_text(json.dumps([p.model_dump() for p in proxies + aliases], indent=2))
    (run_dir / "roomplan.json").write_text(json.dumps({
        "floor_shift_m": floor_shift, "floor_points": len(on_floor), "walls": {k: {"shift_m": v[0], "points": v[1]} for k, v in snaps.items()},
        "from_roomplan": sorted(roomplan_ids), "aliases": {a.id: next(p.id for p in proxies if p.translate == a.translate) for a in aliases},
        "from_photo": sorted(unheld), "left_to_stage_2": left, "scan_to_run": scan_to_run.round(6).tolist(),
    }, indent=2))
    (run_dir / "layout.json").write_text(scene.model_dump_json(indent=2))
    draw_wireframes(image, proxies, camera, BUILT_IN_EDGE).save(run_dir / "debug" / "layout.jpg")
    _compose_and_render(run_dir, scene)

    faces = _wall_faces(run_dir, camera)
    agreement = wall_agreement(points_world[wall & ~built_area & valid], faces, eye)
    hidden = {
        oid: round(hidden_by_walls(points_world[mask & valid], faces, eye), 3)
        for oid, mask in built_masks.items() if (mask & valid).any()
    }
    print(f"camera {camera_to_world[2, 3]:.2f} m above the floor")
    print(f"{len(walls)} walls, {len(proxies)} built-ins", flush=True)
    if agreement:
        print(
            f"wall pixels: {agreement['agree']:.0%} agree with a wall, {agreement['too_near']:.0%} have a wall in front, "
            f"{agreement['too_far']:.0%} a wall behind, {agreement['missed']:.0%} no wall at all"
        )
    for oid, share in hidden.items():
        if share >= 0.05:
            print(f"{oid}: {share:.0%} of it is behind a wall")
    checked = depth_check(run_dir)
    (run_dir / "shell.json").write_text(json.dumps(
        {"walls": len(walls), "built_ins": len(proxies), "depth_scale": 1.0, "wall_pixels": agreement,
         "built_ins_behind_walls": hidden, "depth_check": {"agree": checked["agree"], "room": checked["room"], "objects": checked["objects"]},
         "roomplan": True},
        indent=2,
    ))
    print(f"-> {run_dir}/shell.json, roomplan.json, debug/compare.jpg, debug/masks.jpg")


def _box_corners_px(run_dir: Path, object_id: str, shape: tuple[int, int]) -> np.ndarray | None:
    """An item's box's 8 corners as pixel positions (x, y) in a mask of `shape`, which may lie past the photo's
    edges; None for an item with no box, or one reaching behind the camera."""
    from aomma.stages.forms import project_px
    from aomma.stages.score import box_corners

    proxies = [Proxy(**p) for p in json.loads((run_dir / "proxies.json").read_text()) if p["id"] == object_id]
    if not proxies:
        return None
    camera = SceneLayout.model_validate_json((run_dir / "layout.json").read_text()).camera
    corners = box_corners(proxies[0])
    if (((corners - camera.matrix[:3, 3]) @ camera.matrix[:3, :3])[:, 2] >= 0).any():
        return None
    return project_px(camera, corners) * np.array([shape[1] / camera.width, shape[0] / camera.height])


def _taller(mask: np.ndarray, height: int) -> np.ndarray:
    """`mask` with empty rows added below to `height` (see `repair.extend_below`)."""
    return np.pad(mask, ((0, height - mask.shape[0]), (0, 0))) if height > mask.shape[0] else mask


def _generate_asset(run_dir: Path, object_id: str, config: Config, comfy: ComfyClient, ticket: int | None = None) -> None:
    """Crop one object, generate its model on `comfy` and convert it to USD.

    Writes only this object's own files, so several can run at once. `ticket`
    is a place in the job queue taken earlier for the model.
    """
    source = load_source(run_source(run_dir))
    mask = np.asarray(Image.open(run_dir / "masks" / f"{object_id}.png")) > 0
    asset_dir = run_dir / "assets" / object_id
    asset_dir.mkdir(parents=True, exist_ok=True)
    glb_path = asset_dir / f"{object_id}.glb"
    # Generation takes many minutes, so an existing model is reused; delete it to regenerate.
    if not glb_path.exists():
        override = overrides(run_dir).get(object_id, {})
        # A run imported with a seed offset draws every model from other seeds (a second draw of the same shot).
        attempt = override.get("attempt", 0) + json.loads((run_dir / "run.json").read_text()).get("seed_offset", 0)
        occlusion = json.loads((run_dir / "occlusion.json").read_text())[object_id]
        (asset_dir / "repaired").unlink(missing_ok=True)
        background = crop_background(config.image_to_3d_workflow)
        inventory = Inventory.model_validate_json((run_dir / "inventory.json").read_text())
        entry = next(o for o in inventory.objects if o.id == object_id)
        # The mask stopping well short of the box's bottom is a hidden bottom, whatever hides it.
        bottom_hidden = entry.on == "floor" and bottom_short(mask, entry.bbox_2d) > BOTTOM_SHORT
        # Or the photo's bottom edge cuts it off, which its box shows reaching below the frame.
        corners = _box_corners_px(run_dir, object_id, mask.shape) if entry.on == "floor" else None
        cut_rows = cut_below_frame(mask, corners) if corners is not None else 0
        if occlusion["hidden_edge"] >= REPAIR_THRESHOLD or bottom_hidden or cut_rows or override.get("repair"):
            # Too much of the object is hidden to model it as seen: redraw it whole first.
            holds_things = contents_id(object_id) in occlusion["occluders"] or any(
                o.on == object_id for o in inventory.objects + _listed_small(run_dir) if o.id in occlusion["occluders"]
            )
            beyond = None
            if cut_rows:
                # The redraw is given the photo carried down to the box's foot, black there, to draw what the edge cuts.
                source, mask, beyond = extend_below(source, mask, cut_rows, corners)
                print(f"{object_id}: the photo's edge cuts its lower {cut_rows} px off; redrawn down to the floor", flush=True)
            context, context_side = context_crop(source, mask)
            redraw = functools.partial(
                repair_object, comfy_client(config.comfy.light[0], run_dir), config.repair_workflow, context,
                entry.description, holds_things, f"{run_dir.name}_{object_id}", seed=1 + attempt, to_floor=beyond is not None,
            )
            picture, subject = redraw()
            if is_light(picture, in_crop(mask, mask, picture.width)):
                picture, subject = redraw(backdrop="black")
            occluders = [_taller(np.asarray(Image.open(run_dir / "masks" / f"{oid}.png")) > 0, mask.shape[0])
                         for oid in occlusion["occluders"]]
            if bottom_hidden:
                occluders.append(below_mask(mask, entry.bbox_2d))
            if beyond is not None:
                occluders.append(beyond)
            subject &= in_crop(where_it_can_be(mask, occluders), mask, picture.width)
            before_after = Image.new("RGB", (context.width * 2, context.height))
            before_after.paste(context, (0, 0))
            before_after.paste(picture, (context.width, 0))
            before_after.save(run_dir / "debug" / f"repair_{object_id}.jpg")
            crop, side = crop_object(picture, subject, background=background)
            # The redrawn picture covers the same view as the context crop.
            side = side * context_side / picture.width
            (asset_dir / "repaired").touch()
        elif contents_id(object_id) in occlusion["occluders"]:
            # Things stand inside it (stage 3's), cut out of its crop: the holes are filled before it is modelled.
            holes = np.asarray(Image.open(run_dir / "masks" / f"{contents_id(object_id)}.png")) > 0
            cutout, side = crop_object(source, mask, background=background)
            outline = mask | holes
            picture, subject = fill_contents(
                comfy_client(config.comfy.light[0], run_dir), config.repair_workflow, cutout,
                in_object_crop(holes, outline, source.size), in_object_crop(outline, outline, source.size),
                entry.description, f"{run_dir.name}_{object_id}", seed=1 + attempt,
            )
            before_after = Image.new("RGB", (cutout.width * 2, cutout.height))
            before_after.paste(cutout, (0, 0))
            before_after.paste(picture, (cutout.width, 0))
            before_after.save(run_dir / "debug" / f"repair_{object_id}.jpg")
            crop, filled_side = crop_object(picture, subject, background=background)
            # The filled picture covers the same view as the cut-out.
            side = filled_side * side / picture.width
            # Modelled from a redrawn picture: posed from its box, judged only where nothing stands in front.
            (asset_dir / "repaired").touch()
        else:
            crop, side = crop_object(source, mask, background=background)
        crop.save(run_dir / "debug" / f"crop_{object_id}.jpg")
        # The crop is a narrow slice of the photo's view: its field of view follows
        # from how many source pixels it spans and the focal length in those pixels.
        intrinsics = Intrinsics.model_validate_json((run_dir / "camera.json").read_text())
        focal_px = intrinsics.focal_length_mm / intrinsics.sensor_width_mm * source.width
        fov = crop_fov_deg(side, focal_px)
        key = model_key(config.image_to_3d_workflow, crop, fov, attempt)
        from_cache = cached_model(key) is not None
        if not from_cache:
            comfy.wait_for_memory(config.min_free_memory_gb)
        glb_path.write_bytes(
            generate_model(comfy, config.image_to_3d_workflow, crop, fov, f"{run_dir.name}_{object_id}", attempt, ticket)
        )
        (asset_dir / "model.json").write_text(json.dumps({"model_key": key, "from_cache": from_cache}) + "\n")
        if from_cache:
            print(f"{object_id}: model {key} taken from {MODEL_CACHE}", flush=True)
    glb_to_usd(glb_path, asset_dir / f"{object_id}.usdc")
    print(f"{object_id}: {glb_path.stat().st_size / 1e6:.1f} MB model -> {asset_dir}", flush=True)


def _wall_faces(run_dir: Path, camera: Camera) -> list[Wall]:
    """The room-facing face of each wall the layout found, for the pose solver."""
    path = run_dir / "walls.json"
    faces = []
    for wall in json.loads(path.read_text()) if path.exists() else []:
        faces.append(sight_face(Proxy(**wall), camera.matrix[:3, 3]))
    return faces


def placement_facts(run_dir: Path, object_id: str) -> dict:
    """How an object's model sits in the scene now: its fit to the photo, and whether it is where a thing can be.

    `hidden_by_wall` is the share of the model that a wall stands in front of,
    seen from the photo's camera, and `off_floor_m` how far its lowest surface
    is above the floor (below, if negative), not counting a stray spike below
    it (see `lowest`). `distance_vs_photo` is how far from the camera its
    visible surface is, over where the depth map puts the object (1 is right,
    below 1 too near), and `size_vs_estimate` its size over the size the VLM
    estimated for the object. `inside_other` is the largest share of its volume
    (or the other's, if smaller) that another model also fills, and
    `inside_which` that model.
    """
    scene = SceneLayout.model_validate_json((run_dir / "layout.json").read_text())
    placement = next(o for o in scene.objects if o.name == object_id)
    solved = {p["id"]: p for p in json.loads((run_dir / "poses.json").read_text())}[object_id]
    points, _ = sample_surface(run_dir / placement.asset, 20000)
    world = placement.scale * points @ matrix_from_euler_xyz_deg(placement.rotate_xyz_deg).T + np.array(placement.translate)
    estimated = next(o for o in Inventory.model_validate_json((run_dir / "inventory.json").read_text()).objects if o.id == object_id).size_m
    return {
        "outline_fit": round(solved["iou"], 2),
        "colour_error": round(solved["color_error"], 2),
        "hidden_by_wall": round(hidden_by_walls(world, _wall_faces(run_dir, scene.camera), scene.camera.matrix[:3, 3]), 2),
        "off_floor_m": round(lowest(world[:, 2], trim=True), 2),
        "longest_side_m": round(placement.scale, 2),
        "distance_vs_photo": _distance_vs_photo(run_dir, object_id, world, scene.camera),
        "size_vs_estimate": round(placement.scale / max(estimated), 2),
        **_inside_other(run_dir, scene, object_id),
    }


def _inside_other(run_dir: Path, scene: SceneLayout, object_id: str) -> dict:
    """How much of an object's model another model also fills, at most, and which (see stages/collision.py)."""
    from aomma.stages.collision import VOXEL, Solid

    placed = _placed_models(run_dir, scene, 20000)
    if object_id not in placed:
        return {"inside_other": None, "inside_which": None}
    mine = Solid(placed.pop(object_id))
    worst, which = 0.0, None
    for other_id, world in placed.items():
        # Only models whose boxes meet can share a voxel; skip the rest cheaply.
        low, high = np.floor(world.min(axis=0) / VOXEL), np.floor(world.max(axis=0) / VOXEL)
        if np.any(high < mine.origin) or np.any(low > mine.origin + mine.grid.shape):
            continue
        other = Solid(world)
        smaller = min(mine.volume, other.volume)
        share = mine.overlap(other) / smaller if smaller else 0.0
        if share > worst:
            worst, which = share, other_id
    return {"inside_other": round(worst, 2), "inside_which": which if worst > 0 else None}


def _distance_vs_photo(run_dir: Path, object_id: str, world: np.ndarray, camera: Camera) -> float | None:
    """Median, over the object's visible pixels the model covers, of the model's distance there over the depth map's."""
    mask = np.asarray(Image.open(run_dir / "masks" / f"{object_id}.png")) > 0
    occluders = json.loads((run_dir / "occlusion.json").read_text()).get(object_id, {}).get("occluders", [])
    visible = mask & ~_in_front(run_dir, occluders, mask.shape)
    measured = -_load_points(run_dir)[..., 2]
    pixels, depth = project(world, camera)
    height, width = mask.shape
    inside = (pixels >= 0).all(axis=1) & (pixels[:, 0] < width) & (pixels[:, 1] < height) & (depth > 0)
    columns, rows, depth = pixels[inside, 0].astype(int), pixels[inside, 1].astype(int), depth[inside]
    # The nearest point on each pixel is the surface the camera sees.
    order = np.argsort(depth)
    _, first = np.unique((rows * width + columns)[order], return_index=True)
    rows, columns, depth = rows[order][first], columns[order][first], depth[order][first]
    judged = visible[rows, columns] & np.isfinite(measured[rows, columns]) & (measured[rows, columns] > 0)
    if judged.sum() < 10:
        return None
    return round(float(np.median(depth[judged] / measured[rows[judged], columns[judged]])), 2)


def _nearest_wall(run_dir: Path, object_id: str, camera: Camera) -> tuple[np.ndarray, float] | None:
    """The wall an object's proxy box is nearest, as a plane (normal into the room, offset)."""
    proxies = {p["id"]: p for p in json.loads((run_dir / "proxies.json").read_text())}
    centre = np.array(proxies[object_id]["translate"])
    planes = []
    for wall in json.loads((run_dir / "walls.json").read_text()):
        origin, _, normal = wall_face(Proxy(**wall), camera.matrix[:3, 3])
        planes.append((normal, float(normal @ origin)))
    return min(planes, key=lambda plane: abs(plane[0] @ centre - plane[1]), default=None)


def _build_flat_art(run_dir: Path, object_id: str) -> bool:
    """Build a flat wall object (picture, poster) as a slab painted from the photo.

    Returns False, having done nothing, for any other kind of object. The slab's
    place on the wall follows from its mask, so it is saved here for `pose` to use.
    """
    inventory = Inventory.model_validate_json((run_dir / "inventory.json").read_text())
    obj = next(o for o in inventory.objects if o.id == object_id)
    camera = SceneLayout.model_validate_json((run_dir / "layout.json").read_text()).camera
    wanted = is_flat_art(obj) or overrides(run_dir).get(object_id, {}).get("flat")
    wall = _nearest_wall(run_dir, object_id, camera) if wanted else None
    if wall is None:
        return False
    mask = np.asarray(Image.open(run_dir / "masks" / f"{object_id}.png")) > 0
    art = fit_flat_art(mask, camera, wall, min(obj.size_m), Image.open(run_dir / "input.jpg"))
    source = load_source(run_source(run_dir))
    asset_dir = run_dir / "assets" / object_id
    asset_dir.mkdir(parents=True, exist_ok=True)
    texture = flat_art_texture(source, art, camera)
    texture.save(asset_dir / f"{object_id}.png")
    # A frame found in the picture is raised with a normal map; without one the slab stays smooth.
    normal_path = asset_dir / f"{object_id}_normal.png"
    normal_map = frame_normal_map(texture)
    if normal_map is not None:
        normal_map.save(normal_path)
    else:
        normal_path.unlink(missing_ok=True)
    # Like every asset, the slab's longest side is 1 m; its placement scales it back up.
    longest = max(art.size)
    write_picture_slab(
        asset_dir / f"{object_id}.usdc", tuple(side / longest for side in art.size), asset_dir / f"{object_id}.png",
        normal_path if normal_map is not None else None,
    )
    (asset_dir / "flat.json").write_text(art.model_dump_json(indent=2))
    print(
        f"{object_id}: flat, {art.size[0]:.2f} x {art.size[2]:.2f} m, painted from the photo, "
        f"{'raised frame' if normal_map is not None else 'no frame found'} -> {asset_dir}",
        flush=True,
    )
    return True


def _place_asset(run_dir: Path, object_id: str) -> SceneLayout:
    """Swap an object's generated model into the layout in place of its proxy box."""
    # The asset's longest side is 1 m, so scaling by the proxy's longest side
    # gives it the right size. `pose` then solves its orientation and position.
    scene = SceneLayout.model_validate_json((run_dir / "layout.json").read_text())
    proxies = {p["id"]: p for p in json.loads((run_dir / "proxies.json").read_text())}
    placement = next(o for o in scene.objects if o.name == object_id)
    placement.asset = f"./assets/{object_id}/{object_id}.usdc"
    placement.scale = max(proxies[object_id]["size"])
    (run_dir / "layout.json").write_text(scene.model_dump_json(indent=2))
    return scene


def asset(run_dir: Path, object_id: str, config_path: Path) -> None:
    """Generate a real 3D model for one object and swap it in for its proxy box (milestone M3)."""
    config = load_config(config_path)
    if not _build_flat_art(run_dir, object_id):
        _generate_asset(run_dir, object_id, config, comfy_client(config.comfy.heavy[0], run_dir, heavy=True))
    _compose_and_render(run_dir, _place_asset(run_dir, object_id))


# A pose fitting the outline worse than this is no pose (see `pose`). The garage's right slipper, posed from
# a mask of scraps, got 0.00; the lowest of 200 other poses was a pillow at 0.09.
MIN_PLACED_FIT = 0.05


def pose(run_dir: Path, object_ids: list[str], settings: PoseSettings | None = None) -> None:
    """Solve rotation, scale and position for generated assets and re-render (milestone M4).

    Without `settings`, each object is posed with the defaults changed by its
    own "pose" override, if the review has set one (see agent.py).
    """
    chosen = overrides(run_dir)
    image = Image.open(run_dir / "input.jpg")
    inventory = Inventory.model_validate_json((run_dir / "inventory.json").read_text())
    scene = SceneLayout.model_validate_json((run_dir / "layout.json").read_text())
    points = _load_points(run_dir)
    on = {o.id: o.on for o in inventory.objects}
    occlusions = json.loads((run_dir / "occlusion.json").read_text())
    proxies = {p["id"]: p for p in json.loads((run_dir / "proxies.json").read_text())}
    walls, eye = _wall_faces(run_dir, scene.camera), scene.camera.matrix[:3, 3]
    # A staged run has a shell to hold models to: the floor at 0 and the built-ins' tops.
    staged = (run_dir / "survey.json").exists()
    supports = {"floor": 0.0}
    if staged:
        built = {o.id for o in Survey.model_validate_json((run_dir / "survey.json").read_text()).built_ins}
        supports |= {oid: p["translate"][2] + p["size"][2] for oid, p in proxies.items() if oid in built}
    # Measured distance along the camera's axis per pixel, for the hidden objects' search.
    distances = -points[..., 2]
    estimated = {o.id: max(o.size_m) for o in inventory.objects}
    listed = {o.id: tuple(o.size_m) for o in inventory.objects}
    # Default to every object in the scene that has a generated asset.
    def usdc(object_id: str) -> Path:
        return run_dir / "assets" / object_id / f"{object_id}.usdc"

    in_scene = {o.name for o in scene.objects}
    ids = object_ids or [o.id for o in inventory.objects if usdc(o.id).exists() and o.id in in_scene]
    # Items whose box is RoomPlan's (`depth+objects`): whole where the photo cuts the item, and not squared.
    priors = roomplan_priors(run_dir)
    # Small items' gate 4 boxes (`aomma boxes`, `aomma small-models`): each such model is also tried stretched into its box.
    boxes_path = run_dir / "boxes.json"
    small_boxes = {b["proxy"]["id"]: Proxy(**b["proxy"]) for b in json.loads(boxes_path.read_text())["boxes"]
                   if b["kind"] == "small"} if boxes_path.exists() else {}

    poses = []
    cut_priors = set()
    # How each object's outline is judged: its mask, and the window and things in front of it, if any.
    judged: dict[str, tuple[np.ndarray, np.ndarray | None, np.ndarray | None]] = {}
    given = settings
    for object_id in ids:
        settings = given or PoseSettings(**chosen.get(object_id, {}).get("pose", {}))
        trim = settings.contact == "trim"
        mask = np.asarray(Image.open(run_dir / "masks" / f"{object_id}.png")) > 0
        if object_id in priors:
            own = priors[object_id]["own"]["size"]
            if cut_by_frame(mask, own, listed[object_id]):
                cut_priors.add(object_id)
                # RoomPlan's box only places it (`_hold_to_shell`, `prior`), so its size is the solve's, as without RoomPlan.
                settings = settings.model_copy(update={"size_weight": max(settings.size_weight, CUT_SIZE_WEIGHT)})
                print(f"{object_id}: cut by the photo's edge, so sized from its listed size; placed in RoomPlan's box", flush=True)
        elif object_id in proxies and cut_by_frame(mask, proxies[object_id]["size"], listed[object_id]):
            # The box measured for it is cut by the photo's edge: sized from its listed size, not stretched onto the box.
            settings = settings.model_copy(update={"fit_box": False, "size_weight": max(settings.size_weight, CUT_SIZE_WEIGHT)})
            print(f"{object_id}: cut by the photo's edge, so not stretched onto its measured box", flush=True)
        flat = run_dir / "assets" / object_id / "flat.json"
        asset = f"./{usdc(object_id).relative_to(run_dir)}"
        if flat.exists():
            # A slab painted from the photo was fitted to the wall when it was built.
            art = FlatArt.model_validate_json(flat.read_text())
            solved = Pose(
                id=object_id, translate=art.translate, rotate_xyz_deg=(0.0, 0.0, art.yaw_deg),
                scale=max(art.size), iou=art.iou, color_error=0.0,
            )
        else:
            samples, colors = sample_surface(usdc(object_id), 60000)
            facets = surface_facets(usdc(object_id))
            depth = float(np.nanmedian(-points[mask][:, 2]))
            judged[object_id] = (mask, None, None)
            if (run_dir / "assets" / object_id / "repaired").exists():
                # A complete model of a partly hidden object: place it from its proxy
                # box, judged only where the things in front of it leave it visible.
                hidden_by = occlusions[object_id]["occluders"]
                in_front = _in_front(run_dir, hidden_by, mask.shape)
                proxy = Proxy(**proxies[object_id])
                outline = silhouette(proxy, scene.camera)
                judged[object_id] = (mask, outline, in_front)
                facing = rotation_z(settings.turn_deg) @ view_rotation(mask, scene.camera)
                solved = solve_occluded_pose(
                    object_id, samples, colors, mask, in_front, outline, image, scene.camera,
                    start_scale=max(proxy.size), start_base=proxy.translate,
                    guesses=(facing,), facing=facing, facets=facets,
                    walls=walls, wall_weight=settings.wall_weight, trim=trim,
                    distances=distances, depth_weight=settings.depth_weight,
                    expected_size=estimated[object_id], size_weight=settings.size_weight,
                )
                if settings.pull_forward:
                    # Compared with the stretch where it will stand: brought out of the wall, not buried in it.
                    solved = pull_forward(solved, samples, walls, eye)
                # The redrawn picture gives the look; the depth map's box gives the proportions.
                fitted = fit_to_box(
                    solved, samples, colors, proxy.size, proxy.translate, proxy.yaw_deg,
                    mask, in_front, outline, image, scene.camera, facets, walls, eye,
                ) if settings.fit_box else None
                if fitted is not None:
                    solved, reshape = fitted
                    asset = f"./assets/{object_id}/{object_id}_fitted.usda"
                    write_reshaped_asset(run_dir / asset, run_dir / "assets" / object_id / f"{object_id}.usdc", reshape.tolist())
            else:
                if on[object_id] == "floor":
                    contact = FLOOR
                elif staged and on[object_id] in supports:
                    contact = (np.array([0.0, 0.0, 1.0]), supports[on[object_id]])
                elif on[object_id] == "wall":
                    # Hang it on the wall whose surface its box is nearest.
                    contact = _nearest_wall(run_dir, object_id, scene.camera)
                else:
                    contact = None
                facing = rotation_z(settings.turn_deg) @ view_rotation(mask, scene.camera)
                solved = solve_pose(
                    object_id, samples, colors, mask, image, scene.camera, depth, contact,
                    guesses=(facing,), facing=facing,
                    settle=on[object_id] != "wall", facets=facets, trim=trim,
                )
        if settings.pull_forward and asset.endswith(".usdc") and not flat.exists():
            solved = pull_forward(solved, samples, walls, eye)
        if settings.reseat != "none" and on[object_id] == "floor" and asset.endswith(".usdc") and not flat.exists():
            solved, asset = _reseat(
                run_dir, object_id, solved, asset, samples, colors, settings.reseat, mask, image, scene.camera, occlusions, proxies
            )
        if staged and settings.hold_to_shell and on[object_id] in supports and not flat.exists():
            box = Proxy(**proxies[object_id]) if settings.fit_box and asset.endswith(".usdc") and object_id in proxies else None
            unheld = solved
            solved, reshape = _hold_to_shell(
                run_dir, solved, asset, judged.get(object_id, (mask, None, None)), image, scene.camera, walls,
                supports[on[object_id]], trim, estimated[object_id], points, settings.turn_deg, box,
                [Proxy(**proxies[oid]) for oid in built if oid in proxies], listed[object_id], object_id in priors,
            )
            if object_id in priors and (solved.shell_moves or {}).get("wall_or_model_wrong"):
                # RoomPlan's box cost too much of the fit: held to its own box instead, as without RoomPlan's.
                own = Proxy(**priors[object_id]["own"])
                print(f"{object_id}: RoomPlan's box undone; held to its own", flush=True)
                solved, reshape = _hold_to_shell(
                    run_dir, unheld, asset, judged.get(object_id, (mask, None, None)), image, scene.camera, walls,
                    supports[on[object_id]], trim, estimated[object_id], points, settings.turn_deg,
                    own if box is not None and object_id not in cut_priors else None,
                    [Proxy(**proxies[oid]) for oid in built if oid in proxies], listed[object_id],
                )
                solved.shell_moves["prior_undone"] = True
            if reshape is not None:
                write_reshaped_asset(run_dir / f"assets/{object_id}/{object_id}_fitted.usda", usdc(object_id), reshape.tolist())
                asset = f"./assets/{object_id}/{object_id}_fitted.usda"
        if object_id in small_boxes and not flat.exists():
            solved, asset = _small_in_box(run_dir, solved, asset, small_boxes[object_id], samples, colors,
                                          judged[object_id], image, scene.camera, walls, eye)
        placement = next(o for o in scene.objects if o.name == object_id)
        placement.asset = asset
        placement.translate, placement.rotate_xyz_deg, placement.scale = solved.translate, solved.rotate_xyz_deg, solved.scale
        # No pose found that covers the object's outline at all: the model would stand somewhere arbitrary, at
        # any size, in front of other things. It is kept out of the scene until a fix gives it a pose.
        # One that stands beyond the walls is in another room, seen through a doorway. One whose moves the
        # shell's check undid stands where the wall or its model is wrong.
        moved = solved.shell_moves or {}
        placement.left_out = solved.iou < MIN_PLACED_FIT or bool(moved.get("beyond_walls") or moved.get("wall_or_model_wrong"))
        poses.append(solved)

    if staged:
        # Box-shaped large items lined up with their neighbours and the built-ins (see `_line_up_items`).
        lined = _line_up_items(run_dir, scene, {oid: Proxy(**p) for oid, p in proxies.items()}, built, image, walls, occlusions)
        placements = {o.name: o for o in scene.objects}
        for oid, found in lined.items():
            if not found["kept"]:
                continue
            index = next((i for i, solved in enumerate(poses) if solved.id == oid), None)
            if index is None:
                _update_pose(run_dir, oid, placements[oid], found["fit"][1], found["hidden_by_wall"])
                continue
            poses[index] = poses[index].model_copy(update={
                "translate": placements[oid].translate, "rotate_xyz_deg": placements[oid].rotate_xyz_deg, "iou": found["fit"][1],
            })

    _rest_objects(run_dir, scene, inventory, proxies, {p.id: p.translate for p in poses})

    # The solvers judge a model's outline with no walls in the way, but a model a wall stands in front of
    # does not show: judge it again as the camera sees it, so its fit says so (a desk behind the wall
    # "fitted" at 0.84 while the render showed only wall).
    placements = {o.name: o for o in scene.objects}
    for index, solved in enumerate(poses):
        if solved.id in judged:
            placement = placements[solved.id]
            samples, colors = sample_surface(run_dir / placement.asset, 20000)
            world = placement.scale * samples @ matrix_from_euler_xyz_deg(placement.rotate_xyz_deg).T + np.array(placement.translate)
            hidden = behind_walls(world, walls, eye)
            if hidden.any():
                mask, frame, occluders = judged[solved.id]
                iou, color_error = (0.0, 1.0) if hidden.all() else outline_fit(
                    world[~hidden], colors[~hidden], mask, image, scene.camera, frame, occluders
                )
                poses[index] = solved = solved.model_copy(update={
                    "iou": round(float(iou), 3), "color_error": round(float(color_error), 3),
                    "hidden_by_wall": round(float(hidden.mean()), 2),
                })
        print(
            f"{solved.id}: silhouette IoU {solved.iou:.2f}, colour error {solved.color_error:.2f}, "
            f"longest side {solved.scale:.2f} m" + (f", {solved.hidden_by_wall:.0%} behind a wall" if solved.hidden_by_wall else "")
            + (", left out of the scene" if placements[solved.id].left_out else ""),
            flush=True,
        )

    # Keep poses solved earlier for other objects.
    poses_path = run_dir / "poses.json"
    saved = {p["id"]: p for p in json.loads(poses_path.read_text())} if poses_path.exists() else {}
    saved.update({p.id: p.model_dump() for p in poses})
    poses_path.write_text(json.dumps(list(saved.values()), indent=2))
    (run_dir / "layout.json").write_text(scene.model_dump_json(indent=2))
    _compose_and_render(run_dir, scene)
    print(f"-> {run_dir}/scene.usda, debug/compare.jpg")


def _hold_to_shell(
    run_dir: Path,
    solved: Pose,
    asset: str,
    judged: tuple[np.ndarray, np.ndarray | None, np.ndarray | None],
    image: Image.Image,
    camera: Camera,
    walls: list[Wall],
    support_top: float,
    trim: bool,
    expected_size: float,
    points: np.ndarray,
    turn_deg: float = 0.0,
    box: Proxy | None = None,
    built_ins: list[Proxy] = (),
    listed: tuple[float, float, float] | None = None,
    prior: bool = False,
) -> tuple[Pose, np.ndarray | None]:
    """Hold a solved model to the room's shell (see `hold_to_shell`), and judge its outline again where it now stands.

    `points` is the depth map's camera-frame points per pixel. A model standing far past a wall while the item's
    own points do not was placed at the wrong distance: it is posed again, upright, held to the depth map, to
    `expected_size` and to the walls, starting from where the points put the item with the side the generator
    painted from the photo turned to the camera (turned by `turn_deg`, as `PoseSettings.turn_deg`), and then
    held again. With a `box`, the box the depth map measured for the item, the model is then stretched onto it
    where that fits its outline better (see `fit_to_box`): generated from the photo, a model cannot see the
    object's depth (the kitchen's fridge came out 1.61 m deep, listed 0.7, and stuck out towards the camera).
    Where one of `built_ins` stands over the item, the box is first squared to that built-in's outline (see
    `square_to_built_in`, with `listed`, the item's listed size), and the model may be stretched further onto it.
    If the moves lose more than `UNDO_FIT_DROP` of the fit it was solved with, they are undone and the model
    marked `wall_or_model_wrong`. With `prior`, the box is RoomPlan's (`depth+objects`) and only guides the model
    (`fit_to_box`'s `guide`): it is not squared to a built-in, and the model, keeping its own shape and size, is turned
    onto the box's heading and stood in it whether or not that fits the outline better (the outline barely ever
    prefers a box from elsewhere to the pose it chose, so as a mere candidate the box changed nothing on OfficeScan 006),
    tried on each of the box's four sides, as the outline chose its heading only up to which side is the front
    (BedroomScan's bed came out 87-88 deg off under `depth+roomplan`); the outline places it within `PRIOR_SHIFT_M` of
    the box in plan (the office chair, 2 cm from its reference by the photo, lost 0.81 -> 0.43 of fit held to RoomPlan's
    box exactly, 6-9 cm off). `UNDO_FIT_DROP` still undoes it; the caller then holds it to its own box. Returns the
    pose and the stretch, if any (see `fit_to_box`; never with `prior`).
    """
    samples, colors = sample_surface(run_dir / asset, 20000)
    facets = surface_facets(run_dir / asset)
    eye = camera.matrix[:3, 3]
    mask, frame, occluders = judged
    seen = to_world(points[mask], camera.matrix)
    seen = seen[np.isfinite(seen).all(axis=1)]
    # Squared to the walls only if the photo does not show it standing at an angle.
    seen_turn = seen_turn_deg(to_world(points, camera.matrix), mask, walls)
    square = seen_turn is None or abs(seen_turn) <= SQUARE_WITHIN_DEG

    def seen_fit(pose: Pose) -> tuple[float, float]:
        world = placed_points(pose, samples)
        hidden = behind_walls(world, walls, eye)
        return (0.0, 1.0) if hidden.all() else outline_fit(world[~hidden], colors[~hidden], mask, image, camera, frame, occluders)

    def hold(pose: Pose, **options) -> list[tuple[Pose, ShellMoves]]:
        # Where it was moved back to a wall, also as it stands without that move: each is finished (stretched onto
        # its box) and the better fit kept, as the stretch itself is. The bookshelf stands a few centimetres off its
        # wall, and backing it onto the wall cost 0.09 of fit; another model fitted its box only from the wall (0.76).
        backed, moves = hold_to_shell(pose, samples, facets, walls, support_top, trim, square=square, **options)
        if not moves.backed_m or moves.far_past or moves.beyond_walls:
            return [(backed, moves)]
        unbacked, kept = hold_to_shell(pose, samples, facets, walls, support_top, trim, square=square, back=False, **options)
        return [(backed, moves), (unbacked, kept.model_copy(update={"not_backed_m": moves.backed_m}))]

    candidates = hold(solved, seen=seen)
    far_past = candidates[0][1].far_past

    def repose(held: Pose) -> Pose:
        """Posed again, upright, held to the depth map, its listed size and the walls."""
        # The generator paints the photo onto the side it builds facing the photo's view: that side is turned to
        # the camera, as the search's start, rather than the heading the outline chose.
        # The camera's pitch is in that rotation; only its heading is taken, onto the upright the solve found.
        facing = rotation_z(turn_deg) @ view_rotation(mask, camera)
        heading = euler_xyz_deg(facing)[2] - held.rotate_xyz_deg[2]
        upright = rotation_z(heading) @ matrix_from_euler_xyz_deg(held.rotate_xyz_deg)
        middle = np.median(seen, axis=0)
        return solve_occluded_pose(
            solved.id, samples, colors, mask, np.zeros_like(mask) if occluders is None else occluders, frame, image, camera,
            start_scale=expected_size, start_base=(float(middle[0]), float(middle[1]), support_top),
            guesses=(upright,), facets=facets, walls=walls, wall_weight=REPOSE_WALL_WEIGHT, trim=trim,
            distances=-points[..., 2], depth_weight=REPOSE_DEPTH_WEIGHT,
            expected_size=expected_size, size_weight=REPOSE_SIZE_WEIGHT, facing=facing,
            turn_within_deg=REPOSE_TURN_WITHIN_DEG,
        )

    if far_past:
        # Held again without `seen`, so what is left past a wall is pushed out.
        candidates = [(p, m.model_copy(update={"far_past": True, "reposed": True})) for p, m in hold(repose(candidates[0][0]))]

    squared = None
    if box is not None and listed is not None and not prior and not candidates[0][1].beyond_walls:
        squared = square_to_built_in(box, to_world(points, camera.matrix), mask, built_ins, listed, eye)
        if squared is not None:
            box = squared[0]

    def finish(held: Pose, moves: ShellMoves) -> tuple[Pose, ShellMoves, np.ndarray | None, float, float]:
        """Stretched onto its box where that fits its outline better; returns the pose, moves, stretch and fit."""
        iou, color_error = seen_fit(held)
        reshape = None
        if box is None or moves.beyond_walls:
            return held, moves, reshape, iou, color_error
        moves = moves.model_copy(update={"squared_to": squared[1] if squared is not None else None})
        fitted = fit_to_box(
            held, samples, colors, box.size, box.translate, box.yaw_deg,
            mask, np.zeros_like(mask) if occluders is None else occluders, frame, image, camera, facets, walls, eye,
            max_stretch=SQUARED_MAX_STRETCH if squared is not None else _MAX_STRETCH,
            turns=(0.0, 90.0, 180.0, 270.0) if prior else (0.0,),
            facing=rotation_z(turn_deg) @ view_rotation(mask, camera) if prior else None, guide=prior,
            shift_within=PRIOR_SHIFT_M if prior else 0.0,
        )
        if fitted is not None:
            (held, reshape), iou, color_error = fitted, fitted[0].iou, fitted[0].color_error
            moves.fitted_to_box = True
            # The stretch stands the model on the box's bottom by its very lowest point; settled again by its
            # lowest surface, as above, so that a thin base does not lift it (the kitchen's fridge, 6 cm).
            shaped = samples if reshape is None else samples @ reshape[:3, :3].T + reshape[:3, 3]
            held, rise = settle_on_support(held, shaped, support_top, trim)
            if abs(rise) > 0.005:
                moves.settled_m = round(moves.settled_m + rise, 3)
                world = placed_points(held, shaped)
                hidden = behind_walls(world, walls, eye)
                iou, color_error = (0.0, 1.0) if hidden.all() else outline_fit(
                    world[~hidden], colors[~hidden], mask, image, camera, frame, occluders
                )
        return held, moves, reshape, iou, color_error

    # The first (moved back) wins a tie.
    done = max((finish(*c) for c in candidates), key=lambda d: d[3])

    def depth_ratio(finished: tuple) -> float | None:
        held, _, reshape = finished[:3]
        shaped = samples if reshape is None else samples @ reshape[:3, :3].T + reshape[:3, 3]
        visible = mask if occluders is None else mask & ~occluders
        return distance_ratio(placed_points(held, shaped), visible, -points[..., 2], camera.matrix, camera.focal_px)

    # Finished (stretched onto its box), a model on the floor at the wrong distance is posed again the same way, and
    # kept so only if that brings it nearer the depth map's distance. Judged before the stretch, OfficeScan 006's desk
    # read 1.10 where it finished at 1.04, and was moved from where it stood right.
    ratio = depth_ratio(done) if not far_past and support_top == 0.0 else None
    if ratio is not None and abs(np.log(ratio)) > np.log(OFF_DEPTH_RATIO):
        flagged = [(p, m.model_copy(update={"reposed": True, "off_depth": round(ratio, 3)})) for p, m in hold(repose(done[0]))]
        again = max((finish(*c) for c in flagged), key=lambda d: d[3])
        if abs(np.log(depth_ratio(again) or ratio)) < abs(np.log(ratio)):
            done = again
        else:
            done = (done[0], done[1].model_copy(update={"off_depth": round(ratio, 3)}), *done[2:])
    held, moves, reshape, iou, color_error = done
    moved_fit = iou
    if solved.iou - iou > UNDO_FIT_DROP:
        held, moves, reshape = solved, moves.model_copy(update={"wall_or_model_wrong": True, "fitted_to_box": False}), None
        iou, color_error = solved.iou, solved.color_error
    print(
        f"{solved.id}: held to the shell (settled {moves.settled_m:+.2f} m, turned {moves.turned_deg:+.0f} deg, "
        f"out of walls {moves.cleared_m:.2f} m, back to a wall {moves.backed_m:.2f} m"
        + (f", not {moves.not_backed_m:.2f} m back, fitting better off it" if moves.not_backed_m else "") + "): "
        f"fit {solved.iou:.2f} -> {iou:.2f}" + (", stands beyond the walls" if moves.beyond_walls else "")
        + (f", at {moves.off_depth:.2f} of the depth map's distance" if moves.off_depth else "")
        + (", posed again, held to the depth map" if moves.reposed else "")
        + (f", box squared to {moves.squared_to}" if moves.squared_to else "")
        + ((", placed in RoomPlan's box" if prior else ", stretched onto its measured box") if moves.fitted_to_box else "")
        + (f", undone (moved, fit {moved_fit:.2f}) and left out: the wall or the model is wrong" if moves.wall_or_model_wrong else ""),
        flush=True,
    )
    return held.model_copy(update={
        "iou": round(float(iou), 3), "color_error": round(float(color_error), 3), "shell_moves": moves.model_dump(),
    }), reshape


def _ceiling(run_dir: Path, image: Image.Image, points_world: np.ndarray, camera: Camera, walls: list[Proxy]) -> Ceiling | None:
    """The ceiling from ceiling.png, if the run has one and enough of it shows; written to ceiling.json and assets/ceiling.usda."""
    path = run_dir / "ceiling.png"
    if not path.exists():
        return None
    mask = np.asarray(Image.open(path)) > 0
    photo = np.asarray(image.convert("RGB"))
    with np.errstate(invalid="ignore"):
        floor = (np.asarray(Image.open(run_dir / "floor.png")) > 0) & (np.abs(points_world[..., 2]) < _ON_FLOOR_M)
    floor_seen = points_world[floor][:, :2]
    ceiling = build_ceiling(
        points_world[mask], photo[mask], walls, camera.matrix[:3, 3], floor_seen[np.isfinite(floor_seen).all(axis=1)], mask.size,
    )
    if ceiling is None:
        print("ceiling: too little of it shows to build", flush=True)
        (run_dir / "ceiling.json").unlink(missing_ok=True)
        return None
    write_floor(
        run_dir / "assets" / "ceiling.usda", ceiling.polygon, ceiling.height_m + CEILING_THICKNESS_M, CEILING_THICKNESS_M,
        ceiling.color, from_below=True,
    )
    (run_dir / "ceiling.json").write_text(ceiling.model_dump_json(indent=2))
    print(f"ceiling: {ceiling.height_m:.2f} m up, from {ceiling.points} points; bounded by {', '.join(ceiling.bounded_by) or 'no wall'}", flush=True)
    return ceiling


def _write_wall_asset(run_dir: Path, wall: Proxy, color: tuple[float, float, float], board: dict | None, eye: np.ndarray) -> None:
    """Write a wall's slab asset as `_build_room` does, with its baseboard (`board`, as walls.json has it) on the room's side."""
    _, _, normal = wall_face(wall, eye)
    yaw = np.radians(wall.yaw_deg)
    room_side = 1.0 if normal @ np.array([-np.sin(yaw), np.cos(yaw), 0.0]) > 0 else -1.0
    write_wall(
        run_dir / "assets" / f"{wall.id}.usda", wall.size, color,
        (board["height"], BASEBOARD_THICKNESS_M, tuple(board["color"])) if board else None, room_side, opening_spans(wall),
    )


def _draw_shell_masks(run_dir: Path, image: Image.Image) -> None:
    """Draw debug/masks.jpg, the dashboard's Masks picture, as the shell does: built-ins, openings, floor, ceiling, walls."""
    survey = Survey.model_validate_json((run_dir / "survey.json").read_text())
    infos = json.loads((run_dir / "masks.json").read_text())
    usable = {
        oid: np.asarray(Image.open(run_dir / "masks" / f"{oid}.png")) > 0
        for oid, info in infos.items() if info["box_iou"] >= MIN_USABLE_BOX_IOU and (run_dir / "masks" / f"{oid}.png").exists()
    }
    wall_masks = {o.id: usable[o.id] for o in survey.walls if o.id in usable}
    if not wall_masks and (run_dir / "masks" / "wall_text.png").exists():
        wall_masks = {"wall": np.asarray(Image.open(run_dir / "masks" / "wall_text.png")) > 0}
    shown = {o.id: usable[o.id] for o in survey.built_ins + survey.openings if o.id in usable}
    shown["FLOOR"] = np.asarray(Image.open(run_dir / "floor.png")) > 0
    if (run_dir / "ceiling.png").exists():
        shown["CEILING"] = np.asarray(Image.open(run_dir / "ceiling.png")) > 0
    draw_masks(image, shown | {f"WALL {oid}": mask for oid, mask in wall_masks.items()}).save(run_dir / "debug" / "masks.jpg")


def add_ceiling(run_dir: Path, config_path: Path = DEFAULT_CONFIG) -> Ceiling | None:
    """Give a finished staged run a ceiling: segment it if the run has no ceiling.png, build it, stand the walls up to it, render.

    For runs surveyed before the ceiling was segmented; a new run builds it with its shell.
    """
    image = Image.open(run_dir / "input.jpg")
    if not (run_dir / "ceiling.png").exists():
        config = load_config(config_path)
        comfy = comfy_client(config.comfy.light[0], run_dir)
        image_name = comfy.upload_image(to_jpeg(image), f"aomma_{run_dir.name}.jpg")
        Image.fromarray(segment_surface(comfy, image_name, image.size, "ceiling")).save(run_dir / "ceiling.png")
    _draw_shell_masks(run_dir, image)
    scene = SceneLayout.model_validate_json((run_dir / "layout.json").read_text())
    camera, eye = scene.camera, scene.camera.matrix[:3, 3]
    saved = json.loads((run_dir / "walls.json").read_text())
    walls = [Proxy(**w) for w in saved]
    ceiling = _ceiling(run_dir, image, to_world(_load_points(run_dir), camera.matrix), camera, walls)
    scene.objects = [o for o in scene.objects if o.name != "ceiling"]
    if ceiling is not None:
        placed = {o.name: o for o in scene.objects}
        for entry in saved:
            entry["size"][2] = ceiling.height_m
            wall = Proxy(**entry)
            if wall.id in placed:
                _write_wall_asset(run_dir, wall, _asset_color(run_dir / "assets" / f"{wall.id}.usda"), entry.get("baseboard"), eye)
        (run_dir / "walls.json").write_text(json.dumps(saved, indent=2))
        scene.objects.append(Placement(name="ceiling", asset="./assets/ceiling.usda"))
    (run_dir / "layout.json").write_text(scene.model_dump_json(indent=2))
    _compose_and_render(run_dir, scene)
    depth_check(run_dir)
    return ceiling


def _square_built_ins(
    proxies: dict[str, Proxy], survey: Survey, built_masks: dict[str, np.ndarray], world: np.ndarray,
    faces: list[Wall], grid: float, eye: np.ndarray,
) -> dict:
    """Turn the built-ins onto the room's grid (see `square_built_in`), replacing each kept one in `proxies`.

    A built-in and what rests on it (a counter on its cabinets, a sink in the counter) are squared or put back
    together, kept only if their depth points (`world`, per pixel) lie no further from their boxes, on median
    and weighted by points, than `BUILT_IN_KEEP_M` more than before: the kitchen's back counter fitted the grid
    better alone, its cabinets worse, and squared apart they disagreed. Returns what was tried, by id.
    """
    resting = {o.id: o.on for o in survey.built_ins}

    def base(oid: str) -> str:
        seen_ids = set()
        while resting.get(oid) in built_masks and oid not in seen_ids:
            seen_ids.add(oid)
            oid = resting[oid]
        return oid

    trials: dict[str, list] = {}
    for oid, mask in exclusive_masks(built_masks).items():
        # A built-in whose mask a smaller one covers entirely was given no box: KitchenScan 007's island base and
        # counter were surveyed with one box and segmented to one mask.
        if oid not in proxies:
            continue
        listed = next(o for o in survey.built_ins if o.id == oid)
        seen = own_points_of(world, mask)
        before = proxies[oid]
        squared = square_built_in(before, seen, grid, faces, min(listed.size_m[0], listed.size_m[1]), eye)
        after = before if squared is None else squared
        trials.setdefault(base(oid), []).append((oid, before, after, len(seen), box_error_m(before, seen), box_error_m(after, seen)))
    record = {}
    for members in trials.values():
        if all(after is before for _, before, after, *_ in members):
            continue
        count = sum(m[3] for m in members) or 1
        error_before = sum(m[3] * m[4] for m in members) / count
        error_after = sum(m[3] * m[5] for m in members) / count
        kept = error_after <= error_before + BUILT_IN_KEEP_M
        for oid, before, after, _, own_before, own_after in members:
            if after is before:
                continue
            record[oid] = {
                "turned_deg": round(after.yaw_deg - before.yaw_deg, 1), "error_m": [round(own_before, 3), round(own_after, 3)],
                "with": [m[0] for m in members], "together_error_m": [round(error_before, 3), round(error_after, 3)],
                "size": [round(v, 2) for v in after.size], "corner": after.footprint is not None, "kept": kept,
            }
            print(
                f"{oid}: turned {after.yaw_deg - before.yaw_deg:+.1f} deg onto the room's grid, depth points off its box "
                f"{own_before:.3f} -> {own_after:.3f} m" + (f" (with {', '.join(m[0] for m in members if m[0] != oid)}: "
                f"{error_before:.3f} -> {error_after:.3f} m)" if len(members) > 1 else "") + ("" if kept else ", put back"),
                flush=True,
            )
            if kept:
                proxies[oid] = after
    return record


def _line_up_items(
    run_dir: Path, scene: SceneLayout, proxies: dict[str, Proxy], built: set[str], image: Image.Image,
    faces: list[Wall], occlusions: dict,
) -> dict:
    """Line up a staged run's box-shaped large items with their neighbours, built-ins included (see `line_up`).

    Each move is kept only if it loses no more than `ITEM_KEEP_FIT_DROP` of the item's outline fit as the
    camera sees it past the walls, and puts no more of it past a wall; a kept one is moved in `scene`.
    Returns what was tried, by id, with the fit before and after and the share behind a wall.
    """
    camera, eye = scene.camera, scene.camera.matrix[:3, 3]
    placed = {o.name: o for o in scene.objects}
    large = LargeSurvey.model_validate_json((run_dir / "large_survey.json").read_text()) if (run_dir / "large_survey.json").exists() else None
    footprints, models = [], {}
    for oid in [o.id for o in large.objects] if large else []:
        placement = placed.get(oid)
        # Only items placed as their own model: one not yet posed still stands as its box from the list (in the
        # large stage, the stool's while the fridge was posed first).
        if placement is None or placement.left_out or Path(placement.asset).parent.name != oid:
            continue
        rotation = matrix_from_euler_xyz_deg(placement.rotate_xyz_deg)
        normals, areas, _ = surface_facets(run_dir / placement.asset)
        heading, share = squareness(normals @ rotation.T, areas)
        if share < SQUARE_SHARE:
            continue
        samples, colors = sample_surface(run_dir / placement.asset, 20000)
        models[oid] = (samples, colors)
        at = placement.scale * samples @ rotation.T + np.array(placement.translate)
        axes = np.array([[np.cos(np.radians(heading)), np.sin(np.radians(heading))], [-np.sin(np.radians(heading)), np.cos(np.radians(heading))]])
        low, high = np.percentile(at[:, :2] @ axes.T, [1, 99], axis=0)
        footprints.append(Footprint(
            id=oid, heading_deg=heading, centre=(low + high) / 2 @ axes, size=high - low,
            z=tuple(float(v) for v in np.percentile(at[:, 2], [1, 99])), steadiness=share,
        ))
    if footprints:
        footprints += [
            Footprint(id=oid, heading_deg=p.yaw_deg, centre=np.array(p.translate[:2]), size=np.array(p.size[:2]),
                      z=(p.translate[2], p.translate[2] + p.size[2]), fixed=True, steadiness=float(p.size[0] * p.size[1]))
            for oid, p in proxies.items() if oid in built
        ]
    record = {}
    for footprint in line_up(footprints, eye):
        if footprint.fixed or (abs(footprint.turn_deg) < 0.5 and np.abs(footprint.shift).max() < 0.01):
            continue
        oid, placement = footprint.id, placed[footprint.id]
        samples, colors = models[oid]
        mask = np.asarray(Image.open(run_dir / "masks" / f"{oid}.png")) > 0
        frame = occluders = None
        if (run_dir / "assets" / oid / "repaired").exists():
            frame = silhouette(proxies[oid], camera)
            occluders = _in_front(run_dir, occlusions[oid]["occluders"], mask.shape)

        def judged(translate, rotate) -> tuple[float, float, float]:
            at = placement.scale * samples @ matrix_from_euler_xyz_deg(rotate).T + np.array(translate)
            hidden = behind_walls(at, faces, eye)
            fit = 0.0 if hidden.all() else outline_fit(at[~hidden], colors[~hidden], mask, image, camera, frame, occluders)[0]
            return float(fit), share_past_walls(at, faces), float(hidden.mean())

        spin = rotation_z(footprint.turn_deg)
        centre = np.array([*footprint.centre, 0.0])
        translate = centre + spin @ (np.array(placement.translate) - centre) + np.array([*footprint.shift, 0.0])
        rotate = euler_xyz_deg(spin @ matrix_from_euler_xyz_deg(placement.rotate_xyz_deg))
        (fit_before, past_before, _), (fit_after, past_after, hidden) = judged(placement.translate, placement.rotate_xyz_deg), judged(translate, rotate)
        kept = fit_after >= fit_before - ITEM_KEEP_FIT_DROP and past_after <= past_before + 0.02
        record[oid] = {
            "turned_deg": round(footprint.turn_deg, 1), "moved_m": [round(float(v), 3) for v in footprint.shift],
            "fit": [round(fit_before, 3), round(fit_after, 3)], "hidden_by_wall": round(hidden, 2), "kept": kept,
        }
        print(
            f"{oid}: turned {footprint.turn_deg:+.1f} deg and moved {np.linalg.norm(footprint.shift):.2f} m to line up with "
            f"its neighbours, fit {fit_before:.2f} -> {fit_after:.2f}" + ("" if kept else ", put back"), flush=True,
        )
        if kept:
            placement.translate, placement.rotate_xyz_deg = tuple(float(v) for v in translate), rotate
    return record


def square_up(run_dir: Path) -> dict:
    """Square a staged run's built-ins with its walls, and its large items with each other; saves square_up.json.

    For a run built before the shell and `pose` did it: walls cut at a corner first (see `chamfered_corners`),
    then the built-ins (see `_square_built_ins`), then the box-shaped large items (see `_line_up_items`).
    Then the scene is rendered and its depth checked.
    """
    image = Image.open(run_dir / "input.jpg")
    scene = SceneLayout.model_validate_json((run_dir / "layout.json").read_text())
    camera, eye = scene.camera, scene.camera.matrix[:3, 3]
    survey = Survey.model_validate_json((run_dir / "survey.json").read_text())
    proxies = {p["id"]: Proxy(**p) for p in json.loads((run_dir / "proxies.json").read_text())}
    occlusions = json.loads((run_dir / "occlusion.json").read_text())
    world = to_world(_load_points(run_dir), camera.matrix)
    grid = room_grid_deg([Proxy(**w) for w in json.loads((run_dir / "walls.json").read_text())])
    placed = {o.name: o for o in scene.objects}
    record: dict = {"grid_deg": grid, "walls": {}, "built_ins": {}, "items": {}}

    # Walls first: a wall cutting a corner at 45 degrees, where the wall pixels show one (see `chamfered_corners`).
    saved_walls = {w["id"]: w for w in json.loads((run_dir / "walls.json").read_text())}
    wall_masks = [run_dir / "masks" / f"{o.id}.png" for o in survey.walls if (run_dir / "masks" / f"{o.id}.png").exists()]
    if wall_masks and len(saved_walls) >= 2:
        built_area = np.any([np.asarray(Image.open(run_dir / "masks" / f"{o.id}.png")) > 0 for o in survey.built_ins
                             if (run_dir / "masks" / f"{o.id}.png").exists()] or [np.zeros(image.size[::-1], dtype=bool)], axis=0)
        floor = np.asarray(Image.open(run_dir / "floor.png")) > 0
        wall_pixels = np.any([np.asarray(Image.open(path)) > 0 for path in wall_masks], axis=0) & ~built_area & ~floor
        if (run_dir / "ceiling.png").exists():
            wall_pixels &= ~(np.asarray(Image.open(run_dir / "ceiling.png")) > 0)
        seen = own_points_of(world, wall_pixels)
        seen = seen[seen[:, 2] > 0.15]
        walls, added = chamfered_corners([Proxy(**w) for w in saved_walls.values()], seen, eye)
        # A new wall takes the paint of the larger wall it was cut from, by their lengths before any change.
        before = dict(saved_walls)
        for wall in walls:
            if wall in added or wall.model_dump() != Proxy(**saved_walls[wall.id]).model_dump():
                like = before.get(wall.id) or max(
                    (w for w in before.values() if Proxy(**w).color == wall.color), key=lambda w: w["size"][0],
                    default=max(before.values(), key=lambda w: w["size"][0]),
                )
                color = _asset_color(run_dir / "assets" / f"{like['id']}.usda")
                board = like.get("baseboard")
                _write_wall_asset(run_dir, wall, color, board, eye)
                saved_walls[wall.id] = wall.model_dump() | {"baseboard": board}
                if wall.id in placed:
                    placed[wall.id].translate, placed[wall.id].rotate_xyz_deg = wall.translate, (0.0, 0.0, wall.yaw_deg)
                else:
                    placement = Placement(name=wall.id, asset=f"./assets/{wall.id}.usda", translate=wall.translate,
                                          rotate_xyz_deg=(0.0, 0.0, wall.yaw_deg))
                    scene.objects.append(placement)
                    placed[wall.id] = placement
                record["walls"][wall.id] = {"length_m": round(wall.size[0], 2), "yaw_deg": round(wall.yaw_deg, 1), "added": wall in added}
                print(f"{wall.id}: " + ("a wall cutting the corner, " if wall in added else "now ") + f"{wall.size[0]:.2f} m long", flush=True)
        (run_dir / "walls.json").write_text(json.dumps(list(saved_walls.values()), indent=2))
    faces = _wall_faces(run_dir, camera)

    built_masks = {
        o.id: np.asarray(Image.open(run_dir / "masks" / f"{o.id}.png")) > 0
        for o in survey.built_ins if o.id in proxies and (run_dir / "masks" / f"{o.id}.png").exists()
    }
    photo = np.asarray(image.convert("RGB"))
    if grid is not None:
        record["built_ins"] = _square_built_ins(proxies, survey, built_masks, world, faces, grid, eye)
    for oid, found in record["built_ins"].items():
        if found["kept"]:
            after = proxies[oid]
            color = tuple(float(c) for c in (np.median(photo[built_masks[oid]], axis=0) / 255) ** 2.2)
            write_proxy_box(run_dir / "assets" / f"{oid}.usda", after.size, color, after.footprint)
            placed[oid].translate, placed[oid].rotate_xyz_deg = after.translate, (0.0, 0.0, after.yaw_deg)
    (run_dir / "proxies.json").write_text(json.dumps([p.model_dump() for p in proxies.values()], indent=2))

    record["items"] = _line_up_items(run_dir, scene, proxies, set(built_masks), image, faces, occlusions)
    for oid, found in record["items"].items():
        if found["kept"]:
            _update_pose(run_dir, oid, placed[oid], found["fit"][1], found["hidden_by_wall"])

    inventory = Inventory.model_validate_json((run_dir / "inventory.json").read_text())
    _rest_objects(run_dir, scene, inventory, {oid: p.model_dump() for oid, p in proxies.items()})
    (run_dir / "layout.json").write_text(scene.model_dump_json(indent=2))
    draw_wireframes(image, [proxies[oid] for oid in built_masks], camera, BUILT_IN_EDGE).save(run_dir / "debug" / "layout.jpg")
    _compose_and_render(run_dir, scene)
    checked = depth_check(run_dir)
    record["depth_check"] = {"agree": checked["agree"], "objects": checked["objects"]}
    (run_dir / "square_up.json").write_text(json.dumps(record, indent=2))
    print(f"-> {run_dir}/square_up.json, debug/compare.jpg")
    return record


def own_points_of(world: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """The world points of a mask's pixels that the depth map knows."""
    found = world[mask]
    return found[np.isfinite(found).all(axis=1)]


def _asset_color(path: Path) -> tuple[float, float, float]:
    """The first diffuse colour in a USD asset (a wall's paint); grey if it has none."""
    found = re.search(r"diffuseColor = \(([^)]*)\)", path.read_text()) if path.exists() else None
    return tuple(float(v) for v in found.group(1).split(",")) if found else (0.6, 0.6, 0.6)


def _update_pose(run_dir: Path, object_id: str, placement: Placement, fit: float, hidden: float) -> None:
    """Carry a moved placement, and its fit as now judged, into poses.json."""
    path = run_dir / "poses.json"
    poses = json.loads(path.read_text())
    for saved in poses:
        if saved["id"] == object_id:
            saved.update({
                "translate": list(placement.translate), "rotate_xyz_deg": list(placement.rotate_xyz_deg),
                "iou": round(fit, 3), "hidden_by_wall": round(hidden, 2),
            })
    path.write_text(json.dumps(poses, indent=2))


def _in_front(run_dir: Path, occluders: list[str], shape: tuple[int, int]) -> np.ndarray:
    """Where any of `occluders` stands in front of an object, as one mask; nowhere if there are none.

    An object can be redrawn whole with nothing in front of it: the review asks
    for that when a model comes out badly shaped.
    """
    in_front = np.zeros(shape, dtype=bool)
    for occluder in occluders:
        in_front |= np.asarray(Image.open(run_dir / "masks" / f"{occluder}.png")) > 0
    return in_front


def _reseat(
    run_dir: Path,
    object_id: str,
    solved: Pose,
    asset: str,
    samples: np.ndarray,
    colors: np.ndarray,
    mode: str,
    mask: np.ndarray,
    image: Image.Image,
    camera: Camera,
    occlusions: dict,
    proxies: dict[str, dict],
) -> tuple[Pose, str]:
    """Stand a floor-standing model that floats back on the floor (see `reseat_on_floor`); returns its pose and asset."""
    reseated = reseat_on_floor(solved, samples, mode)
    if reseated is None:
        return solved, asset
    pose, reshape = reseated
    shaped = samples if reshape is None else samples @ reshape[:3, :3].T + reshape[:3, 3]
    # The solvers score a pose each in their own way. The change re-seating makes is
    # measured here, judging both placements alike, and carried over to the solver's score.
    hidden_by = occlusions[object_id]["occluders"]
    in_front = _in_front(run_dir, hidden_by, mask.shape)
    frame = silhouette(Proxy(**proxies[object_id]), camera)
    before = outline_fit(placed_points(solved, samples), colors, mask, image, camera, frame, in_front)
    after = outline_fit(placed_points(pose, shaped), colors, mask, image, camera, frame, in_front)
    pose = pose.model_copy(update={
        "iou": round(solved.iou + after[0] - before[0], 3),
        "color_error": round(solved.color_error + after[1] - before[1], 3),
    })
    if reshape is not None:
        # Named like the models fitted to their boxes, so everything that reads those reads this.
        source = run_dir / asset
        reshaped = source.with_name(f"{object_id}_fitted.usda")
        write_reshaped_asset(reshaped, source, reshape.tolist())
        asset = f"./{reshaped.relative_to(run_dir)}"
    return pose, asset


def _small_in_box(
    run_dir: Path, solved: Pose, asset: str, box: Proxy, samples: np.ndarray, colors: np.ndarray,
    judged: tuple[np.ndarray, np.ndarray | None, np.ndarray | None], image: Image.Image, camera: Camera,
    walls: list[Wall], eye: np.ndarray,
) -> tuple[Pose, str]:
    """A small item's model as the photo posed it, or stretched into its gate 4 `box` (`stages.pose.in_box`), whichever
    fits the photo's outline better as it will stand: the photo's pose stood at the box's bottom, as `_rest_objects`
    stands it. Either alone lost: the photo's pose kept models generated too thick or turned onto a side, and stood
    at their level they lost their outline (OfficeScan 007's notebook 0.09); in the box, well-shaped models lost theirs
    (the fish figure 0.92 -> 0.31). The better of the two: median 0.56 -> 0.69 on the office's 51 models.
    Returns the pose and its asset."""
    mask, frame, occluders = judged

    def seen(world: np.ndarray, cols: np.ndarray) -> tuple[float, float]:
        hidden = behind_walls(world, walls, eye)
        return (0.0, 1.0) if hidden.all() else outline_fit(world[~hidden], cols[~hidden], mask, image, camera, frame, occluders)

    own, own_colors = sample_surface(run_dir / asset, 20000)
    world = solved.scale * own @ matrix_from_euler_xyz_deg(solved.rotate_xyz_deg).T + np.array(solved.translate)
    world[:, 2] += box.translate[2] - world[:, 2].min()
    moves = dict(solved.shell_moves or {})
    # A pose the shell's check leaves out of the scene (`pose`: beyond the walls, or its moves undone) is no candidate.
    photo = 0.0 if moves.get("beyond_walls") or moves.get("wall_or_model_wrong") else seen(world, own_colors)[0]
    pick = np.random.default_rng(0).choice(len(samples), min(len(samples), 20000), replace=False)
    # The asset's side the photo's pose stood up (its rotation's inverse on the world's up), kept up in the box.
    up = matrix_from_euler_xyz_deg(solved.rotate_xyz_deg).T @ np.array([0.0, 0.0, 1.0])
    boxed, reshape = in_box(solved.id, samples[pick], colors[pick], box.size, box.translate, box.yaw_deg, seen,
                            up if asset.endswith(".usdc") else None)
    if boxed.iou <= photo:
        print(f"{solved.id}: the photo's pose kept, {photo:.2f} stood at its level (in its box {boxed.iou:.2f})", flush=True)
        return solved.model_copy(update={"shell_moves": moves | {"in_box_fit": boxed.iou, "photo_fit": round(photo, 3)}}), asset
    print(f"{solved.id}: stretched into its gate 4 box, {boxed.iou:.2f} (the photo's pose stood at its level {photo:.2f})", flush=True)
    fitted = f"./assets/{solved.id}/{solved.id}_fitted.usda"
    write_reshaped_asset(run_dir / fitted, run_dir / "assets" / solved.id / f"{solved.id}.usdc", reshape.tolist())
    return boxed.model_copy(update={"shell_moves": {"in_box": True, "photo_fit": round(photo, 3)}}), fitted


def _rest_objects(
    run_dir: Path, scene: SceneLayout, inventory: Inventory, proxies: dict[str, dict],
    solved: dict[str, Vec3] | None = None,
) -> None:
    """Stand every object that rests on another object on it, supports first.

    Runs over the whole scene each time: moving a shelf should take its books
    along. Each object is rested from where it was solved (poses.json, or
    `solved` for poses not saved yet), not from where the last rest left it:
    otherwise every rest adds its small move to the last, and over many
    re-poses of the shelf a book climbs onto the shelf above.

    A small item with a gate 4 box (`boxes.json`, `aomma small-models`) is stood at its box's bottom instead: the
    support's level the depth shows (`stages.boxes.stand_on`), not its support's model, whose surface can be
    centimetres off (OfficeScan's cabinets 4-5 cm short dropped a 7 cm shell off its outline, 0.95 -> 0.16).
    It is not nudged inside the support model's edges either.
    """
    on = {o.id: o.on for o in inventory.objects}
    boxes_path = run_dir / "boxes.json"
    standing = {b["proxy"]["id"]: b["proxy"]["translate"][2] for b in json.loads(boxes_path.read_text())["boxes"]
                if b["kind"] == "small"} if boxes_path.exists() else {}
    placed = {o.name: o for o in scene.objects}
    poses_path = run_dir / "poses.json"
    start = {p["id"]: tuple(p["translate"]) for p in json.loads(poses_path.read_text())} if poses_path.exists() else {}
    start |= solved or {}

    def depth(object_id: str) -> int:
        chain = 0
        while on.get(object_id) in on and chain < len(on):
            object_id, chain = on[object_id], chain + 1
        return chain

    def world(placement: Placement) -> tuple[np.ndarray, np.ndarray]:
        """Rotation and offset taking the placement's asset into the world, with its scale folded in."""
        return placement.scale * matrix_from_euler_xyz_deg(placement.rotate_xyz_deg), np.array(placement.translate)

    for object_id in sorted((i for i in on if on[i] in placed and i in placed), key=depth):
        resting, support = placed[object_id], placed[on[object_id]]
        if not (resting.asset.endswith(".usdc") or "_fitted" in resting.asset) or (run_dir / "assets" / object_id / "flat.json").exists():
            continue  # still a box, or wall art
        if object_id in standing:
            if object_id in start:
                resting.translate = tuple(float(v) for v in start[object_id])
            turn, offset = world(resting)
            bottom = (sample_surface(run_dir / resting.asset, 5000)[0] @ turn.T + offset)[:, 2].min()
            resting.translate = (*resting.translate[:2], float(resting.translate[2] + standing[object_id] - bottom))
            continue
        if not (support.asset.endswith(".usdc") or "_fitted" in support.asset):
            continue
        if object_id in start:
            resting.translate = tuple(float(v) for v in start[object_id])
        turn, offset = world(support)
        normals, areas, centres = surface_facets(run_dir / support.asset)
        facets = (normals @ (turn / support.scale).T, areas * support.scale**2, centres @ turn.T + offset)
        turn, offset = world(resting)
        points = sample_surface(run_dir / resting.asset, 5000)[0] @ turn.T + offset
        move = rest_on(points.min(axis=0), points.max(axis=0), facets, proxies[on[object_id]]["yaw_deg"])
        resting.translate = tuple(float(v) for v in np.array(resting.translate) + move)


def assets(run_dir: Path, object_ids: list[str], config_path: Path) -> None:
    """Generate and place a model for every object in a run, largest first.

    One object generates at a time on each heavy ComfyUI server. The scene is
    re-rendered as each finishes, and a failure on one object does not stop the
    rest. Existing models are reused.
    """
    config = load_config(config_path)
    inventory = Inventory.model_validate_json((run_dir / "inventory.json").read_text())
    chosen = overrides(run_dir)
    placed = {p["id"] for p in json.loads((run_dir / "proxies.json").read_text())} - {i for i, o in chosen.items() if o.get("drop")}
    area = {
        o.id: int((np.asarray(Image.open(run_dir / "masks" / f"{o.id}.png")) > 0).sum()) for o in inventory.objects
    }
    ids = sorted([i for i in (object_ids or area) if i in placed], key=lambda i: -area[i])

    # Each worker borrows a server for one object, so a server never has two of our jobs.
    idle: queue.Queue[ComfyClient] = queue.Queue()
    for url in config.comfy.heavy:
        idle.put(comfy_client(url, run_dir, heavy=True))

    progress = BatchProgress(run_dir, ids)

    # Flat wall art is painted from the photo in a moment, with no model to generate.
    failed = []
    total = len(ids)
    for object_id in list(ids):
        try:
            if not _build_flat_art(run_dir, object_id):
                continue
            progress.set(object_id, "placing")
            _place_asset(run_dir, object_id)
            pose(run_dir, [object_id])
            progress.set(object_id, "done")
        except Exception as error:
            failed.append(object_id)
            progress.set(object_id, "failed")
            print(f"{object_id}: FAILED: {error}", flush=True)
        ids.remove(object_id)

    # Take every model's place in the queue now, so the whole batch is in line
    # (and on show) from the start, not one object at a time.
    tickets = {object_id: QUEUE.enqueue("3d", f"3D model {run_dir.name}_{object_id}", run_dir.name) for object_id in ids}

    def generate(object_id: str) -> None:
        comfy = idle.get()
        progress.set(object_id, "generating")
        try:
            _generate_asset(run_dir, object_id, config, comfy, tickets[object_id])
        finally:
            # A place not used (the model already existed, or something failed first) is given up.
            QUEUE.cancel(tickets[object_id])
            idle.put(comfy)

    with ThreadPoolExecutor(max_workers=len(config.comfy.heavy)) as pool:
        jobs = {pool.submit(generate, object_id): object_id for object_id in ids}
        # Placing and posing edit shared files, so they happen here, one at a time.
        for number, job in enumerate(as_completed(jobs), start=total - len(ids) + 1):
            object_id = jobs[job]
            print(f"[{number}/{total}] {object_id}", flush=True)
            try:
                job.result()
                progress.set(object_id, "placing")
                _place_asset(run_dir, object_id)
                pose(run_dir, [object_id])
                progress.set(object_id, "done")
            except Exception as error:
                failed.append(object_id)
                progress.set(object_id, "failed")
                print(f"{object_id}: FAILED: {error}", flush=True)
    print(f"done: {total - len(failed)} of {total} objects placed" + (f"; failed: {', '.join(failed)}" if failed else ""))


def split(run_dir: Path, object_ids: list[str], config_path: Path, vlm_name: str | None) -> list[str]:
    """Look again at suspect inventory entries: split each into its parts, or group the parts under one mask.

    With no ids, every entry whose box the masks leave mostly uncovered is
    looked at. The layout is then rebuilt and the existing models placed
    again; new and regrouped objects still need models (`aomma assets`), and
    their ids are returned.
    """
    config = load_config(config_path)
    image = Image.open(run_dir / "input.jpg")
    inventory = Inventory.model_validate_json((run_dir / "inventory.json").read_text())
    masks = {o.id: np.asarray(Image.open(run_dir / "masks" / f"{o.id}.png")) > 0 for o in inventory.objects}
    infos = {oid: MaskInfo(**info) for oid, info in json.loads((run_dir / "masks.json").read_text()).items()}
    fills = box_fills(inventory, masks)
    ids = object_ids or [oid for oid, fill in fills.items() if fill < MIN_FILL]
    if not ids:
        print("every box is at least " + f"{MIN_FILL:.0%} covered by masks; nothing to look at again")
        return []
    for oid in ids:
        print(f"{oid}: box {fills[oid]:.0%} covered")

    comfy = comfy_client(config.comfy.light[0], run_dir)
    image_name = comfy.upload_image(to_jpeg(image), f"aomma_{run_dir.name}.jpg")
    replaced = _relist(
        image, inventory, ids, vlm_for(config, vlm_name, run_dir),
        Ledger(run_dir / "usage.jsonl", config.max_billed_tokens), comfy, image_name, masks, infos,
    )
    for oid, fill in box_fills(inventory, masks).items():
        infos[oid].fill = fill

    kept = {o.id for o in inventory.objects}
    for old, (done, _) in replaced.items():
        if old not in kept:
            (run_dir / "masks" / f"{old}.png").unlink(missing_ok=True)
        # The model of an entry that was replaced, or whose mask changed, is set aside, not deleted.
        if done != "kept" and (run_dir / "assets" / old).is_dir():
            (run_dir / "assets_replaced").mkdir(exist_ok=True)
            (run_dir / "assets" / old).rename(run_dir / "assets_replaced" / old)
    for oid in kept:
        Image.fromarray(masks[oid]).save(run_dir / "masks" / f"{oid}.png")
    (run_dir / "inventory.json").write_text(inventory.model_dump_json(indent=2))
    (run_dir / "masks.json").write_text(json.dumps({oid: infos[oid].model_dump() for oid in infos if oid in kept}, indent=2))
    draw_inventory(image, inventory).save(run_dir / "debug" / "inventory.jpg")
    draw_masks(image, {oid: masks[oid] for oid in kept}).save(run_dir / "debug" / "masks.jpg")
    poses_path = run_dir / "poses.json"
    if poses_path.exists():
        poses_path.write_text(json.dumps([p for p in json.loads(poses_path.read_text()) if p["id"] in kept], indent=2))

    new = []
    for old, (done, entries) in replaced.items():
        print(f"{old}: {done} -> " + ", ".join(f"{oid} (box {infos[oid].fill:.0%} covered)" for oid in entries))
        new += entries if done != "kept" else []
    layout(run_dir)
    pose(run_dir, [])
    if new:
        print(f"next: aomma assets {run_dir} {' '.join(new)}")
    return new


def new_run(photo: Path) -> Path:
    """Make the folder for a new run on `photo`, holding so far only a note of the photo; `aomma run` takes it from there."""
    if not photo.is_file():
        raise SystemExit(f"{photo} is neither a photo nor a run folder")
    run_dir = Path("runs") / f"{datetime.now():%Y%m%d-%H%M%S}-{photo.stem}"
    run_dir.mkdir(parents=True)
    note_source(run_dir, photo)
    return run_dir


def note_source(run_dir: Path, photo: Path, **fields) -> None:
    """Start `run_dir`'s manifest: the photo it is made from (relative to the project when it lies inside, so a moved
    checkout still finds it), any `fields` (a capture run's scan, shot and condition), and the code it was started with."""
    photo = photo.resolve()
    if photo.is_relative_to(Path.cwd()):
        photo = photo.relative_to(Path.cwd())
    (run_dir / "run.json").write_text(json.dumps({"source": str(photo), **fields, **code_version(run_dir)}))


def note_made(run_dir: Path, file: str, made_by: str, **details) -> None:
    """Say in `run_dir`'s provenance.json what made `file` (e.g. "lidar_aligned(moge)"), from which inputs, and how."""
    path = run_dir / "provenance.json"
    made = json.loads(path.read_text()) if path.exists() else {}
    made[file] = {"made_by": made_by, **details}
    path.write_text(json.dumps(made, indent=2))


def code_version(run_dir: Path) -> dict:
    """The commit checked out and, with uncommitted changes, a hash of them, whose patch is kept in `run_dir/code/`.

    Asked of the checkout this code was loaded from, which for a round is the round's own (rounds.py).
    Untracked files are not in the patch. Without git, both are None.
    """
    git = ["git", "-C", str(CODE_ROOT)]
    try:
        commit = subprocess.run([*git, "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
        patch = subprocess.run([*git, "diff", "HEAD"], capture_output=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "diff": None}
    if not patch:
        return {"commit": commit, "diff": None}
    digest = hashlib.sha256(patch).hexdigest()[:16]
    kept = run_dir / "code" / f"{digest}.patch"
    if not kept.exists():
        kept.parent.mkdir(exist_ok=True)
        kept.write_bytes(patch)
    return {"commit": commit, "diff": digest}


# Commands that change a run folder; each time one runs, the code it ran is added to the folder's code.jsonl.
_CHANGES_RUN = ("run", "staged", "review", "shell", "large", "layout", "asset", "assets", "pose", "ceiling", "square-up",
                "collisions", "depth-check", "split")


def note_code(run_dir: Path, command: str) -> None:
    """Add which code `command` runs on `run_dir` with to its code.jsonl, since a run is often carried on with later code."""
    entry = {"at": f"{datetime.now():%Y-%m-%d %H:%M:%S}", "command": command, **code_version(run_dir)}
    with open(run_dir / "code.jsonl", "a") as log:
        log.write(json.dumps(entry) + "\n")


def run_source(run_dir: Path) -> Path:
    """The photo `run_dir` is made from."""
    return Path(json.loads((run_dir / "run.json").read_text())["source"])


def stop(target: str, config_path: Path) -> None:
    """End the command at work in a run or experiment folder (or the dashboard), and clear up after it."""
    if target == "dashboard":
        pid = background.stop(Path("runs") / "dashboard.pid")
        print(f"stopped the dashboard (process {pid})" if pid else "the dashboard is not running")
        return
    folder = Path(target)
    pid_file = folder / background.PID
    stages = folder / "progress_run.json"
    if not background.alive(pid_file) and stages.exists():
        # A run started before process ids were kept still says who is running it here.
        pid_file.write_text(str(json.loads(stages.read_text())["pid"]))
    pid = background.stop(pid_file)
    if pid is None:
        print(f"nothing is running in {folder}")
        return
    # Its jobs would otherwise run on for nobody, alongside whatever the queue starts next.
    config = load_config(config_path)
    cancelled = sum(ComfyClient(url, JOB_LOG).cancel_labelled(folder.name) for url in config.comfy.all)
    QUEUE.snapshot()  # reading the queue drops the tickets of processes that are gone
    print(f"stopped process {pid} in {folder}" + (f"; took {cancelled} of its jobs off the servers" if cancelled else ""))


# Commands that go to the background unless told `--foreground`; the rest only with `--detach`.
_DETACHED_BY_DEFAULT = ("run", "staged", "shell", "large", "dashboard", "experiment", "round")


def _placed_models(run_dir: Path, scene: SceneLayout, samples: int) -> dict[str, np.ndarray]:
    """Surface samples of every generated model in the scene, where it is placed, by object id.

    Flat pictures painted from the photo, walls and the floor are left out:
    they are not models that can pass through one another.
    """
    placed = {}
    for placement in scene.objects:
        is_model = placement.asset.endswith(".usdc") or "_fitted" in placement.asset
        if not is_model or placement.left_out or (run_dir / "assets" / placement.name / "flat.json").exists():
            continue
        points, _ = sample_surface(run_dir / placement.asset, samples)
        placed[placement.name] = placement.scale * points @ matrix_from_euler_xyz_deg(placement.rotate_xyz_deg).T + np.array(placement.translate)
    return placed


def collisions(run_dir: Path) -> list[dict]:
    """Find generated models that pass through each other; saves collisions.json."""
    from aomma.stages.collision import REPORT_SHARE, find_collisions

    scene = SceneLayout.model_validate_json((run_dir / "layout.json").read_text())
    found = find_collisions(_placed_models(run_dir, scene, 40000))
    (run_dir / "collisions.json").write_text(json.dumps(found, indent=2))
    print(f"{len(found)} pairs of models pass through each other by {REPORT_SHARE:.0%} or more of the smaller one")
    for entry in found:
        a, b = entry["objects"]
        print(f"  {a} and {b}: {entry['litres']} litres shared, {entry['share_of_smaller']:.0%} of the smaller")
    print(f"-> {run_dir}/collisions.json")
    return found


def depth_check(run_dir: Path) -> dict:
    """Compare the whole scene, as placed, with the photo's depth map; saves depth_check.json and debug/depth_check.jpg."""
    from aomma.render.blender import render_depth
    from aomma.stages import depthcheck

    scene = SceneLayout.model_validate_json((run_dir / "layout.json").read_text())
    inventory = Inventory.model_validate_json((run_dir / "inventory.json").read_text())
    masks = exclusive_masks(_usable_masks(run_dir, inventory)[0])
    render_depth(
        run_dir / "scene.usda", run_dir / "debug" / "depth.npy", scene.camera.width, scene.camera.height,
        objects=run_dir / "debug" / "depth_objects.npz",
    )
    rendered = np.load(run_dir / "debug" / "depth.npy")
    with np.load(run_dir / "debug" / "depth_objects.npz") as found:
        shown = {str(name): found["ids"] == index + 1 for index, name in enumerate(found["names"])}
    summary, rendered_small, measured_small = depthcheck.check_scene(
        rendered, -_load_points(run_dir)[..., 2], masks, shown=shown
    )
    (run_dir / "depth_check.json").write_text(json.dumps(summary, indent=2))
    depthcheck.draw(Image.open(run_dir / "input.jpg"), rendered_small, measured_small).save(run_dir / "debug" / "depth_check.jpg", quality=88)

    def line(name: str, found: dict) -> str:
        covering = [
            f"{found[key]:.0%} {words}" for key, words in
            (("hidden_by_others", "under other models"),) if found.get(key, 0) >= 0.05
        ] + ([f"covers {found['over_others']:.1f} times its own size of other objects"] if found.get("over_others", 0) >= 0.1 else [])
        return (f"  {name:30s} distance {found['ratio']:5.2f} of the photo's: {found['agree']:4.0%} agrees, "
                f"{found['too_near']:4.0%} too near, {found['too_far']:4.0%} too far ({found['pixels']} cells)"
                + (f"; {', '.join(covering)}" if covering else ""))

    print(f"{summary['agree']:.0%} of the photo within {depthcheck.AGREE - 1:.0%} of the depth map")
    if summary["room"]:
        print(line("(room: no object there)", summary["room"]))
    for object_id, found in sorted(summary["objects"].items(), key=lambda item: item[1]["agree"]):
        print(line(object_id, found))
    print(f"-> {run_dir}/depth_check.json, debug/depth_check.jpg")
    return summary


def _detach(args: argparse.Namespace) -> None:
    """Start this same command in the background and say where to follow it."""
    # The background copy must not detach in its turn.
    arguments = [a for a in sys.argv[1:] if a != "--detach"]
    if args.command in _DETACHED_BY_DEFAULT:
        arguments.append("--foreground")
    if args.command == "dashboard":
        folder, log, pid_file = Path("runs"), Path("runs") / "dashboard.log", Path("runs") / "dashboard.pid"
    else:
        if args.command in ("run", "staged", "shell", "large"):
            folder = args.target if args.target.is_dir() else new_run(args.target)
            # The run's folder exists now, so the background command is told to pick it up.
            arguments = [str(folder) if Path(a) == args.target else a for a in arguments]
        elif args.command == "review":
            folder = args.run_dir
        elif args.command == "round":
            from aomma import rounds

            folder = args.target if args.target.is_dir() else rounds.prepare(args.target, pinned=not args.working_tree)
            # The background copy takes up the folder made here, not the plan.
            arguments = [str(folder) if Path(a) == args.target else a for a in arguments]
        else:
            from aomma import lab

            folder = lab.new_folder(args.run_dir)
            arguments += ["--into", str(folder)]
        log, pid_file = folder / background.LOG, folder / background.PID
    pid = background.detach(arguments, log, pid_file)
    print(f"{args.command} started in the background (process {pid})")
    if args.command != "dashboard":
        print(f"  folder: {folder}")
    print(f"  follow: tail -f {log}")
    if args.command == "experiment":
        print(f"  check:  aomma experiment-status {folder}")
    if args.command == "round":
        print(f"  state:  {folder}/round.json, and the dashboard's Jobs tab")
    print(f"  stop:   aomma stop {'dashboard' if args.command == 'dashboard' else folder}")


def main() -> None:
    parser = argparse.ArgumentParser(prog="aomma")
    sub = parser.add_subparsers(dest="command", required=True)
    detach_help = "carry on in the background, with output in a log file, and return at once"
    foreground_help = "stay in this terminal, printing as it goes, instead of carrying on in the background"
    sub.add_parser("demo", help="compose and render a hand-authored proxy scene")
    p = sub.add_parser("run", help="take a photo all the way to a reviewed 3D scene, or finish a run left part-way")
    p.add_argument("target", type=Path, help="a photo, or an existing run folder to pick up")
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--vlm", help="name of an entry under `vlms` in the config")
    p.add_argument("--foreground", action="store_true", help=foreground_help)
    p = sub.add_parser("staged", help="a photo through the layout-first build steps: shell, large and small items' masks, labels, relook, box layout (no models); picks up a run left part-way")
    p.add_argument("target", type=Path, help="a photo, or an existing run folder to pick up")
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--vlm", help="name of an entry under `vlms` in the config")
    p.add_argument("--foreground", action="store_true", help=foreground_help)
    p = sub.add_parser("shell", help="first stage of a staged run: survey a photo and build only the room (floor, walls, built-ins)")
    p.add_argument("target", type=Path, help="a photo, or an existing run folder to build again")
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--vlm", help="name of an entry under `vlms` in the config")
    p.add_argument("--foreground", action="store_true", help=foreground_help)
    p = sub.add_parser("large", help="second stage of a staged run: list, model and place the furniture and appliances inside the shell")
    p.add_argument("target", type=Path, help="a photo, or a run folder (its shell is built first if it has none)")
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--vlm", help="name of an entry under `vlms` in the config")
    p.add_argument("--foreground", action="store_true", help=foreground_help)
    p = sub.add_parser("small", help="gate 3: list and segment a run's small items, support by support (no models); writes small_survey.json and masks")
    p.add_argument("runs", type=Path, nargs="+")
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--vlm", help="name of an entry under `vlms` in the config")
    p = sub.add_parser("relook", help="after the small items and the labels: list what a panoptic segmenter flags as unclaimed and no mask took; adds masks and survey entries, labels again; writes relook.json, debug/relook.jpg")
    p.add_argument("runs", type=Path, nargs="+")
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--vlm", help="name of an entry under `vlms` in the config")
    p = sub.add_parser("small-models", help="gate 5: generate and pose a model for each small item the box layout built as a box (not one thinner than the depth's spread)")
    p.add_argument("run_dir", type=Path)
    p.add_argument("object_ids", nargs="*", help="items to do (default: all)")
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p = sub.add_parser("boxes", help="gate 4: every large and small item as a box in the shell, from the label map, stood on its support (no models)")
    p.add_argument("runs", type=Path, nargs="+")
    p = sub.add_parser("review", help="judge a run's scene against its photo, object by object, and fix what can be fixed")
    p.add_argument("run_dir", type=Path)
    p.add_argument("--stage", choices=["large"], help="in a staged run, judge only that stage's objects, with its own budget and log")
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--vlm", help="name of an entry under `vlms` in the config")
    p.add_argument("--detach", action="store_true", help=detach_help)
    p = sub.add_parser("stop", help="end the run, review or experiment at work in a folder, or the dashboard")
    p.add_argument("target", help="a run or experiment folder, or `dashboard`")
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p = sub.add_parser("experiment", help="have the local model answer a question about a run by devising and running trials")
    p.add_argument("run_dir", type=Path)
    p.add_argument("question", help="what the trials should find out")
    p.add_argument("--trials", type=int, default=3, help="how many models it may generate (each takes minutes to an hour)")
    p.add_argument("--pose-trials", type=int, default=0, help="how many times it may pose a model again (seconds each)")
    p.add_argument("--into", type=Path, help=argparse.SUPPRESS)  # the folder a detached experiment was given
    p.add_argument("--foreground", action="store_true", help=foreground_help)
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--vlm", help="name of an entry under `vlms` in the config")
    p = sub.add_parser("round", help="run a round: a plan's commands in order, unwatched, each on the runs an earlier one made")
    p.add_argument("target", type=Path, help="a plan (.toml, see lab/), or a round's folder to take up where it stopped")
    p.add_argument("--foreground", action="store_true", help=foreground_help)
    p.add_argument("--retry-failed", action="store_true",
                   help="taking up a round, run its failed steps again too (they may overwrite runs later steps used)")
    p.add_argument("--working-tree", action="store_true",
                   help="run the working tree as each step finds it, not the commit checked out now (uncommitted code allowed)")
    p = sub.add_parser("measure-change", help="judge a gate's baseline with its measure at an earlier commit and as the code stands, side by side (lab/measures.toml)")
    p.add_argument("gate", help="a gate named in lab/measures.toml, e.g. gate2")
    p.add_argument("--from", dest="since", help="the commit whose measure to compare with (default: the gate's last recorded)")
    p.add_argument("--runs", type=Path, nargs="+", help="runs to judge instead of the gate's baseline")
    p = sub.add_parser("measure-record", help="add an agreed measure-change comparison to lab/measures.toml as the gate's newest version")
    p.add_argument("folder", type=Path, help="the comparison's folder, as measure-change printed it")
    p.add_argument("--section", required=True, help="where the experiment notes record the change and its agreement")
    p.add_argument("--why", required=True, help="what changed in the measure and why")
    p = sub.add_parser("experiment-status", help="say briefly how an experiment stands: running or not, its trials, and its findings")
    p.add_argument("folder", type=Path, nargs="?", help="an experiment folder (default: the newest)")
    p.add_argument("--unreviewed", action="store_true", help="instead, list the experiments that have ended but not been reviewed")
    p.add_argument("--open", action="store_true", help="instead, list the reviewed experiments whose action is still to be done")
    p = sub.add_parser("experiment-reviewed", help="mark an experiment reviewed, saying whether its findings held and what was done")
    p.add_argument("folder", type=Path)
    p.add_argument("--section", required=True, help="where the experiment notes take it up, e.g. \"14. Why the garage run's poses look wrong\"")
    p.add_argument("--verdict", required=True, help="whether the worker's findings held against the harness's numbers and pictures")
    p.add_argument("--action", required=True, help="what was done because of it: a change, a follow-up experiment, or \"none\" and why")
    p.add_argument("--open", action="store_true", help="the action is decided but not yet done; record it with experiment-acted")
    p = sub.add_parser("experiment-acted", help="record what was done about a reviewed experiment, closing its action")
    p.add_argument("folder", type=Path)
    p.add_argument("--action", required=True, help="what was done: the change or commit, the follow-up started, or the decision taken")
    p.add_argument("--still-open", action="store_true", help="part of the action is still to be done")
    p = sub.add_parser("perceive", help="inventory, segment and estimate depth for a photo")
    p.add_argument("image", type=Path)
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--vlm", help="name of an entry under `vlms` in the config")
    p = sub.add_parser("layout", help="build a proxy-box USD scene from a perceived run")
    p.add_argument("run_dir", type=Path)
    p = sub.add_parser("asset", help="generate a 3D model for one object and swap it into the scene")
    p.add_argument("run_dir", type=Path)
    p.add_argument("object_id")
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p = sub.add_parser("assets", help="generate and place models for every object in a run")
    p.add_argument("run_dir", type=Path)
    p.add_argument("object_ids", nargs="*", help="objects to do (default: all)")
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p = sub.add_parser("pose", help="fit rotation, scale and position of generated assets to the photo")
    p.add_argument("run_dir", type=Path)
    p.add_argument("object_ids", nargs="*", help="objects to solve (default: all with a generated asset)")
    p = sub.add_parser("ceiling", help="give a finished staged run a ceiling: segment it if need be, build it, render")
    p.add_argument("run_dir", type=Path)
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p = sub.add_parser("square-up", help="square a staged run's built-ins with its walls and its large items with each other")
    p.add_argument("run_dir", type=Path)
    p = sub.add_parser("collisions", help="find generated models in a run that pass through each other")
    p.add_argument("run_dir", type=Path)
    p = sub.add_parser("depth-check", help="compare the whole scene, as placed, with the photo's depth map")
    p.add_argument("run_dir", type=Path)
    p = sub.add_parser("split", help="look again at suspect inventory entries and list separately the things each covers")
    p.add_argument("run_dir", type=Path)
    p.add_argument("object_ids", nargs="*", help="entries to look at (default: those whose box the masks leave mostly uncovered)")
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--vlm", help="name of an entry under `vlms` in the config")
    p = sub.add_parser("import-capture", help="make a run for each input shot in an AommACapture scan; what the capture measured goes to refs/, for scoring only")
    p.add_argument("scan", type=Path, help="the scan folder, as shared from the app")
    p.add_argument("--seed-offset", type=int, default=0, help="move every model's seeds by this much, for another draw of the same shots (the noise floor)")
    p = sub.add_parser("scan-check", help="check a scan's shots against its mesh, and its LiDAR's scale against the printed marker; writes refs/<scan>/scan_check.json")
    p.add_argument("scans", type=Path, nargs="+", help="scan folders")
    p = sub.add_parser("ref-items", help="list, segment and lift each large item in a scan's reference shots, for scoring; writes refs/<scan>/items.json")
    p.add_argument("scan", type=Path)
    p.add_argument("--all-shots", action="store_true", help="every shot that passed the scan check, not only the reference shots")
    p.add_argument("--merge-only", action="store_true", help="merge the saved lists and masks again (a new merge rule), with no listing or segmenting")
    p.add_argument("--small", action="store_true", help="the small items instead (gate 3's reference), listed per support in the shots the large items were; writes refs/<scan>/small_items.json")
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--vlm", help="name of an entry under `vlms` in the config")
    p = sub.add_parser("ref-large", help="which of RoomPlan's furniture, lone storage and appliance boxes each pixel of a scan's input shots shows (gate 2's reference); writes refs/<scan>/<shot>/large_labels.png")
    p.add_argument("scans", type=Path, nargs="+", help="scan folders")
    p = sub.add_parser("judge-models", help="each run's posed models against the scan's reference items, by gate 4's matching and bars (gate 5); writes <run>/judge_models.json")
    p.add_argument("runs", type=Path, nargs="+")
    p = sub.add_parser("judge-boxes", help="each run's box layout against the scan's reference items (gate 4); writes <run>/judge_boxes.json")
    p.add_argument("runs", type=Path, nargs="+")
    p = sub.add_parser("ref-small", help="which small reference item each pixel of a scan's input shots shows (gate 3's reference); writes refs/<scan>/<shot>/small_labels.png")
    p.add_argument("scans", type=Path, nargs="+", help="scan folders")
    p = sub.add_parser("judge-small", help="each run's label map against its shot's small-item reference (gate 3); writes <run>/judge_small.json")
    p.add_argument("runs", type=Path, nargs="+")
    p = sub.add_parser("judge-large", help="each run's label map against its shot's large-item reference, pixel by pixel (gate 2); writes <run>/judge_large.json")
    p.add_argument("runs", type=Path, nargs="+")
    p = sub.add_parser("judge-shell", help="each run's label map against its shot's shell reference, pixel by pixel (gate 1); writes <run>/judge_shell.json")
    p.add_argument("runs", type=Path, nargs="+")
    p = sub.add_parser("ref-shell", help="what each pixel of a scan's input shots is, for the shell, from RoomPlan snapped to the shot's LiDAR; writes refs/<scan>/<shot>/shell_labels.png")
    p.add_argument("scans", type=Path, nargs="+", help="scan folders")
    p = sub.add_parser("pose-check", help="whether each shot's pose puts its LiDAR where RoomPlan and the reference shots say the room is; writes refs/<scan>/pose_check.json")
    p.add_argument("scans", type=Path, nargs="+", help="scan folders")
    p = sub.add_parser("labels", help="one label for every pixel of each run's photo, from its masks; writes labels.png, labels.json, debug/labels.jpg")
    p.add_argument("runs", type=Path, nargs="+")
    p = sub.add_parser("score", help="judge capture runs at views of their scan they were not given; adds rows to <into>/results.jsonl")
    p.add_argument("runs", type=Path, nargs="+")
    p.add_argument("--keyframes", type=int, default=4, help="keyframes scored from each distance group (default 4)")
    p.add_argument("--into", type=Path, help="folder for results.jsonl and the pictures (default lab/<stamp>-score)")
    p.add_argument("--label", default="", help='put on every row, to tell scorings of the same runs apart (e.g. "after review")')
    p.add_argument("--scene", choices=["scene", "boxes"], default="scene",
                   help="what to score: scene.usda and its models (default), or boxes: gate 4's box layout (boxes.usda)")
    p = sub.add_parser("jobs", help="watch one ComfyUI server live, or list every server's jobs with --once")
    p.add_argument("--once", action="store_true", help="print the current jobs and exit")
    p.add_argument("--url", help="server to watch live (default: the first heavy server)")
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p = sub.add_parser("dashboard", help="serve a web page showing live ComfyUI jobs and each run's results")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument(
        "--host", action="append", default=[], metavar="ADDRESS",
        help="also serve on this address of this machine, e.g. its Tailscale IP (repeatable)",
    )
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--foreground", action="store_true", help=foreground_help)
    args = parser.parse_args()
    stamp = f"{datetime.now():%Y%m%d-%H%M%S}"

    detaching = getattr(args, "detach", False) or (args.command in _DETACHED_BY_DEFAULT and not args.foreground)
    if not detaching and args.command in _CHANGES_RUN:
        # A photo gets its run folder, and with it its code, from `new_run`.
        folder = getattr(args, "run_dir", None) or args.target
        if folder.is_dir():
            note_code(folder, args.command)

    if detaching:
        _detach(args)
    elif args.command == "stop":
        stop(args.target, args.config)
    elif args.command == "demo":
        demo(Path("runs") / f"{stamp}-demo")
    elif args.command in ("run", "review"):
        # Imported here because the agent drives the commands in this module.
        from aomma import agent

        if args.command == "run":
            run_dir = args.target if args.target.is_dir() else new_run(args.target)
            with background.recorded(run_dir / background.PID):
                agent.run(run_dir, args.config, args.vlm)
        else:
            with background.recorded(args.run_dir / background.PID):
                agent.review(args.run_dir, args.config, args.vlm, args.stage)
    elif args.command == "staged":
        run_dir = args.target if args.target.is_dir() else new_run(args.target)
        with background.recorded(run_dir / background.PID):
            staged(run_dir, args.config, args.vlm)
    elif args.command in ("shell", "large"):
        run_dir = args.target if args.target.is_dir() else new_run(args.target)
        with background.recorded(run_dir / background.PID):
            (shell if args.command == "shell" else large)(run_dir, args.config, args.vlm)
    elif args.command == "small":
        for run_dir in args.runs:
            note_code(run_dir, "small")
            with background.recorded(run_dir / background.PID):
                small(run_dir, args.config, args.vlm)
    elif args.command == "relook":
        from aomma.stages.relook import describe as describe_relook, predict, relook

        config = load_config(args.config)
        predict(args.runs)
        for run_dir in args.runs:
            note_code(run_dir, "relook")
            with background.recorded(run_dir / background.PID):
                ledger = Ledger(run_dir / "usage.jsonl", config.max_billed_tokens)
                result = relook(run_dir, vlm_for(config, args.vlm, run_dir), ledger, comfy_client(config.comfy.light[0], run_dir))
            print(describe_relook(run_dir, result), flush=True)
    elif args.command == "small-models":
        small_models(args.run_dir, args.object_ids, args.config)
    elif args.command == "boxes":
        for run_dir in args.runs:
            note_code(run_dir, "boxes")
            print(f"== {run_dir}", flush=True)
            box_layout(run_dir)
    elif args.command == "experiment":
        # Imported here for the same reason as the agent.
        from aomma import lab

        folder = args.into or lab.new_folder(args.run_dir)
        # Experiments run unwatched for hours, so the end, however it comes, is announced, with what to ask
        # Claude next: the user clears the session and pastes it.
        try:
            with background.recorded(folder / background.PID):
                lab.experiment(args.run_dir, args.question, args.config, args.vlm, args.trials, args.pose_trials, folder)
        except BaseException as error:
            background.notify("aomma experiment stopped", f"Find out why the experiment {folder} stopped ({error!r})")
            raise
        background.notify("aomma experiment finished", f"Review the finished experiment {folder}")
    elif args.command == "round":
        from aomma import rounds

        folder = args.target if args.target.is_dir() else rounds.prepare(args.target, pinned=not args.working_tree)
        with background.recorded(folder / background.PID):
            rounds.run(folder, retry_failed=args.retry_failed)
    elif args.command == "measure-change":
        from aomma import measures

        folder = measures.compare(args.gate, args.since, args.runs)
        print((folder / "compare.txt").read_text(), end="")
        print(f"-> {folder}/compare.txt; once agreed: aomma measure-record {folder} --section ... --why ...")
    elif args.command == "measure-record":
        from aomma import measures

        entry = measures.record(args.folder, args.section, args.why)
        print(f"recorded in {measures.LEDGER}: {entry['commit'][:12]}, {entry['after_pass']} of {entry['after_of']} pass; commit the ledger")
    elif args.command == "experiment-status":
        from aomma import lab

        if args.unreviewed:
            print(unreviewed(lab.DEFAULT_ROOT))
        elif args.open:
            print(open_actions(lab.DEFAULT_ROOT))
        else:
            print(experiment_status(args.folder or newest_experiment(lab.DEFAULT_ROOT)))
    elif args.command == "experiment-reviewed":
        record_review(args.folder, args.section, args.verdict, args.action, args.open)
        print(experiment_status(args.folder))
    elif args.command == "experiment-acted":
        record_action(args.folder, args.action, args.still_open)
        print(experiment_status(args.folder))
    elif args.command == "perceive":
        perceive(args.image, Path("runs") / f"{stamp}-{args.image.stem}", args.config, args.vlm)
    elif args.command == "layout":
        layout(args.run_dir)
    elif args.command == "asset":
        asset(args.run_dir, args.object_id, args.config)
    elif args.command == "assets":
        assets(args.run_dir, args.object_ids, args.config)
    elif args.command == "pose":
        pose(args.run_dir, args.object_ids)
    elif args.command == "ceiling":
        add_ceiling(args.run_dir, args.config)
    elif args.command == "square-up":
        square_up(args.run_dir)
    elif args.command == "collisions":
        collisions(args.run_dir)
    elif args.command == "depth-check":
        depth_check(args.run_dir)
    elif args.command == "split":
        split(args.run_dir, args.object_ids, args.config, args.vlm)
    elif args.command == "dashboard":
        try:
            with background.recorded(Path("runs") / "dashboard.pid"):
                from aomma import lab

                serve(load_config(args.config), JOB_LOG, Path("runs"), ["127.0.0.1", *args.host], args.port, lab.DEFAULT_ROOT)
        except KeyboardInterrupt:
            print()
    elif args.command == "import-capture":
        # Imported here because it reads this module's run-folder helpers.
        from aomma.capture import import_capture

        for run_dir in import_capture(args.scan, seed_offset=args.seed_offset):
            print(run_dir)
    elif args.command == "scan-check":
        from aomma.capture import Scan
        from aomma.scancheck import check_scan, describe

        for scan in args.scans:
            print(f"== {scan.name}")
            print("\n".join(describe(check_scan(Scan(scan)))))
    elif args.command == "ref-items":
        from aomma.capture import Scan
        from aomma.refitems import merge_saved, reference_items, reference_small

        if args.small:
            for item in reference_small(Scan(args.scan), args.config, args.vlm):
                print(f"  {item['id']}: {len(item['shots'])} shots, size {item['size']}"
                      + (f"; left out shots {', '.join(item['left_out'])}" if item["left_out"] else ""))
            return
        made = merge_saved(Scan(args.scan)) if args.merge_only else reference_items(
            Scan(args.scan), args.config, args.vlm, shots="all" if args.all_shots else "reference")
        for item in made:
            print(f"  {item['id']} ({item['kind']}): {len(item['shots'])} shots, size {item['size']}, heading {item['heading_deg']} deg"
                  + (f"; left out shots {', '.join(item['left_out'])}" if item["left_out"] else ""))
    elif args.command == "ref-large":
        from aomma.capture import Scan
        from aomma.refthings import reference_large

        for scan in args.scans:
            print(f"== {scan.name}")
            for shot, boxes in reference_large(Scan(scan)).items():
                print(f"  {shot}: " + ", ".join(f"{b['id']} {b['pixels']} px" for b in boxes))
    elif args.command == "judge-models":
        from aomma.judgemodels import describe as describe_models, judge_run as judge_models

        for run_dir in args.runs:
            print(describe_models(run_dir, judge_models(run_dir)))
    elif args.command == "judge-boxes":
        from aomma.judgeboxes import describe as describe_boxes, judge_run as judge_boxes

        for run_dir in args.runs:
            print(describe_boxes(run_dir, judge_boxes(run_dir)))
    elif args.command == "ref-small":
        from aomma.capture import Scan
        from aomma.judgesmall import reference_small as reference_small_labels

        for scan in args.scans:
            print(f"== {scan.name}")
            for shot, found in reference_small_labels(Scan(scan)).items():
                print(f"  {shot}: " + (", ".join(f"{f['id']} {f['pixels']} px" for f in found) or "-"))
    elif args.command == "judge-small":
        from aomma.judgesmall import describe as describe_small, judge_run as judge_small

        marked = []
        for run_dir in args.runs:
            result = judge_small(run_dir)
            print(describe_small(run_dir, result))
            if result.get("marked"):
                marked.append(result["marked"])
        if marked:
            found, real = sum(m["found"] for m in marked), sum(m["real"] for m in marked)
            print(f"hand-marked small things, {len(marked)} shots: {found} of {real} found ({100 * found / max(1, real):.0f}%)")
    elif args.command == "judge-large":
        from aomma.judgelarge import describe as describe_large, judge_run as judge_large

        for run_dir in args.runs:
            print(describe_large(run_dir, judge_large(run_dir)))
    elif args.command == "judge-shell":
        from aomma.judgeshell import describe as describe_shell, judge_run

        for run_dir in args.runs:
            print(describe_shell(run_dir, judge_run(run_dir)))
    elif args.command == "ref-shell":
        from aomma.capture import Scan
        from aomma.refshell import reference_shell

        for scan in args.scans:
            print(f"== {scan.name}")
            for shot, made in reference_shell(Scan(scan)).items():
                moved = ", ".join(sid + (" unseen" if s["moved_m"] is None else f" {100 * s['moved_m']:+.1f} cm")
                                  for sid, s in made["snaps"].items() if s["points"] >= 100)
                top = ", ".join(f"{k} {100 * v:.0f}%" for k, v in sorted(made["shares"].items(), key=lambda kv: -kv[1]))
                print(f"  {shot}: {moved}\n    {top}")
    elif args.command == "pose-check":
        from aomma.capture import Scan
        from aomma.posecheck import check_poses, describe as describe_poses

        for scan in args.scans:
            print(f"== {scan.name}")
            print("\n".join(describe_poses(check_poses(Scan(scan)))))
    elif args.command == "labels":
        from aomma.labels import describe, label_run

        for run_dir in args.runs:
            print(describe(run_dir, label_run(run_dir)))
    elif args.command == "score":
        from aomma.scoring import score

        print(f"-> {score(args.runs, args.into or Path('lab') / f'{stamp}-score', args.keyframes, label=args.label, scene=args.scene)}")
    elif args.command == "jobs":
        servers = load_config(args.config).comfy
        try:
            if args.once:
                snapshot(servers.all, JOB_LOG)
            else:
                watch(args.url or servers.heavy[0], JOB_LOG)
        except KeyboardInterrupt:
            print()
