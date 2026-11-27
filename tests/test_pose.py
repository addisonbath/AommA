import numpy as np
import pytest
from scipy.spatial import ConvexHull
from PIL import Image
from pxr import Usd, UsdGeom

from aomma.layout import Camera, Placement, SceneLayout
from aomma.stages.pose import (
    FLOOR,
    UPRIGHTS,
    _Target,
    euler_xyz_deg,
    matrix_from_euler_xyz_deg,
    rotation_z,
    settle_rotation,
    solve_pose,
)
from aomma.stages.score import project
from aomma.usd.assets import write_proxy_box
from aomma.usd.compose import compose_scene


def test_euler_angles_round_trip_for_every_candidate_rotation():
    for upright in UPRIGHTS:
        for yaw in (0, 33, 90, 215):
            rotation = rotation_z(yaw) @ upright
            np.testing.assert_allclose(matrix_from_euler_xyz_deg(euler_xyz_deg(rotation)), rotation, atol=1e-9)


def test_euler_angles_mean_the_same_thing_to_usd(tmp_path):
    rotation = rotation_z(40) @ UPRIGHTS[1]
    write_proxy_box(tmp_path / "box.usda", (1.0, 1.0, 1.0))
    layout = SceneLayout(
        camera=Camera.looking_at((0.0, -5.0, 1.0), (0.0, 0.0, 1.0)),
        objects=[Placement(name="box", asset="./box.usda", rotate_xyz_deg=euler_xyz_deg(rotation))],
    )
    compose_scene(layout, tmp_path / "scene.usda")
    stage = Usd.Stage.Open(str(tmp_path / "scene.usda"))
    prim = stage.GetPrimAtPath("/World/Objects/box")
    # USD matrices act on row vectors, so ours is the transpose.
    usd_matrix = np.array(UsdGeom.XformCache().GetLocalToWorldTransform(prim))[:3, :3].T
    np.testing.assert_allclose(usd_matrix, rotation, atol=1e-6)


def test_solve_pose_recovers_a_known_pose():
    # An L-shaped, two-coloured object: asymmetric, so only one pose fits.
    rng = np.random.default_rng(0)
    a = rng.random((6000, 3)) * (1.0, 0.3, 0.2)
    b = rng.random((3000, 3)) * (0.3, 0.6, 0.2) + (0.0, 0.3, 0.0)
    points = np.concatenate([a, b]) - (0.5, 0.45, 0.0)
    colors = np.concatenate([np.tile((0.9, 0.2, 0.2), (6000, 1)), np.tile((0.2, 0.3, 0.9), (3000, 1))])

    camera = Camera.looking_at((0.0, -2.0, 1.5), (0.0, 0.0, 0.0), width=400, height=300)
    true_rotation, true_scale, true_translation = rotation_z(60), 0.4, np.array([0.1, 0.2, 0.0])
    world = true_scale * points @ true_rotation.T + true_translation
    pixels, depth = project(world, camera)

    photo = np.zeros((300, 400, 3), dtype=np.uint8)
    mask = np.zeros((300, 400), dtype=bool)
    for dx in (-1, 0, 1):  # splat a little wide so the picture has no pinholes
        for dy in (-1, 0, 1):
            cols = np.clip(pixels[:, 0].astype(int) + dx, 0, 399)
            rows = np.clip(pixels[:, 1].astype(int) + dy, 0, 299)
            mask[rows, cols] = True
            photo[rows, cols] = (colors * 255).astype(np.uint8)

    pose = solve_pose("thing", points, colors, mask, Image.fromarray(photo), camera, float(np.median(depth)), contact=FLOOR)

    assert pose.iou > 0.8
    np.testing.assert_allclose(matrix_from_euler_xyz_deg(pose.rotate_xyz_deg), true_rotation, atol=0.12)
    np.testing.assert_allclose(pose.scale, true_scale, rtol=0.1)
    np.testing.assert_allclose(pose.translate, true_translation, atol=0.05)


def test_solve_pose_handles_a_mask_touching_the_photo_edge():
    rng = np.random.default_rng(0)
    points = rng.random((4000, 3)) - (0.5, 0.5, 0.0)
    colors = np.full((4000, 3), 0.5)
    camera = Camera.looking_at((0.0, -2.0, 1.0), (0.0, 0.0, 0.0), width=200, height=150)
    mask = np.zeros((150, 200), dtype=bool)
    mask[60:150, 0:80] = True  # runs off the bottom-left corner
    photo = Image.new("RGB", (200, 150), (128, 128, 128))

    pose = solve_pose("thing", points, colors, mask, photo, camera, 2.0, contact=FLOOR)

    assert pose.iou > 0.5


def test_solve_pose_places_an_object_cut_by_the_photo_bottom_where_it_stands():
    # A box whose lower part runs off the bottom of the photo: fitted whole inside
    # the cut mask, it came out too small and, put on the floor, too far back.
    rng = np.random.default_rng(0)
    points = rng.random((9000, 3)) * (0.6, 0.6, 1.0) - (0.3, 0.3, 0.0)
    colors = np.full((9000, 3), 0.5)
    camera = Camera.looking_at((0.0, -2.0, 1.5), (0.0, 0.0, 1.1), width=400, height=300)
    true_scale, true_translation = 0.9, np.array([0.05, 0.1, 0.0])
    pixels, _ = project(true_scale * points + true_translation, camera)
    shown = (pixels >= 0).all(axis=1) & (pixels[:, 0] < 400) & (pixels[:, 1] < 300)
    assert 0.1 < 1 - shown.mean() < 0.5  # a good share of it is below the frame

    mask = np.zeros((300, 400), dtype=bool)
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            cols = pixels[shown, 0].astype(int) + dx
            rows = pixels[shown, 1].astype(int) + dy
            keep = (cols >= 0) & (cols < 400) & (rows >= 0) & (rows < 300)
            mask[rows[keep], cols[keep]] = True
    photo = Image.new("RGB", (400, 300), (128, 128, 128))

    pose = solve_pose("thing", points, colors, mask, photo, camera, 2.0, contact=FLOOR, settle=False)

    np.testing.assert_allclose(pose.scale, true_scale, rtol=0.1)
    np.testing.assert_allclose(pose.translate[:2], true_translation[:2], atol=0.08)


def test_a_model_cut_by_the_frame_still_reaches_the_edge():
    # The mask runs off the bottom; the model's box is as wide as the mask but, lined up by
    # its top, stops 20 px short of the edge: it is sized up until it reaches it.
    mask = np.zeros((150, 200), dtype=bool)
    mask[50:150, 60:120] = True
    target = _Target(mask, Image.new("RGB", (200, 150)))

    factor, shift = target.box_match(np.array([60.0, 50.0]), np.array([120.0, 130.0]))

    assert factor == pytest.approx(100 / 80)
    lo_row = 90 + factor * (50 - 90) + shift[1]
    assert lo_row == pytest.approx(50)  # its top stays on the mask's


def test_settle_rotation_lays_a_tilted_slab_flat():
    rng = np.random.default_rng(0)
    slab = rng.random((5000, 3)) * (0.2, 0.3, 0.03)
    tilted = matrix_from_euler_xyz_deg((14.0, -9.0, 25.0))

    settled = settle_rotation(slab, tilted)

    heights = (slab @ settled.T)[:, 2]
    assert heights.max() - heights.min() < 0.032  # the slab's own thickness, not more
    assert settle_rotation(slab, settled) is None  # already flat: nothing to do


def test_solve_occluded_pose_sizes_a_half_hidden_object_from_its_visible_part():
    from aomma.stages.pose import solve_occluded_pose

    rng = np.random.default_rng(0)
    points = rng.random((8000, 3)) * (1.0, 0.5, 0.4) - (0.5, 0.25, 0.0)
    colors = np.full((8000, 3), 0.6)
    camera = Camera.looking_at((0.0, -3.0, 1.6), (0.0, 0.0, 0.2), width=400, height=300)
    true_scale, true_base = 0.8, np.array([0.2, 0.1, 0.0])
    pixels, depth = project(true_scale * points + true_base, camera)

    whole = np.zeros((300, 400), dtype=bool)
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            whole[np.clip(pixels[:, 1].astype(int) + dy, 0, 299), np.clip(pixels[:, 0].astype(int) + dx, 0, 399)] = True
    # Something in front hides the object's right half.
    occluders = np.zeros_like(whole)
    occluders[:, int(pixels[:, 0].mean()):] = True
    visible = whole & ~occluders
    photo = Image.fromarray(np.where(visible[..., None], 153, 0).astype(np.uint8).repeat(3, axis=2))

    pose = solve_occluded_pose(
        "thing", points, colors, visible, occluders, whole, photo, camera,
        start_scale=1.0, start_base=(0.05, 0.25, 0.0), guesses=(np.eye(3),),
    )

    assert pose.iou > 0.85
    np.testing.assert_allclose(pose.scale, true_scale, rtol=0.12)
    # Compare where the placed box stands, which does not depend on which way round it is.
    placed = pose.scale * points @ matrix_from_euler_xyz_deg(pose.rotate_xyz_deg).T + pose.translate
    low, high = placed.min(axis=0), placed.max(axis=0)
    np.testing.assert_allclose(((low[0] + high[0]) / 2, (low[1] + high[1]) / 2, low[2]), true_base, atol=0.08)


def test_level_rotation_levels_the_main_flat_surface():
    from aomma.stages.pose import level_rotation

    # A table: a big top facing up, its underside, and four small sides.
    normals = np.array([[0, 0, 1], [0, 0, -1], [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0]], dtype=float)
    areas = np.array([1.0, 1.0, 0.1, 0.1, 0.1, 0.1])
    tilted = matrix_from_euler_xyz_deg((14.0, -6.0, 40.0))
    levelled = level_rotation((normals, areas), tilted)
    np.testing.assert_allclose(levelled @ (0, 0, 1), (0, 0, 1), atol=1e-6)
    # The turn is the smallest one: about 15 degrees, not a flip or a spin.
    assert np.degrees(np.arccos((np.trace(levelled @ tilted.T) - 1) / 2)) < 16
    assert level_rotation((normals, areas), levelled) is None  # already level

    # No surface big enough, or one leaning too far to be meant level: leave it alone.
    assert level_rotation((normals, np.array([0.01, 0.01, 1, 1, 1, 1])), tilted) is None
    assert level_rotation((normals, areas), matrix_from_euler_xyz_deg((45.0, 0.0, 0.0))) is None


def test_fit_to_box_reshapes_a_model_onto_its_measured_box(tmp_path):
    from aomma.stages.pose import Pose, fit_to_box, placed_points
    from aomma.usd.assets import write_reshaped_asset

    rng = np.random.default_rng(0)
    # A model 1 x 0.5 x 0.5, posed 1.2 m long; the real thing is 1.5 x 0.4 x 0.7, turned 20 degrees.
    points = rng.random((4000, 3)) * (1.0, 0.5, 0.5) - (0.5, 0.25, 0.0)
    # It was posed 8 degrees askew of the box, which must not shear it.
    solved = Pose(id="shelf", translate=(0, 3, 0), rotate_xyz_deg=(0, 0, 28), scale=1.2, iou=0.0, color_error=0.1)
    camera = Camera.looking_at((0.0, 0.0, 1.2), (0.0, 3.0, 0.4), width=320, height=240)
    everything = np.ones((240, 320), dtype=bool)

    def outline(world):
        pixels, _ = project(world, camera)
        mask = np.zeros((240, 320), dtype=bool)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                mask[np.clip(pixels[:, 1].astype(int) + dy, 0, 239), np.clip(pixels[:, 0].astype(int) + dx, 0, 319)] = True
        return mask

    def fit(photo):
        return fit_to_box(
            solved, points, np.full((4000, 3), 0.5), (1.5, 0.4, 0.7), (0.2, 3.0, 0.0), 20.0,
            photo, ~everything, everything, Image.new("RGB", (320, 240), "grey"), camera,
        )

    # Posed both ways, the better outline is kept: the box's where the photo shows the box...
    box = rng.random((4000, 3)) * (1.5, 0.4, 0.7) - (0.75, 0.2, 0.0)
    pose, reshape = fit(outline(box @ rotation_z(20).T + (0.2, 3.0, 0.0)))
    # ...and the model's own proportions where it shows the model as posed.
    assert fit(outline(placed_points(solved, points))) is None
    assert pose.translate == (0.2, 3.0, 0.0) and pose.rotate_xyz_deg == (0.0, 0.0, 20.0)
    np.testing.assert_allclose(pose.scale, 1.5, atol=0.01)
    # The reshaped asset is the box at 1 m long: upright, centred, standing on its pivot.
    shaped = points @ reshape[:3, :3].T + reshape[:3, 3]
    np.testing.assert_allclose(shaped.max(axis=0) - shaped.min(axis=0), np.array([1.5, 0.4, 0.7]) / 1.5, atol=0.01)
    np.testing.assert_allclose(shaped.min(axis=0), (-0.5, -0.4 / 3, 0.0), atol=0.01)
    # Still a box: its corners are square, so its footprint fills its bounding rectangle.
    assert ConvexHull(shaped[:, :2]).volume > 0.97 * (1.0 * 0.4 / 1.5)

    write_proxy_box(tmp_path / "shelf.usda", (1.0, 0.5, 0.5))
    write_reshaped_asset(tmp_path / "shelf_fitted.usda", tmp_path / "shelf.usda", reshape.tolist())
    # Placed in a scene, which sets its own transform on the referencing prim, the reshaping survives.
    scene = SceneLayout(
        camera=camera,
        objects=[Placement(name="shelf", asset="./shelf_fitted.usda", translate=pose.translate, rotate_xyz_deg=pose.rotate_xyz_deg, scale=pose.scale)],
    )
    compose_scene(scene, tmp_path / "scene.usda")
    stage = Usd.Stage.Open(str(tmp_path / "scene.usda"))
    prim = next(p for p in stage.Traverse() if p.GetName() == "shelf")
    box = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_]).ComputeWorldBound(prim).ComputeAlignedRange()
    # Turned 20 degrees, a 1.5 x 0.4 box spans this much along X; it stands 0.7 tall on the floor.
    across = 1.5 * np.cos(np.radians(20)) + 0.4 * np.sin(np.radians(20))
    np.testing.assert_allclose(box.GetSize()[0], across, atol=0.03)
    np.testing.assert_allclose([box.GetMin()[2], box.GetMax()[2]], [0.0, 0.7], atol=0.02)


def test_upright_faces_say_how_far_a_model_is_askew():
    from aomma.stages.pose import _upright_faces_angle_deg

    # A cabinet's four sides, top and bottom, with a big sloping face that should not sway the answer.
    normals = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1], [0.6, 0, 0.8]], dtype=float)
    areas = np.array([0.3, 0.3, 1.0, 1.0, 0.5, 0.5, 2.0])
    for askew in (-30.0, -7.0, 0.0, 12.0, 40.0):
        assert abs(_upright_faces_angle_deg((normals, areas), rotation_z(askew)) - askew) < 1e-6


def test_a_model_is_not_marked_down_for_carrying_on_past_the_photo_edge():
    from aomma.stages.pose import _Target, _project

    # A unit cube standing on the floor, its left half outside the photo.
    rng = np.random.default_rng(0)
    points = rng.random((20000, 3)) - (0.5, 0.5, 0.0)
    camera = Camera.looking_at((0.0, -3.0, 1.0), (0.0, 0.0, 0.5), width=200, height=150)
    world = points + (-1.1, 0.0, 0.0)
    pixels, depth = _project((world - camera.matrix[:3, 3]) @ camera.matrix[:3, :3], camera)
    assert pixels[:, 0].min() < -20  # it does run off the left edge
    on_photo = (pixels >= 0).all(axis=1) & (pixels[:, 0] < 200) & (pixels[:, 1] < 150)
    mask = np.zeros((150, 200), dtype=bool)
    mask[pixels[on_photo, 1].astype(int), pixels[on_photo, 0].astype(int)] = True

    # Placed exactly where it is, it matches what the photo shows of it.
    iou, _ = _Target(mask, Image.new("RGB", (200, 150), (128, 128, 128))).compare(pixels, depth, np.full((20000, 3), 0.5))
    assert iou > 0.9


def test_a_wall_hides_what_is_behind_its_face_and_a_hidden_model_can_be_pulled_clear():
    from aomma.stages.pose import Pose, hidden_by_walls, pull_forward

    eye = np.array([0.0, 0.0, 1.2])
    # A wall 2 m wide across the view, 3 m away, facing the camera.
    wall = (np.array([-1.0, 3.0, 0.0]), np.array([2.0, 0.0, 0.0]), np.array([0.0, -1.0, 0.0]), 2.4)
    rng = np.random.default_rng(0)
    cube = rng.random((5000, 3)) - (0.5, 0.5, 0.0)  # a unit cube, pivot bottom-centre

    def placed(pose):
        return pose.scale * cube + np.array(pose.translate)

    in_front = Pose(id="box", translate=(0.0, 2.0, 0.0), rotate_xyz_deg=(0.0, 0.0, 0.0), scale=1.0, iou=0.8, color_error=0.1)
    through = in_front.model_copy(update={"translate": (0.0, 3.0, 0.0)})
    beside = in_front.model_copy(update={"translate": (3.0, 4.0, 0.0)})
    assert hidden_by_walls(placed(in_front), [wall], eye) == 0.0
    assert 0.4 < hidden_by_walls(placed(through), [wall], eye) < 0.6  # the half beyond the face
    # Past the wall's end the camera sees it, so no wall hides it: a "wall" may be an open door.
    assert hidden_by_walls(placed(beside), [wall], eye) == 0.0

    pulled = pull_forward(through, cube, [wall], eye)
    assert hidden_by_walls(placed(pulled), [wall], eye) <= 0.02
    # It came towards the camera along its lines of sight, shrinking in step, so it looks the same from there.
    share = pulled.scale / through.scale
    assert 0.8 < share < 0.9
    np.testing.assert_allclose(np.array(pulled.translate) - eye, share * (np.array(through.translate) - eye), atol=1e-9)
    assert pull_forward(in_front, cube, [wall], eye) == in_front


def test_a_floating_model_is_reseated_on_the_floor_by_dropping_or_stretching_it():
    from aomma.stages.pose import Pose, placed_points, reseat_on_floor

    rng = np.random.default_rng(0)
    box = rng.random((4000, 3)) * (0.6, 0.4, 1.0) - (0.3, 0.2, 0.0)  # 1 m tall, pivot bottom-centre
    turned = matrix_from_euler_xyz_deg((0.0, 0.0, 30.0))
    floating = Pose(id="desk", translate=(1.0, 2.0, 0.3), rotate_xyz_deg=(0.0, 0.0, 30.0), scale=0.8, iou=0.8, color_error=0.1)
    before = placed_points(floating, box)
    assert np.isclose(before[:, 2].min(), 0.3, atol=0.01)

    dropped, reshape = reseat_on_floor(floating, box, "drop")
    assert reshape is None
    np.testing.assert_allclose(placed_points(dropped, box), before - (0.0, 0.0, before[:, 2].min()), atol=1e-9)

    # Stretched: the top stays put, the bottom reaches the floor, and nothing moves sideways.
    stretched, reshape = reseat_on_floor(floating, box, "stretch")
    shaped = box @ reshape[:3, :3].T + reshape[:3, 3]
    assert np.isclose((shaped.max(axis=0) - shaped.min(axis=0)).max(), 1.0)  # and its longest side 1 m
    assert np.isclose(shaped[:, 2].min(), 0.0, atol=1e-9)  # the reshaped asset keeps the pivot at its bottom
    after = placed_points(stretched, shaped)
    np.testing.assert_allclose(after[:, :2], before[:, :2], atol=1e-9)
    np.testing.assert_allclose(after[:, 2].max(), before[:, 2].max(), atol=1e-9)
    np.testing.assert_allclose(after[:, 2].min(), 0.0, atol=1e-9)

    # A stretch is capped; what it cannot cover is dropped, so the model still stands on the floor.
    high = floating.model_copy(update={"translate": (1.0, 2.0, 0.7)})
    capped, reshape = reseat_on_floor(high, box, "stretch")
    after = placed_points(capped, box @ reshape[:3, :3].T + reshape[:3, 3])
    height = after[:, 2].max() - after[:, 2].min()
    assert np.isclose(after[:, 2].min(), 0.0, atol=1e-9) and np.isclose(height, 1.6 * 0.8, atol=0.01)

    # A model already on the floor is left alone.
    assert reseat_on_floor(dropped, box, "stretch") is None


def test_a_spike_below_a_model_does_not_hold_it_up_when_trimmed():
    from aomma.stages.pose import lowest

    rng = np.random.default_rng(0)
    body = rng.random((5000, 3)) * (2.0, 1.6, 0.5) + (0.0, 0.0, 0.5)  # a bed's body, 0.5 to 1.0 m up
    spike = np.column_stack([np.full(30, 0.1), np.full(30, 0.1), np.linspace(0.0, 0.5, 30)])  # a sliver down to the floor
    heights = np.concatenate([body, spike])[:, 2]
    assert lowest(heights) == 0.0
    assert 0.49 < lowest(heights, trim=True) < 0.52  # the body's own bottom, give or take


def test_a_hidden_objects_search_can_be_held_to_the_measured_distance():
    from aomma.stages.pose import _project, _Target

    # A unit cube's surface, 3 m in front of the camera; the depth map says 2.5 m, its near face, on its pixels.
    rng = np.random.default_rng(0)
    cube = rng.random((20000, 3)) - 0.5
    face = rng.integers(0, 3, len(cube))
    cube[np.arange(len(cube)), face] = np.sign(cube[np.arange(len(cube)), face]) * 0.5  # push each point onto a face
    camera = Camera.looking_at((0.0, 0.0, 0.0), (0.0, 1.0, 0.0), width=200, height=150)
    pixels, depth = _project((cube + (0.0, 3.0, 0.0) - camera.matrix[:3, 3]) @ camera.matrix[:3, :3], camera)
    mask = np.zeros((150, 200), dtype=bool)
    mask[pixels[:, 1].astype(int).clip(0, 149), pixels[:, 0].astype(int).clip(0, 199)] = True
    distances = np.where(mask, 2.5, np.nan)  # the cube's near face is at 2.5 m
    target = _Target(mask, Image.new("RGB", (200, 150)), distances=distances)
    # Not exactly 0: where a cell happens to hold no point of the near face, a point further back stands in.
    assert target.depth_disagreement(pixels, depth) < 0.05

    # The same outline from a cube twice the size, twice as far: as good an outline, but the wrong distance.
    far_pixels, far_depth = _project((2 * cube + (0.0, 6.0, 0.0) - camera.matrix[:3, 3]) @ camera.matrix[:3, :3], camera)
    assert abs(target.depth_disagreement(far_pixels, far_depth) - np.log(2)) < 0.08
    assert _Target(mask, Image.new("RGB", (200, 150))).depth_disagreement(pixels, depth) is None


def test_the_depth_check_sorts_each_objects_pixels_into_agreeing_too_near_and_too_far():
    from aomma.stages.depthcheck import check_scene

    measured = np.full((40, 40), 3.0)
    measured[:, :4] = np.nan  # the depth map knows nothing here
    rendered = measured.copy()
    rendered[:, :4] = 3.0
    box = np.zeros((40, 40), dtype=bool)
    box[10:30, 10:30] = True
    rendered[10:30, 10:20] = 2.0  # the left half of the object placed a third too near
    rendered[30:34, 10:30] = 2.0  # and the model pokes out below its outline, in front of the room
    rendered[0:4, 30:40] = np.inf  # a hole in the scene: nothing there at all
    summary, _, _ = check_scene(rendered, measured, {"box": box}, factor=2)

    found = summary["objects"]["box"]
    assert found["agree"] == 0.5 and found["too_near"] == 0.5 and found["too_far"] == 0.0
    room = summary["room"]
    assert room["too_near"] > 0 and room["too_far"] > 0  # the overhang and the hole both show, outside the object
    assert 0 < summary["agree"] < 1


def test_the_depth_check_blames_a_model_standing_in_front_of_others_not_the_others():
    from aomma.stages.depthcheck import check_scene

    measured = np.full((40, 40), 3.0)
    washer = np.zeros((40, 40), dtype=bool)
    washer[0:40, 20:40] = True
    slipper = np.zeros((40, 40), dtype=bool)
    slipper[36:40, 0:4] = True
    rendered = measured.copy()
    blob = np.zeros((40, 40), dtype=bool)
    blob[10:30, 10:30] = True  # a model far too big, standing in front of a quarter of the washer
    rendered[blob] = 1.5
    summary, _, _ = check_scene(rendered, measured, {"washer": washer, "slipper": slipper}, factor=2, shown={"slipper": blob})

    found = summary["objects"]
    # The washer is judged only where it shows, and agrees there.
    assert found["washer"]["agree"] == 1.0 and found["washer"]["hidden_by_others"] == 0.25
    assert found["slipper"]["over_others"] == 12.5  # 50 of the washer's cells, against 4 of its own
    # Without the render's map of which model is where, the washer would be blamed for the blob.
    assert check_scene(rendered, measured, {"washer": washer, "slipper": slipper}, factor=2)[0]["objects"]["washer"]["agree"] == 0.75


def test_models_passing_through_each_other_are_found_and_touching_ones_are_not():
    from aomma.stages.collision import find_collisions

    rng = np.random.default_rng(0)

    def box_surface(size, at, count=30000):
        points = rng.random((count, 3)) - 0.5
        face = rng.integers(0, 3, count)
        points[np.arange(count), face] = np.sign(points[np.arange(count), face]) * 0.5
        return points * size + at

    bed = box_surface((2.0, 1.6, 0.5), (0.0, 0.0, 0.25))
    nightstand_inside = box_surface((0.5, 0.4, 0.5), (1.1, 0.0, 0.25))  # 0.2 m of its 0.5 m width inside the bed
    pillow_on_top = box_surface((0.6, 0.4, 0.2), (0.0, 0.0, 0.6))  # rests on the bed's top
    found = find_collisions({"bed": bed, "nightstand": nightstand_inside, "pillow": pillow_on_top})
    assert [entry["objects"] for entry in found] == [["bed", "nightstand"]]
    # 0.2 of its 0.5 m width is inside, 40% of it; the allowance for touching (a voxel off every face) takes some away.
    assert 0.15 < found[0]["share_of_smaller"] < 0.45


def test_a_model_generated_from_the_photo_is_never_turned_to_show_its_back():
    from aomma.stages.pose import _rotation_x, turned_away

    facing = rotation_z(10) @ _rotation_x(-30)  # looked down on, a little to the side
    assert not turned_away(facing, facing)
    assert not turned_away(rotation_z(80) @ facing, facing)
    assert turned_away(rotation_z(100) @ facing, facing) and turned_away(rotation_z(180) @ facing, facing)
    # A front pointing at the sky (a book on its back) has no turn about the vertical to judge.
    assert not turned_away(_rotation_x(90), np.eye(3))



def test_fit_to_box_tries_the_boxs_four_sides_when_asked():
    from aomma.stages.pose import Pose, fit_to_box

    rng = np.random.default_rng(0)
    # A bed: a mattress 2 x 1.4 x 0.4 with a headboard 1 m tall at its -x end; posed lying across its box (RoomPlan's,
    # turned 90 degrees), as BedroomScan's bed was.
    mattress = rng.random((4000, 3)) * (2.0, 1.4, 0.4) - (1.0, 0.7, 0.0)
    headboard = rng.random((1500, 3)) * (0.1, 1.4, 1.0) - (1.0, 0.7, 0.0)
    points = np.concatenate([mattress, headboard]) / 2.0  # the asset's longest side 1 m
    solved = Pose(id="bed", translate=(0, 4, 0), rotate_xyz_deg=(0, 0, 0), scale=2.0, iou=0.0, color_error=0.1)
    camera = Camera.looking_at((2.5, 1.0, 1.6), (0.0, 4.0, 0.3), width=320, height=240)
    everything = np.ones((240, 320), dtype=bool)
    # The real bed stands along the box, its headboard at the box's far end.
    real = (points * 2.0) @ rotation_z(90).T + (0.0, 4.0, 0.0)
    pixels, _ = project(real, camera)
    photo = np.zeros((240, 320), dtype=bool)
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            photo[np.clip(pixels[:, 1].astype(int) + dy, 0, 239), np.clip(pixels[:, 0].astype(int) + dx, 0, 319)] = True

    def fit(turns):
        return fit_to_box(
            solved, points, np.full((len(points), 3), 0.5), (2.0, 1.4, 1.0), (0.0, 4.0, 0.0), 90.0,
            photo, ~everything, everything, Image.new("RGB", (320, 240), "grey"), camera, turns=turns,
        )

    squashed, turned = fit((0.0,)), fit((0.0, 90.0, 180.0, 270.0))
    assert turned is not None and turned[0].iou > 0.8
    assert squashed is None or squashed[0].iou < turned[0].iou - 0.1
    # Turned along the box, its headboard where the real one stands.
    pose, reshape = turned
    shaped = pose.scale * (points @ reshape[:3, :3].T + reshape[:3, 3]) @ rotation_z(pose.rotate_xyz_deg[2]).T + pose.translate
    np.testing.assert_allclose(shaped[4000:].mean(axis=0), real[4000:].mean(axis=0), atol=0.1)


def test_a_guiding_box_is_taken_even_where_the_models_own_pose_fits_better():
    from aomma.stages.pose import Pose, fit_to_box, placed_points

    rng = np.random.default_rng(1)
    points = rng.random((3000, 3)) * (1.0, 0.5, 0.5) - (0.5, 0.25, 0.0)
    solved = Pose(id="sofa", translate=(0, 3, 0), rotate_xyz_deg=(0, 0, 0), scale=1.0, iou=0.0, color_error=0.1)
    camera = Camera.looking_at((0.0, 0.0, 1.2), (0.0, 3.0, 0.4), width=320, height=240)
    everything = np.ones((240, 320), dtype=bool)
    pixels, _ = project(placed_points(solved, points), camera)
    photo = np.zeros((240, 320), dtype=bool)
    photo[np.clip(pixels[:, 1].astype(int), 0, 239), np.clip(pixels[:, 0].astype(int), 0, 319)] = True

    def fit(guide):
        return fit_to_box(
            solved, points, np.full((3000, 3), 0.5), (1.6, 0.9, 0.8), (0.1, 3.2, 0.0), 0.0,
            photo, ~everything, everything, Image.new("RGB", (320, 240), "grey"), camera, guide=guide,
        )

    # The photo shows the model as posed, so a measured box loses; RoomPlan's is taken all the same, as a place only:
    # the model keeps its own size and shape, its bottom-centre where the box's is.
    assert fit(False) is None
    pose, reshape = fit(True)
    assert reshape is None and pose.scale == 1.0
    placed = placed_points(pose, points)
    low, high = placed.min(axis=0), placed.max(axis=0)
    np.testing.assert_allclose(((low[0] + high[0]) / 2, (low[1] + high[1]) / 2, low[2]), (0.1, 3.2, 0.0), atol=1e-6)
    # Placed rigidly: every point keeps its distance from the others, and the height is the model's own.
    np.testing.assert_allclose(np.linalg.norm(placed - placed[0], axis=1), np.linalg.norm(points - points[0], axis=1), atol=1e-6)
    np.testing.assert_allclose(high[2] - low[2], np.ptp(points[:, 2]), atol=1e-6)


def test_a_guiding_box_turns_the_model_without_reshaping_it():
    from aomma.stages.pose import Pose, fit_to_box, placed_points

    # A chair 0.5 x 0.5 x 1.0 m, its seat a level surface at 0.65 of its height; RoomPlan's box turned 90 deg and
    # taller than the chair. Stretched, the seat was taken for the box's top and the chair stood 1.5 times too tall.
    rng = np.random.default_rng(3)
    points = rng.random((3000, 3)) * (0.5, 0.5, 1.0) - (0.25, 0.25, 0.0)
    normals = np.tile((0.0, 0.0, 1.0), (100, 1))
    centres = np.column_stack([rng.random((100, 2)) * 0.5 - 0.25, np.full(100, 0.65)])
    facets = (normals, np.ones(100), centres)
    solved = Pose(id="chair", translate=(0, 3, 0), rotate_xyz_deg=(0, 0, 0), scale=1.0, iou=0.5, color_error=0.1)
    camera = Camera.looking_at((0.0, 0.0, 1.2), (0.0, 3.0, 0.4), width=320, height=240)
    everything = np.ones((240, 320), dtype=bool)
    pose, reshape = fit_to_box(
        solved, points, np.full((3000, 3), 0.5), (0.6, 0.6, 1.0), (0.05, 3.0, 0.0), 90.0,
        everything, ~everything, everything, Image.new("RGB", (320, 240), "grey"), camera, facets, guide=True,
    )
    assert reshape is None and pose.scale == 1.0
    placed = placed_points(pose, points)
    np.testing.assert_allclose(np.ptp(placed[:, 2]), np.ptp(points[:, 2]), atol=1e-6)
    assert abs(pose.rotate_xyz_deg[2] % 90.0) < 1e-6 or abs(pose.rotate_xyz_deg[2] % 90.0 - 90.0) < 1e-6


def test_the_outline_places_a_guiding_box_within_its_slack():
    from aomma.stages.pose import Pose, fit_to_box, placed_points

    rng = np.random.default_rng(2)
    points = rng.random((4000, 3)) * (1.0, 0.5, 0.5) - (0.5, 0.25, 0.0)
    solved = Pose(id="chair", translate=(0, 3, 0), rotate_xyz_deg=(0, 0, 0), scale=1.0, iou=0.0, color_error=0.1)
    camera = Camera.looking_at((0.0, 0.0, 1.2), (0.0, 3.0, 0.4), width=320, height=240)
    everything = np.ones((240, 320), dtype=bool)
    pixels, _ = project(placed_points(solved, points), camera)
    photo = np.zeros((240, 320), dtype=bool)
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            photo[np.clip(pixels[:, 1].astype(int) + dy, 0, 239), np.clip(pixels[:, 0].astype(int) + dx, 0, 319)] = True
    # RoomPlan's box is the right size, 7 cm off where the chair stands.
    pose, _ = fit_to_box(
        solved, points, np.full((4000, 3), 0.5), (1.0, 0.5, 0.5), (0.06, 3.035, 0.0), 0.0,
        photo, ~everything, everything, Image.new("RGB", (320, 240), "grey"), camera, guide=True, shift_within=0.1,
    )
    np.testing.assert_allclose(pose.translate, (0.0, 3.0, 0.0), atol=0.02)
