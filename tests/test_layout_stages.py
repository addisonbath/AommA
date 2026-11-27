import numpy as np

from aomma.layout import Camera
from aomma.stages.inventory import Inventory, InventoryObject
from aomma.stages.proxy import Proxy, depth_scale_correction
from aomma.stages.score import project, silhouette
from aomma.stages.world import fit_camera_to_world, fit_floor, to_world


def test_floor_fit_recovers_camera_height_and_tilt():
    # A camera 1.5 m up, pitched 30 degrees down, looking at a flat floor.
    camera = Camera.looking_at((0.0, 0.0, 1.5), (0.0, 1.5 / np.tan(np.radians(30)), 0.0))
    world_to_camera = np.linalg.inv(camera.matrix)
    xs, ys = np.meshgrid(np.linspace(-1, 1, 40), np.linspace(1, 4, 40))
    floor_world = np.stack([xs, ys, np.zeros_like(xs)], axis=-1)
    points = to_world(floor_world, world_to_camera).astype(np.float32)

    everything = np.ones(points.shape[:2], dtype=bool)
    fitted = fit_camera_to_world(points, everything, ~everything)

    np.testing.assert_allclose(fitted, camera.matrix, atol=1e-4)


def test_with_the_up_known_only_the_floors_height_is_found_and_a_measured_floor_wins():
    # A camera 1.5 m up, pitched 30 degrees down. The point map's floor is tilted 3 degrees and 6 cm high,
    # as MoGe's can be; the up comes from the pose.
    camera = Camera.looking_at((0.0, 0.0, 1.5), (0.0, 1.5 / np.tan(np.radians(30)), 0.0))
    world_to_camera = np.linalg.inv(camera.matrix)
    up = world_to_camera[:3, 2]
    xs, ys = np.meshgrid(np.linspace(-1, 1, 40), np.linspace(1, 4, 40))
    at = lambda z: to_world(np.stack([xs, ys, z], axis=-1), world_to_camera).astype(np.float32)
    points = at(0.06 + np.tan(np.radians(3)) * (ys - 2.5))
    floor = np.ones(points.shape[:2], dtype=bool)
    nothing = np.zeros_like(floor)

    fitted = fit_camera_to_world(points, floor, nothing, up=up)
    np.testing.assert_allclose(fitted[2, :3], up, atol=1e-6)  # its tilt the pose's
    assert abs(fitted[2, 3] - 1.44) < 0.005  # its height the point map's floor

    lidar = at(np.zeros_like(xs)).reshape(-1, 3)
    fitted = fit_camera_to_world(points, floor, nothing, up=up, measured=(lidar, lidar))
    assert abs(fitted[2, 3] - 1.5) < 1e-3

    # No floor mask: the lowest level that many points share, not the bed top above it.
    level = np.concatenate([at(np.full_like(xs, 0.06)), at(np.full_like(xs, 0.5))])
    fitted = fit_camera_to_world(level, np.zeros(level.shape[:2], bool), np.zeros(level.shape[:2], bool), up=up)
    assert abs(fitted[2, 3] - 1.44) < 1e-3


def test_projection_and_silhouette():
    camera = Camera.looking_at((0.0, -4.0, 0.5), (0.0, 0.0, 0.5), width=200, height=100)
    pixels, depth = project(np.array([[0.0, 0.0, 0.5]]), camera)
    np.testing.assert_allclose(pixels[0], (100, 50), atol=1e-6)
    np.testing.assert_allclose(depth[0], 4.0)

    box = Proxy(id="box", size=(1.0, 1.0, 1.0), translate=(0.0, 0.0, 0.0), yaw_deg=0.0, color=(1, 1, 1))
    shape = silhouette(box, camera)
    assert shape[50, 100] and not shape[5, 5]


def test_depth_scale_correction_is_the_median_size_ratio():
    def obj(oid, longest):
        return InventoryObject(id=oid, bbox_2d=(0, 0, 1, 1), description="", size_m=(longest, 0.1, 0.1), on="floor")

    def box(oid, longest):
        return Proxy(id=oid, size=(0.1, longest, 0.1), translate=(0, 0, 0), yaw_deg=0, color=(1, 1, 1))

    inventory = Inventory(objects=[obj("a", 0.8), obj("b", 0.4), obj("c", 5.0)])
    proxies = [box("a", 1.0), box("b", 0.5), box("c", 1.0)]  # "c" is a wild size guess
    assert depth_scale_correction(proxies, inventory) == 0.8


def test_fit_walls_finds_two_walls_meeting_at_a_corner():
    from PIL import Image

    from aomma.stages.room import fit_walls

    rng = np.random.default_rng(1)
    back = np.stack([rng.uniform(-2, 2, 3000), np.full(3000, 3.0), rng.uniform(0.2, 2.0, 3000)], axis=1)
    side = np.stack([np.full(2000, 2.0), rng.uniform(0, 3, 2000), rng.uniform(0.2, 2.0, 2000)], axis=1)
    points = np.concatenate([back, side]).reshape(50, 100, 3)
    image = Image.new("RGB", (100, 50), (40, 90, 160))

    walls = fit_walls(image, points, np.ones((50, 100), dtype=bool), np.array([0.0, 0.0, 1.5]))

    assert len(walls) == 2
    back_wall, side_wall = walls
    assert abs(back_wall.size[0] - 4.0) < 0.2 and abs(side_wall.size[0] - 3.0) < 0.2
    # Lengths run along x for the back wall and along y for the side wall.
    assert min(back_wall.yaw_deg % 180, 180 - back_wall.yaw_deg % 180) < 2
    assert abs(side_wall.yaw_deg % 180 - 90) < 2
    # Each slab sits just behind its plane, away from the camera.
    assert 3.0 < back_wall.translate[1] < 3.06 and 2.0 < side_wall.translate[0] < 2.06


def test_fit_walls_squares_up_and_joins_walls_that_stop_short_of_the_corner():
    from PIL import Image

    from aomma.stages.room import fit_walls

    rng = np.random.default_rng(2)
    # The back wall stops 30 cm short of the corner at (2, 3); the side wall is 3 degrees off square.
    back = np.stack([rng.uniform(-2, 1.7, 3000), np.full(3000, 3.0), rng.uniform(0.2, 2.0, 3000)], axis=1)
    t = rng.uniform(0.0, 2.7, 2000)
    lean = np.radians(3)
    side = np.stack([2.0 + np.sin(lean) * (2.7 - t), 0.3 + np.cos(lean) * t, rng.uniform(0.2, 1.6, 2000)], axis=1)
    points = np.concatenate([back, side]).reshape(50, 100, 3)

    walls = fit_walls(Image.new("RGB", (100, 50)), points, np.ones((50, 100), dtype=bool), np.array([0.0, 0.0, 1.5]))

    assert len(walls) == 2
    assert abs((walls[0].yaw_deg - walls[1].yaw_deg) % 180 - 90) < 1e-6  # exactly square
    assert walls[0].size[2] == walls[1].size[2]  # same height

    def ends(wall):
        along = np.array([np.cos(np.radians(wall.yaw_deg)), np.sin(np.radians(wall.yaw_deg))]) * wall.size[0] / 2
        return [np.array(wall.translate[:2]) - along, np.array(wall.translate[:2]) + along]

    # Some end of one wall coincides with some end of the other: they meet at a corner.
    assert min(np.linalg.norm(p - q) for p in ends(walls[0]) for q in ends(walls[1])) < 1e-6


def test_flat_art_is_fitted_to_the_wall_and_painted_from_the_photo(tmp_path):
    from PIL import Image, ImageDraw
    from pxr import Usd, UsdGeom

    from aomma.stages.flat import fit_flat_art, flat_art_texture, front_corners
    from aomma.usd.assets import write_picture_slab

    # A wall at y = 2 facing the camera, seen off-centre so the picture is in perspective.
    camera = Camera.looking_at((0.8, -2.0, 1.4), (0.0, 2.0, 1.4), width=400, height=300)
    wall = (np.array([0.0, -1.0, 0.0]), -2.0)
    # A 0.6 x 0.9 m picture, 3 cm deep: its front face, left half red and right half blue.
    front = np.array([[-0.3, 1.97, 1.0], [0.3, 1.97, 1.0], [0.3, 1.97, 1.9], [-0.3, 1.97, 1.9]])
    photo = Image.new("RGB", (400, 300), "white")
    draw = ImageDraw.Draw(photo)
    corners = project(front, camera)[0]
    middle = [(corners[0] + corners[1]) / 2, (corners[2] + corners[3]) / 2]
    draw.polygon([tuple(corners[0]), tuple(middle[0]), tuple(middle[1]), tuple(corners[3])], fill="red")
    draw.polygon([tuple(middle[0]), tuple(corners[1]), tuple(corners[2]), tuple(middle[1])], fill="blue")
    mask = np.asarray(photo.convert("L")) < 250

    art = fit_flat_art(mask, camera, wall, 0.03)
    np.testing.assert_allclose(art.size, (0.6, 0.03, 0.9), atol=0.02)
    np.testing.assert_allclose(art.translate, (0.0, 1.985, 1.0), atol=0.02)
    np.testing.assert_allclose(front_corners(art), front, atol=0.02)
    assert art.iou > 0.95

    # Straightened, the picture is upright and unmirrored: red on the left, blue on the right.
    texture = flat_art_texture(photo, art, camera)
    assert abs(texture.width / texture.height - 0.6 / 0.9) < 0.02
    assert texture.getpixel((texture.width // 4, texture.height // 2))[0] > 200
    assert texture.getpixel((3 * texture.width // 4, texture.height // 2))[2] > 200

    texture.save(tmp_path / "picture.png")
    write_picture_slab(tmp_path / "picture.usda", (0.6, 0.03, 0.9), tmp_path / "picture.png")
    stage = Usd.Stage.Open(str(tmp_path / "picture.usda"))
    mesh = UsdGeom.Mesh(stage.GetPrimAtPath("/Asset/Geom"))
    points = np.array(mesh.GetPointsAttr().Get())
    indices = np.array(mesh.GetFaceVertexIndicesAttr().Get()).reshape(6, 4)
    st = np.array(UsdGeom.PrimvarsAPI(mesh).GetPrimvar("st").Get()).reshape(6, 4, 2)
    # The +Y face carries the whole picture, with u running from +X (the viewer's left) to -X.
    for corner, (u, v) in zip(points[indices[3]], st[3]):
        assert corner[1] > 0
        assert u == (0.0 if corner[0] > 0 else 1.0) and v == (1.0 if corner[2] > 0 else 0.0)


def test_outline_snaps_to_the_frame_and_the_frame_is_raised():
    from PIL import Image, ImageDraw

    from aomma.stages.flat import find_outline, frame_normal_map, frame_widths, snap_outline

    # A dark-framed pale picture on a blue wall; the mask misses the frame on two sides.
    photo = Image.new("RGB", (300, 300), (30, 90, 120))
    draw = ImageDraw.Draw(photo)
    draw.rectangle((60, 50, 239, 249), fill=(70, 40, 20))
    draw.rectangle((75, 65, 224, 234), fill=(235, 230, 220))
    mask = np.zeros((300, 300), dtype=bool)
    mask[50:242, 68:240] = True

    outline = find_outline(mask)
    np.testing.assert_allclose(outline, [(68.5, 241.5), (239.5, 241.5), (239.5, 50.5), (68.5, 50.5)])
    snapped = snap_outline(photo, outline)
    np.testing.assert_allclose(snapped, [(60, 250), (240, 250), (240, 50), (60, 50)], atol=1.5)

    picture = photo.crop((60, 50, 240, 250))
    widths = frame_widths(picture)
    np.testing.assert_allclose(widths, (15, 15, 15, 15), atol=2)
    normals = np.asarray(frame_normal_map(picture)).astype(float) / 255 * 2 - 1
    # Flat in the middle and on the frame's top; on the left bevel the surface leans right, into the picture.
    np.testing.assert_allclose(normals[125, 90], (0, 0, 1), atol=0.02)
    assert normals[125, 15, 0] > 0.2
    # No frame, no relief.
    assert frame_normal_map(Image.new("RGB", (200, 260), (235, 230, 220))) is None



def test_resting_objects_are_stood_on_their_support_and_kept_inside_it():
    from aomma.stages.support import rest_on, support_height

    # A shelf unit 1 m wide and 0.3 m deep: a top at 0.6 and a lower board at 0.3, as fine grids of level facets.
    xs, ys = np.meshgrid(np.linspace(-0.5, 0.5, 51), np.linspace(-0.15, 0.15, 16))
    grid = np.stack([xs.ravel(), ys.ravel()], axis=1)
    centres = np.concatenate([np.column_stack([grid, np.full(len(grid), height)]) for height in (0.3, 0.6)])
    facets = (np.tile([0.0, 0.0, 1.0], (len(centres), 1)), np.full(len(centres), 0.0004), centres)

    # A lamp hovering 4 cm over the top comes down onto it; a book sunk 3 cm into the lower board comes up.
    lamp = (np.array([-0.05, -0.05, 0.64]), np.array([0.05, 0.05, 0.9]))
    book = (np.array([0.2, -0.1, 0.27]), np.array([0.25, 0.1, 0.5]))
    assert abs(support_height(facets, *lamp) - 0.6) < 0.01
    assert abs(support_height(facets, *book) - 0.3) < 0.01
    np.testing.assert_allclose(rest_on(*lamp, facets, 0.0), (0, 0, -0.04), atol=0.01)
    np.testing.assert_allclose(rest_on(*book, facets, 0.0), (0, 0, 0.03), atol=0.01)
    # One sticking 5 cm out of the front is pushed back in; one half a metre up is not this shelf's business.
    np.testing.assert_allclose(rest_on(np.array([0.0, -0.2, 0.6]), np.array([0.1, -0.1, 0.7]), facets, 0.0), (0, 0.05, 0), atol=0.011)
    np.testing.assert_allclose(rest_on(np.array([0.0, 0.0, 1.1]), np.array([0.1, 0.1, 1.3]), facets, 0.0), (0, 0, 0), atol=1e-9)


def test_a_wall_smeared_into_a_thick_layer_is_one_wall_behind_what_stands_against_it():
    from PIL import Image

    from aomma.stages.room import fit_walls

    rng = np.random.default_rng(3)
    # The depth map spreads the wall at y = 3 over 15 cm; a lamp against it reaches 8 cm further back.
    wall = np.stack([rng.uniform(-2, 2, 5000), 3.0 + rng.uniform(-0.075, 0.075, 5000), rng.uniform(0.2, 2.0, 5000)], axis=1)
    lamp = np.stack([rng.uniform(-0.2, 0.2, 300), rng.uniform(2.9, 3.155, 300), rng.uniform(0.5, 0.9, 300)], axis=1)

    walls = fit_walls(Image.new("RGB", (100, 50)), wall.reshape(50, 100, 3), np.ones((50, 100), dtype=bool),
                      np.array([0.0, 0.0, 1.5]), in_front=lamp)

    assert len(walls) == 1
    face = walls[0].translate[1] - walls[0].size[1] / 2
    assert 3.15 < face < 3.25


def test_a_wall_turning_away_past_another_wall_end_does_not_set_that_wall_back():
    from PIL import Image

    from aomma.stages.room import fit_walls

    rng = np.random.default_rng(5)
    # One wall's mask: the wall at y = 3 from x = -2 to 1, and a little of the wall turning away behind it at x = 1.
    wall = np.stack([rng.uniform(-2, 1, 4700), np.full(4700, 3.0), rng.uniform(0.2, 2.0, 4700)], axis=1)
    turn = np.stack([np.full(300, 1.0), rng.uniform(3.0, 3.25, 300), rng.uniform(0.2, 2.0, 300)], axis=1)

    walls = fit_walls(Image.new("RGB", (100, 50)), np.concatenate([wall, turn]).reshape(50, 100, 3),
                      np.ones((50, 100), dtype=bool), np.array([0.0, 0.0, 1.5]), max_walls=1)

    face = walls[0].translate[1] - walls[0].size[1] / 2
    assert 2.99 < face < 3.02


def test_walls_on_one_line_are_merged_and_a_chimney_breast_is_not():
    from aomma.stages.proxy import Proxy
    from aomma.stages.room import merge_walls

    def wall(wid, x0, x1, y, yaw=0.0):
        return Proxy(id=wid, size=(x1 - x0, 0.05, 2.4), translate=((x0 + x1) / 2, y, 0.0), yaw_deg=yaw, color=(0.5, 0.5, 0.5))

    # A wall from x = -2 to 1, and its part at the window, 1 to 2, fitted 4 cm behind and a degree off.
    merged = merge_walls([wall("window_part", 1.0, 2.0, 3.04, 1.0), wall("desk_wall", -2.0, 1.0, 3.0)])
    assert len(merged) == 1 and merged[0].id == "desk_wall"
    assert abs(merged[0].size[0] - 4.0) < 0.02 and abs(merged[0].translate[0]) < 0.02 and merged[0].translate[1] == 3.0

    # A chimney breast stands 35 cm out from the wall either side of it, which is one wall running on behind it.
    merged = merge_walls([wall("left", -2.0, -0.5, 3.0), wall("breast", -0.5, 0.5, 2.65), wall("right", 0.5, 2.0, 3.0)])
    assert sorted(w.id for w in merged) == ["breast", "left"] and abs(merged[0].size[0] - 4.0) < 1e-6


def _seen(camera: Camera, *patches: tuple[np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:
    """A point map of `patches` (each world x, y, z grids), stacked down the image, in `camera`'s frame."""
    world_to_camera = np.linalg.inv(camera.matrix)
    return np.concatenate([to_world(np.stack(patch, axis=-1), world_to_camera) for patch in patches]).astype(np.float32)


def _risers(count: int) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Upright faces stepping back and up, 10 cm each: a backsplash over a counter, cabinet fronts, an oven door."""
    patches = []
    for n in range(count):
        xs, zs = np.meshgrid(np.linspace(-1, 1, 30), np.linspace(0.5, 0.6, 30) + 0.1 * n)
        patches.append((xs, np.full_like(xs, 2.5 + 0.1 * n), zs))
    return patches


def test_without_a_floor_mask_the_floor_is_found_among_the_background_points():
    camera = Camera.looking_at((0.0, 0.0, 1.5), (0.0, 3.0, 0.8))
    xs, ys = np.meshgrid(np.linspace(-1, 1, 30), np.linspace(1.0, 2.0, 30))
    # A little floor, and many upright faces whose edges a plane 45 degrees off level runs along: it passes within
    # 3 cm of more points than the floor has, but through surfaces facing across it (KitchenScan 007).
    points = _seen(camera, (xs, ys, np.zeros_like(xs)), *_risers(20))
    nothing = np.zeros(points.shape[:2], dtype=bool)

    found = fit_floor(points, nothing, nothing)
    np.testing.assert_allclose(found.camera_to_world, camera.matrix, atol=1e-3)
    assert not found.seen  # found, not shown by a mask: it may be a counter's top


def test_a_floor_mask_on_a_wall_is_not_the_floor():
    # KitchenScan 011: no floor in view, and the floor mask was the tiled backsplash.
    camera = Camera.looking_at((0.0, 0.0, 1.5), (0.0, 3.0, 0.8))
    xs, ys = np.meshgrid(np.linspace(-1, 1, 30), np.linspace(1.0, 2.0, 30))
    wall_x, wall_z = np.meshgrid(np.linspace(-1, 1, 30), np.linspace(0.9, 1.4, 40))
    points = _seen(camera, (xs, ys, np.zeros_like(xs)), (wall_x, np.full_like(wall_x, 2.4), wall_z))
    on_wall = np.zeros(points.shape[:2], dtype=bool)
    on_wall[30:] = True
    nothing = np.zeros_like(on_wall)

    found = fit_floor(points, on_wall, nothing)
    assert not found.seen
    np.testing.assert_allclose(found.camera_to_world, camera.matrix, atol=1e-3)  # the floor, by the search

    # With the up known, the wall's points do not set the floor's height: the lowest level does.
    up = np.linalg.inv(camera.matrix)[:3, 2]
    found = fit_floor(points, on_wall, nothing, up=up)
    assert not found.seen and abs(found.camera_to_world[2, 3] - 1.5) < 1e-3


def test_a_baseboard_on_a_wall_the_depth_map_bows_is_found_and_given_to_the_wall():
    from PIL import Image

    from aomma.stages.room import find_baseboard, wall_foot_pixels

    # A blue wall whose depth map bows out towards the floor (as MoGe's does; here 0 at 1 m up, 6 cm at the foot, twice
    # BedroomScan 007's back wall), with a white baseboard 10 cm high standing 1.5 cm proud of it. The baseboard is
    # 6-7.5 cm out, farther from the wall's median surface (about 2 cm out) than a baseboard is let stand, but on the
    # surface just above it.
    wall = Proxy(id="wall_1", size=(4.0, 0.05, 2.4), translate=(0.0, 3.025, 0.0), yaw_deg=0.0, color=(0, 0, 1))
    eye = np.array([0.0, 0.0, 1.4])
    xs, zs = np.meshgrid(np.linspace(-1.5, 1.5, 120), np.linspace(1.0, 0.0, 200))
    bow = 0.06 * (1 - zs)
    proud = np.where(zs < 0.10, 0.015, 0.0)
    points = np.stack([xs, 3.0 - bow - proud, zs], axis=-1)
    blue, white = np.array([40, 90, 140]), np.array([240, 240, 240])
    colours = np.where((zs < 0.10)[..., None], white, blue).astype(np.uint8)
    everywhere = np.ones(xs.shape, dtype=bool)
    paint = zs >= 0.10  # the segmenter's wall mask: the paint, not the baseboard

    found = find_baseboard(Image.fromarray(colours), points, wall, eye, everywhere, paint, (blue / 255) ** 2.2).baseboard
    assert found is not None and abs(found.height - 0.10) < 0.011
    shown = wall_foot_pixels(points, wall, eye, everywhere, paint)
    assert shown[zs < 0.10].all()
    # A nightstand against the wall, its front 40 cm out, is not given to it.
    stand = (np.abs(xs) < 0.3) & (zs < 0.5)
    out = points.copy()
    out[stand, 1] -= 0.4
    shown = wall_foot_pixels(out, wall, eye, everywhere, paint & ~stand)
    assert not shown[stand].any() and shown[~stand & (zs < 0.10)].all()


def test_baseboard_is_the_band_at_the_bottom_of_a_wall_in_another_colour():
    from PIL import Image

    from aomma.stages.room import find_baseboard

    # A blue wall 3 m in front of the camera, its face at y = 3, with a white baseboard 10 cm high.
    wall = Proxy(id="wall_1", size=(4.0, 0.05, 2.4), translate=(0.0, 3.025, 0.0), yaw_deg=0.0, color=(0, 0, 1))
    eye = np.array([0.0, 0.0, 1.4])
    xs, zs = np.meshgrid(np.linspace(-1.5, 1.5, 120), np.linspace(1.0, 0.0, 200))
    points = np.stack([xs, np.full_like(xs, 3.0), zs], axis=-1)
    blue, white = np.array([40, 90, 140]), np.array([240, 240, 240])
    linear_blue = (blue / 255) ** 2.2
    everywhere = np.ones(xs.shape, dtype=bool)

    def look(colours: np.ndarray, at: np.ndarray = points, allowed: np.ndarray = everywhere, wall_pixels: np.ndarray = everywhere):
        return find_baseboard(Image.fromarray(colours.astype(np.uint8)), at, wall, eye, allowed, wall_pixels, linear_blue)

    def srgb(color) -> np.ndarray:
        return np.array(color) ** (1 / 2.2) * 255

    found = look(np.where((zs < 0.10)[..., None], white, blue)).baseboard
    assert found is not None and abs(found.height - 0.10) < 0.011
    np.testing.assert_allclose(srgb(found.color), white, atol=1)
    # Half of it in shade: its colour is that of the lit half.
    shaded = np.where((zs < 0.10)[..., None], np.where((xs < 0)[..., None], white * 0.5, white), blue)
    np.testing.assert_allclose(srgb(look(shaded).baseboard.color), white, atol=1)
    # A wall darker near the floor than its median is compared with itself just above the band.
    darker = np.where((zs < 0.10)[..., None], white, np.where((zs < 0.5)[..., None], blue * 0.6, blue))
    assert abs(look(darker).baseboard.height - 0.10) < 0.011
    # A nightstand against it, its front 40 cm out, is not a baseboard.
    stand = (np.abs(xs) < 0.3) & (zs < 0.5)
    out = points.copy()
    out[stand, 1] -= 0.4
    nightstand = np.where(stand[..., None], [90, 60, 30], np.where((zs < 0.10)[..., None], white, blue))
    np.testing.assert_allclose(srgb(look(nightstand, out, wall_pixels=~stand).baseboard.color), white, atol=1)
    # Its top edge blended with the wall in the photo does not tint it.
    blended = np.where((zs < 0.10)[..., None], np.where((zs > 0.085)[..., None], (white + blue) / 2 + [0, 20, 60], white), blue)
    np.testing.assert_allclose(srgb(look(blended).baseboard.color), white, atol=1)
    # Light trim on a light wall, too faint to be found as a baseboard, is not taken for plain wall either.
    beige, trim = np.array([125, 118, 108]), np.array([137, 131, 122])
    faint = look(np.where((zs < 0.10)[..., None], trim, beige))
    assert faint.baseboard is None and not faint.plain
    # Nothing of the wall shows above the band (a shelf in front), and its bottom, lit from under the shelf, is
    # lighter than its median: the baseboard stops where the white does.
    shelf = (zs >= 0.25) & (zs < 0.6)
    under = np.where((zs < 0.10)[..., None], white, np.where((zs < 0.25)[..., None], [120, 170, 210], blue))
    found = look(under, wall_pixels=~shelf, allowed=~shelf).baseboard
    assert abs(found.height - 0.10) < 0.011
    np.testing.assert_allclose(srgb(found.color), white, atol=1)
    # The wall's colour down to the floor: plain, no baseboard.
    plain = look(np.broadcast_to(blue, xs.shape + (3,)))
    assert plain.baseboard is None and plain.plain
    # Its bottom hidden (a bed in front): nothing seen either way.
    hidden = look(np.where((zs < 0.10)[..., None], white, blue), allowed=zs > 0.4)
    assert hidden.baseboard is None and not hidden.plain


def test_a_room_shares_the_baseboard_seen_most_clearly():
    from aomma.stages.room import Baseboard, BaseboardSighting, room_baseboards

    clear = Baseboard(height=0.12, color=(0.9, 0.9, 0.9))
    shaded = Baseboard(height=0.08, color=(0.3, 0.3, 0.3))
    walls = room_baseboards({
        "seen_well": BaseboardSighting(baseboard=clear, points=900),
        "seen_in_shade": BaseboardSighting(baseboard=shaded, points=100),
        "behind_a_bed": BaseboardSighting(),
        "bare_to_the_floor": BaseboardSighting(plain=True),
    })
    assert walls == {"seen_well": clear, "seen_in_shade": clear, "behind_a_bed": clear, "bare_to_the_floor": None}
    assert room_baseboards({"a": BaseboardSighting(), "b": BaseboardSighting(plain=True)}) == {"a": None, "b": None}


def test_a_wall_running_off_the_photos_edge_is_carried_past_the_frame():
    from aomma.stages.room import reach_frame

    # A camera 1.5 m up looking along +y at a back wall 3 m away, fitted only 1 m wide; the photo shows it edge to edge.
    camera = Camera.looking_at((0.0, 0.0, 1.5), (0.0, 3.0, 1.5), width=200, height=150)
    wall = Proxy(id="wall_1", size=(1.0, 0.05, 2.4), translate=(0.0, 3.025, 0.0), yaw_deg=0.0, color=(0.5, 0.5, 0.5))
    rows, cols = np.mgrid[0:150, 0:200]
    sight = np.stack([cols + 0.5 - 100, 75 - rows - 0.5, np.full(rows.shape, -camera.focal_px)], axis=-1) @ camera.matrix[:3, :3].T
    points = camera.matrix[:3, 3] + sight * (3.0 / sight[..., 1])[..., None]
    pixels = np.ones((150, 200), dtype=bool)
    reached = reach_frame([wall], pixels, points, camera)[0]
    # The frame's edges meet the wall at x = -/+ 3 * 100 / focal_px; carried 0.5 m past each.
    half = 3.0 * 100 / camera.focal_px
    assert abs(reached.size[0] - 2 * (half + 0.5)) < 0.05
    assert abs(reached.translate[0]) < 0.02 and reached.translate[1] == wall.translate[1]
    # A wall already reaching past the frame, or one no pixel at the edge stands on, is left as it is.
    wide = wall.model_copy(update={"size": (10.0, 0.05, 2.4)})
    assert reach_frame([wide], pixels, points, camera)[0] == wide
    assert reach_frame([wall], np.zeros_like(pixels), points, camera)[0] == wall


def test_a_doorway_at_the_photos_edge_is_cut_through_its_wall_and_the_wall_carried_past_it():
    from aomma.stages.pose import behind_walls
    from aomma.stages.room import cut_openings, opening_spans, sight_face

    # A back wall 3 m away, fitted only 1 m wide; the photo's right edge shows a doorway in it, 2 m high, cut
    # off by the frame, with the next room 2 m behind it. The wall's own pixels stand left of the doorway.
    camera = Camera.looking_at((0.0, 0.0, 1.5), (0.0, 3.0, 1.5), width=200, height=150)
    wall = Proxy(id="wall_1", size=(1.0, 0.05, 2.4), translate=(0.0, 3.025, 0.0), yaw_deg=0.0, color=(0.5, 0.5, 0.5))
    rows, cols = np.mgrid[0:150, 0:200]
    sight = np.stack([cols + 0.5 - 100, 75 - rows - 0.5, np.full(rows.shape, -camera.focal_px)], axis=-1) @ camera.matrix[:3, :3].T
    on_wall = camera.matrix[:3, 3] + sight * (3.0 / sight[..., 1])[..., None]
    doorway = (on_wall[..., 0] > 0.8) & (on_wall[..., 2] < 2.0)
    points = np.where(doorway[..., None], camera.matrix[:3, 3] + sight * (5.0 / sight[..., 1])[..., None], on_wall)
    edge_x = 3.0 * 100 / camera.focal_px

    cut = cut_openings([wall], {"doorway": (doorway, 0.9, False)}, ~doorway, points, camera)[0]
    (opening,) = cut.openings
    sides = sorted(end[0] for end in opening.ends)
    # Its near side is where the photo shows it; its far side is off the frame, so it takes its listed width.
    assert abs(sides[0] - 0.8) < 0.05 and abs(sides[1] - max(1.7, edge_x)) < 0.05
    assert abs(opening.top - 2.0) < 0.05
    # The wall reaches 0.5 m past the doorway's far side.
    assert abs(cut.translate[0] + cut.size[0] / 2 - (sides[1] + 0.5)) < 0.02
    (low, high, bottom, top, pane, *_), = opening_spans(cut)
    assert abs(low - (sides[0] - cut.translate[0])) < 1e-6 and top == opening.top and bottom == 0.0 and pane is None

    # The next room seen through the doorway is not behind the wall; what lies beyond the wall beside it is.
    face = sight_face(cut, camera.matrix[:3, 3])
    # Lines of sight that cross the wall at x 0.96 low, x -0.3, and x 0.96 2.28 m up, through the lintel.
    through = np.array([[1.6, 5.0, 1.0], [-0.5, 5.0, 1.0], [1.6, 5.0, 2.8]])
    assert behind_walls(through, [face], camera.matrix[:3, 3]).tolist() == [False, True, True]
    # A wall the openings miss is left as it is.
    assert cut_openings([wall], {"doorway": (np.zeros_like(doorway), 0.9, False)}, ~doorway, points, camera)[0] == wall


def test_a_window_is_cut_through_its_wall_above_its_sill_and_glazed_with_its_frame():
    from aomma.stages.pose import behind_walls
    from PIL import Image

    from aomma.stages.room import cut_openings, glaze, opening_spans, sight_face

    # A yellow back wall 3 m away, 4 m wide; a window in it from x -0.5 to 0.5, 0.9 to 2.0 m up, blue glass with the
    # garden 4 m behind it. A white frame 6 cm wide stands round it, its outer edge 2 cm past the window's mask.
    camera = Camera.looking_at((0.0, 0.0, 1.5), (0.0, 3.0, 1.5), width=800, height=600)
    wall = Proxy(id="wall_1", size=(4.0, 0.05, 2.4), translate=(0.0, 3.025, 0.0), yaw_deg=0.0, color=(0.5, 0.5, 0.5))
    rows, cols = np.mgrid[0:600, 0:800]
    sight = np.stack([cols + 0.5 - 400, 300 - rows - 0.5, np.full(rows.shape, -camera.focal_px)], axis=-1) @ camera.matrix[:3, :3].T
    on_wall = camera.matrix[:3, 3] + sight * (3.0 / sight[..., 1])[..., None]
    x, z = on_wall[..., 0], on_wall[..., 2]
    window = (np.abs(x) < 0.5) & (z > 0.9) & (z < 2.0)
    framed = (np.abs(x) < 0.52) & (z > 0.88) & (z < 2.02) & ~((np.abs(x) < 0.46) & (z > 0.94) & (z < 1.96))
    glass = window & ~framed
    points = np.where(glass[..., None], camera.matrix[:3, 3] + sight * (7.0 / sight[..., 1])[..., None], on_wall)
    pixels = np.where(framed[..., None], [240, 240, 235], np.where(glass[..., None], [60, 90, 160], [235, 160, 10]))
    photo = Image.fromarray(pixels.astype(np.uint8))

    cut = cut_openings([wall], {"window": (window, 1.0, True)}, ~window, points, camera)[0]
    (opening,) = cut.openings
    assert opening.window and abs(opening.bottom - 0.9) < 0.05 and abs(opening.top - 2.0) < 0.05
    # The wall already holds it, so it is not carried on past it as a door's wall is.
    assert cut.size == wall.size and cut.translate == wall.translate

    (opening,) = glaze(cut, photo, points, camera).openings
    # The cut stands about 2 cm inside the mask's edge (its 2nd and 98th percentiles), so the frame's edge 4 cm past it.
    assert abs(opening.frame_out - 0.04) <= 0.015 and abs(opening.frame_width - 0.06) <= 0.02
    assert min(opening.frame) > 0.8 and opening.pane[2] > opening.pane[0]
    (span,) = opening_spans(cut.model_copy(update={"openings": [opening]}))
    assert span[2:4] == (opening.bottom, opening.top) and span[4:] == (opening.pane, opening.frame_out, opening.frame_width, opening.frame)

    # Without the frame, the wall's colour runs to the glass: none is found.
    bare = Image.fromarray(np.where(window[..., None], [60, 90, 160], [235, 160, 10]).astype(np.uint8))
    assert glaze(cut, bare, points, camera).openings[0].frame is None

    # The glass stops lines of sight as the wall does: the garden through it is behind the wall.
    face = sight_face(cut, camera.matrix[:3, 3])
    assert behind_walls(np.array([[0.0, 7.0, 1.5]]), [face], camera.matrix[:3, 3]).tolist() == [True]


def test_overlapping_openings_are_one_span():
    from aomma.stages.proxy import Opening
    from aomma.stages.room import opening_spans

    # The same window listed twice, and a door apart from it.
    wall = Proxy(id="wall_1", size=(4.0, 0.05, 2.4), translate=(0.0, 3.0, 0.0), yaw_deg=0.0, color=(0.5, 0.5, 0.5), openings=[
        Opening(id="window", ends=((-1.0, 3.0), (0.0, 3.0)), bottom=0.9, top=2.0),
        Opening(id="window_2", ends=((-0.5, 3.0), (0.3, 3.0)), bottom=1.0, top=2.1),
        Opening(id="door", ends=((1.0, 3.0), (1.8, 3.0)), top=2.0),
    ])
    assert [span[:4] for span in opening_spans(wall)] == [(-1.0, 0.3, 0.9, 2.1), (1.0, 1.8, 0.0, 2.0)]


def test_a_wall_with_a_doorway_is_built_around_it(tmp_path):
    from pxr import Usd, UsdGeom

    from aomma.usd.assets import write_wall

    path = tmp_path / "wall.usda"
    write_wall(path, (4.0, 0.05, 2.4), (0.5, 0.5, 0.5), (0.1, 0.015, (1.0, 1.0, 1.0)), 1.0, [(0.5, 1.4, 0.0, 2.0)])
    stage = Usd.Stage.Open(str(path))
    extents = {
        prim.GetName(): [tuple(round(v, 3) for v in corner) for corner in UsdGeom.Mesh(prim).GetExtentAttr().Get()]
        for prim in stage.Traverse() if prim.IsA(UsdGeom.Mesh)
    }
    assert extents["Wall"] == [(-2.0, -0.025, 0.0), (0.5, 0.025, 2.4)]
    assert extents["Wall1"] == [(1.4, -0.025, 0.0), (2.0, 0.025, 2.4)]
    assert extents["Lintel0"] == [(0.5, -0.025, 2.0), (1.4, 0.025, 2.4)]
    assert {"Baseboard", "Baseboard1"} <= set(extents) and len(extents) == 5


def test_a_wall_with_a_window_is_built_around_it_and_its_baseboard_runs_under_it(tmp_path):
    from pxr import Usd, UsdGeom

    from aomma.usd.assets import write_wall

    path = tmp_path / "wall.usda"
    window = (-1.0, 0.0, 0.9, 2.0, (0.2, 0.3, 0.6), 0.02, 0.06, (0.9, 0.9, 0.9))
    write_wall(path, (4.0, 0.05, 2.4), (0.5, 0.5, 0.5), (0.1, 0.015, (1.0, 1.0, 1.0)), 1.0, [window, (1.0, 1.8, 0.0, 2.0)])
    stage = Usd.Stage.Open(str(path))
    extents = {
        prim.GetName(): [tuple(round(v, 3) for v in corner) for corner in UsdGeom.Mesh(prim).GetExtentAttr().Get()]
        for prim in stage.Traverse() if prim.IsA(UsdGeom.Mesh)
    }
    assert extents["Sill0"] == [(-1.0, -0.025, 0.0), (0.0, 0.025, 0.9)]
    assert extents["Lintel0"] == [(-1.0, -0.025, 2.0), (0.0, 0.025, 2.4)]
    assert "Sill1" not in extents and "Pane1" not in extents
    # The pane fills the window at the slab's middle; the frame stands 2 cm proud of the room's face, 2 cm past the window.
    assert extents["Pane0"] == [(-1.0, -0.005, 0.9), (0.0, 0.005, 2.0)]
    assert extents["Frame0Left"] == [(-1.02, 0.025, 0.88), (-0.96, 0.045, 2.02)]
    assert extents["Frame0Top"] == [(-0.96, 0.025, 1.96), (-0.04, 0.045, 2.02)]
    assert {"Frame0Right", "Frame0Bottom"} <= set(extents)
    # One board from the wall's end under the window to the door, and one past the door.
    assert extents["Baseboard"] == [(-2.0, 0.025, 0.0), (1.0, 0.04, 0.1)]
    assert extents["Baseboard1"] == [(1.8, 0.025, 0.0), (2.0, 0.04, 0.1)]


def _plan_scene(end_depth_m: float):
    """A wall along x at y = 0, a built-in 0.6 m deep standing on it (front at y = 0.6, x 0-1), and a surface square to
    the wall at x = 1 running from the wall to `end_depth_m`; each as its own column block of a point map, 1 m high."""
    import numpy as np

    rows = 40
    def block(xs, ys):
        x, z = np.meshgrid(xs, np.linspace(0.3, 1.3, rows))
        y = np.broadcast_to(ys, x.shape) if np.ndim(ys) else np.full(x.shape, ys)
        return np.stack([x, y, z], axis=-1)

    wall = block(np.linspace(-1.0, 0.0, 30), 0.0)
    front = block(np.linspace(0.0, 1.0, 30), 0.6)
    ys = np.linspace(0.0, end_depth_m, 30)
    end = np.stack([np.ones((rows, 30)), np.broadcast_to(ys, (rows, 30)), np.broadcast_to(np.linspace(0.3, 1.3, rows)[:, None], (rows, 30))], axis=-1)
    points = np.concatenate([wall, front, end], axis=1)
    masks = {name: np.zeros(points.shape[:2], bool) for name in ("wall", "front", "end")}
    masks["wall"][:, :30], masks["front"][:, 30:60], masks["end"][:, 60:] = True, True, True
    return points, masks


def test_built_in_ends_takes_a_panel_no_deeper_than_the_built_in():
    from aomma.stages.room import built_in_ends

    points, masks = _plan_scene(0.6)
    walls = {"back": masks["wall"], "side": masks["end"]}
    assert built_in_ends(walls, {"cabinet": masks["front"]}, points, 0.03) == {"side": "cabinet"}


def test_built_in_ends_leaves_a_wall_that_runs_past_the_built_in():
    from aomma.stages.room import built_in_ends

    points, masks = _plan_scene(2.0)
    walls = {"back": masks["wall"], "side": masks["end"]}
    assert built_in_ends(walls, {"cabinet": masks["front"]}, points, 0.03) == {}


def test_parse_small_takes_a_description_given_as_a_list():
    from aomma.stages.small import parse_small

    text = '{"objects": [{"id": "cushion", "bbox_2d": [1, 2, 3, 4], "description": ["Orange round", "floor cushion"], "size_m": [0.5, 0.5, 0.2]}]}'
    found = parse_small(text, None, set())
    assert found[0].description == "Orange round floor cushion"
