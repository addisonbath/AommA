"""Per-object USD assets.

Every asset file follows the same conventions so the scene can reference any
of them interchangeably: Z-up, meters, default prim `/Asset`, pivot at
bottom-centre, UsdPreviewSurface materials.
"""

import math
from pathlib import Path
from typing import NamedTuple

from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade

from aomma.layout import Vec3

# Outward-facing (counter-clockwise) quads over corners indexed x + 2y + 4z.
_BOX_FACES = [
    0, 2, 3, 1,  # bottom
    4, 5, 7, 6,  # top
    0, 1, 5, 4,  # -y
    2, 6, 7, 3,  # +y
    0, 4, 6, 2,  # -x
    1, 3, 7, 5,  # +x
]
# How far inside a picture's border its slab's edges take their colour, as a share of its side.
_EDGE_INSET = 0.02


def new_stage(path: Path) -> Usd.Stage:
    """Create a stage with the project-wide units and up axis."""
    path.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    return stage


def write_proxy_box(
    path: Path, size: Vec3, color: Vec3 = (0.6, 0.6, 0.6), footprint: list[tuple[float, float]] | None = None,
) -> None:
    """Write a box asset of `size` meters, standing in for a real model.

    With a `footprint`, an outline in plan about the pivot, counter-clockwise, it is that outline stood up
    `size[2]` tall instead (a corner cabinet's 45-degree front).
    """
    stage = new_stage(path)
    root = UsdGeom.Xform.Define(stage, "/Asset")
    stage.SetDefaultPrim(root.GetPrim())

    hx, hy, sz = size[0] / 2, size[1] / 2, size[2]
    mesh = UsdGeom.Mesh.Define(stage, "/Asset/Geom")
    if footprint is None:
        mesh.CreatePointsAttr([Gf.Vec3f(x, y, z) for z in (0.0, sz) for y in (-hy, hy) for x in (-hx, hx)])
        mesh.CreateFaceVertexCountsAttr([4] * 6)
        mesh.CreateFaceVertexIndicesAttr(_BOX_FACES)
    else:
        # Bottom ring then top ring; the bottom face turned to face down, a quad up each side.
        count = len(footprint)
        mesh.CreatePointsAttr([Gf.Vec3f(x, y, z) for z in (0.0, sz) for x, y in footprint])
        sides = [i for k in range(count) for i in (k, (k + 1) % count, count + (k + 1) % count, count + k)]
        mesh.CreateFaceVertexCountsAttr([count, count] + [4] * count)
        mesh.CreateFaceVertexIndicesAttr(list(range(count))[::-1] + list(range(count, 2 * count)) + sides)
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    mesh.CreateExtentAttr([Gf.Vec3f(-hx, -hy, 0.0), Gf.Vec3f(hx, hy, sz)])

    material = UsdShade.Material.Define(stage, "/Asset/Looks/Proxy")
    shader = UsdShade.Shader.Define(stage, "/Asset/Looks/Proxy/Surface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.7)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)

    stage.GetRootLayer().Save()


# How far a window's frame stands proud of the wall, and how thick its pane is.
_FRAME_DEPTH_M = 0.02
_PANE_M = 0.01


class Span(NamedTuple):
    """An opening through a wall slab (see `write_wall`): from `low` to `high` along its x axis, `bottom` to `top` up it."""

    low: float
    high: float
    bottom: float
    top: float
    # A window's pane colour, and its frame: how far the frame's outer edge stands past the opening, its width, colour.
    pane: Vec3 | None = None
    frame_out: float = 0.0
    frame_width: float = 0.0
    frame: Vec3 | None = None


def write_wall(
    path: Path, size: Vec3, color: Vec3, baseboard: tuple[float, float, Vec3] | None = None, room_side: float = 1.0,
    openings: list[tuple] = (),
) -> None:
    """Write a wall slab of `size` meters, with a baseboard along the bottom of the face on `room_side` (+1: +Y, -1: -Y).

    `baseboard` is its height, thickness and colour. `openings` are doorways and windows through it, each a
    `Span` (or a tuple in its order), sorted and apart: the slab is built of the pieces beside them, the lintels
    over them and the sills under the windows. A window has a pane at the middle of the slab and, with a frame
    colour, a frame round it on the room's side. The baseboard runs under a window and stops at an opening lower
    than itself. The pivot is bottom-centre of the slab, as for a proxy box.
    """
    openings = [Span(*span) for span in openings]
    stage = new_stage(path)
    root = UsdGeom.Xform.Define(stage, "/Asset")
    stage.SetDefaultPrim(root.GetPrim())
    hx, hy = size[0] / 2, size[1] / 2
    # The stretches of floor-to-top wall between the openings.
    edges = [-hx] + [x for span in openings for x in (span.low, span.high)] + [hx]
    solid = [(edges[k], edges[k + 1]) for k in range(0, len(edges), 2) if edges[k + 1] > edges[k]]
    boxes = [(f"Wall{k}" if k else "Wall", (low, -hy, 0.0), (high, hy, size[2]), color) for k, (low, high) in enumerate(solid)]
    boxes += [(f"Lintel{k}", (o.low, -hy, o.top), (o.high, hy, size[2]), color) for k, o in enumerate(openings) if o.top < size[2]]
    boxes += [(f"Sill{k}", (o.low, -hy, 0.0), (o.high, hy, o.bottom), color) for k, o in enumerate(openings) if o.bottom > 0]
    if baseboard is not None:
        height, thickness, board_color = baseboard
        face = room_side * hy
        front, back = min(face, face + room_side * thickness), max(face, face + room_side * thickness)
        # The board's stretches: between the openings it cannot run under.
        stops = [-hx] + [x for o in openings if o.bottom < height for x in (o.low, o.high)] + [hx]
        runs = [(stops[k], stops[k + 1]) for k in range(0, len(stops), 2) if stops[k + 1] > stops[k]]
        boxes += [(f"Baseboard{k}" if k else "Baseboard", (low, front, 0.0), (high, back, height), board_color) for k, (low, high) in enumerate(runs)]
    face = room_side * hy
    front, back = min(face, face + room_side * _FRAME_DEPTH_M), max(face, face + room_side * _FRAME_DEPTH_M)
    for k, span in enumerate(openings):
        if span.pane is not None:
            boxes.append((f"Pane{k}", (span.low, -_PANE_M / 2, span.bottom), (span.high, _PANE_M / 2, span.top), span.pane))
        if span.frame is None or span.frame_width <= 0:
            continue
        # The frame's outer and inner edges: left, right, bottom, top.
        x0, x1 = max(span.low - span.frame_out, -hx), min(span.high + span.frame_out, hx)
        z0, z1 = max(span.bottom - span.frame_out, 0.0), min(span.top + span.frame_out, size[2])
        w = span.frame_width
        parts = {
            "Left": ((x0, z0), (x0 + w, z1)),
            "Right": ((x1 - w, z0), (x1, z1)),
            "Bottom": ((x0 + w, z0), (x1 - w, z0 + w)),
            "Top": ((x0 + w, z1 - w), (x1 - w, z1)),
        }
        boxes += [
            (f"Frame{k}{name}", (a[0], front, a[1]), (b[0], back, b[1]), span.frame)
            for name, (a, b) in parts.items() if b[0] > a[0] and b[1] > a[1]
        ]

    for name, low, high, box_color in boxes:
        points = [Gf.Vec3f(x, y, z) for z in (low[2], high[2]) for y in (low[1], high[1]) for x in (low[0], high[0])]
        mesh = UsdGeom.Mesh.Define(stage, f"/Asset/{name}")
        mesh.CreatePointsAttr(points)
        mesh.CreateFaceVertexCountsAttr([4] * 6)
        mesh.CreateFaceVertexIndicesAttr(_BOX_FACES)
        mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
        mesh.CreateExtentAttr([Gf.Vec3f(*low), Gf.Vec3f(*high)])
        material = UsdShade.Material.Define(stage, f"/Asset/Looks/{name}")
        shader = UsdShade.Shader.Define(stage, f"/Asset/Looks/{name}/Surface")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*box_color))
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.7)
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)

    stage.GetRootLayer().Save()


def write_floor(
    path: Path,
    polygon: list[tuple[float, float]],
    top: float,
    thickness: float,
    color: Vec3,
    texture: Path | None = None,
    tile_m: float = 1.0,
    angle_deg: float = 0.0,
    from_below: bool = False,
) -> None:
    """Write a flat slab over the convex `polygon` (world x, y, counter-clockwise), its top at `top`.

    It is meant to be placed untransformed. `texture`, an image file next to
    `path`, repeats over its top every `tile_m` meters along axes turned by
    `angle_deg`; without one, the slab is `color`. A slab seen `from_below`
    (a ceiling) has its bottom face where a floor has its top.
    """
    stage = new_stage(path)
    root = UsdGeom.Xform.Define(stage, "/Asset")
    stage.SetDefaultPrim(root.GetPrim())
    count = len(polygon)
    bottom = top - thickness
    points = [Gf.Vec3f(x, y, top) for x, y in polygon] + [Gf.Vec3f(x, y, bottom) for x, y in polygon]
    # The top, then a quad up each side; the bottom is never seen. Seen from below, the bottom instead, facing down.
    faces = [list(range(count, 2 * count))[::-1] if from_below else list(range(count))]
    faces += [[count + i, count + (i + 1) % count, (i + 1) % count, i] for i in range(count)]
    mesh = UsdGeom.Mesh.Define(stage, "/Asset/Geom")
    mesh.CreatePointsAttr(points)
    mesh.CreateFaceVertexCountsAttr([len(face) for face in faces])
    mesh.CreateFaceVertexIndicesAttr([index for face in faces for index in face])
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    xs, ys = [x for x, _ in polygon], [y for _, y in polygon]
    mesh.CreateExtentAttr([Gf.Vec3f(min(xs), min(ys), bottom), Gf.Vec3f(max(xs), max(ys), top)])
    if texture is None:
        material = UsdShade.Material.Define(stage, "/Asset/Looks/Surface")
        shader = UsdShade.Shader.Define(stage, "/Asset/Looks/Surface/Shader")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.8)
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
    else:
        turn = math.radians(angle_deg)
        def st(x: float, y: float) -> Gf.Vec2f:
            # Image rows run down the texture, so "down" in the grid it was drawn on is -t.
            return Gf.Vec2f((x * math.cos(turn) + y * math.sin(turn)) / tile_m, -(-x * math.sin(turn) + y * math.cos(turn)) / tile_m)
        coords = [st(x, y) for x, y in polygon] * 2
        primvar = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex)
        primvar.Set(coords)
        _bind_texture(stage, mesh, texture, repeat=True)
    stage.GetRootLayer().Save()


def write_textured_quad(path: Path, corners: list[Vec3], texture: Path) -> None:
    """Write a flat four-cornered surface showing `texture`, an image file next to `path`.

    `corners` run bottom-left, bottom-right, top-right, top-left as the image is
    seen. They may be given in world coordinates and the asset placed untransformed.
    """
    stage = new_stage(path)
    root = UsdGeom.Xform.Define(stage, "/Asset")
    stage.SetDefaultPrim(root.GetPrim())

    mesh = UsdGeom.Mesh.Define(stage, "/Asset/Geom")
    mesh.CreatePointsAttr([Gf.Vec3f(*corner) for corner in corners])
    mesh.CreateFaceVertexCountsAttr([4])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    mesh.CreateDoubleSidedAttr(True)
    st = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex)
    st.Set([Gf.Vec2f(0, 0), Gf.Vec2f(1, 0), Gf.Vec2f(1, 1), Gf.Vec2f(0, 1)])

    _bind_texture(stage, mesh, texture)

    stage.GetRootLayer().Save()


def write_picture_slab(path: Path, size: Vec3, texture: Path, normal_map: Path | None = None) -> None:
    """Write a thin box whose +Y face shows `texture`, an image file next to `path`.

    `normal_map`, another image next to `path`, adds relief such as a raised frame.

    `size` is width, depth and height. The pivot is bottom-centre. The four
    edges repeat the picture's outermost pixels and the back takes one of them,
    so a framed picture gets frame-coloured sides.
    """
    stage = new_stage(path)
    root = UsdGeom.Xform.Define(stage, "/Asset")
    stage.SetDefaultPrim(root.GetPrim())

    hx, hy, sz = size[0] / 2, size[1] / 2, size[2]
    mesh = UsdGeom.Mesh.Define(stage, "/Asset/Geom")
    mesh.CreatePointsAttr([Gf.Vec3f(x, y, z) for z in (0.0, sz) for y in (-hy, hy) for x in (-hx, hx)])
    mesh.CreateFaceVertexCountsAttr([4] * 6)
    mesh.CreateFaceVertexIndicesAttr(_BOX_FACES)
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    mesh.CreateExtentAttr([Gf.Vec3f(-hx, -hy, 0.0), Gf.Vec3f(hx, hy, sz)])

    # Facing the +Y face, +X is on the left, so u runs against X; v runs up Z.
    # Each other face pins one coordinate just inside the picture's border.
    pinned = [(None, _EDGE_INSET), (None, 1 - _EDGE_INSET), None, None, (1 - _EDGE_INSET, None), (_EDGE_INSET, None)]
    coords = []
    for face, pin in enumerate(pinned):
        for corner in _BOX_FACES[face * 4 : face * 4 + 4]:
            u, v = 1.0 - (corner & 1), float(corner >> 2)
            if face == 2:  # the back: one frame-coloured texel
                u, v = _EDGE_INSET, _EDGE_INSET
            elif pin is not None:
                u, v = (u if pin[0] is None else pin[0]), (v if pin[1] is None else pin[1])
            coords.append(Gf.Vec2f(u, v))
    st = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying)
    st.Set(coords)
    _bind_texture(stage, mesh, texture, normal_map)

    stage.GetRootLayer().Save()


def write_painted_sheet(path: Path, corners: list[Vec3], texture: Path) -> None:
    """Write a sheet (`stages.forms`): the one quad `corners` (bottom-left, bottom-right, top-right, top-left as seen,
    about the pivot) showing `texture`, an RGBA image next to `path` whose alpha cuts it out. One quad and no back: a
    back 3 mm behind it, given the texture's middle, was what Blender drew in its place."""
    stage = new_stage(path)
    root = UsdGeom.Xform.Define(stage, "/Asset")
    stage.SetDefaultPrim(root.GetPrim())
    front = [Gf.Vec3f(*c) for c in corners]
    mesh = UsdGeom.Mesh.Define(stage, "/Asset/Geom")
    mesh.CreatePointsAttr(front)
    mesh.CreateFaceVertexCountsAttr([4])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    mesh.CreateExtentAttr(UsdGeom.PointBased.ComputeExtent(front))
    st = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying)
    st.Set([Gf.Vec2f(0, 0), Gf.Vec2f(1, 0), Gf.Vec2f(1, 1), Gf.Vec2f(0, 1)])
    _bind_texture(stage, mesh, texture, cutout=True)
    stage.GetRootLayer().Save()


def write_reshaped_asset(path: Path, source: Path, matrix: list[list[float]]) -> None:
    """Write an asset that is `source`, a USD file next to `path`, with its shape remade by a 4x4 `matrix`.

    The matrix acts on column vectors and may stretch each direction
    differently. Such a matrix cannot be left as a transform: applications
    that keep position, rotation and scale separately (Blender) cannot hold the
    skew it has when combined with a rotation. So the new asset references the
    source for materials and textures and overrides the mesh points themselves.
    """
    original = Usd.Stage.Open(str(source))
    start = original.GetDefaultPrim()
    stage = new_stage(path)
    root = stage.OverridePrim("/Asset")
    stage.SetDefaultPrim(root)
    root.GetReferences().AddReference(f"./{source.name}")

    linear = [[matrix[row][column] for column in range(3)] for row in range(3)]
    offset = [matrix[row][3] for row in range(3)]
    transforms = UsdGeom.XformCache()
    for prim in Usd.PrimRange(start):
        over = stage.OverridePrim(prim.GetPath().ReplacePrefix(start.GetPath(), root.GetPath()))
        if prim.IsA(UsdGeom.Xformable):
            # The points below are final, so nothing above them may move them again.
            UsdGeom.Xformable(over).ClearXformOpOrder()
        if not prim.IsA(UsdGeom.Mesh):
            continue
        # USD matrices act on row vectors.
        placed = transforms.GetLocalToWorldTransform(prim)
        points = [placed.Transform(Gf.Vec3d(*point)) for point in UsdGeom.Mesh(prim).GetPointsAttr().Get()]
        points = [
            Gf.Vec3f(*(sum(linear[row][k] * point[k] for k in range(3)) + offset[row] for row in range(3)))
            for point in points
        ]
        mesh = UsdGeom.Mesh(over)
        mesh.CreatePointsAttr(points)
        mesh.CreateExtentAttr(UsdGeom.PointBased.ComputeExtent(points))
        # Stretching changes which way surfaces face: drop stored normals so they are worked out afresh.
        mesh.CreateNormalsAttr().Block()
        over.CreateAttribute("primvars:normals", Sdf.ValueTypeNames.Normal3fArray).Block()
    stage.GetRootLayer().Save()


def _bind_texture(stage: Usd.Stage, mesh: UsdGeom.Mesh, texture: Path, normal_map: Path | None = None, repeat: bool = False,
                  cutout: bool = False) -> None:
    """Give `mesh` a matte material showing `texture` through its "st" coordinates, repeated past 0-1 if `repeat`;
    with `cutout`, the texture's alpha cuts the surface away where it is under a half."""
    material = UsdShade.Material.Define(stage, "/Asset/Looks/Surface")
    reader = UsdShade.Shader.Define(stage, "/Asset/Looks/Surface/TexCoords")
    reader.CreateIdAttr("UsdPrimvarReader_float2")
    reader.CreateInput("varname", Sdf.ValueTypeNames.String).Set("st")
    image = UsdShade.Shader.Define(stage, "/Asset/Looks/Surface/Image")
    image.CreateIdAttr("UsdUVTexture")
    image.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(f"./{texture.name}")
    image.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
    if repeat:
        image.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
        image.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
    image.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(
        reader.CreateOutput("result", Sdf.ValueTypeNames.Float2)
    )
    shader = UsdShade.Shader.Define(stage, "/Asset/Looks/Surface/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
        image.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
    )
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.9)
    if cutout:
        shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).ConnectToSource(
            image.CreateOutput("a", Sdf.ValueTypeNames.Float)
        )
        shader.CreateInput("opacityThreshold", Sdf.ValueTypeNames.Float).Set(0.5)
    if normal_map is not None:
        normals = UsdShade.Shader.Define(stage, "/Asset/Looks/Surface/Normals")
        normals.CreateIdAttr("UsdUVTexture")
        normals.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(f"./{normal_map.name}")
        normals.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("raw")
        # Stored 0-1, meant -1 to 1.
        normals.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(2, 2, 2, 1))
        normals.CreateInput("bias", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(-1, -1, -1, 0))
        normals.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(
            reader.CreateOutput("result", Sdf.ValueTypeNames.Float2)
        )
        shader.CreateInput("normal", Sdf.ValueTypeNames.Normal3f).ConnectToSource(
            normals.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
        )
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
