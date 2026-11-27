"""Sample coloured points from a USD asset's surface, for fast software rendering."""

from pathlib import Path

import numpy as np
from PIL import Image
from pxr import Usd, UsdGeom, UsdShade


def sample_surface(usd_path: Path, count: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Return `count` points spread evenly over the asset's surface, with colours.

    Points are (count, 3) in the asset's own frame; colours are (count, 3)
    sRGB 0-1, read from the base-colour texture (mid grey if there is none).
    """
    stage = Usd.Stage.Open(str(usd_path))
    prim = next(p for p in stage.Traverse() if p.IsA(UsdGeom.Mesh))
    mesh = UsdGeom.Mesh(prim)
    if set(mesh.GetFaceVertexCountsAttr().Get()) != {3}:
        raise ValueError(f"{usd_path}: only triangle meshes are supported")

    # USD matrices act on row vectors.
    to_asset = np.array(UsdGeom.XformCache().GetLocalToWorldTransform(prim))
    vertices = np.array(mesh.GetPointsAttr().Get())
    vertices = vertices @ to_asset[:3, :3] + to_asset[3, :3]
    triangles = np.array(mesh.GetFaceVertexIndicesAttr().Get()).reshape(-1, 3)
    corners = vertices[triangles]

    # Pick triangles in proportion to area, then a uniform point inside each.
    rng = np.random.default_rng(seed)
    areas = np.linalg.norm(np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]), axis=1)
    chosen = rng.choice(len(triangles), size=count, p=areas / areas.sum())
    root = np.sqrt(rng.random(count))
    other = rng.random(count)
    weights = np.stack([1 - root, root * (1 - other), root * other], axis=1)
    points = np.einsum("nk,nkd->nd", weights, corners[chosen])

    colors = np.full((count, 3), 0.5)
    uv = _triangle_uvs(mesh, triangles)
    texture = _base_color_texture(stage)
    if uv is not None and texture is not None:
        st = np.einsum("nk,nkd->nd", weights, uv[chosen])
        height, width = texture.shape[:2]
        cols = np.clip((st[:, 0] * width).astype(int), 0, width - 1)
        # Texture coordinates start at the bottom-left; image rows start at the top.
        rows = np.clip(((1 - st[:, 1]) * height).astype(int), 0, height - 1)
        colors = texture[rows, cols]
    return points, colors


def surface_facets(usd_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The asset's triangles as unit normals (T, 3), areas (T,) and centres (T, 3), in the asset's own frame."""
    stage = Usd.Stage.Open(str(usd_path))
    prim = next(p for p in stage.Traverse() if p.IsA(UsdGeom.Mesh))
    mesh = UsdGeom.Mesh(prim)
    to_asset = np.array(UsdGeom.XformCache().GetLocalToWorldTransform(prim))
    vertices = np.array(mesh.GetPointsAttr().Get()) @ to_asset[:3, :3] + to_asset[3, :3]
    counts = np.array(mesh.GetFaceVertexCountsAttr().Get())
    indices = np.array(mesh.GetFaceVertexIndicesAttr().Get())
    if set(counts) == {4}:
        # Split each quad along a diagonal.
        quads = indices.reshape(-1, 4)
        indices = np.concatenate([quads[:, [0, 1, 2]], quads[:, [0, 2, 3]]]).reshape(-1)
    corners = vertices[indices.reshape(-1, 3)]
    normals = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    lengths = np.linalg.norm(normals, axis=1)
    return normals / np.maximum(lengths[:, None], 1e-12), lengths / 2, corners.mean(axis=1)


def _triangle_uvs(mesh: UsdGeom.Mesh, triangles: np.ndarray) -> np.ndarray | None:
    """Texture coordinates as (triangles, 3, 2), or None if the mesh has none."""
    for primvar in UsdGeom.PrimvarsAPI(mesh).GetPrimvars():
        if "texCoord2" not in str(primvar.GetTypeName()):
            continue
        values = np.array(primvar.ComputeFlattened())
        if primvar.GetInterpolation() == UsdGeom.Tokens.faceVarying:
            return values.reshape(-1, 3, 2)
        return values[triangles]
    return None


def _base_color_texture(stage: Usd.Stage) -> np.ndarray | None:
    """The image feeding a preview surface's diffuse colour, as (H, W, 3) sRGB 0-1."""
    for prim in stage.Traverse():
        shader = UsdShade.Shader(prim)
        if not prim.IsA(UsdShade.Shader) or shader.GetIdAttr().Get() != "UsdPreviewSurface":
            continue
        diffuse = shader.GetInput("diffuseColor")
        for source in diffuse.GetConnectedSources()[0] if diffuse else []:
            file_input = UsdShade.Shader(source.source.GetPrim()).GetInput("file")
            if file_input:
                image = Image.open(file_input.Get().resolvedPath).convert("RGB")
                return np.asarray(image).astype(np.float32) / 255
    return None
