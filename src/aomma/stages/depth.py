"""Depth stage: a metric 3D point for every pixel, from MoGe running in ComfyUI."""

import numpy as np
from PIL import Image

from aomma.backends.comfy import ComfyClient
from aomma.glb import read_vertex_attributes

MOGE_MODEL = "moge_2_vitl_normal_fp16.safetensors"


def estimate_points(
    comfy: ComfyClient, image_name: str, size: tuple[int, int], fov_x_deg: float | None
) -> tuple[np.ndarray, float]:
    """Return an (H, W, 3) point map in meters and the horizontal FOV it assumes.

    Points are in the camera's frame: x right, y up, looking down -z (the same
    convention as a USD or Blender camera). Pixels MoGe rejects are NaN. Pass
    `fov_x_deg` when the true field of view is known; otherwise MoGe estimates it.
    """
    outputs = comfy.run(
        {
            "image": {"class_type": "LoadImage", "inputs": {"image": image_name}},
            "model": {"class_type": "LoadMoGeModel", "inputs": {"model_name": MOGE_MODEL}},
            "moge": {
                "class_type": "MoGeInference",
                "inputs": {
                    "moge_model": ["model", 0],
                    "image": ["image", 0],
                    "resolution_level": 9,
                    # 0 tells MoGe to estimate the field of view itself.
                    "fov_x_degrees": fov_x_deg or 0.0,
                    "batch_size": 4,
                    "force_projection": True,
                    "apply_mask": True,
                    "refine_steps": 3,
                },
            },
            # ComfyUI has no lossless depth output, so export the point map as
            # a mesh: one vertex per pixel, with the pixel recoverable from its UV.
            "mesh": {
                "class_type": "MoGePointMapToMesh",
                "inputs": {
                    "moge_geometry": ["moge", 0],
                    "batch_index": 0,
                    "decimation": 1,
                    "discontinuity_threshold": 0.04,
                    "texture": False,
                },
            },
            "glb": {
                "class_type": "SaveGLB",
                "inputs": {"mesh": ["mesh", 0], "filename_prefix": "aomma/points"},
            },
            "fov": {
                "class_type": "MoGeGeometryToFOV",
                "inputs": {"moge_geometry": ["moge", 0], "axis": "horizontal", "unit": "degrees"},
            },
            "fov_out": {"class_type": "PreviewAny", "inputs": {"source": ["fov", 0]}},
        },
        label=f"depth {image_name}",
    )
    vertices = read_vertex_attributes(comfy.fetch(outputs["glb"]["3d"][0]))

    width, height = size
    uv = vertices["TEXCOORD_0"]
    cols = np.rint(uv[:, 0] * (width - 1)).astype(int)
    rows = np.rint(uv[:, 1] * (height - 1)).astype(int)
    points = np.full((height, width, 3), np.nan, dtype=np.float32)
    points[rows, cols] = vertices["POSITION"]
    return points, float(outputs["fov_out"]["text"][0])


# ARKit's confidence in a LiDAR depth it is sure of.
LIDAR_SURE = 2
# A fit needs at least this many sure LiDAR pixels that MoGe also has.
_MIN_LIDAR_PIXELS = 500


def lidar_pixels(lidar_shape: tuple[int, int], image_shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """The rows and columns of an image of `image_shape` (height, width) at the centres of the LiDAR map's rows and columns."""
    rows, cols = lidar_shape
    height, width = image_shape
    return ((np.arange(rows) + 0.5) * height / rows).astype(int), ((np.arange(cols) + 0.5) * width / cols).astype(int)


def lidar_points(lidar: np.ndarray, intrinsics: dict) -> np.ndarray:
    """The LiDAR map as (rows, cols, 3) camera-frame points (x right, y up, looking down -z), from the photo's
    `intrinsics` (fx, fy, cx, cy at its `width` x `height`), each pixel taken at its centre as the scorers do."""
    rows, cols = lidar.shape
    v, u = np.mgrid[0:rows, 0:cols]
    x = ((u + 0.5) * intrinsics["width"] / cols - intrinsics["cx"]) / intrinsics["fx"]
    y = -((v + 0.5) * intrinsics["height"] / rows - intrinsics["cy"]) / intrinsics["fy"]
    return np.stack([x * lidar, y * lidar, -lidar], -1)


def fit_to_lidar(points: np.ndarray, lidar: np.ndarray, conf: np.ndarray) -> tuple[np.ndarray, dict]:
    """`points` moved along their rays so their depth agrees with the shot's LiDAR: lidar = scale * depth + shift.

    `lidar` (metres) and its confidence `conf` cover the photo's view at their own
    smaller size; each of their pixels is matched with the photo pixel at its centre,
    as the scorers match them. The fit uses the sure pixels only, and leaves out those
    far off it (an edge, a mirror, a window) over three rounds. Returns the points and
    the fit: scale, shift, the pixels it used of those it had, and the median error.
    """
    v, u = lidar_pixels(lidar.shape, points.shape[:2])
    moge = -points[v][:, u, 2]
    with np.errstate(invalid="ignore"):
        usable = (conf == LIDAR_SURE) & np.isfinite(moge) & (moge > 0) & (lidar > 0)
    if usable.sum() < _MIN_LIDAR_PIXELS:
        raise ValueError(f"only {usable.sum()} sure LiDAR pixels with a depth from MoGe; {_MIN_LIDAR_PIXELS} needed")
    x, y = moge[usable], lidar[usable]
    keep = np.ones(len(x), bool)
    for _ in range(3):
        (scale, shift), *_ = np.linalg.lstsq(np.c_[x[keep], np.ones(keep.sum())], y[keep], rcond=None)
        error = np.abs(scale * x + shift - y)
        # Three robust deviations, but never under 2 cm, the LiDAR's own noise.
        keep = error <= max(3 * 1.4826 * float(np.median(error[keep])), 0.02)
    depth = -points[..., 2]
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = (scale * depth + shift) / depth
    aligned = (points * np.where(ratio > 0, ratio, np.nan)[..., None]).astype(np.float32)
    return aligned, {
        "scale": round(float(scale), 4), "shift_m": round(float(shift), 4),
        "pixels": int(keep.sum()), "of": int(usable.sum()),
        "median_error_m": round(float(np.median(error[keep])), 4),
    }


def draw_depth(points: np.ndarray) -> Image.Image:
    """Greyscale picture of distance from the camera: near is bright, missing is black."""
    depth = -points[..., 2]
    valid = np.isfinite(depth)
    near, far = np.percentile(depth[valid], [1, 99])
    shade = 1.0 - np.clip((depth - near) / (far - near), 0.0, 1.0)
    return Image.fromarray(np.where(valid, 40 + shade * 215, 0).astype(np.uint8))
