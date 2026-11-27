import numpy as np
import pytest

from aomma.stages.depth import LIDAR_SURE, fit_to_lidar


def point_map(depth: np.ndarray, focal: float) -> np.ndarray:
    """The camera-frame points (x right, y up, looking down -z) of a depth map."""
    rows, cols = depth.shape
    v, u = np.mgrid[0:rows, 0:cols]
    x = (u + 0.5 - cols / 2) / focal
    y = -(v + 0.5 - rows / 2) / focal
    return np.stack([x * depth, y * depth, -depth], -1).astype(np.float32)


def test_the_fit_recovers_scale_and_shift_and_keeps_each_point_on_its_ray():
    rng = np.random.default_rng(0)
    true = rng.uniform(1.0, 5.0, (192, 256))
    moge = point_map((true - 0.1) / 1.2, 200.0)  # MoGe's depth: lidar = 1.2 * depth + 0.1
    lidar = true[2::4, 2::4].copy()  # the LiDAR at a quarter the size, each pixel the photo's at its centre
    conf = np.full(lidar.shape, LIDAR_SURE, np.uint8)
    lidar[:5] = 9.0  # a window the LiDAR saw through
    conf[-3:] = 0  # and pixels it is unsure of, which are wrong too
    lidar[-3:] = 0.3

    aligned, fit = fit_to_lidar(moge, lidar, conf)
    assert fit["scale"] == pytest.approx(1.2, abs=1e-3) and fit["shift_m"] == pytest.approx(0.1, abs=1e-3)
    assert fit["pixels"] == lidar.size - 8 * lidar.shape[1]
    assert np.allclose(-aligned[..., 2], true, atol=1e-3)
    directions = lambda p: p / np.linalg.norm(p, axis=-1, keepdims=True)
    assert np.allclose(directions(aligned), directions(moge), atol=1e-5)


def test_too_little_sure_lidar_is_no_fit():
    moge = point_map(np.full((40, 40), 2.0), 40.0)
    with pytest.raises(ValueError):
        fit_to_lidar(moge, np.full((10, 10), 2.0), np.zeros((10, 10), np.uint8))
