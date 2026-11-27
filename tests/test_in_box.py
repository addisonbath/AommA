import numpy as np

from aomma.stages.pose import SQUARE_TURNINGS, in_box


def _placed(points, pose, reshape):
    return (np.c_[points, np.ones(len(points))] @ reshape.T)[:, :3] * pose.scale


def test_the_24_turnings_are_distinct_rotations():
    assert len({t.tobytes() for t in SQUARE_TURNINGS}) == 24
    assert all(np.isclose(np.linalg.det(t), 1.0) for t in SQUARE_TURNINGS)


def test_stretched_onto_the_box_its_bottom_on_the_base():
    points = np.random.default_rng(0).uniform(-0.5, 0.5, (500, 3)) * [0.3, 0.3, 0.13]
    pose, reshape = in_box("notebook", points, np.zeros_like(points), (0.26, 0.29, 0.03), (1.0, 2.0, 0.75), 30.0,
                           lambda world, colors: (0.5, 0.1))
    shaped = _placed(points, pose, reshape)
    assert np.allclose(np.ptp(shaped, axis=0), (0.26, 0.29, 0.03), atol=1e-6)
    assert np.isclose(shaped[:, 2].min(), 0.0) and pose.translate == (1.0, 2.0, 0.75) and pose.rotate_xyz_deg[2] == 30.0


def test_up_keeps_the_side_the_photo_stood_up():
    points = np.random.default_rng(1).uniform(-0.5, 0.5, (500, 3)) * [0.2, 0.2, 0.6]
    # The outline would rather lay it on its side: it scores a model whose asset x points up.
    def seen(world, colors):
        return (1.0 if np.ptp(world[:, 2]) > 0 and np.argmax(np.abs(np.corrcoef(world[:, 2], points[:, 0])[0, 1])) == 0
                and abs(np.corrcoef(world[:, 2], points[:, 0])[0, 1]) > 0.9 else 0.5, 0.0)
    _, laid = in_box("plant", points, np.zeros_like(points), (0.3, 0.3, 0.4), (0, 0, 0), 0.0, seen)
    _, kept = in_box("plant", points, np.zeros_like(points), (0.3, 0.3, 0.4), (0, 0, 0), 0.0, seen, up=np.array([0.0, 0.1, 0.9]))
    assert abs(laid[2, 0]) > 0 and np.allclose(kept[2, :3] / np.linalg.norm(kept[2, :3]), (0, 0, 1))
