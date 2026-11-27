"""The layout contract: where every asset and the camera sit in the scene.

This is the JSON that the layout stage will produce and the agent will edit.
Conventions for the whole project: Z-up, meters, right-handed (Blender's
native frame). Assets have their pivot at bottom-centre, so `translate` is
the point where the object rests on its support.
"""

import numpy as np
from pydantic import BaseModel, Field

Vec3 = tuple[float, float, float]


class Camera(BaseModel):
    # 4x4 camera-to-world matrix, row-major, for column vectors (p_world = M @ p_cam).
    # The camera looks down its local -Z with +Y up, as in USD and Blender.
    camera_to_world: tuple[float, ...] = Field(min_length=16, max_length=16)
    focal_length_mm: float = 35.0
    sensor_width_mm: float = 36.0
    width: int = 960
    height: int = 640

    @property
    def matrix(self) -> np.ndarray:
        return np.array(self.camera_to_world).reshape(4, 4)

    @property
    def focal_px(self) -> float:
        return self.focal_length_mm / self.sensor_width_mm * self.width

    @classmethod
    def looking_at(cls, position: Vec3, target: Vec3, **kwargs) -> "Camera":
        """A level camera (no roll) at `position` aimed at `target`."""
        eye = np.array(position, dtype=float)
        forward = np.array(target, dtype=float) - eye
        forward /= np.linalg.norm(forward)
        right = np.cross(forward, (0.0, 0.0, 1.0))
        right /= np.linalg.norm(right)
        matrix = np.eye(4)
        matrix[:3, 0], matrix[:3, 1] = right, np.cross(right, forward)
        matrix[:3, 2], matrix[:3, 3] = -forward, eye
        return cls(camera_to_world=tuple(matrix.flatten()), **kwargs)


class Placement(BaseModel):
    # Becomes the prim name, so it must be a valid USD identifier.
    name: str = Field(pattern=r"^[A-Za-z_][A-Za-z0-9_]*$")
    # Path to the asset's USD file, relative to the scene file.
    asset: str
    translate: Vec3 = (0.0, 0.0, 0.0)
    rotate_xyz_deg: Vec3 = (0.0, 0.0, 0.0)
    scale: float = 1.0
    # Kept in the layout, so a later pose can bring it back, but not put in the scene.
    left_out: bool = False


class SceneLayout(BaseModel):
    camera: Camera
    objects: list[Placement]
