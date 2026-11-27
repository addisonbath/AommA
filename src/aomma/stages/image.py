"""Prepare the input photo: upright working copy plus camera intrinsics."""

import io
import math
from pathlib import Path

from PIL import Image, ImageOps
from pydantic import BaseModel

_EXIF_IFD = 0x8769
_FOCAL_LENGTH_35MM = 0xA405
# Diagonal of the 36x24mm frame that "35mm-equivalent" focal lengths refer to.
_FULL_FRAME_DIAGONAL_MM = math.hypot(36.0, 24.0)
# Used when the photo has no EXIF: a typical phone main camera.
_FALLBACK_FOCAL_35MM = 26.0


class Intrinsics(BaseModel):
    focal_length_mm: float
    sensor_width_mm: float
    width: int
    height: int
    source: str

    @property
    def fov_x_deg(self) -> float:
        return math.degrees(2 * math.atan(self.sensor_width_mm / (2 * self.focal_length_mm)))

    def with_fov_x(self, fov_x_deg: float, source: str) -> "Intrinsics":
        """Copy with the focal length that gives this horizontal field of view."""
        focal = self.sensor_width_mm / (2 * math.tan(math.radians(fov_x_deg) / 2))
        return self.model_copy(update={"focal_length_mm": focal, "source": source})


def load_source(path: Path) -> Image.Image:
    """The full-resolution photo, rotated upright."""
    return ImageOps.exif_transpose(Image.open(path)).convert("RGB")


def load_working_image(path: Path, max_side: int) -> tuple[Image.Image, Intrinsics]:
    """Return the photo rotated upright and downscaled, with its intrinsics."""
    raw = Image.open(path)
    focal = raw.getexif().get_ifd(_EXIF_IFD).get(_FOCAL_LENGTH_35MM)
    image = ImageOps.exif_transpose(raw).convert("RGB")
    image.thumbnail((max_side, max_side))

    width, height = image.size
    intrinsics = Intrinsics(
        focal_length_mm=float(focal or _FALLBACK_FOCAL_35MM),
        # The 35mm-equivalent focal length is defined against the frame's
        # diagonal, so derive the width of an equivalent sensor at this aspect.
        sensor_width_mm=_FULL_FRAME_DIAGONAL_MM * width / math.hypot(width, height),
        width=width,
        height=height,
        source="exif" if focal else "fallback",
    )
    return image, intrinsics


def to_jpeg(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=85)
    return buffer.getvalue()
