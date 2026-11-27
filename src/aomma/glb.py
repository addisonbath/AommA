"""Just enough of a binary glTF (.glb) reader to pull vertex data out."""

import json
import struct

import numpy as np

_COMPONENTS = {5121: "u1", 5123: "<u2", 5125: "<u4", 5126: "<f4"}
_WIDTHS = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}


def read_vertex_attributes(data: bytes) -> dict[str, np.ndarray]:
    """Return the vertex attributes (POSITION, TEXCOORD_0, ...) of the first mesh primitive."""
    json_length = struct.unpack_from("<I", data, 12)[0]
    gltf = json.loads(data[20 : 20 + json_length])
    # Skip the 12-byte file header, the JSON chunk and both 8-byte chunk headers.
    binary_start = 20 + json_length + 8

    attributes = {}
    for name, index in gltf["meshes"][0]["primitives"][0]["attributes"].items():
        accessor = gltf["accessors"][index]
        view = gltf["bufferViews"][accessor["bufferView"]]
        width = _WIDTHS[accessor["type"]]
        attributes[name] = np.frombuffer(
            data,
            dtype=_COMPONENTS[accessor["componentType"]],
            count=accessor["count"] * width,
            offset=binary_start + view.get("byteOffset", 0) + accessor.get("byteOffset", 0),
        ).reshape(-1, width)
    return attributes
