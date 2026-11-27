"""`aomma pose-check`: whether each shot's pose puts its LiDAR where the rest of the scan says the room is
(gate 0).

Two measures, each independent of any run:

- *Against RoomPlan's room.* For each wall and floor the shot sees, its sure LiDAR points near that surface (within
  `BAND_M`, inside its extents, outside its doors, windows and openings) are taken by their signed distance from it,
  positive toward the room. The one shift of the shot that lays the most of them on their surfaces is found by a
  robust fit (`robust_shift`): from no shift, with a tolerance narrowed from 15 to 4 cm, so that furniture standing in
  front of a wall stops counting; each surface counts once. Its length is how far the pose is off. Each surface's
  share of points that end on it says whether it was seen or mostly hidden. A direction no surface faces is not
  measured, and is reported as such. (A first version took each surface's densest layer: on a wall seen mostly
  behind a dresser, that was the dresser's front, 28 cm out.)
- *Against the reference shots*: the shot's LiDAR, placed by its pose, projected into each
  reference view and set against that view's LiDAR; the median ratio and error. A reference shot is set against the
  others.

Writes `refs/<scan>/pose_check.json`, with a verdict for each shot against gate 0's bar (set after the
baseline): a shot fails when its LiDAR lies more than `MAX_ERROR_M` from the reference
shots' (median), or it shares too little with them to say (fewer than `MIN_VIEWS` views). The RoomPlan measure informs
but does not judge: RoomPlan's own errors are as large as the poses'. The reference shots are judged
first, against each other; those that fail stop judging, and every shot is measured again against the rest. A shot
that fails is not mended. An input shot that fails still gets its run, judged and scored like any other, each result
noting `pose_off` (a photo run is given no pose, so only its comparisons with the references
suffer). A shot that fails is left out only where its pose would spread to other shots: as a reference shot judging
or scoring others, or as a source of item references (`scancheck.usable_shots`).
"""

import json
from pathlib import Path

import numpy as np

from aomma.capture import REFS, Scan, View, _matrix, to_frame
from aomma.scancheck import usable_shots

# Gate 0's bar: the median error against the reference shots, and the fewest views to say it from.
# On the baseline: 0.5-3.7 cm for every shot judged good, 5.7-9.1 cm for the rest; nothing between.
MAX_ERROR_M = 0.05
MIN_VIEWS = 2

# Points further than this from a surface are not its.
BAND_M = 0.3
# The tolerances the robust fit narrows through: wide enough at first to reach a few cm of pose error, narrow at the
# end so that furniture standing in front of a wall no longer counts. The last is about the LiDAR's own spread on a wall.
_TOLERANCES_M = (0.15, 0.08, 0.04)
# Past a wall's or floor's edges, a point may lie this far and still count for it.
_EDGE_M = 0.1
# The fewest points for a surface to say anything.
MIN_SURFACE_POINTS = 200
# A direction is measured when the surfaces seen face it this much (the smallest eigenvalue of the sum of their normals'
# outer products; 1 is one surface square on).
_MEASURED_FACING = 0.25
# The fewest pixels two views must share for a comparison.
_MIN_SHARED = 200


def _surfaces(room: dict) -> list[dict]:
    """RoomPlan's walls and floors, each as a frame (local x, y in its plane, z its normal) with its extents, and the
    doors, windows and openings in it."""
    cut = [c for kind in ("doors", "windows", "openings") for c in room.get(kind, [])]
    found = []
    for kind in ("walls", "floors"):
        for index, element in enumerate(room.get(kind, [])):
            transform = _matrix(element["transform"])
            holes = [c for c in cut if c.get("parentIdentifier") == element["identifier"]]
            found.append({"id": f"{kind[:-1]}_{index}", "kind": kind[:-1], "transform": transform,
                          "size": element["dimensions"][:2], "corners": element.get("polygonCorners") or [], "holes": holes})
    return found


def _inside_polygon(xy: np.ndarray, corners: list) -> np.ndarray:
    """Which local points lie inside the polygon (even-odd rule), in the plane's own x, y."""
    poly = np.asarray(corners, float)[:, :2]
    inside = np.zeros(len(xy), bool)
    for a, b in zip(poly, np.roll(poly, -1, axis=0)):
        crosses = (a[1] > xy[:, 1]) != (b[1] > xy[:, 1])
        with np.errstate(divide="ignore", invalid="ignore"):
            x = a[0] + (xy[:, 1] - a[1]) * (b[0] - a[0]) / (b[1] - a[1])
        inside ^= crosses & (xy[:, 0] < x)
    return inside


def robust_shift(distances: list[np.ndarray], normals: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    """The shift t that lays the most points on their surfaces: each point's residual is its signed distance d plus
    n . t, minimised under Tukey's biweight, starting from no shift with a wide tolerance and narrowing it (`_TOLERANCES_M`),
    so that furniture standing in front of a wall stops counting. Each surface counts equally, however many points it
    has. Only the directions the surfaces face are solved. Returns t and each surface's final weights."""
    facing = normals.T @ normals
    values, vectors = np.linalg.eigh(facing)
    solve_along = vectors[:, values >= _MEASURED_FACING]
    shift = np.zeros(3)
    weights = [np.ones(len(d)) for d in distances]
    for tolerance in _TOLERANCES_M:
        for _ in range(20):
            a, b = np.zeros((3, 3)), np.zeros(3)
            for d, n, i in zip(distances, normals, range(len(distances))):
                r = d + n @ shift
                w = np.where(np.abs(r) < tolerance, (1 - (r / tolerance) ** 2) ** 2, 0.0)
                weights[i] = w
                if w.sum() == 0:
                    continue
                w = w / w.sum()  # each surface counts once
                a += w.sum() * np.outer(n, n)
                b += -(w * d).sum() * n
            # Restricted to the directions solved: a is nearly singular along the others.
            a_s = solve_along.T @ a @ solve_along
            if np.linalg.matrix_rank(a_s) < solve_along.shape[1]:
                break
            new = solve_along @ np.linalg.solve(a_s, solve_along.T @ b)
            if np.linalg.norm(new - shift) < 1e-4:
                shift = new
                break
            shift = new
    return shift, weights


def against_room(points: np.ndarray, camera: np.ndarray, room: dict) -> dict:
    """The shot's LiDAR against RoomPlan's walls and floors, and the shift that best lays it on them (see the module's
    notes)."""
    seen, distances = [], []
    for surface in _surfaces(room):
        local = to_frame(points, np.linalg.inv(surface["transform"]))
        normal = surface["transform"][:3, 2].copy()
        # Positive toward the room, i.e. toward the camera, whichever way RoomPlan's normal points.
        side = np.sign(to_frame(camera[None], np.linalg.inv(surface["transform"]))[0, 2]) or 1.0
        distance = local[:, 2] * side
        width, height = surface["size"]
        if surface["kind"] == "floor" and surface["corners"]:
            within = _inside_polygon(local[:, :2], surface["corners"])
        else:
            within = (np.abs(local[:, 0]) <= width / 2 + _EDGE_M) & (np.abs(local[:, 1]) <= height / 2 + _EDGE_M)
        for hole in surface["holes"]:
            h = to_frame(points, np.linalg.inv(_matrix(hole["transform"])))
            within &= ~((np.abs(h[:, 0]) <= hole["dimensions"][0] / 2) & (np.abs(h[:, 1]) <= hole["dimensions"][1] / 2))
        near = within & (np.abs(distance) <= BAND_M)
        if near.sum() < MIN_SURFACE_POINTS:
            continue
        seen.append({"id": surface["id"], "kind": surface["kind"], "points": int(near.sum()),
                     "normal": (normal * side).round(4).tolist()})
        distances.append(distance[near])
    result = {"surfaces": seen, "shift_m": None, "shift_cm": None, "unmeasured": None}
    if not seen:
        return result
    normals = np.array([s["normal"] for s in seen])
    shift, weights = robust_shift(distances, normals)
    values, vectors = np.linalg.eigh(normals.T @ normals)
    for surface, d, n, w in zip(seen, distances, normals, weights):
        r = d + n @ shift
        on = w > 0
        # How much of what lies near the surface ends on it, and where that layer sits once shifted.
        surface["on_share"] = round(float(on.mean()), 3)
        surface["offset_m"] = round(float(np.median(d[on])), 4) if on.any() else None
        surface["left_m"] = round(float(np.median(r[on])), 4) if on.any() else None
    result.update({
        "shift_m": round(float(np.linalg.norm(shift)), 4),
        "shift_cm": (shift * 100).round(1).tolist(),
        "unmeasured": [vectors[:, i].round(2).tolist() for i in range(3) if values[i] < _MEASURED_FACING],
    })
    return result


def _zbuffer(view: View, depth_size: tuple[int, int], points: np.ndarray) -> np.ndarray:
    """World points drawn into `view` at its LiDAR map's size, nearest kept (depth along the axis; inf where none)."""
    camera = to_frame(points, np.linalg.inv(view.depth_to_world))
    d = -camera[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = (view.k[0, 0] * camera[:, 0] / d + view.k[0, 2]) * depth_size[0] / view.size[0]
        v = (-view.k[1, 1] * camera[:, 1] / d + view.k[1, 2]) * depth_size[1] / view.size[1]
    ok = (d > 0.1) & (u >= 0) & (v >= 0) & (u < depth_size[0]) & (v < depth_size[1])
    zb = np.full((depth_size[1], depth_size[0]), np.inf)
    np.minimum.at(zb, (v[ok].astype(int), u[ok].astype(int)), d[ok])
    return zb


def against_references(scan: Scan, shot: View, points: np.ndarray, references: list[View]) -> dict:
    """The shot's LiDAR at each reference view against that view's own: the medians over the views that share enough."""
    found = []
    for ref in references:
        if ref.name == shot.name:
            continue
        lidar, conf = scan.depth(ref)
        zb = _zbuffer(ref, ref.depth_size, points)
        both = np.isfinite(zb) & (conf == 2) & (lidar > 0)
        if both.sum() < _MIN_SHARED:
            continue
        ratio = zb[both] / lidar[both]
        found.append({"view": ref.name, "pixels": int(both.sum()), "ratio": round(float(np.median(ratio)), 4),
                      "error_m": round(float(np.median(np.abs(zb[both] - lidar[both]))), 4)})
    if not found:
        return {"views": [], "ratio": None, "error_m": None}
    return {"views": found, "ratio": round(float(np.median([f["ratio"] for f in found])), 4),
            "error_m": round(float(np.median([f["error_m"] for f in found])), 4)}


def verdict(references: dict) -> dict:
    """Whether a shot's pose passes gate 0's bar, and why not."""
    views = len(references["views"])
    if views < MIN_VIEWS:
        return {"ok": False, "why": f"shares {views} reference view(s) with the others, under {MIN_VIEWS}: its pose can't be judged"}
    if references["error_m"] > MAX_ERROR_M:
        return {"ok": False, "why": f"its LiDAR lies {100 * references['error_m']:.1f} cm from the reference shots' "
                                    f"(x{references['ratio']:.3f}), over {100 * MAX_ERROR_M:.0f} cm"}
    return {"ok": True, "why": None}


def check_poses(scan: Scan, refs: Path = REFS) -> dict:
    """Both measures and a verdict for every input and reference shot that passed the scan check; writes
    `pose_check.json`."""
    usable = usable_shots(scan.name, refs, poses=False)
    shots = [v for v in scan.shots if v.role in ("input", "reference") and v.depth_size
             and (usable is None or v.name in usable)]
    points = {shot.name: scan.lidar_world(shot)[0] for shot in shots}
    references = [v for v in shots if v.role == "reference"]
    # The judges first: a reference shot that fails against the others stops judging.
    judges = [r for r in references if verdict(against_references(scan, r, points[r.name], references))["ok"]]
    found = {}
    for shot in shots:
        measured = against_references(scan, shot, points[shot.name], judges)
        found[shot.name] = {
            "role": shot.role,
            "room": against_room(points[shot.name], shot.depth_to_world[:3, 3], scan.room),
            "references": measured,
            "verdict": verdict(measured),
        }
    result = {"scan": scan.name, "shots": found, "judges": [r.name for r in judges],
              "bar": {"max_error_m": MAX_ERROR_M, "min_views": MIN_VIEWS}}
    out = refs / scan.name / "pose_check.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    return result


def describe(result: dict) -> list[str]:
    lines = []
    for name, shot in result["shots"].items():
        room, ref = shot["room"], shot["references"]
        shift = f"{100 * room['shift_m']:.1f} cm {room['shift_cm']}" if room["shift_m"] is not None else "-"
        unmeasured = f", {len(room['unmeasured'])} direction(s) unseen" if room["unmeasured"] else ""
        shares = " ".join(f"{s['id']}:{s['on_share']:.0%}" for s in room["surfaces"])
        refs = f"x{ref['ratio']:.3f}, {100 * ref['error_m']:.1f} cm over {len(ref['views'])}" if ref["ratio"] else "no overlap"
        flag = "" if shot["verdict"]["ok"] else f"\n      POSE OFF: {shot['verdict']['why']}"
        lines.append(f"  {name} ({shot['role']}): room shift {shift}{unmeasured} [{shares}] | references {refs}{flag}")
    return lines
