"""Scan check: whether a scan's shots can be trusted, before any run is made from them or judged at them.

Each shot's LiDAR is set against the scan's LiDAR mesh, made over the whole mesh pass:
a shot whose pose is off lies away from it. All five scans so far lie within 1.4-3.0 cm
of it (median) on every shot. RoomPlan's walls are also checked (as
`Tools/check_shot_alignment.py` in AommACapture does), but only noted: furniture against
a wall and RoomPlan's own errors put a shot 5-27 cm from them that lies on the mesh
(Kitchen 002, OfficeScan 002). Its tracking and depth are read, and the printed marker is
looked for in it. Found in an input shot, the board would be in a run's photo, so the shot is
left out. Found in reference shots, it gives the phone's own scale: the board's distance
from its printed size alone (OpenCV), against the LiDAR's depth at its corners. A scan
with no marker scale is not used: depth work stands on scans whose LiDAR the board has
checked (the first scans, made before the board was used, are in
`scans/archive/`). The scan's session events (`diagnostics.json`) are listed when they fall during the shots.

Two files of ours beside the references say what the app cannot (the app's files stay
read-only to us): `marker_measured.json`, the board's square as measured with a ruler
(`{"square_mm": 35.0, "measured": "<when, how>"}`), since a typed value equal to the
designed one may never have been measured; and `use_anyway.json`, a failed check the
user has chosen to run on all the same (`{"problems": ["marker scale"], "why": ...,
"decided": ...}`): the scan still fails, says so, and is used.

Writes `refs/<scan>/scan_check.json`.
"""

import json
from pathlib import Path

import numpy as np

from scipy.spatial import cKDTree

from aomma.capture import REFS, Scan, View, _matrix, to_frame

# A LiDAR point further than this from every wall's plane is furniture or clutter.
_ON_WALL_M = 0.3
# How far past a wall's edges a point may lie and still count for it.
_EDGE_MARGIN_M = 0.1
# The fewest points on walls for a shot to say anything of them.
_MIN_WALL_POINTS = 200
# A shot whose LiDAR lies further than this from the scan's mesh (median) does not line up with the scan.
MESH_MAX_M = 0.05
# A shot whose wall points lie further than this from RoomPlan's walls (median) is noted.
WALL_MAX_M = 0.05
# A depth taken further than this from its image is noted: it is unprojected from its own pose, so it is not wrong.
_DEPTH_GAP_S = 0.1
# The fewest board corners found for a shot to count, as found or for scale.
_MIN_CORNERS = 6
# The LiDAR may be this far off the board's scale before the scan is not used.
SCALE_TOLERANCE = 0.01
# Session events that say the scan's tracking or mesh may have broken.
_BAD_EVENTS = ("interrupted", "failed", "memoryWarning", "meshAnchorRemoved", "meshAnchorCountDropped")


def wall_distances(scan: Scan, view: View) -> dict:
    """How far the view's sure LiDAR points on walls lie from RoomPlan's walls: the median over all of them, and by wall."""
    points, _ = scan.lidar_world(view)
    walls = scan.room.get("walls", [])
    distances = np.full((len(points), len(walls)), np.inf)
    for index, wall in enumerate(walls):
        local = to_frame(points, np.linalg.inv(_matrix(wall["transform"])))
        width, height = wall["dimensions"][:2]
        within = (np.abs(local[:, 0]) <= width / 2 + _EDGE_MARGIN_M) & (np.abs(local[:, 1]) <= height / 2 + _EDGE_MARGIN_M)
        distances[within, index] = np.abs(local[within, 2])
    if not walls:
        return {"median_m": None, "points": 0, "walls": {}}
    nearest = distances.argmin(axis=1)
    distance = distances[np.arange(len(points)), nearest]
    on = distance <= _ON_WALL_M
    by_wall = {
        f"wall_{index}": {"median_m": round(float(np.median(distance[on & (nearest == index)])), 4), "points": int((on & (nearest == index)).sum())}
        for index in range(len(walls)) if (on & (nearest == index)).sum() >= 50
    }
    return {"median_m": round(float(np.median(distance[on])), 4) if on.sum() >= _MIN_WALL_POINTS else None,
            "points": int(on.sum()), "walls": by_wall}


def mesh_distance(scan: Scan, view: View, tree: cKDTree) -> dict:
    """How far the view's sure LiDAR points lie from the scan's mesh: the median, and the share within 3 cm."""
    points, _ = scan.lidar_world(view)
    distance, _ = tree.query(points)
    return {"median_m": round(float(np.median(distance)), 4), "within_3cm": round(float((distance < 0.03).mean()), 3)}


def board(marker: dict):
    """The OpenCV ChArUco board `capture.json`'s marker describes, at its measured square size if there is one."""
    import cv2

    square = (marker.get("measuredSquareMillimetres") or marker["squareMillimetres"]) / 1000
    # The printed marker shrinks with its square: the board is printed whole, at one scale.
    printed_marker = marker["markerMillimetres"] / 1000 * square / (marker["squareMillimetres"] / 1000)
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, marker["dictionary"]))
    return cv2.aruco.CharucoBoard((marker["squaresX"], marker["squaresY"]), square, printed_marker, dictionary)


def find_board(scan: Scan, view: View, charuco) -> dict | None:
    """The board in the view's image, if found: how many corners, its distance by its printed size, and the LiDAR's
    depth over that distance at the corners it is sure of (1 is right; the median)."""
    import cv2

    image = cv2.imread(str(view.image), cv2.IMREAD_GRAYSCALE)
    corners, ids, _, _ = cv2.aruco.CharucoDetector(charuco).detectBoard(image)
    if ids is None or len(ids) < _MIN_CORNERS:
        return None
    on_board, in_image = charuco.matchImagePoints(corners, ids)
    solved, rotation, translation = cv2.solvePnP(on_board, in_image, view.k, None)
    if not solved:
        return None
    turn, _ = cv2.Rodrigues(rotation)
    # OpenCV's camera looks down +z, so z is the depth along the camera's axis, as the LiDAR's is.
    depth = (on_board.reshape(-1, 3) @ turn.T + translation.ravel())[:, 2]
    # OpenCV's pixel centres are whole numbers; `depth_at` takes the first pixel's centre at 0.5.
    lidar = scan.depth_at(view, in_image.reshape(-1, 2) + 0.5) if view.depth_size else np.full(len(depth), np.nan)
    sure = np.isfinite(lidar)
    return {
        "corners": int(len(ids)),
        "distance_m": round(float(depth.mean()), 3),
        "lidar_over_board": round(float(np.median(lidar[sure] / depth[sure])), 4) if sure.sum() >= _MIN_CORNERS else None,
    }


def check_shot(scan: Scan, view: View, charuco, tree: cKDTree | None) -> dict:
    """One shot's facts and its problems; a shot with a problem is not used (as an input or to judge at)."""
    problems, notes = [], []
    found = {"role": view.role, "label": view.label, "high_resolution": view.info.get("highResolution"),
             "tracking": view.info.get("trackingState"), "mapping": view.info.get("worldMappingStatus")}
    gap = abs(view.info.get("timestamp", 0) - view.info.get("depthTimestamp", view.info.get("timestamp", 0)))
    found["depth_gap_s"] = round(gap, 3)
    if gap > _DEPTH_GAP_S:
        notes.append(f"its depth is {gap:.2f} s from its image")
    if found["tracking"] not in (None, "normal"):
        problems.append(f"tracking was {found['tracking']}")
    if found["mapping"] in ("notAvailable", "limited"):
        notes.append(f"world mapping was {found['mapping']}")
    if view.depth_size is None:
        problems.append("no depth")
    else:
        if tree is not None:
            found["mesh"] = mesh_distance(scan, view, tree)
            if found["mesh"]["median_m"] > MESH_MAX_M:
                problems.append(f"its LiDAR lies {found['mesh']['median_m'] * 100:.1f} cm from the scan's mesh (median)")
        else:
            notes.append("the scan has no mesh to check it against")
        found["walls"] = wall_distances(scan, view)
        median = found["walls"]["median_m"]
        if median is not None and median > WALL_MAX_M:
            notes.append(f"its LiDAR lies {median * 100:.1f} cm from RoomPlan's walls (median)")
    found["board"] = find_board(scan, view, charuco) if charuco is not None else None
    if found["board"] and view.role == "input":
        problems.append("the marker board is in it, and a run would see it")
    found["problems"], found["notes"] = problems, notes
    found["ok"] = not problems
    return found


def marker_scale(marker: dict | None, views: dict[str, dict], measured: bool | None = None) -> dict:
    """The phone's scale from the board in the reference shots and keyframes: the LiDAR's depth over the board's, the
    median over them. Each view alone scatters by about 1% (OfficeScan: 0.994-1.015 over 12 views), so all are taken.
    `measured` says the square was measured; without it, a typed value unlike the designed one is taken as measured."""
    if marker is None:
        return {"status": "no marker: the scan is from before the protocol", "ok": False}
    ratios = {name: s["board"]["lidar_over_board"] for name, s in views.items()
              if s["role"] in ("reference", "keyframe") and s["board"] and s["board"]["lidar_over_board"] is not None}
    # The app starts from the designed size, so a value equal to it may not have been measured (scan-contract.md).
    if measured is None:
        measured = marker.get("measuredSquareMillimetres") not in (None, marker["squareMillimetres"])
    if not ratios:
        return {"status": "board in no reference shot: no marker scale", "measured": measured, "ok": False}
    ratio = float(np.median(list(ratios.values())))
    off = abs(ratio - 1) > SCALE_TOLERANCE
    if not off:
        status = "within tolerance"
    elif measured:
        status = f"off by {ratio - 1:+.1%}: not used until that is understood"
    else:
        status = f"off by {ratio - 1:+.1%}, but the board's square may never have been measured"
    return {"status": status, "lidar_over_board": round(ratio, 4), "by_view": ratios,
            "spread": round(float(np.ptp(list(ratios.values()))), 4),
            "per_view_sd": round(float(np.std(list(ratios.values()))), 4), "measured": measured,
            "square_mm": marker.get("measuredSquareMillimetres") or marker["squareMillimetres"],
            # The square the LiDAR says was printed: what the board would need to be for a ratio of 1.
            "square_from_lidar_mm": round(ratio * (marker.get("measuredSquareMillimetres") or marker["squareMillimetres"]), 1),
            "ok": not (off and measured)}


def session_events(scan: Scan) -> list[dict]:
    """Session events that may have broken tracking or the mesh, from the first shot or keyframe on, one entry for each
    kind and detail between two shots, with how many there were."""
    views = scan.shots + scan.keyframes
    start = min((v.info["timestamp"] for v in views), default=0.0)
    found = []
    for event in scan.diagnostics.get("sessionEvents", []):
        bad = event["kind"] in _BAD_EVENTS or (event["kind"] == "trackingState" and event.get("detail") != "normal")
        if bad and event["time"] >= start:
            after = [s.name for s in scan.shots if s.info["timestamp"] <= event["time"]]
            # A removed mesh anchor's detail is its id: those are counted together.
            detail = None if event["kind"] == "meshAnchorRemoved" else event.get("detail")
            entry = {"kind": event["kind"], "detail": detail, "after_shot": after[-1] if after else None}
            same = [f for f in found if {k: f[k] for k in entry} == entry]
            if same:
                same[0]["count"] += 1
            else:
                found.append(entry | {"count": 1})
    return found


def check_scan(scan: Scan, refs: Path = REFS) -> dict:
    """Check every shot and the scan as a whole; writes and returns `refs/<scan>/scan_check.json`."""
    marker = scan.capture.get("marker")
    ours = refs / scan.name
    measured = json.loads((ours / "marker_measured.json").read_text()) if (ours / "marker_measured.json").exists() else None
    if marker and measured:
        marker = marker | {"measuredSquareMillimetres": measured["square_mm"]}
    charuco = board(marker) if marker else None
    tree = cKDTree(scan.mesh) if (scan.folder / "mesh.usda").exists() and len(scan.mesh) else None
    shots = {view.name: check_shot(scan, view, charuco, tree) for view in scan.shots}
    keyframes = {
        f"keyframe {view.name}": {"role": "keyframe", "board": find_board(scan, view, charuco)}
        for view in scan.keyframes if charuco is not None and view.depth_size
    }
    scale = marker_scale(marker, shots | keyframes, measured=True if measured else None)
    if measured:
        scale["measured_how"] = measured.get("measured")
    events = session_events(scan)
    problems = [] if scale["ok"] else [f"marker scale: {scale['status']}"]
    if not any(s["ok"] and s["role"] == "input" for s in shots.values()):
        problems.append("no usable input shot")
    result = {
        "scan": str(scan.folder), "shots": shots, "marker_scale": scale, "events": events,
        "problems": problems, "ok": not problems,
    }
    anyway = json.loads((ours / "use_anyway.json").read_text()) if (ours / "use_anyway.json").exists() else None
    if problems and anyway and all(any(p.startswith(allowed) for allowed in anyway["problems"]) for p in problems):
        result["used_anyway"] = anyway
    out = refs / scan.name / "scan_check.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    return result


def usable(result: dict) -> bool:
    """Whether runs may be made from a checked scan: it passed, or the user chose to use it despite what failed."""
    return result["ok"] or "used_anyway" in result


def load_check(scan_name: str, refs: Path = REFS) -> dict | None:
    """The scan's check, if it has been made."""
    path = refs / scan_name / "scan_check.json"
    return json.loads(path.read_text()) if path.exists() else None


def usable_shots(scan_name: str, refs: Path = REFS, poses: bool = True) -> set[str] | None:
    """The shots that passed the scan's check, or None if it has not been checked (every shot is then used). With
    `poses`, less those whose pose failed gate 0 (`aomma pose-check`): the shots fit to judge, score or give references
    for others. A shot's own run, judges and score don't use this: a failed pose only notes them (`pose_failures`)."""
    check = load_check(scan_name, refs)
    if check is None:
        return None
    left = pose_failures(scan_name, refs) if poses else {}
    return {name for name, shot in check["shots"].items() if shot["ok"] and name not in left}


def pose_failures(scan_name: str, refs: Path = REFS) -> dict[str, str]:
    """The shots whose pose failed gate 0, each with why (`refs/<scan>/pose_check.json`). Empty if
    the scan's poses have not been checked or no bar is set yet."""
    path = refs / scan_name / "pose_check.json"
    if not path.exists():
        return {}
    found = json.loads(path.read_text())
    return {name: shot["verdict"]["why"] for name, shot in found["shots"].items()
            if shot.get("verdict") and not shot["verdict"]["ok"]}


def describe(result: dict) -> list[str]:
    """The check as lines to print."""
    lines = []
    for name, shot in result["shots"].items():
        mesh = shot.get("mesh", {}).get("median_m")
        facts = [shot["role"], f"mesh {mesh * 100:.1f} cm" if mesh is not None else "mesh -"]
        if shot["board"]:
            ratio = shot["board"]["lidar_over_board"]
            facts.append(f"board at {shot['board']['distance_m']:.2f} m" + (f", LiDAR x{ratio:.3f}" if ratio else ""))
        verdict = "ok" if shot["ok"] else "LEFT OUT: " + "; ".join(shot["problems"])
        lines.append(f"  shot {name} ({', '.join(facts)}): {verdict}" + (f" ({'; '.join(shot['notes'])})" if shot["notes"] else ""))
    lines.append(f"  marker scale: {result['marker_scale']['status']}"
                 + (f" (LiDAR x{result['marker_scale']['lidar_over_board']:.3f} the board's over {len(result['marker_scale']['by_view'])} "
                    f"views, each +-{result['marker_scale']['per_view_sd']:.1%}; printed square by the LiDAR "
                    f"{result['marker_scale']['square_from_lidar_mm']} mm against {result['marker_scale']['square_mm']} mm"
                    + (", measured" if result["marker_scale"]["measured"] else "") + ")"
                    if "lidar_over_board" in result["marker_scale"] else ""))
    for event in result["events"]:
        lines.append(f"  event {event['kind']}" + (f" ({event['detail']})" if event["detail"] else "")
                     + (f" x{event['count']}" if event["count"] > 1 else "") + f", after shot {event['after_shot']}")
    if result["ok"]:
        lines.append("scan ok")
    elif "used_anyway" in result:
        lines.append(f"SCAN FAILED, USED ANYWAY: {'; '.join(result['problems'])} (why: {result['used_anyway']['why']})")
    else:
        lines.append("SCAN NOT USED: " + "; ".join(result["problems"]))
    return lines
