"""`aomma labels`: one label for every pixel of a run's photo.

Each mask a run made is given a class from the list it came from: the survey's walls, openings and built-ins, stage 2's
large items, the floor survey's regions (a rug is a `covering`), and the shell's own `floor.png` and `ceiling.png`.
A mask no list names is `other`, and reported.

Where masks overlap, the pixel goes to the claimant whose surface its point lies on: for each claimant, a plane
through its own unclaimed points nearby (a small window first, then a wider one, then all of them), and the claimant
whose plane the point lies nearest takes it. A plane, not a distance, so that a wall and the ceiling where they meet
(both about as far from the camera) are told apart. A mask with no pixels of its own (an item wholly inside a wall's
mask) is taken as the plane through all of its points. Only where depth cannot tell them apart (no depth, or the two
nearest within `SAME_DEPTH_M` of each other) does the more specific one take it (`RANK`): an item over its support, a
support over the shell. The shell (floor, walls, ceiling, openings) bounds the room: where an item claims the pixel too,
a shell surface the point stands more than `SAME_DEPTH_M` off, measured along the camera ray, cannot take it.

Except where the more specific one is a small item lying flat on the surface under it (a paper on a fridge door, a
coaster on a worktop): where the pixel's point lies on the surface of a claimant below the item, as near as that
surface's own spread (`_tolerance`), the item adds nothing to its shape. That claimant keeps the pixel and the item is
kept as a layer on it (`SETTLED_LAYER`, `layer_px`, `debug/labels_layers.png`): what is printed on the surface, not a
thing standing on it. Whether the surface faces up or sideways does not enter: only how far the item stands off it.

Each small item is also set against the support its listing names (`relations` in the report): a layer on it,
touching it (somewhere along their shared border the depth runs on from one to the other, within `SAME_DEPTH_M`), or
not touching it (a jump in depth all along that border: the item is in front of it, not on it). Reported, not acted on.

A pixel no mask claims is then grown into from its neighbours (`grow`): it goes to a neighbouring claimant whose own
surface it lies on, as near that surface as most of the claimant's own edge lies to the plane its inside gives (its
own spread, so the same on the LiDAR's depth and on a monocular one). The plane is fitted to the pixels the claimant kept, so a mask that leaked onto
another's surface does not carry its leak on. What is still unclaimed after that is `unknown`, and counted.

`floor.png` (the shell's plain "floor" mask) only adds the floor the floor survey's regions leave out: the two are
the same floor, and their overlap is not a conflict.

Reads only what the run already holds (masks, `points.npy`); nothing is segmented again.
Writes `labels.png` (0 is unknown), `labels.json` (what each value is, and the report), `debug/labels.jpg` and
`debug/labels_settled.png` (how each overlap was settled), which the dashboard's Labels tab reads.
"""

import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation, binary_erosion, label as connected, uniform_filter

UNKNOWN = "unknown"
# How each pixel was settled, as saved in `debug/labels_settled.png`.
SETTLED_ALONE, SETTLED_BY_DEPTH, SETTLED_BY_RANK, SETTLED_GROWN, SETTLED_LAYER = 0, 1, 2, 3, 4
# The more specific a class, the higher: it takes a pixel only where depth cannot decide.
RANK = {
    "ceiling": 0,
    "floor": 1,
    "covering": 2,
    "wall": 3,
    "opening": 4,
    "builtin": 5,
    "large": 6,
    "other": 6,
    "small": 7,
}
# Two claimants whose surfaces near a pixel lie closer than this cannot be told apart by depth: the depth map's own
# spread on one flat surface (`provenance.json` puts the LiDAR fit's median error at 0.5-2 cm) with room to spare.
SAME_DEPTH_M = 0.03
# The windows a claimant's own surface is read in, as shares of the image's width: near first, then wider, then (past
# these) all of its own pixels.
_WINDOWS = (0.03, 0.1)
# Fewer of a claimant's own pixels than this in a window fit no plane.
_MIN_OWN = 10
# Points whose second spread is under this share of their first lie along a line, which fixes no plane (a conditioning
# limit of the fit, not a property of rooms).
_LINE_SHARE = 0.01
# How near its own surface a pixel must lie to be grown into: as near as this share of the claimant's own edge lies to
# the plane through its inside ("most of them"; a statistic of the claimant's own spread, not a distance).
_OWN_SPREAD_SHARE = 0.9
# The least tolerance: on exactly flat depth (a rendered or synthetic map) a claimant's own edge lies off its plane by
# rounding alone, which a pixel just past it may not match. A tenth of a millimetre is below any depth source's
# precision, so it decides nothing on a real map.
_MIN_TOLERANCE_M = 1e-4
# A gap along the camera ray is the gap square to the plane over the cosine between them; a plane seen exactly edge-on
# would make it endless. The cosine is held to this at least (a plane within 3 degrees of the ray: a gap 20 times the
# square one), which only keeps the division finite.
_MIN_RAY_COS = 0.05
# The classes that may lie on another claimant's surface as a layer: small items only. A large item flat on a wall (a
# mirror) or an opening flush with its wall is a thing of its own the layout places, not a print on its support.
LAYER_CLASSES = ("small",)
# The share of the photo `unknown` may take (gate 3).
UNKNOWN_BAR = 0.10

COLOURS = {
    "ceiling": (150, 150, 255),
    "floor": (200, 150, 60),
    "covering": (230, 190, 90),
    "wall": (70, 160, 230),
    "opening": (120, 230, 230),
    "builtin": (60, 200, 90),
    "large": (240, 120, 40),
    "other": (240, 220, 40),
    "small": (230, 60, 60),
}


def classes(run_dir: Path) -> dict[str, tuple[str, Path]]:
    """Every mask of the run, by label id: its class and its file."""
    def ids(path: Path, key: str) -> list[str]:
        if not path.exists():
            return []
        return [o["id"] for o in json.loads(path.read_text()).get(key, [])]

    survey = run_dir / "survey.json"
    named: dict[str, str] = {}
    for kind, key in (("wall", "walls"), ("opening", "openings"), ("builtin", "built_ins")):
        named |= {oid: kind for oid in ids(survey, key)}
    named |= {oid: "large" for oid in ids(run_dir / "large_survey.json", "objects")}
    named |= {oid: "small" for oid in ids(run_dir / "small_survey.json", "objects")}
    floors = run_dir / "floors.json"
    if floors.exists():
        for region in json.loads(floors.read_text()):
            named[f"floor_{region['id']}"] = "covering" if region.get("kind") == "covering" else "floor"

    found: dict[str, tuple[str, Path]] = {}
    for path in sorted((run_dir / "masks").glob("*.png")):
        oid = path.stem
        if oid.endswith("_contents"):
            continue  # the holes a holder's things stand in: not a thing, left to whatever fills them
        found[oid] = (named.get(oid, "other"), path)
    for name, kind in (("floor", "floor"), ("ceiling", "ceiling")):
        path = run_dir / f"{name}.png"
        if path.exists() and name not in found:
            found[name] = (kind, path)
    return found


def _load(path: Path, size: tuple[int, int]) -> np.ndarray:
    image = Image.open(path).convert("L")
    if image.size != size:
        image = image.resize(size, Image.NEAREST)
    return np.asarray(image) > 127


def resolve(masks: list[np.ndarray], ranks: list[int], points: np.ndarray,
            layers: list[bool] | None = None) -> tuple[np.ndarray, dict[str, int], np.ndarray]:
    """One label per pixel: 0 where no mask claims it, else 1 + the index of the mask that takes it.

    `points` is the camera-frame point map (h, w, 3; NaN where unknown). `layers` marks the masks that may lie on
    another's surface as a layer (`LAYER_CLASSES`; none if not given). Also returns how each pixel was settled:
    `SETTLED_ALONE` (one claimant or none), `SETTLED_BY_DEPTH`, `SETTLED_BY_RANK` or `SETTLED_LAYER` (kept by the
    surface under a layer mask; the layer is the highest-ranked mask claiming the pixel, as `layer_map` gives).
    """
    stack = np.stack(masks)
    count = stack.sum(axis=0)
    labels = np.zeros(count.shape, np.int32)
    sole = count == 1
    labels[sole] = np.argmax(stack[:, sole], axis=0) + 1
    contested = count > 1
    how = {"by_depth": 0, "by_rank": 0, "layer": 0}
    settled = np.full(count.shape, SETTLED_ALONE, np.uint8)
    if not contested.any():
        return labels, how, settled

    width = count.shape[1]
    known = np.isfinite(points).all(axis=-1)
    p = np.where(known[..., None], points, 0.0)
    here = p[contested]
    # Each claimant's own surface at each contested pixel: a plane through its unclaimed pixels nearby. How far the
    # pixel's point lies off that plane is its gap to the claimant.
    gap = np.full((len(masks), len(here)), np.nan)
    for k, mask in enumerate(masks):
        own = mask & sole & known
        if not own.any():
            # A mask lying wholly inside others (an item inside a wall's mask) has no pixels of its own: its surface is
            # the plane through all of its points.
            if (mask & known).sum() >= _MIN_OWN:
                gap[k] = _plane_gap(mask & known, p, contested, here, None)
            continue
        # Near, then wider, then all of its own points: a mask may run onto another further than any window reaches.
        for size in [max(3, int(round(width * share)) | 1) for share in _WINDOWS] + [None]:
            todo = np.isnan(gap[k])
            if not todo.any():
                break
            gap[k, todo] = _plane_gap(own, p, contested, here, size)[todo]
    gap[:, ~known[contested]] = np.nan  # a pixel with no depth lies on no surface

    claims = stack[:, contested]
    # The shell bounds the room, so a point that stands in front of a shell surface is not on it, whatever an item's
    # own surface says (a round lampshade fits no plane, and its mask may hold few pixels of its own beside a wall's).
    # In front by the depth: along the camera ray, where the depth's own error lies. Measured square to the plane, a
    # point 9 cm in front of a wall seen edge-on is 1.3 cm off it (BedroomScan 006).
    shell = np.asarray(ranks) <= RANK["opening"]
    item_claims = claims[~shell].any(axis=0)
    for k in np.flatnonzero(shell):
        asked = claims[k] & item_claims & np.isfinite(gap[k])
        if not asked.any():
            continue
        own = masks[k] & sole & known
        along = np.full(len(here), np.nan)
        for size in [max(3, int(round(width * share)) | 1) for share in _WINDOWS] + [None]:
            todo = np.isnan(along)
            if not todo.any() or not own.any():
                break
            along[todo] = _plane_gap(own, p, contested, here, size, along_ray=True)[todo]
        with np.errstate(invalid="ignore"):
            gap[k, asked & (along > SAME_DEPTH_M)] = np.nan

    gap[~claims | np.isnan(gap)] = np.inf
    order = np.argsort(gap, axis=0)
    best, second = order[0], order[1]
    cols = np.arange(gap.shape[1])
    near, next_near = gap[best, cols], gap[second, cols]
    with np.errstate(invalid="ignore"):
        decided = np.isfinite(near) & (next_near - near >= SAME_DEPTH_M)
    rank = np.asarray(ranks)[:, None] * np.ones_like(claims, dtype=np.int64)
    rank[~claims] = -1
    by_rank = np.argmax(rank, axis=0)
    pick = np.where(decided, best, by_rank)
    layer = np.zeros(len(here), bool)
    if layers is not None and any(layers):
        # Where rank would give the pixel to a layer mask, a claimant ranked below it whose surface the point lies on,
        # within that surface's own spread, keeps it: the layer adds nothing to the shape there.
        ranks_ = np.asarray(ranks)
        sizes = [max(3, int(round(width * share)) | 1) for share in _WINDOWS]
        undecided = ~decided & np.asarray(layers)[by_rank]
        under_gap = np.full(gap.shape, np.inf)
        rows, cols_ = np.nonzero(contested)
        for k in np.unique(np.flatnonzero(claims[:, undecided].any(axis=1))):
            asked = undecided & claims[k] & (ranks_[k] < ranks_[by_rank])
            own = masks[k] & sole & known
            tolerance = _tolerance(own, p, sizes)
            if tolerance is None and ranks_[k] <= RANK["opening"]:
                # A shell surface with too few pixels of its own to read (OfficeScan 006's window: 33 of 57 475, inside
                # its wall's and its blind's masks) is all of its points, as above: a plane. Not an item's: a bed's
                # whole mask holds its quilt and pillows, which would then lie on it (BedroomScan 005).
                own = masks[k] & known
                tolerance = _tolerance(own, p, sizes)
            if tolerance is None or not asked.any():
                continue
            # As growth reads a surface carried on past its edge: a window whose own points lie along a line (a strip
            # of the surface beside the item) is passed over for a wider one, not turned to face the camera.
            at = np.zeros(count.shape, bool)
            at[rows[asked], cols_[asked]] = True
            local = np.full(len(here), np.inf)
            local[asked] = np.nan_to_num(_local_gap(own, p, at, sizes), nan=np.inf)
            lies_on = asked & (local <= tolerance)
            under_gap[k, lies_on] = local[lies_on]
        # Of the surfaces the layer lies on, the most specific keeps it, as rank decides elsewhere where depth cannot: a
        # blind lies on its window, which lies in its wall (OfficeScan 006). Nearest among
        # those of one rank.
        keeper = np.argmax(np.where(np.isfinite(under_gap), ranks_[:, None] * 1e6 - under_gap, -np.inf), axis=0)
        layer = undecided & np.isfinite(under_gap[keeper, cols])
        pick = np.where(layer, keeper, pick)
    labels[contested] = pick + 1
    how["by_depth"] = int(decided.sum())
    how["by_rank"] = int((~decided & ~layer).sum())
    how["layer"] = int(layer.sum())
    settled[contested] = np.where(decided, SETTLED_BY_DEPTH, np.where(layer, SETTLED_LAYER, SETTLED_BY_RANK))
    return labels, how, settled


def grow(labels: np.ndarray, ranks: list[int], points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Give unclaimed pixels (0) to a neighbouring claimant whose surface they lie on.

    For each claimant next to unclaimed pixels: its tolerance is how far `_OWN_SPREAD_SHARE` of its own edge pixels lie
    off the local plane through its inside (its kept pixels a window's half-width or more from its edge), i.e. how
    well its surface carries on past its edge, measured on its own pixels. Every unclaimed pixel
    within that of the claimant's local plane is a candidate, and of those it takes the ones connected to it. A pixel
    two claimants reach goes to the one whose plane it lies nearer, then by `RANK` within `SAME_DEPTH_M`. Pixels with
    no depth lie on no surface and are left. Returns the new labels and the mask of grown pixels.
    """
    known = np.isfinite(points).all(axis=-1)
    p = np.where(known[..., None], points, 0.0)
    open_ = (labels == 0) & known
    if not open_.any():
        return labels, np.zeros(labels.shape, bool)
    width = labels.shape[1]
    sizes = [max(3, int(round(width * share)) | 1) for share in _WINDOWS]
    best_gap = np.full(labels.shape, np.inf)
    best_rank = np.full(labels.shape, -1)
    best = np.zeros(labels.shape, np.int32)
    four = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], bool)
    regions, _ = connected(open_, four)
    for value in np.unique(labels):
        if value == 0:
            continue
        own = (labels == value) & known
        if own.sum() < _MIN_OWN:
            continue
        touching = binary_dilation(labels == value, four) & open_
        if not touching.any():
            continue
        # The unclaimed regions next to it: only their pixels can be reached.
        reach = np.isin(regions, np.unique(regions[touching])) & open_
        tolerance = _tolerance(own, p, sizes)
        if tolerance is None:
            continue
        gap = np.full(labels.shape, np.inf)
        gap[reach] = np.nan_to_num(_local_gap(own, p, reach, sizes), nan=np.inf)
        candidate = gap <= tolerance
        parts, _ = connected(candidate, four)
        joined = np.unique(parts[binary_dilation(labels == value, four) & candidate])
        taken = np.isin(parts, joined[joined > 0])
        rank = ranks[value - 1]
        better = taken & ((gap + SAME_DEPTH_M <= best_gap) | ((gap < best_gap + SAME_DEPTH_M) & (rank > best_rank)))
        best[better], best_gap[better], best_rank[better] = value, gap[better], rank
    grown = best > 0
    out = labels.copy()
    out[grown] = best[grown]
    return out, grown


def _tolerance(own: np.ndarray, p: np.ndarray, sizes: list[int]) -> float | None:
    """How near its own surface a point must lie to be on it, for the claimant whose kept pixels (with depth) are
    `own`: how far `_OWN_SPREAD_SHARE` of its edge lies off the local plane its inside gives, i.e. the error of carrying
    its surface on past its edge by a window's half-width. None where it has too little inside or edge to tell."""
    four = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], bool)
    inside = binary_erosion(own, four, iterations=sizes[0] // 2)
    edge = own & ~inside
    if inside.sum() < _MIN_OWN or not edge.any():
        return None
    spread = _local_gap(inside, p, edge, sizes)
    spread = spread[np.isfinite(spread)]
    if not len(spread):
        return None
    return max(float(np.quantile(spread, _OWN_SPREAD_SHARE)), _MIN_TOLERANCE_M)


def layer_map(masks: list[np.ndarray], ranks: list[int], settled: np.ndarray) -> np.ndarray:
    """At each pixel settled as a layer, 1 + the index of the layer mask (the highest-ranked claimant); 0 elsewhere."""
    out = np.zeros(settled.shape, np.int32)
    at = settled == SETTLED_LAYER
    if at.any():
        rank = np.where(np.stack(masks)[:, at], np.asarray(ranks)[:, None], -1)
        out[at] = np.argmax(rank, axis=0) + 1
    return out


def relations(labels: np.ndarray, layers: np.ndarray, ids: list[str], on: dict[str, str], points: np.ndarray,
              kinds: list[str] | None = None) -> dict:
    """Each small item against the support its listing names (`on`): `layer` (it lies on it, none of its pixels its
    own), `touching` (somewhere along their shared border the two points lie within `SAME_DEPTH_M`: the depth runs on
    from one to the other), `not touching` (they share a border, but depth jumps all along it: in front, not on),
    `apart` (no shared border in the photo) or `no support` (none named, or not masked). A support named by a class
    (the listing's "wall", "floor") is every label of that class (`kinds`)."""
    value = {oid: i + 1 for i, oid in enumerate(ids)}
    by_class = {}
    for i, kind in enumerate(kinds or []):
        by_class.setdefault(kind, []).append(i + 1)
    out = {}
    for item, support in on.items():
        if item not in value:
            continue
        mine, kept = labels == value[item], int((labels == value[item]).sum())
        layer_px = int((layers == value[item]).sum())
        support_values = [value[support]] if support in value else by_class.get(support, [])
        if not support_values:
            out[item] = {"on": support, "relation": "no support", "kept_px": kept, "layer_px": layer_px}
            continue
        theirs = np.isin(labels, support_values)
        if not kept and layer_px:
            out[item] = {"on": support, "relation": "layer", "kept_px": 0, "layer_px": layer_px}
            continue
        shared = continuous = 0
        for dy, dx in ((0, 1), (1, 0), (0, -1), (-1, 0)):
            a = mine[max(0, -dy):labels.shape[0] - max(0, dy), max(0, -dx):labels.shape[1] - max(0, dx)]
            b = theirs[max(0, dy):, max(0, dx):][:a.shape[0], :a.shape[1]]
            pa = points[max(0, -dy):labels.shape[0] - max(0, dy), max(0, -dx):labels.shape[1] - max(0, dx)]
            pb = points[max(0, dy):, max(0, dx):][:a.shape[0], :a.shape[1]]
            both = a & b & np.isfinite(pa).all(-1) & np.isfinite(pb).all(-1)
            shared += int(both.sum())
            continuous += int((np.linalg.norm(pa[both] - pb[both], axis=-1) < SAME_DEPTH_M).sum())
        relation = "apart" if not shared else "touching" if continuous else "not touching"
        out[item] = {"on": support, "relation": relation, "kept_px": kept, "layer_px": layer_px,
                     "shared_border_px": shared, "continuous_px": continuous}
    return out


def _local_gap(own: np.ndarray, p: np.ndarray, at: np.ndarray, sizes: list[int]) -> np.ndarray:
    """At each pixel of `at`, how far its point lies off `own`'s plane: near windows first, then all of `own`.

    A window whose own points lie along a line (a thin strip of the claimant at its edge) fixes no plane: it is passed
    over for a wider one, since growing carries the plane on further than settling an overlap does. Only all of
    `own` may stand as a line, as `_plane_gap` takes it."""
    here = p[at]
    gap = np.full(len(here), np.nan)
    for size in sizes + [None]:
        todo = np.isnan(gap)
        if not todo.any():
            break
        gap[todo] = _plane_gap(own, p, at, here, size, lines=size is None)[todo]
    return gap


def _plane_gap(own: np.ndarray, p: np.ndarray, contested: np.ndarray, here: np.ndarray, size: int | None,
               lines: bool = True, along_ray: bool = False) -> np.ndarray:
    """At each contested pixel, how far its point lies off the plane through `own`'s points in a `size` window, or
    through all of them when `size` is None (NaN where the window holds fewer than `_MIN_OWN` of them, or, with
    `lines` False, where they lie along a line). Square to the plane, or with `along_ray` along the camera ray: how
    far the depth would have to move for the point to lie on the plane."""
    w = own.astype(np.float64)
    if size is None:
        def mean(a: np.ndarray) -> np.ndarray:
            return np.full(len(here), a.mean())
        area = own.size
    else:
        def mean(a: np.ndarray) -> np.ndarray:
            return uniform_filter(a, size)[contested]
        area = size * size
    n = mean(w) * area
    usable = n >= _MIN_OWN
    out = np.full(len(here), np.nan)
    if not usable.any():
        return out
    first = np.stack([mean(p[..., i] * w) for i in range(3)], axis=-1)
    second = np.empty((len(here), 3, 3))
    for i in range(3):
        for j in range(i, 3):
            second[:, i, j] = second[:, j, i] = mean(p[..., i] * p[..., j] * w)
    share = n[usable] / area
    mean = first[usable] / share[:, None]
    cov = second[usable] / share[:, None, None] - mean[:, :, None] * mean[:, None, :]
    values, vectors = np.linalg.eigh(cov)
    normal = vectors[:, :, 0]  # the direction the points spread least along
    # Points along a line (a mask's edge, a strip) leave the plane free to turn about it: there the normal is taken
    # in that free plane, facing the camera, i.e. the surface goes on at the depth it shows.
    line = values[:, 1] < _LINE_SHARE * np.maximum(values[:, 2], 1e-12)
    def off(normal: np.ndarray, mean: np.ndarray, at: np.ndarray) -> np.ndarray:
        square = np.abs(np.einsum("ij,ij->i", normal, at - mean))
        if not along_ray:
            return square
        ray = at / np.maximum(np.linalg.norm(at, axis=1, keepdims=True), 1e-12)
        return square / np.maximum(np.abs(np.einsum("ij,ij->i", normal, ray)), _MIN_RAY_COS)

    if line.any() and not lines:
        keep = np.flatnonzero(usable)
        usable[keep[line]] = False
        out[usable] = off(normal[~line], mean[~line], here[usable])
        return out
    if line.any():
        ray = here[usable][line] / np.maximum(np.linalg.norm(here[usable][line], axis=1, keepdims=True), 1e-12)
        e0, e1 = vectors[line, :, 0], vectors[line, :, 1]
        facing = np.einsum("ij,ij->i", ray, e0)[:, None] * e0 + np.einsum("ij,ij->i", ray, e1)[:, None] * e1
        normal[line] = facing / np.maximum(np.linalg.norm(facing, axis=1, keepdims=True), 1e-12)
    out[usable] = off(normal, mean, here[usable])
    return out


def label_run(run_dir: Path) -> dict:
    """Build the run's label map; returns its report (also in `labels.json`)."""
    photo = Image.open(run_dir / "input.jpg").convert("RGB")
    size = photo.size
    found = classes(run_dir)
    if not found:
        raise SystemExit(f"{run_dir}: no masks")
    ids = list(found)
    masks = [_load(found[oid][1], size) for oid in ids]
    kinds = [found[oid][0] for oid in ids]
    # A survey wall the shell found to be a built-in's end panel (`room.built_in_ends`) is that built-in's.
    shell = run_dir / "shell.json"
    ends = json.loads(shell.read_text()).get("built_in_ends", {}) if shell.exists() else {}
    for oid, built_id in ends.items():
        if oid in ids and built_id in ids:
            masks[ids.index(built_id)] = masks[ids.index(built_id)] | masks[ids.index(oid)]
    for oid in [oid for oid in ends if oid in ids and ends[oid] in ids]:
        i = ids.index(oid)
        del ids[i], masks[i], kinds[i]
    # A wall's mask stops at its baseboard (the segmenter takes the paint); the shell found the pixels at the wall's
    # foot from its surface and gives them back to the wall (`cli._save_wall_feet`).
    for i, oid in enumerate(ids):
        board = run_dir / "masks" / "wall_foot" / f"{oid}.png"
        if kinds[i] == "wall" and board.exists():
            masks[i] = masks[i] | _load(board, size)
    # The survey's wall masks follow the paint; the shell found which wall each wall pixel lies on
    # (`cli._save_wall_feet`). A pixel on a fitted wall goes to that wall's survey wall, and leaves the others.
    surfaces = {i: _load(run_dir / "masks" / "wall_surface" / f"{oid}.png", size) for i, oid in enumerate(ids)
                if kinds[i] == "wall" and (run_dir / "masks" / "wall_surface" / f"{oid}.png").exists()}
    for i in (i for i, kind in enumerate(kinds) if kind == "wall" and surfaces):
        others = [s for j, s in surfaces.items() if j != i]
        if others:
            masks[i] = masks[i] & ~np.any(others, axis=0)
        if i in surfaces:
            masks[i] = masks[i] | surfaces[i]
    regions = [m for oid, m in zip(ids, masks) if oid.startswith("floor_")]
    if "floor" in ids and regions:
        masks[ids.index("floor")] = masks[ids.index("floor")] & ~np.any(regions, axis=0)
    points = np.load(run_dir / "points.npy").astype(np.float64)
    if points.shape[:2] != (size[1], size[0]):
        raise SystemExit(f"{run_dir}: points.npy is {points.shape[:2]}, the photo {size[::-1]}")

    ranks = [RANK[k] for k in kinds]
    labels, how, settled = resolve(masks, ranks, points, [k in LAYER_CLASSES for k in kinds])
    layers = layer_map(masks, ranks, settled)
    unknown_before = float((labels == 0).mean())
    labels, grown = grow(labels, ranks, points)
    settled[grown] = SETTLED_GROWN
    total = labels.size
    claimed = np.stack(masks).sum(axis=0)
    shares = {}
    for kind in RANK:
        taken = np.isin(labels, [i + 1 for i, k in enumerate(kinds) if k == kind])
        if taken.any():
            shares[kind] = round(float(taken.mean()), 4)
    items = []
    for i, oid in enumerate(ids):
        mask_px = int(masks[i].sum())
        kept = int(((labels == i + 1) & ~grown).sum())
        items.append({"value": i + 1, "id": oid, "class": kinds[i], "mask_px": mask_px, "kept_px": kept,
                      "grown_px": int(((labels == i + 1) & grown).sum()),
                      "layer_px": int((layers == i + 1).sum()),
                      "share": round(float((labels == i + 1).mean()), 4), "lost": round(1 - kept / mask_px, 4) if mask_px else None})
    unknown = float((labels == 0).mean())
    small_survey = run_dir / "small_survey.json"
    on = {o["id"]: o.get("on") for o in json.loads(small_survey.read_text()).get("objects", [])} if small_survey.exists() else {}
    related = relations(labels, layers, ids, on, points, kinds)
    contents = [p for p in (run_dir / "masks").glob("*_contents.png")]
    in_contents = np.zeros(labels.shape, bool)
    for p in contents:
        in_contents |= _load(p, size)
    report = {
        "unknown": round(unknown, 4),
        "unknown_bar": UNKNOWN_BAR,
        "within_bar": unknown <= UNKNOWN_BAR,
        "unknown_before_growth": round(unknown_before, 4),
        "grown": round(float(grown.mean()), 4),
        "grown_by_class": {kind: round(float((grown & np.isin(labels, [i + 1 for i, k in enumerate(kinds) if k == kind])).mean()), 4)
                           for kind in RANK if any(k == kind for k in kinds)},
        "unknown_in_contents": round(float((in_contents & (labels == 0)).mean()), 4),
        # Each wall's foot (`cli._save_wall_feet`): its pixels, and the shares labelled that wall, another class, unknown.
        "wall_foot": _wall_feet(run_dir, size, labels, ids, kinds),
        "overlap": round(float((claimed > 1).mean()), 4),
        "overlap_by_depth": round(how["by_depth"] / total, 4),
        "overlap_by_rank": round(how["by_rank"] / total, 4),
        "layer": round(how["layer"] / total, 4),
        "layer_items": sorted({ids[v - 1] for v in np.unique(layers) if v}),
        "relations": related,
        "classes": shares,
        "unnamed": [oid for oid, k in zip(ids, kinds) if k == "other"],
    }
    if len(ids) > 255:
        raise SystemExit(f"{run_dir}: {len(ids)} labels do not fit an 8-bit map")
    Image.fromarray(labels.astype(np.uint8)).save(run_dir / "labels.png")
    (run_dir / "labels.json").write_text(json.dumps({"labels": items, "report": report}, indent=2))
    (run_dir / "debug").mkdir(exist_ok=True)
    Image.fromarray(settled).save(run_dir / "debug" / "labels_settled.png")
    Image.fromarray(layers.astype(np.uint8)).save(run_dir / "debug" / "labels_layers.png")
    draw_labels(photo, labels, kinds, claimed > 1).save(run_dir / "debug" / "labels.jpg", quality=88)
    return report


def draw_labels(photo: Image.Image, labels: np.ndarray, kinds: list[str], contested: np.ndarray) -> Image.Image:
    """Left: the photo tinted by class, `unknown` in magenta. Right: where masks overlapped, outlined in white."""
    base = np.asarray(photo).astype(np.float32)
    lut = np.array([(255, 0, 255)] + [COLOURS[k] for k in kinds], np.float32)
    tint = lut[labels]
    left = base * 0.55 + tint * 0.45
    left[labels == 0] = base[labels == 0] * 0.3 + np.array([255, 0, 255]) * 0.7
    # Edges between labels, so neighbouring items of one class stay apart.
    edge = np.zeros(labels.shape, bool)
    edge[1:, :] |= labels[1:, :] != labels[:-1, :]
    edge[:, 1:] |= labels[:, 1:] != labels[:, :-1]
    left[edge] = 0
    right = base * 0.4
    right[contested] = 255
    return Image.fromarray(np.concatenate([left, right], axis=1).clip(0, 255).astype(np.uint8))


def pose_off(run_dir: Path) -> str | None:
    """Why the run's shot's pose failed gate 0, or None. The run is used like any other; only its comparisons with the
    references (drawn by that pose) are not to be trusted."""
    from aomma.scancheck import pose_failures

    manifest = run_dir / "run.json"
    run = json.loads(manifest.read_text()) if manifest.exists() else {}
    return pose_failures(Path(run["scan"]).name).get(run.get("shot")) if run.get("scan") else None


def _wall_feet(run_dir: Path, size: tuple[int, int], labels: np.ndarray, ids: list[str], kinds: list[str]) -> dict:
    """For each wall given a foot: how many pixels it has and what share of them the label map gives that wall, any
    other claimant, or no one."""
    found = {}
    for i, oid in enumerate(ids):
        path = run_dir / "masks" / "wall_foot" / f"{oid}.png"
        if kinds[i] != "wall" or not path.exists():
            continue
        foot = _load(path, size)
        n = int(foot.sum())
        if not n:
            continue
        own = float((labels[foot] == i + 1).mean())
        unknown = float((labels[foot] == 0).mean())
        found[oid] = {"px": n, "wall": round(own, 4), "other": round(1 - own - unknown, 4), "unknown": round(unknown, 4)}
    return found


def describe(run_dir: Path, report: dict) -> str:
    classes_line = ", ".join(f"{k} {100 * v:.1f}%" for k, v in report["classes"].items())
    flag = "" if report["within_bar"] else f"  (over the {100 * report['unknown_bar']:.0f}% bar)"
    before = report.get("unknown_before_growth")
    grown = f" (from {100 * before:.1f}% before growth)" if before is not None else ""
    lines = [f"{run_dir.name}: unknown {100 * report['unknown']:.1f}%{grown}{flag}; overlap {100 * report['overlap']:.1f}% "
             f"(by depth {100 * report['overlap_by_depth']:.1f}%, by rank {100 * report['overlap_by_rank']:.1f}%)",
             f"  {classes_line}"]
    if report.get("wall_foot"):
        lines.append("  wall feet: " + ", ".join(f"{w} {f['px']} px, {100 * f['wall']:.0f}% wall, {100 * f['other']:.0f}% other, "
                                                 f"{100 * f['unknown']:.0f}% unknown" for w, f in report["wall_foot"].items()))
    if report.get("layer"):
        lines.append(f"  layers {100 * report['layer']:.1f}%: {', '.join(report['layer_items'])}")
    relations_ = report.get("relations") or {}
    by = {}
    for item, r in relations_.items():
        by.setdefault(r["relation"], []).append(f"{item} on {r['on']}")
    for relation in ("not touching", "apart", "no support"):
        if by.get(relation):
            lines.append(f"  {relation}: {', '.join(by[relation])}")
    if relations_:
        lines.append("  small items: " + ", ".join(f"{len(by.get(r, []))} {r}" for r in
                                                  ("touching", "layer", "not touching", "apart", "no support")))
    if report["unnamed"]:
        lines.append(f"  masks no list names: {', '.join(report['unnamed'])}")
    if why := pose_off(run_dir):
        lines.append(f"  pose off (gate 0; comparisons with the references not to be trusted): {why}")
    return "\n".join(lines)
