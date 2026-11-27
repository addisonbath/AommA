"""AommA's side of the scan and layer contract (docs/scan-contract.md in AommACapture, version 0.5).

What the capture apps ask of the dashboard, apart from sending photos and scans:

- the scans held here, each with its newest run (`list_scans`);
- a scan's own files, as uploaded (served by the dashboard);
- a scan's runs that have a scene, so the person can choose one (`list_runs`);
- the newest run of a scan that has a scene, or a chosen one, as the contract's result (`result_for`, `run_result`): `aomma.usda`, the generated
  objects in the scan's world frame (ARKit's: +Y up, metres) with provenance as `customData`; `scene.usdz`, that
  layer packed with its assets for RealityKit; and `result.json`, the object list RealityKit cannot read from USD;
- a person's review of a run, made on the headset (`save_review`), kept under `reviews/<scan>/`. Its `edits.usda`
  is the user's and is never written here.

A run's own world is Z up (layout.py); it goes to the scan's by its shot's pose (`views.run_to_scan`). Each object's
transform in `aomma.usda` is that and its placement multiplied out, one `xformOp:transform` relative to
`/AommA/Objects`, so an edit's override of it (`edits.usda`) replaces exactly what was written.
"""

import json
import re
import shutil
import tempfile
import threading
import zipfile
from datetime import datetime, timezone
from pathlib import Path

CONTRACT_VERSION = "0.5"
# A manifest whose major version is not one of these is refused (scan-contract.md, "manifest.json").
KNOWN_MAJORS = ("0",)
# Where the generated objects sit in aomma.usda, and so in scene.usdz.
ROOT = "/AommA"
OBJECTS = f"{ROOT}/Objects"
RESULT = "result"  # the folder in a run holding result.json and scene.usdz


def scan_manifest(folder: Path) -> dict:
    """The scan's manifest.json; a scan without one (from before 0.4) reads as version 0.3 from iOS.

    Raises ValueError for a contract major version not known here; unknown fields are ignored.
    """
    path = folder / "manifest.json"
    manifest = json.loads(path.read_text()) if path.is_file() else {"contractVersion": "0.3", "producer": {"platform": "iOS"}}
    check_version(manifest.get("contractVersion"), f"{folder.name}'s manifest")
    return manifest


def check_version(version, what: str) -> None:
    if not isinstance(version, str) or version.split(".")[0] not in KNOWN_MAJORS:
        raise ValueError(f"{what} is contract version {version!r}; this AommA reads {CONTRACT_VERSION} (major {', '.join(KNOWN_MAJORS)})")


def unpack(archive: Path, into: Path) -> Path:
    """Unzip `archive` into the empty folder `into`, leaving out Apple's metadata files; the folder the zip holds,
    or `into` itself when the zip holds its files loose."""
    if not zipfile.is_zipfile(archive):
        raise ValueError("not a zip file")
    with zipfile.ZipFile(archive) as zipped:
        # Apple's zips add metadata files: `__MACOSX/`, `._<name>` beside each file, `.DS_Store`.
        members = [
            member for member in zipped.namelist()
            if member.split("/")[0] != "__MACOSX" and not member.rsplit("/", 1)[-1].startswith(("._", ".DS_Store"))
        ]
        # extractall drops absolute paths and `..`, so nothing lands outside `into`.
        zipped.extractall(into, members)
    entries = list(into.iterdir())
    return entries[0] if len(entries) == 1 and entries[0].is_dir() else into


def clean_name(name: str, limit: int = 80) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "-", name).strip("-")[:limit]


def scan_folder(scans: Path, scan: str) -> Path | None:
    """The scan folder named `scan` directly inside `scans`, or None."""
    folder = (scans / scan).resolve()
    return folder if folder.parent == scans.resolve() and (folder / "capture.json").is_file() else None


def runs_of(scan: str, runs: Path) -> list[Path]:
    """The runs made from the scan named `scan`, newest first (run folders start with the time they were made)."""
    found = []
    for run_dir in sorted(runs.glob("*/"), reverse=True):
        try:
            manifest = json.loads((run_dir / "run.json").read_text())
        except (OSError, ValueError):
            continue
        if Path(manifest.get("scan") or "").name == scan:
            found.append(run_dir)
    return found


def has_scene(run_dir: Path) -> bool:
    return (run_dir / "scene.usda").is_file() and (run_dir / "layout.json").is_file()


def list_scans(scans: Path, runs: Path) -> list[dict]:
    """`GET /scans`: each scan with when it was made, on which platform, and its newest run (or None), newest scan first."""
    listed = []
    for folder in sorted(scans.glob("*/")):
        if folder.name.startswith(".") or not (folder / "capture.json").is_file():
            continue
        try:
            manifest = scan_manifest(folder)
        except ValueError:
            continue  # a scan this AommA cannot read is not offered
        capture = json.loads((folder / "capture.json").read_text())
        made = runs_of(folder.name, runs)
        listed.append({
            "scan": folder.name,
            "created": manifest.get("createdAt") or capture.get("created"),
            "platform": (manifest.get("producer") or {}).get("platform") or capture.get("platform") or "iOS",
            "latestRun": made[0].name if made else None,
        })
    return sorted(listed, key=lambda s: s["created"] or "", reverse=True)


def _made_at(run_dir: Path) -> str | None:
    """When the run was made, from its folder name (`20261126-105914-...`, local time), as ISO 8601 with the offset."""
    try:
        return datetime.strptime(run_dir.name[:15], "%Y%m%d-%H%M%S").astimezone().isoformat()
    except ValueError:
        return None


def list_runs(scan: str, runs: Path) -> list[dict]:
    """`GET /scans/<scan>/runs`: the scan's runs that have a scene, newest first, each with when it was made, the shot
    it was made from and how many objects it places, so the person can choose which to review."""
    from aomma.layout import SceneLayout

    listed = []
    for run_dir in runs_of(scan, runs):
        if not has_scene(run_dir):
            continue
        try:
            layout = SceneLayout.model_validate_json((run_dir / "layout.json").read_text())
        except (OSError, ValueError):
            continue
        manifest = json.loads((run_dir / "run.json").read_text())
        listed.append({
            "run": run_dir.name,
            "created": _made_at(run_dir),
            "shot": manifest.get("shot"),
            "objects": sum(1 for placed in layout.objects if not placed.left_out),
        })
    return listed


def run_result(scan: str, run: str, runs: Path, project: Path) -> Path | None:
    """`GET /scans/<scan>/runs/<run>/result`: the result folder of the named run, or None if it is not a run of this
    scan with a scene."""
    run_dir = (runs / run).resolve()
    if run_dir.parent != runs.resolve() or run_dir not in [r.resolve() for r in runs_of(scan, runs)] or not has_scene(run_dir):
        return None
    return write_result(run_dir, project)


def label_of(name: str) -> str:
    """"chair" from the placement "chair_2"; the name itself when it has no number."""
    return re.sub(r"_\d+$", "", name)


def write_layer(run_dir: Path, project: Path) -> list[dict]:
    """Write the run's `aomma.usda`: its placed objects in its scan's world frame, with provenance; returns the object list.

    The run's own scene.usda is left as it is (Z up, the run's world), for the run's renders and scorers.
    """
    import numpy as np
    from pxr import Gf, Usd, UsdGeom, Vt

    from aomma.capture import Scan
    from aomma.layout import SceneLayout
    from aomma.views import run_to_scan

    manifest = json.loads((run_dir / "run.json").read_text())
    scan_path = Path(manifest["scan"])
    scan = Scan(scan_path if scan_path.is_absolute() else project / scan_path)
    to_scan = run_to_scan(run_dir, scan.shot(manifest["shot"]))
    layout = SceneLayout.model_validate_json((run_dir / "layout.json").read_text())
    provenance = {"aomma:runId": run_dir.name, "aomma:source": scan.name, "aomma:contractVersion": CONTRACT_VERSION}

    path = run_dir / "aomma.usda"
    partial = run_dir / "aomma.partial.usda"
    stage = Usd.Stage.CreateNew(str(partial))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    stage.GetRootLayer().customLayerData = provenance
    root = UsdGeom.Xform.Define(stage, ROOT)
    stage.SetDefaultPrim(root.GetPrim())
    root.GetPrim().SetCustomData(provenance)
    UsdGeom.Xform.Define(stage, OBJECTS)
    # USD matrices act on row vectors: a point goes by the placement, then into the scan's world.
    into_scan = Gf.Matrix4d(*np.asarray(to_scan, float).T.flatten().tolist())
    scratch = Usd.Stage.CreateInMemory()
    objects = []
    for placed in layout.objects:
        if placed.left_out:
            continue
        # The placement's own transform, as compose_scene writes it into scene.usda.
        local = UsdGeom.Xform.Define(scratch, f"/{placed.name}")
        api = UsdGeom.XformCommonAPI(local)
        api.SetTranslate(Gf.Vec3d(*placed.translate))
        api.SetRotate(Gf.Vec3f(*placed.rotate_xyz_deg), UsdGeom.XformCommonAPI.RotationOrderXYZ)
        api.SetScale(Gf.Vec3f(placed.scale))
        matrix = local.GetLocalTransformation() * into_scan

        prim_path = f"{OBJECTS}/{placed.name}"
        xform = UsdGeom.Xform.Define(stage, prim_path)
        xform.GetPrim().GetReferences().AddReference(placed.asset)
        xform.AddTransformOp().Set(matrix)
        xform.GetPrim().SetCustomData({**provenance, "aomma:objectId": placed.name, "aomma:label": label_of(placed.name)})
        objects.append({"objectId": placed.name, "primPath": prim_path, "label": label_of(placed.name)})
    stage.GetRootLayer().Save()
    partial.replace(path)
    return objects


_result_lock = threading.Lock()


def _newest_input(run_dir: Path) -> float:
    paths = [run_dir / "scene.usda", run_dir / "layout.json", *(run_dir / "assets").rglob("*")]
    return max(p.stat().st_mtime for p in paths if p.exists())


def write_result(run_dir: Path, project: Path) -> Path:
    """The run's result folder, holding result.json and scene.usdz, written again whenever the run's scene has changed."""
    from pxr import Sdf, UsdUtils

    folder = run_dir / RESULT
    result = folder / "result.json"
    with _result_lock:
        if result.is_file() and result.stat().st_mtime >= _newest_input(run_dir):
            try:
                current = json.loads(result.read_text()).get("contractVersion") == CONTRACT_VERSION
            except ValueError:
                current = False
            if current:  # one written under an older contract is written again
                return folder
        objects = write_layer(run_dir, project)
        folder.mkdir(exist_ok=True)
        # Packed under another name first, so a failed pack never leaves a broken file behind.
        partial = folder / "scene.partial.usdz"
        if not UsdUtils.CreateNewUsdzPackage(Sdf.AssetPath(str(run_dir / "aomma.usda")), str(partial)):
            raise RuntimeError(f"could not pack {run_dir.name}/aomma.usda as USDZ")
        partial.replace(folder / "scene.usdz")
        manifest = json.loads((run_dir / "run.json").read_text())
        result.write_text(json.dumps({
            "scan": Path(manifest["scan"]).name,
            "runId": run_dir.name,
            "contractVersion": CONTRACT_VERSION,
            "scene": "scene.usdz",
            "objects": objects,
        }, indent=2))
    return folder


def result_for(scan: str, runs: Path, project: Path) -> Path | None:
    """`GET /scans/<scan>/result`: the result folder of the scan's newest run with a scene, or None if it has none."""
    for run_dir in runs_of(scan, runs):
        if has_scene(run_dir):
            return write_result(run_dir, project)
    return None


def save_review(archive: Path, scan: str, reviews: Path) -> str:
    """`POST /scans/<scan>/reviews`: unpack a zipped review folder (review.json and edits.usda) into `reviews/<scan>/`;
    returns the review folder's name. A name already used gets the time added rather than replacing that review.

    Raises ValueError if the zip holds no review.json, or one of an unknown contract version or for another scan.
    """
    into = reviews / scan
    into.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=into, prefix=".upload-") as unpacked:
        unpacked = Path(unpacked)
        folder = unpack(archive, unpacked)
        try:
            review = json.loads((folder / "review.json").read_text())
        except FileNotFoundError:
            raise ValueError("the zip holds no review: review.json is missing") from None
        except ValueError:
            raise ValueError("review.json is not JSON") from None
        check_version(review.get("contractVersion"), "the review")
        if review.get("scan") != scan:
            raise ValueError(f"the review is of {review.get('scan')!r}, sent to {scan!r}")
        stem = clean_name(folder.name if folder != unpacked else "") or f"{scan}-{int(datetime.now(timezone.utc).timestamp())}"
        target = into / stem
        if target.exists():
            target = into / f"{stem}-{datetime.now():%H%M%S}"
        if folder == unpacked:
            target.mkdir()
            for entry in list(unpacked.iterdir()):
                shutil.move(entry, target / entry.name)
        else:
            folder.rename(target)
    return target.name
