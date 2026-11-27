"""Local web page for watching the pipeline: live ComfyUI jobs, each run's results and each experiment's trials.

A background thread follows ComfyUI (websocket events plus the job list) and
keeps a snapshot; the page polls that snapshot as JSON once a second. Run and
experiment results are read straight from their folders on each request.
"""

import hashlib
import io
import json
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import urllib.parse
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from datetime import datetime
from pathlib import Path

import httpx
from PIL import Image, UnidentifiedImageError
from pxr import Sdf, UsdUtils
from websockets.sync.client import connect

from aomma import background, rounds
from aomma.backends.comfy import CLIENT_ID, read_job_labels
from aomma.batch import live_batches
from aomma.config import Config
from aomma.jobqueue import JobQueue
from aomma.models import pipeline_models
from aomma.render.blender import usd_to_glb

_PAGE = Path(__file__).with_name("dashboard.html")
_VIEWER = Path(__file__).with_name("viewer.html")
# How long the Models tab's answer is reused, in seconds.
_MODELS_KEPT_S = 120
# How many finished jobs keep their per-node timings.
_KEEP_TIMELINES = 20


def _page_version() -> str:
    """Changes whenever the page or this server's code does.

    The page carries the version it was served with and reloads itself when
    the server reports a different one, so an open tab never runs a stale script
    against a newer server.
    """
    return hashlib.sha1(_PAGE.read_bytes() + _VIEWER.read_bytes() + Path(__file__).read_bytes()).hexdigest()[:12]


class JobTracker:
    """Follows ComfyUI and holds the latest state for the page."""

    def __init__(self, url: str, lane: str, job_log: Path):
        self.url = url.rstrip("/")
        self.lane = lane
        self.job_log = job_log
        self._free_memory_gb: float | None = None
        self._lock = threading.Lock()
        self._connected = False
        self._active: list[dict] = []
        self._recent: list[dict] = []
        # Job id -> {"started", "ended", "total", "cached", "nodes": [...], "error"}.
        # Each node entry is {"node", "start", "end", "step", "steps"}.
        self._progress: dict[str, dict] = {}

    def snapshot(self) -> dict:
        with self._lock:
            return json.loads(
                json.dumps(
                    {
                        "url": self.url,
                        "lane": self.lane,
                        "connected": self._connected,
                        "free_memory_gb": self._free_memory_gb,
                        "active": self._active,
                        "recent": self._recent,
                        "progress": self._progress,
                    }
                )
            )

    def run_forever(self) -> None:
        threading.Thread(target=self._poll_jobs, daemon=True).start()
        ws_url = self.url.replace("http", "ws", 1) + f"/ws?clientId={CLIENT_ID}"
        while True:
            try:
                with connect(ws_url, max_size=None) as ws:
                    self._connected = True
                    for message in ws:
                        if isinstance(message, str):
                            self._handle(json.loads(message))
            except Exception:
                pass  # ComfyUI restarting or unreachable: retry below
            self._connected = False
            time.sleep(3)

    def _poll_jobs(self) -> None:
        http = httpx.Client(base_url=self.url, timeout=30)
        while True:
            try:
                labels = read_job_labels(self.job_log)
                jobs = http.get("/api/jobs", params={"limit": 15}).json()["jobs"]
                # Node counts come from the queue, which carries each job's workflow.
                queue = http.get("/queue").json()
                totals = {
                    entry[1]: len(entry[2]) for entry in queue["queue_running"] + queue["queue_pending"]
                }
                for job in jobs:
                    job["label"] = labels.get(job["id"], "")
                free = http.get("/system_stats").json()["system"]["ram_free"] / 1e9
                active = [j for j in jobs if j["status"] in ("in_progress", "pending")]
                with self._lock:
                    # Running first, then queued in submission order.
                    self._active = sorted(active, key=lambda j: (j["status"] != "in_progress", j["create_time"]))
                    self._recent = [j for j in jobs if j not in active]
                    self._free_memory_gb = round(free, 1)
                    for job_id, total in totals.items():
                        self._job(job_id)["total"] = total
            except Exception:
                pass
            time.sleep(2)

    def _job(self, job_id: str) -> dict:
        """The progress record for a job, created on first sight. Caller holds the lock."""
        if job_id not in self._progress:
            self._progress[job_id] = {
                "started": None, "ended": None, "total": None, "cached": 0, "nodes": [], "error": None,
            }
            for old in list(self._progress)[:-_KEEP_TIMELINES]:
                del self._progress[old]
        return self._progress[job_id]

    def _handle(self, event: dict) -> None:
        kind, data = event["type"], event["data"]
        job_id = data.get("prompt_id")
        if not job_id:
            return
        now = time.time()
        with self._lock:
            job = self._job(job_id)
            running = job["nodes"][-1] if job["nodes"] and job["nodes"][-1]["end"] is None else None
            if kind == "execution_start":
                job["started"] = now
            elif kind == "execution_cached":
                job["cached"] = len(data["nodes"])
            elif kind == "executing":
                if running:
                    running["end"] = now
                if data["node"]:
                    job["nodes"].append(
                        {"node": data["node"], "start": now, "end": None, "step": None, "steps": None}
                    )
            elif kind == "progress":
                if not running:
                    # Joined partway through a node: start timing it from now.
                    running = {"node": data.get("node"), "start": now, "end": None, "step": None, "steps": None}
                    job["nodes"].append(running)
                running["step"], running["steps"] = data["value"], data["max"]
            elif kind in ("execution_success", "execution_error", "execution_interrupted"):
                if running:
                    running["end"] = now
                job["ended"] = now
                if kind == "execution_error":
                    job["error"] = f"{data.get('node_id')}: {data.get('exception_message', '').strip()}"
                elif kind == "execution_interrupted":
                    job["error"] = "interrupted"


# The pictures each stage leaves in a run folder, in pipeline order: (title, what the stage does, file or files).
_STAGE_IMAGES = [
    ("Photo", "The input, scaled down. Its lens angle is read from the photo's metadata.", "input.jpg"),
    (
        "Shell Inventory",
        "A vision-language model lists the room's shell: walls, floor, ceiling, doors and windows, and built-ins, each with a box.",
        "debug/inventory.jpg",
    ),
    ("Shell Mask", "Each shell piece is segmented into an outline: built-ins, openings, floor, ceiling and walls.", "debug/masks.jpg"),
    # Stage 2 of a staged run (one with a shell.json) lists and segments its large items apart from the shell's.
    (
        "Large Inventory",
        "A vision-language model lists the furniture and appliances, each with what it stands on.",
        "debug/large_inventory.jpg",
    ),
    ("Large Mask", "Each large item's box becomes an outline of it.", "debug/large_masks.jpg"),
    (
        "Small Inventory",
        "A close-up of each support (a worktop, a desk, a shelf) and one look at the floor and walls list what stands on or in them.",
        "debug/small_inventory.jpg",
    ),
    ("Small Mask", "Each small item is segmented in its close-up and carried back to the photo.", "debug/small_masks.jpg"),
    (
        "Depth",
        "A depth model gives every pixel a distance (near is bright). Finding the floor in it fixes the camera's height and tilt.",
        "debug/depth.jpg",
    ),
    # Each picture of the layout is a list of files, the first that exists shown: gate 4's box layout of every item
    # once it has run, else the shell's (built-ins only on a staged run).
    (
        "Layout",
        "Each object becomes a box in the room, walls and floor fitted from the photo. Left: the boxes' edges over the photo, "
        "built-ins green, large items blue, small items yellow "
        "(sheets outlined in violet). Right: the boxes rendered from the photo's camera. Built-ins only until gate 4 has run.",
        [["debug/boxes.jpg", "debug/layout.jpg"], ["boxes.png"]],
    ),
]
_IMAGE_TYPES = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png"}
# The largest photo a client may send to start a run, in bytes.
_MAX_UPLOAD = 40_000_000
# The largest zipped scan a client may send, in bytes; a whole-home scan's keyframes come to a few hundred MB.
_MAX_SCAN_UPLOAD = 8_000_000_000
# What a scan's or a result's files are sent as (scan-contract.md); anything else as bytes.
_SCAN_TYPES = {
    **_IMAGE_TYPES, ".json": "application/json", ".usdz": "model/vnd.usdz+zip", ".usda": "text/plain; charset=utf-8",
}


def start_run(photo: bytes, project: Path, name: str = "") -> str:
    """Save `photo` under `project`/images and start `aomma staged` on it (no models); returns the new run folder's name.

    The run's folder is named for the photo, so `name`, cleaned up to letters, digits, `-` and `_`,
    names both; without one the photo is named for the time.

    Raises ValueError if `photo` is not a picture.
    """
    try:
        image = Image.open(io.BytesIO(photo))
        image.verify()
    except (UnidentifiedImageError, OSError) as error:
        raise ValueError("not an image") from error
    images = project / "images"
    images.mkdir(exist_ok=True)
    suffix = "png" if image.format == "PNG" else "jpg"
    stem = re.sub(r"[^A-Za-z0-9_-]+", "-", name).strip("-")[:60] or f"photo-{datetime.now():%Y%m%d-%H%M%S}"
    path = images / f"{stem}.{suffix}"
    # A name already used does not replace that photo.
    if path.exists():
        path = images / f"{stem}-{datetime.now():%H%M%S}.{suffix}"
    path.write_bytes(photo)
    # `aomma staged` makes the run's folder and goes to the background itself; it takes paths relative to the project.
    # The layout-first build steps, not `aomma run`'s single inventory and models.
    done = subprocess.run(
        [sys.executable, "-c", "from aomma.cli import main; main()", "staged", str(path.relative_to(project))],
        cwd=project, capture_output=True, text=True, timeout=60,
    )
    folder = re.search(r"^\s*folder: (.+)$", done.stdout, re.MULTILINE)
    if done.returncode or not folder:
        raise RuntimeError((done.stderr or done.stdout).strip() or "aomma staged did not start")
    return Path(folder.group(1)).name


def save_scan(archive: Path, scans: Path, name: str = "") -> str:
    """Unpack `archive`, a zipped AommACapture scan folder, into `scans`; returns the new scan folder's name.

    The zip may hold the scan folder itself or just its contents. The folder is named `name`,
    cleaned up to letters, digits, `-` and `_`, or else for the folder in the zip; a name
    already used gets the time added rather than replacing that scan.

    Raises ValueError if `archive` is not a zip holding a scan (one with capture.json), or holds one whose
    manifest.json names a contract major version not known here.
    """
    from aomma.scanapi import scan_manifest, unpack

    if not zipfile.is_zipfile(archive):
        raise ValueError("not a zip file")
    scans.mkdir(parents=True, exist_ok=True)
    # Unpacked beside the scans, so the finished folder is moved into place rather than copied.
    with tempfile.TemporaryDirectory(dir=scans, prefix=".upload-") as unpacked:
        unpacked = Path(unpacked)
        folder = unpack(archive, unpacked)
        entries = list(unpacked.iterdir())
        if not (folder / "capture.json").is_file():
            raise ValueError("the zip holds no scan: capture.json is missing")
        # A scan of a contract version this AommA does not know is refused (scan-contract.md, "manifest.json").
        scan_manifest(folder)
        stem = re.sub(r"[^A-Za-z0-9_-]+", "-", name or (folder.name if folder != unpacked else "")).strip("-")[:80]
        target = scans / (stem or f"Scan-{datetime.now():%Y%m%d-%H%M%S}")
        if target.exists():
            target = scans / f"{target.name}-{datetime.now():%H%M%S}"
        if folder == unpacked:
            target.mkdir()
            for entry in entries:
                shutil.move(entry, target / entry.name)
        else:
            folder.rename(target)
    return target.name


def _load(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


# What an experiment's trials measure, as trials.json names them (see lab.py). Older trials lack some.
_TRIAL_MEASURES = (
    "generation_seconds", "outline_fit", "colour_error", "hidden_by_wall", "off_floor_m", "longest_side_m",
    "distance_vs_photo", "size_vs_estimate", "inside_other", "inside_which",
)
# How many of the worker's latest messages are shown while an experiment runs.
_WORKER_MESSAGES = 4
# Left in an experiment's folder once someone has checked it: when, where the notes
# take it up, whether the worker's findings held, what was decided, whether that is
# still to be done ("open"), and each thing done about it since ("acted").
REVIEW = "review.json"


def _experiment_summary(folder: Path) -> dict:
    """What the experiment list and the top of an experiment's page show: its question, run and state."""
    brief = folder / "brief.txt"
    text = brief.read_text() if brief.exists() else ""
    # The brief ends with the question; older briefs and newer ones alike.
    question = text.rpartition("\nQUESTION\n")[2].strip() if "\nQUESTION\n" in text else None
    report = folder / "report.md"
    header = report.read_text().partition("\n")[0] if report.exists() else ""
    run = header.removeprefix("# Experiment on ").strip() if header.startswith("# Experiment on ") else None
    if background.alive(folder / background.PID):
        state = "running"
    else:
        state = "finished" if report.exists() else "stopped"
    # The brief states the budget: "You have N trials" in older ones, one line per kind of trial in newer ones.
    budget = {}
    if match := re.search(r"You have (\d+) trials", text):
        budget["generate"] = int(match.group(1))
    if match := re.search(r"You may generate (\d+) models", text):
        budget["generate"] = int(match.group(1))
    if match := re.search(r"You may run (\d+) pose trials", text):
        budget["pose"] = int(match.group(1))
    return {
        "name": folder.name,
        "question": question,
        "run": run,
        # A new experiment's folder is named for the run's photo; the run itself is known once it reports.
        "photo": folder.name.split("-", 2)[-1],
        "state": state,
        "budget": budget,
        "review": _load(folder / REVIEW),
    }


def job_owner(name: str, runs_dir: Path, lab_dir: Path | None) -> dict | None:
    """What a queue ticket's work is for, from the name it was taken under: a run, or an experiment and which trial.

    A trial works in a folder of its own inside its experiment's, named after
    the experiment with "-t<number>" added, and its jobs carry that name.
    """
    if not name or name in (".", "..") or "/" in name:
        return None  # not the name of a folder: never a run or an experiment
    if lab_dir:
        trial = re.fullmatch(r"(.+)-t(\d+)", name)
        if trial and (lab_dir / trial[1]).is_dir():
            return {"kind": "experiment", "name": trial[1], "trial": int(trial[2])}
        if (lab_dir / name).is_dir():
            return {"kind": "experiment", "name": name}
    if (runs_dir / name).is_dir():
        return {"kind": "run", "name": name}
    return None


def list_experiments(lab_dir: Path) -> list[dict]:
    """Experiments under `lab_dir` that have at least been briefed, newest first.

    Other folders there (a check run by hand, say) have no brief and are left out.
    """
    if not lab_dir.is_dir():
        return []
    return [
        {**_experiment_summary(folder), "trials": len(_load(folder / "trials.json") or [])}
        for folder in sorted(lab_dir.glob("*/"), reverse=True)
        if (folder / "brief.txt").exists()
    ]


def experiment_detail(folder: Path) -> dict:
    """Everything the page shows for one experiment: its question, its trials with their pictures, the worker's findings."""

    def picture(name: str) -> dict | None:
        path = folder / name
        return {"path": f"{folder.name}/{name}", "version": int(path.stat().st_mtime)} if path.is_file() else None

    trials = []
    for trial in _load(folder / "trials.json") or []:
        trials.append({
            "trial": trial.get("trial"),
            # Trials from before pose trials existed were all generations.
            "kind": "pose" if trial.get("kind") == "pose_trial" else "generate",
            "object_id": trial.get("object_id"),
            "generator": trial.get("generator"),
            "settings": trial.get("settings") or {},
            "reason": trial.get("reason"),
            **{key: trial.get(key) for key in _TRIAL_MEASURES},
            "error": (trial.get("error") or "").strip() or None,
            "picture": picture(trial["picture"]) if trial.get("picture") else None,
        })
    shown = {trial["picture"]["path"] for trial in trials if trial["picture"]}
    # Pictures added to the folder by hand, such as a comparison made afterwards.
    extras = [
        picture(path.name) for path in sorted(folder.glob("*.jpg")) if f"{folder.name}/{path.name}" not in shown
    ]
    report = folder / "report.md"
    findings = None
    if report.exists():
        heading = "## The worker's findings"
        text = report.read_text()
        findings = text.partition(heading)[2].strip() if heading in text else None
    messages = []
    log = folder / "log.jsonl"
    if log.exists():
        for line in log.read_text().splitlines():
            entry = json.loads(line)
            if entry.get("kind") == "worker" and entry.get("text"):
                messages.append(entry["text"][:2000])
    return {
        **_experiment_summary(folder),
        "trials": trials,
        "extras": extras,
        "findings": findings,
        "messages": messages[-_WORKER_MESSAGES:],
    }


def newest_experiment(lab_dir: Path) -> Path:
    """The folder of the experiment started last."""
    found = list_experiments(lab_dir)
    if not found:
        raise SystemExit(f"no experiments in {lab_dir}")
    return lab_dir / found[0]["name"]


def _review_state(review: dict | None) -> str:
    if not review:
        return ", not reviewed"
    return ", reviewed, action open" if review["open"] else ", reviewed, acted on"


def _review_lines(review: dict | None) -> list[str]:
    if not review:
        return []
    lines = [
        f"reviewed {review['date']}, in {review['section']}:",
        f"  verdict: {review['verdict']}",
        f"  action: {review['action']}" + (" (still to do)" if review["open"] else ""),
    ]
    return lines + [f"  acted {entry['date']}: {entry['action']}" for entry in review["acted"]]


def _round_summary(folder: Path) -> dict:
    """A round as the review lists see it: its title, whether it has ended, and its review."""
    state = _load(folder / rounds.STATE) or {}
    if state.get("finished"):
        ended = state.get("outcome") or "finished"
    elif background.alive(folder / background.PID):
        ended = "running"
    else:
        ended = "stopped"
    code = state.get("code")
    return {"name": folder.name, "title": state.get("title"), "state": ended, "steps": state.get("steps", []),
            "code": code["commit"][:12] if code else "working tree", "review": _load(folder / REVIEW)}


def list_rounds(lab_dir: Path) -> list[dict]:
    """Rounds under `lab_dir`, newest first; like experiments, each is reviewed once it has ended."""
    return [_round_summary(path.parent) for path in sorted(lab_dir.glob(f"*/{rounds.STATE}"), reverse=True)]


def round_status(folder: Path) -> str:
    """A round in a few lines of text: its state, a line for each step, and its review."""
    summary = _round_summary(folder)
    state = summary["state"] + ("" if summary["state"] == "running" else _review_state(summary["review"]))
    lines = [f"{folder}: round {state}", f"title: {summary['title'] or 'not known'}", f"code: {summary['code']}"]
    for step in summary["steps"]:
        phases = ", ".join(item["phase"] for item in step["items"]) or "not started"
        lines.append(f"  {step['id']}: {phases}")
    return "\n".join(lines + _review_lines(summary["review"]))


def experiment_status(folder: Path) -> str:
    """An experiment in a few lines of text: its state, a line for each trial and, once it has reported, its findings.

    Short enough to be read again and again while an experiment runs, where
    `report.md` and the log are not. A round's folder is shown as a round.
    """
    if (folder / rounds.STATE).exists():
        return round_status(folder)
    detail = experiment_detail(folder)
    budget = ", ".join(f"{count} {kind}" for kind, count in detail["budget"].items()) or "not known"
    state = detail["state"]
    if state == "running":
        state += f" (process {background.alive(folder / background.PID)})"
    elif state == "stopped":
        state += f" without a report; see {folder / background.LOG}"
    if detail["state"] != "running":
        state += _review_state(detail["review"])
    lines = [
        f"{folder}: {state}",
        f"question: {detail['question'] or 'not known'}",
        f"budget: {budget}; trials so far: {len(detail['trials'])}",
    ]
    for trial in detail["trials"]:
        changed = ", ".join(f"{name} = {value}" for name, value in trial["settings"].items()) or "no changes"
        start = f"  t{trial['trial']} {trial['kind']} {trial['object_id']} with {trial['generator']}, {changed}: "
        if trial["error"]:
            lines.append(start + f"failed: {trial['error'][:200]}")
        else:
            measured = ", ".join(f"{key} {trial[key]}" for key in _TRIAL_MEASURES if trial[key] is not None)
            lines.append(start + measured + (f" [{folder.parent / trial['picture']['path']}]" if trial["picture"] else ""))
    if detail["findings"]:
        lines += ["findings:", detail["findings"]]
    elif detail["messages"]:
        lines += ["worker's latest message:", detail["messages"][-1][:600]]
    return "\n".join(lines + _review_lines(detail["review"]))


def unreviewed(lab_dir: Path) -> str:
    """One line for each experiment or round that has ended but not been reviewed, oldest first."""
    waiting = [e for e in list_experiments(lab_dir) if e["state"] != "running" and not e["review"]]
    waiting += [r for r in list_rounds(lab_dir) if r["state"] != "running" and not r["review"]]
    lines = [
        (e["name"], f"{lab_dir / e['name']}: {e['state']}, {e['trials']} trials. {(e['question'] or '')[:150]}")
        if "trials" in e else (e["name"], f"{lab_dir / e['name']}: round {e['state']}. {(e['title'] or '')[:150]}")
        for e in waiting
    ]
    # Folders are named from when they began, so their names sort oldest first.
    return "\n".join(line for _, line in sorted(lines)) or "every experiment that has ended has been reviewed"


def open_actions(lab_dir: Path) -> str:
    """One line for each reviewed experiment whose action is still to be done, oldest first."""
    waiting = [e for e in list_experiments(lab_dir) + list_rounds(lab_dir) if e["review"] and e["review"]["open"]]
    return "\n".join(
        f"{lab_dir / e['name']} ({e['review']['section']}): {e['review']['action']}"
        for e in sorted(waiting, key=lambda e: e["name"])
    ) or "every reviewed experiment has been acted on"


def record_review(folder: Path, section: str, verdict: str, action: str, open: bool) -> dict:
    """Mark an experiment reviewed, with where the notes take it up, whether its findings held, and what was decided.

    `open` says the action is still to be done; `record_action` records it when it is.
    """
    if not (folder / "brief.txt").exists() and not (folder / rounds.STATE).exists():
        raise SystemExit(f"{folder} is neither an experiment's folder nor a round's")
    review = {"date": f"{datetime.now():%Y-%m-%d}", "section": section, "verdict": verdict, "action": action,
              "open": open, "acted": []}
    (folder / REVIEW).write_text(json.dumps(review, indent=2) + "\n")
    return review


def record_action(folder: Path, action: str, still_open: bool = False) -> dict:
    """Add to a reviewed experiment what has been done about it, and whether anything is still to do."""
    review = _load(folder / REVIEW)
    if review is None:
        raise SystemExit(f"{folder} has not been reviewed; run `aomma experiment-reviewed` first")
    review["acted"].append({"date": f"{datetime.now():%Y-%m-%d}", "action": action})
    review["open"] = still_open
    (folder / REVIEW).write_text(json.dumps(review, indent=2) + "\n")
    return review


def list_runs(runs_dir: Path) -> list[dict]:
    """Runs that have at least been perceived, newest first."""
    runs = []
    for run_dir in sorted(runs_dir.glob("*/"), reverse=True):
        if (run_dir / "input.jpg").exists():
            inventory = _load(run_dir / "inventory.json") or {"objects": []}
            runs.append({"name": run_dir.name, "objects": len(inventory["objects"])})
    return runs


_export_lock = threading.Lock()


def scene_glb(run_dir: Path) -> Path:
    """The run's USD scene as a GLB for the 3D view, exported again whenever the scene has changed.

    Browsers cannot read USD, so the viewer shows a conversion: the same
    objects in the same places, with textures shrunk to keep the file small.
    """
    scene, glb = run_dir / "scene.usda", run_dir / "debug" / "scene.glb"
    # One export at a time: each is a Blender process, and two tabs may ask at once.
    with _export_lock:
        if not glb.exists() or glb.stat().st_mtime < scene.stat().st_mtime:
            usd_to_glb(scene, glb)
    return glb


def scene_usdz(run_dir: Path) -> Path:
    """The run's USD scene packed with its assets into one USDZ file to download, packed again whenever any of them has changed."""
    scene, usdz = run_dir / "scene.usda", run_dir / "debug" / "scene.usdz"
    newest = max(path.stat().st_mtime for path in [scene, *(run_dir / "assets").rglob("*")])
    with _export_lock:
        if not usdz.exists() or usdz.stat().st_mtime < newest:
            usdz.parent.mkdir(exist_ok=True)
            # Packed under another name first, so a failed pack never leaves a broken file behind.
            partial = usdz.with_suffix(".partial.usdz")
            if not UsdUtils.CreateNewUsdzPackage(Sdf.AssetPath(str(scene)), str(partial)):
                raise RuntimeError(f"could not pack {scene} as USDZ")
            partial.replace(usdz)
    return usdz


def _stage_pictures(run_dir: Path, files: str | list[list[str]]) -> list[dict | None]:
    """A stage's pictures: one file, or for each picture the first of its files the run has (None for one it has none of)."""
    choices = [[files]] if isinstance(files, str) else files
    return [next((_picture(run_dir, f) for f in choice if (run_dir / f).exists()), None) for choice in choices]


def run_detail(run_dir: Path) -> dict:
    """Everything the page shows for one run: stage pictures, numbers, per-object results."""

    def picture(relative: str) -> dict | None:
        path = run_dir / relative
        # The modification time lets the page refetch a picture only when it changes.
        return {"path": f"{run_dir.name}/{relative}", "version": int(path.stat().st_mtime)} if path.exists() else None

    inventory = _load(run_dir / "inventory.json") or {"objects": []}
    masks = _load(run_dir / "masks.json") or {}
    scores = _load(run_dir / "scores.json") or {}
    poses = {pose["id"]: pose for pose in _load(run_dir / "poses.json") or []}
    layout = _load(run_dir / "layout.json")
    usage_path = run_dir / "usage.jsonl"
    usage = [json.loads(line) for line in usage_path.read_text().splitlines()] if usage_path.exists() else []

    objects = []
    for obj in inventory["objects"]:
        oid = obj["id"]
        objects.append(
            {
                "id": oid,
                "description": obj["description"],
                "on": obj["on"],
                "mask": masks.get(oid),
                "box_iou": scores.get(oid),
                "has_model": (run_dir / "assets" / oid / f"{oid}.usdc").exists(),
                "viewable": (run_dir / "assets" / oid / f"{oid}.glb").exists(),
                "repaired": (run_dir / "assets" / oid / "repaired").exists(),
                "flat": (run_dir / "assets" / oid / "flat.json").exists(),
                "repair": picture(f"debug/repair_{oid}.jpg"),
                "pose": poses.get(oid),
                "crop": picture(f"debug/crop_{oid}.jpg"),
            }
        )
    return {
        "name": run_dir.name,
        "source": (_load(run_dir / "run.json") or {}).get("source"),
        "input": picture("input.jpg"),
        "scene": picture("scene.usda"),
        # Every stage is listed, with a picture once it has run.
        "stages": [
            {"title": title, "caption": caption, "pictures": _stage_pictures(run_dir, files)}
            for title, caption, files in _STAGE_IMAGES
            if not str(files).startswith(("debug/large_", "debug/small_")) or (run_dir / "shell.json").exists()
        ],
        "result": picture("debug/compare.jpg"),
        # What the review found and did, in order (see agent.py).
        "review": [
            {**entry, "view": picture(f"debug/review_{entry['id']}.jpg") if "id" in entry else None}
            for entry in (json.loads(line) for line in (run_dir / "agent.jsonl").read_text().splitlines())
        ] if (run_dir / "agent.jsonl").exists() else None,
        "camera": _load(run_dir / "camera.json"),
        "depth_scale": (_load(run_dir / "world.json") or {}).get("depth_scale"),
        # Row 3, column 4 of the camera-to-world matrix is the camera's height.
        "camera_height": layout["camera"]["camera_to_world"][11] if layout else None,
        "usage": usage,
        "objects": objects,
    }


def _picture(run_dir: Path, relative: str) -> dict | None:
    """A run's file as the page asks for it; the modification time lets the page refetch it only when it changes."""
    path = run_dir / relative
    return {"path": f"{run_dir.name}/{relative}", "version": int(path.stat().st_mtime)} if path.exists() else None


def list_labelled(runs_dir: Path) -> list[dict]:
    """Every run with a label map (`aomma labels`), newest first, with its report and, if its shot's pose failed gate 0,
    why it is left out: the Labels tab's table."""
    from aomma.labels import pose_off

    found = []
    for run_dir in sorted(runs_dir.glob("*/"), reverse=True):
        saved = _load(run_dir / "labels.json")
        if not saved:
            continue
        run = _load(run_dir / "run.json") or {}
        found.append({
            "name": run_dir.name,
            # Runs made together share their folder's time stamp: one round, shown together.
            "batch": run_dir.name[:15],
            "scan": Path(run["scan"]).name if run.get("scan") else None,
            "shot": run.get("shot"),
            "condition": run.get("condition"),
            "seed_offset": run.get("seed_offset", 0),
            "report": saved["report"],
            # Gate 0: the shot's pose is off, so its gates' comparisons with the references are not to be trusted. The run
            # itself is used like any other (a photo run is given no pose).
            "pose_off": pose_off(run_dir),
            # Gate 1's verdict, once `aomma judge-shell` has run: True, False or None.
            "shell_ok": ((_load(run_dir / "judge_shell.json") or {}).get("verdict") or {}).get("ok"),
            # The gallery's pictures: the photo, its label map drawn over it, and the shell's, large items' and small items' masks.
            "pictures": {
                "photo": _picture(run_dir, "input.jpg"),
                "labels": _picture(run_dir, "debug/labels.jpg"),
                "shell": _picture(run_dir, "debug/masks.jpg"),
                "large": _picture(run_dir, "debug/large_masks.jpg"),
                "small": _picture(run_dir, "debug/small_masks.jpg"),
            },
        })
    return found


def labels_detail(run_dir: Path) -> dict:
    """One run's label map for the Labels tab: what each value is, the report, the colours and the pictures."""
    from aomma.labels import COLOURS, SETTLED_BY_DEPTH, SETTLED_BY_RANK, SETTLED_GROWN, SETTLED_LAYER, pose_off

    saved = _load(run_dir / "labels.json") or {"labels": [], "report": {}}
    return {
        "name": run_dir.name,
        "run": _load(run_dir / "run.json") or {},
        "labels": saved["labels"],
        "report": saved["report"],
        "pose_off": pose_off(run_dir),
        "colours": COLOURS,
        "settled": {"by_depth": SETTLED_BY_DEPTH, "by_rank": SETTLED_BY_RANK, "grown": SETTLED_GROWN, "layer": SETTLED_LAYER},
        "photo": _picture(run_dir, "input.jpg"),
        "map": _picture(run_dir, "labels.png"),
        "how": _picture(run_dir, "debug/labels_settled.png"),
        # Gate 1, once `aomma judge-shell` has run: the shell's reference and where the run disagrees with it.
        "shell": _load(run_dir / "judge_shell.json"),
        "shell_reference": _picture(run_dir, "debug/shell_reference.png"),
        "shell_against": _picture(run_dir, "debug/shell_against.png"),
    }


def serve(config: Config, job_log: Path, runs_dir: Path, hosts: list[str], port: int, lab_dir: Path | None = None) -> None:
    """Serve the dashboard on each address in `hosts` until interrupted.

    Runs are read from `runs_dir` and experiments, if given, from `lab_dir`.

    The default is this machine only. Add another of its addresses (a VPN
    address, say) to reach the page from other devices on that network; it
    shows the run photos, so do not expose it wider than you mean to.
    """
    servers = config.comfy
    trackers = []
    # Asking the servers what they hold takes dozens of requests, so the answer is kept for a while.
    models = {"at": 0.0, "body": b"", "lock": threading.Lock()}
    for url in servers.all:
        lanes = [lane for lane, urls in (("light", servers.light), ("heavy", servers.heavy)) if url in urls]
        trackers.append(JobTracker(url, " + ".join(lanes), job_log))
        threading.Thread(target=trackers[-1].run_forever, daemon=True).start()
    runs_root = runs_dir.resolve()
    lab_root = lab_dir.resolve() if lab_dir else None
    vlm = config.vlms[config.vlm]

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            path = urllib.parse.unquote(self.path.split("?")[0])
            cache, download = "no-store", None
            if path == "/state.json":
                state = {
                    "now": time.time(),
                    "page_version": _page_version(),
                    # The one queue all jobs wait in, and the stage each active run has reached.
                    "queue": [
                        {**ticket, "owner": job_owner(ticket["run"], runs_root, lab_root)}
                        for ticket in JobQueue(runs_root / "queue.json").snapshot()
                    ],
                    # The VLM's turns wait in the same queue; it has no server of its own to ask.
                    "vlm": {"name": config.vlm, "model": vlm.model, "url": vlm.url},
                    "runs": [batch for batch in live_batches(runs_root) if batch["title"] == "Run"],
                    # Rounds under way in lab/, with every step to come; the Rounds tab has them all.
                    "rounds": [r for r in rounds.shown_rounds(lab_root) if r["running"]] if lab_root else [],
                    "servers": [tracker.snapshot() for tracker in trackers],
                }
                body, content_type = json.dumps(state).encode(), "application/json"
            elif path == "/models.json":
                with models["lock"]:
                    if time.time() - models["at"] > _MODELS_KEPT_S:
                        models["body"], models["at"] = json.dumps(pipeline_models(config)).encode(), time.time()
                    body, content_type = models["body"], "application/json"
            elif path == "/runs.json":
                body, content_type = json.dumps(list_runs(runs_root)).encode(), "application/json"
            elif path == "/labelled.json":
                body, content_type = json.dumps(list_labelled(runs_root)).encode(), "application/json"
            elif path == "/rounds.json":
                body, content_type = json.dumps(rounds.shown_rounds(lab_root) if lab_root else []).encode(), "application/json"
            elif path == "/experiments.json":
                experiments = list_experiments(lab_root) if lab_root else []
                body, content_type = json.dumps(experiments).encode(), "application/json"
            elif path == "/scans" or path.startswith("/scans/"):
                self._scan_get(path)
                return
            elif path == "/":
                body = _PAGE.read_bytes().replace(b"__PAGE_VERSION__", _page_version().encode())
                content_type = "text/html; charset=utf-8"
            else:
                kind, _, rest = path.lstrip("/").partition("/")
                # Experiments and their pictures come from the lab folder; everything else from the runs folder.
                root = lab_root if kind in ("experiment", "labfiles") else runs_root
                target = (root / rest).resolve() if root else None
                # Only ever read inside that folder.
                if target is None or root not in target.parents:
                    self.send_error(404)
                    return
                if kind == "run" and target.is_dir():
                    body, content_type = json.dumps(run_detail(target)).encode(), "application/json"
                elif kind == "labels" and (target / "labels.json").is_file():
                    body, content_type = json.dumps(labels_detail(target)).encode(), "application/json"
                elif kind == "log" and target.is_dir():
                    # Served apart from the run's details: it changes every few seconds and they do not.
                    log = {"active": background.alive(target / background.PID) is not None, "lines": background.log_tail(target / background.LOG)}
                    body, content_type = json.dumps(log).encode(), "application/json"
                elif kind == "model" and (glb := target.parent / "assets" / target.name / f"{target.name}.glb").is_file():
                    # One object's own model, as generated, before it is placed in the scene.
                    body, content_type = glb.read_bytes(), "model/gltf-binary"
                elif kind == "viewer" and (target / "scene.usda").is_file():
                    body, content_type = _VIEWER.read_bytes(), "text/html; charset=utf-8"
                elif kind == "scene" and (target / "scene.usda").is_file():
                    try:
                        body, content_type = scene_glb(target).read_bytes(), "model/gltf-binary"
                    except RuntimeError as error:
                        self.send_error(500, explain=str(error))
                        return
                elif kind == "usdz" and (target / "scene.usda").is_file():
                    try:
                        body, content_type = scene_usdz(target).read_bytes(), "model/vnd.usdz+zip"
                    except RuntimeError as error:
                        self.send_error(500, explain=str(error))
                        return
                    download = f"{target.name}.usdz"
                elif kind == "experiment" and target.is_dir():
                    body, content_type = json.dumps(experiment_detail(target)).encode(), "application/json"
                elif kind in ("files", "labfiles") and target.is_file() and target.suffix.lower() in _IMAGE_TYPES:
                    body, content_type = target.read_bytes(), _IMAGE_TYPES[target.suffix.lower()]
                    cache = "max-age=86400"  # the page adds a version to the URL when a file changes
                else:
                    self.send_error(404)
                    return
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", cache)
            if download:
                self.send_header("Content-Disposition", f'attachment; filename="{download}"')
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self) -> None:
            """`POST /runs[?name=<name>]` with a photo as the body starts a run on it and answers {"run": <folder name>}.

            `POST /scans[?name=<name>]` with a zipped AommACapture scan folder as the body unpacks it into
            `scans/` and answers {"scan": <folder name>}.

            `POST /scans/<scan>/reviews` with a zipped review folder from the headset keeps it under
            `reviews/<scan>/` and answers {"review": <folder name>}.

            `POST /queue/move` with {"ticket": <number>, "by": <places>} moves a waiting job later (positive) or sooner (negative).
            """
            url = urllib.parse.urlsplit(self.path)
            if url.path == "/queue/move":
                # Like the photo upload, a type a browser page cannot send to us without our consent.
                if not self.headers.get("Content-Type", "").startswith("application/json"):
                    self.send_error(415, explain="send JSON with an application/json Content-Type")
                    return
                try:
                    request = json.loads(self.rfile.read(min(int(self.headers.get("Content-Length") or 0), 1024)))
                    moved = JobQueue(runs_root / "queue.json").move(int(request["ticket"]), int(request["by"]))
                except (ValueError, KeyError, TypeError):
                    self.send_error(400)
                    return
                body = json.dumps({"moved": moved}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if url.path == "/scans":
                self._receive_scan(url)
                return
            if (review := re.fullmatch(r"/scans/([^/]+)/reviews", urllib.parse.unquote(url.path))):
                self._receive_review(review.group(1))
                return
            if url.path != "/runs":
                self.send_error(404)
                return
            # A browser page cannot send this type to us without our consent, so another site cannot start runs.
            if not self.headers.get("Content-Type", "").startswith("image/"):
                self.send_error(415, explain="send the photo as the body, with an image/* Content-Type")
                return
            length = int(self.headers.get("Content-Length") or 0)
            if not 0 < length <= _MAX_UPLOAD:
                self.send_error(413 if length else 411)
                return
            try:
                name = urllib.parse.parse_qs(url.query).get("name", [""])[0]
                run = start_run(self.rfile.read(length), runs_root.parent, name)
            except ValueError as error:
                self.send_error(400, explain=str(error))
                return
            except (RuntimeError, subprocess.TimeoutExpired) as error:
                self.send_error(500, explain=str(error))
                return
            body = json.dumps({"run": run}).encode()
            self.send_response(201)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _receive_scan(self, url: urllib.parse.SplitResult) -> None:
            name = urllib.parse.parse_qs(url.query).get("name", [""])[0]
            scans = runs_root.parent / "scans"
            self._receive_zip(_MAX_SCAN_UPLOAD, scans, lambda archive: {"scan": save_scan(archive, scans, name)})

        def _scan_get(self, path: str) -> None:
            """The capture apps' reads (scan-contract.md, "AommA endpoints"): `GET /scans`, a scan's files, its runs
            with a scene, and the result of its newest run with a scene or of a chosen one."""
            from aomma import scanapi

            project = runs_root.parent
            scans = project / "scans"
            parts = path.strip("/").split("/")
            if parts == ["scans"]:
                self._send_json(scanapi.list_scans(scans, runs_root))
                return
            folder = scanapi.scan_folder(scans, parts[1])
            if folder is None:
                self.send_error(404, explain="no such scan")
                return
            if len(parts) > 3 and parts[2] == "files":
                self._send_file_within(folder, "/".join(parts[3:]))
            elif parts[2:] == ["runs"]:
                self._send_json(scanapi.list_runs(folder.name, runs_root))
            elif parts[2:3] == ["result"] or (parts[2:3] == ["runs"] and parts[4:5] == ["result"]):
                # `result` is the newest run's; `runs/<run>/result` is the one the person chose.
                chosen = parts[3] if parts[2] == "runs" else None
                rest = parts[5:] if chosen else parts[3:]
                try:
                    if chosen:
                        result = scanapi.run_result(folder.name, chosen, runs_root, project)
                    else:
                        result = scanapi.result_for(folder.name, runs_root, project)
                except (RuntimeError, ValueError, KeyError, OSError) as error:
                    self.send_error(500, explain=str(error))
                    return
                if result is None:
                    self.send_error(404, explain=f"{chosen} is not a run of this scan with a scene" if chosen
                                    else "no run of this scan has a result yet")
                else:
                    self._send_file_within(result, "/".join(rest) or "result.json")
            else:
                self.send_error(404)

        def _send_json(self, value, status: int = 200) -> None:
            body = json.dumps(value).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _send_file_within(self, root: Path, relative: str) -> None:
            """Send the file at `relative` inside `root`, a piece at a time (a scan's mesh can be large); 404 outside it."""
            target = (root / relative).resolve()
            if root.resolve() not in target.parents or not target.is_file():
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", _SCAN_TYPES.get(target.suffix.lower(), "application/octet-stream"))
            self.send_header("Content-Length", str(target.stat().st_size))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            with target.open("rb") as sent:
                shutil.copyfileobj(sent, self.wfile, 1 << 20)

        def _receive_zip(self, limit: int, into: Path, keep) -> None:
            """Take a zip body (only as application/zip, which a browser page cannot send to us without our consent)
            to a file in `into` as it arrives, then answer `keep(<zip path>)`, a JSON value, with 201."""
            if self.headers.get("Content-Type", "") != "application/zip":
                self.send_error(415, explain="send the zipped folder as the body, with an application/zip Content-Type")
                return
            length = int(self.headers.get("Content-Length") or 0)
            if not 0 < length <= limit:
                self.send_error(413 if length else 411)
                return
            into.mkdir(parents=True, exist_ok=True)
            # Too large to hold in memory, so it goes to disk as it arrives.
            with tempfile.NamedTemporaryFile(dir=into, prefix=".upload-", suffix=".zip") as archive:
                remaining = length
                while remaining:
                    chunk = self.rfile.read(min(remaining, 1 << 20))
                    if not chunk:
                        self.send_error(400, explain="the upload ended early")
                        return
                    archive.write(chunk)
                    remaining -= len(chunk)
                archive.flush()
                try:
                    answer = keep(Path(archive.name))
                except (ValueError, zipfile.BadZipFile) as error:
                    self.send_error(400, explain=str(error))
                    return
            self._send_json(answer, 201)

        def _receive_review(self, scan: str) -> None:
            from aomma.scanapi import save_review, scan_folder

            project = runs_root.parent
            if scan_folder(project / "scans", scan) is None:
                self.send_error(404, explain="no such scan")
                return
            reviews = project / "reviews"
            self._receive_zip(_MAX_UPLOAD, reviews, lambda archive: {"review": save_review(archive, scan, reviews)})

        def log_message(self, *args) -> None:
            pass  # one request a second per open tab would flood the terminal

    listeners = [ThreadingHTTPServer((host, port), Handler) for host in hosts]
    for listener in listeners[1:]:
        threading.Thread(target=listener.serve_forever, daemon=True).start()
    print("aomma dashboard: " + ", ".join(f"http://{host}:{port}" for host in hosts) + "  (Ctrl-C to stop)")
    listeners[0].serve_forever()
