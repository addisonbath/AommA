"""Progress of a long piece of work on a run, saved so the dashboard can show what is still to come.

ComfyUI's own queue only ever holds the job in flight, because the harness
submits one object per server at a time, and VLM calls never appear there at
all. The work still waiting (objects to model, objects to judge, stages of a
run not yet begun) lives here.
"""

import json
import os
import threading
from pathlib import Path

# An item is waiting to start, being worked on (live), or finished one way or the other.
_PHASES = ("waiting", "live", "done", "failed")
_UNFINISHED = {"waiting", "live"}
# Phases of the states a models batch has always used.
_KNOWN = {"waiting": "waiting", "done": "done", "failed": "failed"}


class BatchProgress:
    """Tracks each item's state through a piece of work and mirrors it to a file in the run.

    `name` tells kinds of work on one run apart and `title` is what the
    dashboard calls it. A models batch keeps the file name `batch.json`.
    """

    def __init__(self, run_dir: Path, items: list[str], name: str = "models", title: str = "Models"):
        self._path = run_dir / ("batch.json" if name == "models" else f"progress_{name}.json")
        self._run, self._title = run_dir.name, title
        self._states = {item: ("waiting", "waiting") for item in items}
        self._lock = threading.Lock()
        self._save()

    def set(self, item: str, state: str, phase: str | None = None) -> None:
        """Record what an item is doing now. `state` is shown as it is.

        `phase` says whether that is waiting, live, done or failed; without it,
        "waiting", "done" and "failed" mean themselves and anything else is live.
        An item not seen before is added at the end.
        """
        phase = phase or _KNOWN.get(state, "live")
        assert phase in _PHASES, phase
        with self._lock:
            self._states[item] = (state, phase)
            self._save()

    def _save(self) -> None:
        data = {
            "run": self._run,
            "title": self._title,
            # Lets a reader tell work that is still going from work that was killed.
            "pid": os.getpid(),
            "items": [{"id": item, "state": state, "phase": phase} for item, (state, phase) in self._states.items()],
        }
        # Write then rename, so a reader never sees a half-written file.
        scratch = self._path.with_suffix(".tmp")
        scratch.write_text(json.dumps(data, indent=2))
        scratch.replace(self._path)


def live_batches(runs_dir: Path) -> list[dict]:
    """Work under `runs_dir` that still has something to do and whose process is alive."""
    batches = []
    for path in sorted([*runs_dir.glob("*/batch.json"), *runs_dir.glob("*/progress_*.json")]):
        try:
            batch = json.loads(path.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            continue  # its work ended and took the file away after the folder was listed
        for item in batch["items"]:
            # Files written before phases existed carry only the state.
            item.setdefault("phase", _KNOWN.get(item["state"], "live"))
        batch.setdefault("title", "Models")
        if not any(item["phase"] in _UNFINISHED for item in batch["items"]):
            continue
        try:
            os.kill(batch["pid"], 0)  # signal 0 only checks that the process exists
        except OSError:
            continue
        batches.append(batch)
    return batches
