"""How long each GPU job and Blender job took, kept in one file across all runs and processes.

One line per job: what it was, which run it was for, how long it waited for
its turn, how long it ran and how it ended. The queue writes a line for each
job it lets run (VLM calls, masks, depth maps, 3D models), and the Blender
wrapper one for each render or conversion.
"""

import json
import time
from pathlib import Path

DEFAULT_PATH = Path("runs") / "timings.jsonl"


def record(entry: dict, path: Path = DEFAULT_PATH) -> None:
    """Append one job's timing, stamped with when it finished. A line this short is written whole even with other writers."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps({"finished": round(time.time(), 1), **entry}) + "\n")


def read(path: Path = DEFAULT_PATH) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def run_of(path: Path) -> str:
    """The run a file belongs to: the folder just under `runs/` on its path, else ""."""
    parts = path.resolve().parts
    return parts[parts.index("runs") + 1] if "runs" in parts[:-1] else ""
