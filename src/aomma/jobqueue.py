"""One queue for every job the harness sends to the model machine, across all its processes.

Several commands can be running at once (two runs, a review), and each used
to send its jobs straight to the servers. They then fought over one GPU: jobs
ran many times slower, and a caller could give up on a job that had not even
started. Now every job (a VLM call, a mask or depth job, a 3D generation)
takes a ticket here and waits its turn.

Jobs wait in two lanes, and one job runs at a time in each: quick jobs (a VLM
call, a mask, a depth map or an image edit, seconds each) and 3D models (many
minutes each). A quick job need not wait for a model to finish, so a new
photo's first stages are not held up for hours behind another run's models;
the cost is that the two share the GPU while both run. Within a lane, tickets
are served in the order they were taken, unless someone has moved one with `move`.

The queue is a file, guarded by a lock, so that any process can join it and
the dashboard can show it. A ticket whose process has died is dropped.
"""

import fcntl
import json
import os
import time
from contextlib import contextmanager
from pathlib import Path

from aomma import timings

DEFAULT_PATH = Path("runs") / "queue.json"
# Kinds of job, and the lane each waits in. One job runs at a time in each lane.
LANE = {"vlm": "quick", "light": "quick", "3d": "heavy"}
# Lanes in the order the queue lists them.
_LANES = ["quick", "heavy"]
# How many finished tickets are kept for the dashboard's history.
_KEEP_FINISHED = 30
_POLL_S = 0.5


class JobQueue:
    def __init__(self, path: Path = DEFAULT_PATH):
        self.path = path

    @contextmanager
    def _locked(self):
        """The queue's contents, read and written back under an exclusive lock."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path.with_suffix(".lock"), "w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            state = json.loads(self.path.read_text()) if self.path.exists() else {"next": 1, "tickets": []}
            _drop_dead(state)
            yield state
            finished = [t for t in state["tickets"] if t["state"] not in ("waiting", "running")]
            for ticket in finished[:-_KEEP_FINISHED]:
                state["tickets"].remove(ticket)
            scratch = self.path.with_suffix(".tmp")
            scratch.write_text(json.dumps(state))
            scratch.replace(self.path)

    def enqueue(self, kind: str, label: str, run: str = "") -> int:
        """Take a place in the queue for a job to be run later with `turn`; returns the ticket's number."""
        assert kind in LANE, kind
        with self._locked() as state:
            number = state["next"]
            state["next"] += 1
            state["tickets"].append(
                {"id": number, "kind": kind, "label": label, "run": run, "pid": os.getpid(), "state": "waiting", "enqueued": time.time()}
            )
        return number

    def cancel(self, ticket: int) -> None:
        """Give up a place that turned out not to be needed. A ticket already started or finished is left alone."""
        with self._locked() as state:
            for entry in state["tickets"]:
                if entry["id"] == ticket and entry["state"] == "waiting":
                    entry.update(state="cancelled", finished=time.time())

    def move(self, ticket: int, by: int) -> bool:
        """Move a waiting ticket `by` places later (positive) or sooner (negative) among those of its lane.

        Says whether the ticket was waiting and so could be moved.
        """
        with self._locked() as state:
            mine = next((t for t in state["tickets"] if t["id"] == ticket and t["state"] == "waiting"), None)
            if mine is None:
                return False
            others = sorted((t for t in state["tickets"] if t["state"] == "waiting" and _same_lane(t, mine) and t is not mine), key=_order)
            keys = [t.get("rank", t["id"]) for t in others]
            at = min(max(sum(1 for t in others if _order(t) < _order(mine)) + by, 0), len(others))
            # Take a rank between the two neighbours it lands between; ranks are fractions, so nothing else is renumbered.
            if not keys:
                return True
            if at == 0:
                mine["rank"] = keys[0] - 1
            elif at == len(keys):
                mine["rank"] = keys[-1] + 1
            else:
                mine["rank"] = (keys[at - 1] + keys[at]) / 2
            return True

    @contextmanager
    def turn(self, kind: str, label: str, run: str = "", ticket: int | None = None):
        """Wait for this job's turn, hold the machine while the block runs, then release it.

        Pass a `ticket` from `enqueue` to use a place taken earlier; without
        one, the job joins the queue now. The block gets a dict; what it puts
        there is kept with the job's timing (a server's own seconds, say).
        """
        if ticket is None:
            ticket = self.enqueue(kind, label, run)
        while not self._start(ticket):
            time.sleep(_POLL_S)
        notes: dict = {}
        try:
            yield notes
        except BaseException:
            self._finish(ticket, "failed", notes)
            raise
        self._finish(ticket, "done", notes)

    def _start(self, ticket: int) -> bool:
        """Mark the ticket running if nothing else in its lane is and it is next in line there; says whether it did."""
        with self._locked() as state:
            mine = next((t for t in state["tickets"] if t["id"] == ticket and t["state"] == "waiting"), None)
            if mine is None:
                raise RuntimeError(f"queue ticket {ticket} is not waiting")
            lane = [t for t in state["tickets"] if _same_lane(t, mine)]
            if any(t["state"] == "running" for t in lane):
                return False
            waiting = [t for t in lane if t["state"] == "waiting"]
            if min(waiting, key=_order) is not mine:
                return False
            mine.update(state="running", started=time.time())
            return True

    def _finish(self, ticket: int, outcome: str, notes: dict | None = None) -> None:
        """Mark the ticket finished, and record how long it waited and ran (next to the queue, in timings.jsonl)."""
        with self._locked() as state:
            entry = next((t for t in state["tickets"] if t["id"] == ticket), None)
            if entry is None:
                return
            entry.update(state=outcome, finished=time.time())
        timings.record(
            {
                "kind": entry["kind"],
                "label": entry["label"],
                "run": entry["run"],
                "outcome": outcome,
                "waited_s": round(entry["started"] - entry["enqueued"], 2),
                "ran_s": round(entry["finished"] - entry["started"], 2),
                **(notes or {}),
            },
            self.path.with_name("timings.jsonl"),
        )

    def snapshot(self) -> list[dict]:
        """Every ticket: running, then waiting in turn (quick lane first), then finished, newest first."""
        with self._locked() as state:
            tickets = list(state["tickets"])
        running = [t for t in tickets if t["state"] == "running"]
        waiting = sorted((t for t in tickets if t["state"] == "waiting"), key=_order)
        finished = sorted((t for t in tickets if t["state"] not in ("waiting", "running")), key=lambda t: -t["finished"])
        return running + waiting + finished


def _order(ticket: dict) -> tuple:
    """Where a ticket stands among the waiting ones: by lane, then by its place in line (a moved ticket's own rank, else its number)."""
    return _LANES.index(LANE[ticket["kind"]]), ticket.get("rank", ticket["id"])


def _same_lane(a: dict, b: dict) -> bool:
    return LANE[a["kind"]] == LANE[b["kind"]]


def _drop_dead(state: dict) -> None:
    """Mark tickets whose process no longer exists as abandoned, so they hold nobody up."""
    for ticket in state["tickets"]:
        if ticket["state"] in ("waiting", "running"):
            try:
                os.kill(ticket["pid"], 0)  # signal 0 only checks that the process exists
            except OSError:
                ticket.update(state="abandoned", finished=time.time())
