"""Live view of ComfyUI jobs: what is queued, which node is running, step progress."""

import json
import time
from datetime import datetime
from pathlib import Path

import httpx
from websockets.exceptions import ConnectionClosed
from websockets.sync.client import connect

from aomma.backends.comfy import CLIENT_ID, read_job_labels


def _elapsed(seconds: float) -> str:
    minutes, secs = divmod(int(seconds), 60)
    return f"{minutes}m{secs:02d}s" if minutes else f"{secs}s"


def _active_jobs(http: httpx.Client) -> list[dict]:
    """Running jobs first, then pending ones in queue order."""
    jobs = []
    for status in ("in_progress", "pending"):
        jobs += http.get("/api/jobs", params={"status": status}).json()["jobs"]
    return jobs


def _describe(job: dict, labels: dict[str, str]) -> str:
    started = job.get("execution_start_time") or job["create_time"]
    state = "running" if job["status"] == "in_progress" else "queued"
    label = labels.get(job["id"], "(unlabelled)")
    return f"{job['id'][:8]}  {state:8s} {_elapsed(time.time() - started / 1000):>7s}  {label}"


def snapshot(urls: list[str], job_log: Path) -> None:
    """Print the running and queued jobs on every server once."""
    labels = read_job_labels(job_log)
    for url in urls:
        with httpx.Client(base_url=url.rstrip("/"), timeout=30) as http:
            jobs = _active_jobs(http)
            free = http.get("/system_stats").json()["system"]["ram_free"] / 1e9
        print(f"{url}  ({free:.0f} GB free)")
        print("\n".join("  " + _describe(job, labels) for job in jobs) or "  no jobs running or queued")


def watch(url: str, job_log: Path) -> None:
    """Follow jobs until interrupted, printing a line per node and live step counts.

    Node and step detail is only available for jobs submitted by aomma, because
    ComfyUI sends those events to the submitting client alone.
    """
    url = url.rstrip("/")
    http = httpx.Client(base_url=url, timeout=30)
    ws_url = url.replace("http", "ws", 1) + f"/ws?clientId={CLIENT_ID}"
    known: set[str] = set()
    started: dict[str, float] = {}
    last_poll = 0.0

    def say(text: str) -> None:
        # Clear any in-place progress line before printing a permanent one.
        print(f"\r\033[K{datetime.now():%H:%M:%S}  {text}", flush=True)

    with connect(ws_url, max_size=None) as ws:
        say(f"watching {url} (Ctrl-C to stop)")
        while True:
            if time.monotonic() - last_poll > 3:
                last_poll = time.monotonic()
                labels = read_job_labels(job_log)
                jobs = _active_jobs(http)
                for job in jobs:
                    if job["id"] not in known:
                        say(_describe(job, labels))
                if known and not jobs:
                    say("idle")
                known = {job["id"] for job in jobs}

            try:
                message = ws.recv(timeout=1)
            except TimeoutError:
                continue
            except ConnectionClosed:
                say("connection to ComfyUI closed")
                return
            if isinstance(message, bytes):
                continue  # preview images
            event = json.loads(message)
            kind, data = event["type"], event["data"]
            job = data.get("prompt_id", "")[:8]

            if kind == "execution_start":
                started[job] = time.monotonic()
            elif kind == "execution_cached" and data["nodes"]:
                say(f"{job}  reusing {len(data['nodes'])} cached nodes")
            elif kind == "executing" and data["node"]:
                say(f"{job}  {data['node']}")
            elif kind == "progress":
                took = _elapsed(time.monotonic() - started.get(job, time.monotonic()))
                print(f"\r\033[K          {job}  {data['node']}  step {data['value']}/{data['max']}  ({took})", end="", flush=True)
            elif kind == "execution_success":
                say(f"{job}  finished in {_elapsed(time.monotonic() - started.pop(job, time.monotonic()))}")
            elif kind == "execution_error":
                say(f"{job}  FAILED at {data.get('node_id')}: {data.get('exception_message', '').strip()}")
            elif kind == "execution_interrupted":
                say(f"{job}  interrupted")
