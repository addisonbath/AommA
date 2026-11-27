"""Minimal ComfyUI client: upload inputs, run an API-format workflow, fetch outputs.

ComfyUI is the local model server for segmentation, depth and image-to-3D.
A workflow is a dict of {node_id: {"class_type": ..., "inputs": {...}}}, where
an input that comes from another node is written as [node_id, output_index].
"""

import json
import time
from pathlib import Path

import httpx

from aomma.jobqueue import JobQueue

# Every job is submitted under one client id, so `aomma jobs` can subscribe to
# their progress events (ComfyUI sends those only to the submitting client).
CLIENT_ID = "aomma"
# How long to keep retrying a server that cannot be reached, and how often.
_RETRY_FOR_S = 900
_RETRY_PAUSE_S = 5


# How often a waiting job is polled, in seconds, and every how many polls it is asked whether it has started.
_POLL_S = 1.0
_STARTED_CHECK_EVERY = 5


class ComfyError(RuntimeError):
    pass


class ComfyClient:
    def __init__(
        self,
        url: str,
        job_log: Path | None = None,
        timeout_s: float = 3600,
        queue: JobQueue | None = None,
        kind: str = "light",
        run: str = "",
    ):
        # The harness's one job queue, in which every job on this server waits its
        # turn as a `kind` of job ("light" or "3d") for run `run`. None: no waiting.
        self.queue, self.kind, self.run_name = queue, kind, run
        # The longest a job may run once started; time spent queued does not count.
        self.timeout_s = timeout_s
        # Where submitted jobs are recorded with their labels, for the monitor.
        self.job_log = job_log
        self.url = url.rstrip("/")
        # How long the last job `run` waited for spent running on the server, in seconds.
        self.last_seconds: float | None = None
        self._http = httpx.Client(base_url=self.url, timeout=60)

    def _request(self, method: str, path: str, safe_to_repeat: bool = True, **kwargs) -> httpx.Response:
        """Send a request, waiting out network drops instead of failing.

        Jobs take many minutes and the server may be reached over a VPN, so a
        dropped connection is retried for `_RETRY_FOR_S`. A request that is not
        `safe_to_repeat` is retried only if it never reached the server.
        """
        deadline = time.monotonic() + _RETRY_FOR_S
        announced = False
        while True:
            try:
                return self._http.request(method, path, **kwargs)
            except httpx.TransportError as error:
                reached_server = not isinstance(error, (httpx.ConnectError, httpx.ConnectTimeout))
                if time.monotonic() > deadline or (reached_server and not safe_to_repeat):
                    raise
                if not announced:
                    print(f"{self.url}: connection lost ({type(error).__name__}), retrying", flush=True)
                    announced = True
                time.sleep(_RETRY_PAUSE_S)

    def upload_image(self, data: bytes, name: str) -> str:
        """Upload to ComfyUI's input folder; returns the name LoadImage should use."""
        response = self._request(
            "POST", "/upload/image", files={"image": (name, data)}, data={"overwrite": "true"}
        )
        response.raise_for_status()
        info = response.json()
        return f"{info['subfolder']}/{info['name']}" if info.get("subfolder") else info["name"]

    def free_memory_gb(self) -> float:
        """Free memory on the server's machine (on a Mac, also what the GPU can use)."""
        return self._request("GET", "/system_stats").json()["system"]["ram_free"] / 1e9

    def wait_for_memory(self, min_free_gb: float, poll_s: float = 15) -> None:
        """Block until the server has at least `min_free_gb` of memory free."""
        announced = False
        while (free := self.free_memory_gb()) < min_free_gb:
            if not announced:
                print(f"{self.url}: {free:.0f} GB free, waiting for {min_free_gb:.0f} GB", flush=True)
                announced = True
            time.sleep(poll_s)

    def run(self, workflow: dict, label: str = "", ticket: int | None = None) -> dict:
        """Run a workflow and block until it finishes; returns outputs by node id.

        `label` says what the job is for; it is shown by `aomma jobs` and the
        dashboard. `ticket` is a place in the harness's queue taken earlier for
        this job; without one it joins the queue now.
        """
        if self.queue is None:
            return self._run(workflow, label)
        with self.queue.turn(self.kind, label, self.run_name, ticket) as notes:
            self.last_seconds = None
            outputs = self._run(workflow, label)
            if self.last_seconds is not None:
                notes["server_s"] = round(self.last_seconds, 2)
            return outputs

    def _run(self, workflow: dict, label: str) -> dict:
        prompt_id = self._active_job(label)
        if prompt_id:
            # A previous run of ours was interrupted while this job was on the
            # server: wait for that one rather than generating it twice.
            print(f"{self.url}: picking up job already under way for '{label}'", flush=True)
        else:
            response = self._request(
                "POST", "/prompt", safe_to_repeat=False, json={"prompt": workflow, "client_id": CLIENT_ID}
            )
            if response.status_code != 200:
                # Validation failures (bad wiring, missing model file) arrive here.
                raise ComfyError(f"workflow rejected: {response.text}")
            prompt_id = response.json()["prompt_id"]
            if self.job_log:
                self.job_log.parent.mkdir(parents=True, exist_ok=True)
                with self.job_log.open("a") as f:
                    f.write(json.dumps({"id": prompt_id, "label": label}) + "\n")

        # The clock starts when the job starts running, not when it is queued:
        # a server shared with other work may keep it waiting a long time, and
        # giving up on a queued job would leave it there to run for nobody.
        started, checks = None, 0
        while started is None or time.monotonic() - started < self.timeout_s:
            history = self._request("GET", f"/history/{prompt_id}").json()
            if prompt_id in history:
                entry = history[prompt_id]
                if entry["status"]["status_str"] != "success":
                    raise ComfyError(f"workflow failed: {entry['status']['messages'][-1]}")
                self.last_seconds = _seconds(entry)
                return entry["outputs"]
            if started is None and checks % _STARTED_CHECK_EVERY == 0 and self._is_running(prompt_id):
                started = time.monotonic()
            checks += 1
            time.sleep(_POLL_S)
        # A job that runs this long has hung. Stop it, so it does not hold up everything queued behind it.
        self.cancel(prompt_id)
        raise ComfyError(f"workflow {prompt_id} ran for more than {self.timeout_s:.0f}s and was stopped")

    def _is_running(self, prompt_id: str) -> bool:
        running = self._request("GET", "/api/jobs", params={"status": "in_progress"}).json()["jobs"]
        return any(job["id"] == prompt_id for job in running)

    def job_seconds(self, prompt_id: str) -> float | None:
        """How long a job ran, if the server still remembers it and it succeeded."""
        entry = self._request("GET", f"/history/{prompt_id}").json().get(prompt_id)
        return _seconds(entry) if entry and entry["status"]["status_str"] == "success" else None

    def input_limits(self, class_type: str) -> dict[str, dict]:
        """For each numeric input of a node type, what the server accepts: {"min": .., "max": .., "step": ..}."""
        inputs = self._request("GET", f"/object_info/{class_type}").json()[class_type]["input"]
        specs = {**inputs.get("required", {}), **inputs.get("optional", {})}
        return {name: spec[1] if len(spec) > 1 else {} for name, spec in specs.items() if spec[0] in ("INT", "FLOAT")}

    def cancel(self, prompt_id: str) -> None:
        """Take a job off the server, whether it is running or still queued."""
        self._request("POST", "/queue", json={"delete": [prompt_id]})
        if self._is_running(prompt_id):
            self._request("POST", "/interrupt", json={"prompt_id": prompt_id})

    def cancel_labelled(self, fragment: str) -> int:
        """Take every running or queued job of ours whose label contains `fragment` off the server; returns how many."""
        labels = read_job_labels(self.job_log) if self.job_log else {}
        found = [
            job["id"]
            for status in ("in_progress", "pending")
            for job in self._request("GET", "/api/jobs", params={"status": status}).json()["jobs"]
            if fragment in labels.get(job["id"], "")
        ]
        for prompt_id in found:
            self.cancel(prompt_id)
        return len(found)

    def _active_job(self, label: str) -> str | None:
        """Id of a running or queued job on this server that we submitted with `label`."""
        if not label or not self.job_log:
            return None
        ours = {job_id for job_id, known in read_job_labels(self.job_log).items() if known == label}
        for status in ("in_progress", "pending"):
            for job in self._request("GET", "/api/jobs", params={"status": status}).json()["jobs"]:
                if job["id"] in ours:
                    return job["id"]
        return None

    def fetch(self, file: dict) -> bytes:
        """Download one output file, as described by an entry in `run`'s result."""
        response = self._request(
            "GET",
            "/view",
            params={
                "filename": file["filename"],
                "subfolder": file.get("subfolder", ""),
                "type": file.get("type", "output"),
            },
        )
        response.raise_for_status()
        return response.content


def _seconds(entry: dict) -> float | None:
    """How long a finished job ran on the server, from its history entry."""
    stamps = {name: detail.get("timestamp") for name, detail in entry["status"].get("messages", [])}
    if stamps.get("execution_start") and stamps.get("execution_success"):
        return (stamps["execution_success"] - stamps["execution_start"]) / 1000
    return None


def read_job_labels(job_log: Path) -> dict[str, str]:
    """Map ComfyUI job id to the label it was submitted with."""
    if not job_log.exists():
        return {}
    entries = (json.loads(line) for line in job_log.read_text().splitlines())
    return {entry["id"]: entry["label"] for entry in entries}
