import json

import httpx

from aomma.backends import comfy
from aomma.backends.comfy import ComfyClient


def _client(handler, tmp_path) -> ComfyClient:
    client = ComfyClient("http://comfy.test", tmp_path / "jobs.jsonl")
    client._http = httpx.Client(base_url="http://comfy.test", transport=httpx.MockTransport(handler))
    return client


def test_requests_wait_out_a_dropped_connection(tmp_path, monkeypatch):
    monkeypatch.setattr(comfy, "_RETRY_PAUSE_S", 0)
    attempts = []

    def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(request.url.path)
        if len(attempts) < 3:
            raise httpx.ConnectError("network is down")
        return httpx.Response(200, json={"system": {"ram_free": 40e9}})

    assert _client(handler, tmp_path).free_memory_gb() == 40.0
    assert len(attempts) == 3


def test_run_picks_up_a_job_already_under_way_instead_of_resubmitting(tmp_path):
    (tmp_path / "jobs.jsonl").write_text(json.dumps({"id": "abc", "label": "3D model bed"}) + "\n")
    posted = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            posted.append(request.url.path)
            return httpx.Response(200, json={"prompt_id": "new"})
        if request.url.path == "/api/jobs":
            running = request.url.params["status"] == "in_progress"
            return httpx.Response(200, json={"jobs": [{"id": "abc"}] if running else []})
        return httpx.Response(200, json={"abc": {"status": {"status_str": "success"}, "outputs": {"n": 1}}})

    assert _client(handler, tmp_path).run({}, label="3D model bed") == {"n": 1}
    assert posted == []


def test_a_queued_job_is_waited_for_and_a_hung_one_is_stopped(tmp_path, monkeypatch):
    monkeypatch.setattr(comfy, "_POLL_S", 0)
    monkeypatch.setattr(comfy, "_STARTED_CHECK_EVERY", 1)
    seen = {"polls": 0, "cancelled": []}

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if request.method == "POST" and path == "/prompt":
            return httpx.Response(200, json={"prompt_id": "job"})
        if request.method == "POST":
            seen["cancelled"].append((path, json.loads(request.content)))
            return httpx.Response(200, json={})
        if path == "/api/jobs":
            # Queued for the first 50 polls, far longer than the time limit; running after that, for ever.
            started = seen["polls"] > 50
            running = request.url.params["status"] == "in_progress"
            return httpx.Response(200, json={"jobs": [{"id": "job"}] if started == running else []})
        seen["polls"] += 1
        return httpx.Response(200, json={})  # never finishes

    client = _client(handler, tmp_path)
    client.timeout_s = 0.05
    try:
        client.run({"node": {}}, label="3D model shelf")
        raise AssertionError("a hung job should raise")
    except comfy.ComfyError as error:
        assert "was stopped" in str(error)
    # It outlasted its queue wait, and was then taken off the server: out of the queue and interrupted.
    assert seen["polls"] > 50
    assert seen["cancelled"] == [("/queue", {"delete": ["job"]}), ("/interrupt", {"prompt_id": "job"})]
