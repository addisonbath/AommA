import json
from pathlib import Path

from aomma.batch import BatchProgress, live_batches


def test_live_batches_lists_unfinished_work_of_running_batches(tmp_path):
    run = tmp_path / "run_a"
    run.mkdir()
    progress = BatchProgress(run, ["bed", "lamp"])
    progress.set("bed", "generating")

    (batch,) = live_batches(tmp_path)
    assert batch["run"] == "run_a"
    assert batch["title"] == "Models"
    assert batch["items"] == [
        {"id": "bed", "state": "generating", "phase": "live"}, {"id": "lamp", "state": "waiting", "phase": "waiting"},
    ]

    progress.set("bed", "done")
    progress.set("lamp", "failed")
    assert live_batches(tmp_path) == []  # nothing left to do


def test_live_batches_skips_a_batch_whose_process_died(tmp_path):
    run = tmp_path / "run_b"
    run.mkdir()
    BatchProgress(run, ["bed"])
    stale = json.loads((run / "batch.json").read_text())
    stale["pid"] = 2**22 + 12345  # beyond any real process id here
    (run / "batch.json").write_text(json.dumps(stale))

    assert live_batches(tmp_path) == []


def test_other_kinds_of_work_show_what_is_still_to_come(tmp_path):
    run = tmp_path / "run_c"
    run.mkdir()
    review = BatchProgress(run, ["judge shelf", "judge lamp"], name="review", title="Review")
    review.set("judge shelf", "wrong shape", phase="done")
    review.set("fix shelf", "repair queued", phase="waiting")  # an item added along the way

    (batch,) = live_batches(tmp_path)
    assert batch["title"] == "Review"
    assert [(item["id"], item["phase"]) for item in batch["items"]] == [
        ("judge shelf", "done"), ("judge lamp", "waiting"), ("fix shelf", "waiting"),
    ]
    # A models batch on the same run is listed beside it, and a file from before phases existed still reads.
    (run / "batch.json").write_text(json.dumps({"run": "run_c", "pid": __import__("os").getpid(), "items": [{"id": "bed", "state": "placing"}]}))
    assert [(b["title"], b["items"][0]["phase"]) for b in live_batches(tmp_path)] == [("Models", "live"), ("Review", "done")]


def test_live_batches_skips_a_file_removed_after_the_folder_was_listed(tmp_path, monkeypatch):
    (tmp_path / "run").mkdir()
    BatchProgress(tmp_path / "run", ["a"], name="review", title="Review")
    gone = tmp_path / "run" / "progress_review.json"
    read = Path.read_text
    monkeypatch.setattr(Path, "read_text", lambda self, *a, **k: (_ for _ in ()).throw(FileNotFoundError()) if self == gone else read(self, *a, **k))
    assert live_batches(tmp_path) == []
