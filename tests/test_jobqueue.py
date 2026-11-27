import threading
import time

from aomma import jobqueue, timings
from aomma.jobqueue import JobQueue


def test_one_job_runs_at_a_time_in_each_lane(tmp_path, monkeypatch):
    monkeypatch.setattr(jobqueue, "_POLL_S", 0.005)
    queue = JobQueue(tmp_path / "queue.json")
    order, running, most = [], {"quick": 0, "heavy": 0}, {"quick": 0, "heavy": 0}
    lock = threading.Lock()

    def job(kind: str, label: str, ticket: int | None = None, seconds: float = 0.02) -> None:
        lane = jobqueue.LANE[kind]
        with queue.turn(kind, label, ticket=ticket):
            with lock:
                running[lane] += 1
                most[lane] = max(most[lane], running[lane])
                order.append(label)
            time.sleep(seconds)
            with lock:
                running[lane] -= 1

    # A long job holds the heavy lane while two more long jobs and then two quick ones join the queue.
    first = threading.Thread(target=job, args=("3d", "model A", None, 0.2))
    first.start()
    while not order:
        time.sleep(0.002)
    reserved = queue.enqueue("3d", "model B")
    later = [
        threading.Thread(target=job, args=("3d", "model B", reserved)),
        threading.Thread(target=job, args=("3d", "model C")),
        threading.Thread(target=job, args=("vlm", "judge shelf")),
        threading.Thread(target=job, args=("light", "mask shelf")),
    ]
    for thread in later:
        thread.start()
        time.sleep(0.004)
    # The quick jobs run, one after the other, while model A is still going.
    while len(order) < 3:
        time.sleep(0.002)
    assert first.is_alive()
    for thread in [first, *later]:
        thread.join()

    assert most == {"quick": 1, "heavy": 1}
    assert order == ["model A", "judge shelf", "mask shelf", "model B", "model C"]
    assert [t["state"] for t in queue.snapshot()] == ["done"] * 5


def test_the_queue_lists_what_is_to_come_and_drops_the_dead(tmp_path):
    queue = JobQueue(tmp_path / "queue.json")
    a, b, c = queue.enqueue("3d", "model bed", "run_1"), queue.enqueue("3d", "model lamp", "run_1"), queue.enqueue("vlm", "inventory", "run_2")
    queue.cancel(b)
    listed = queue.snapshot()
    # Waiting in the order they will be served (quick first), then what has finished.
    assert [(t["label"], t["state"]) for t in listed] == [("inventory", "waiting"), ("model bed", "waiting"), ("model lamp", "cancelled")]

    # A ticket whose process is gone stops counting.
    import json
    state = json.loads(queue.path.read_text())
    state["tickets"][0]["pid"] = 2**22 + 4321
    queue.path.write_text(json.dumps(state))
    assert [(t["label"], t["state"]) for t in queue.snapshot() if t["id"] == a] == [("model bed", "abandoned")]

    # A failure inside a turn releases the machine and is recorded.
    try:
        with queue.turn("vlm", "inventory", ticket=c):
            raise ValueError("model unreachable")
    except ValueError:
        pass
    assert next(t["state"] for t in queue.snapshot() if t["id"] == c) == "failed"


def test_a_waiting_job_can_be_moved_within_its_lane_only(tmp_path):
    queue = JobQueue(tmp_path / "queue.json")
    a, b, c = (queue.enqueue("3d", name) for name in "abc")
    quick = queue.enqueue("vlm", "quick")
    mask = queue.enqueue("light", "mask")
    waiting = lambda: [ticket["label"] for ticket in queue.snapshot()]

    assert queue.move(c, -1) and waiting() == ["quick", "mask", "a", "c", "b"]
    assert queue.move(c, -5) and waiting() == ["quick", "mask", "c", "a", "b"]
    assert queue.move(c, 9) and waiting() == ["quick", "mask", "a", "b", "c"]
    # VLM calls and masks share the quick lane, so either can pass the other.
    assert queue.move(mask, -1) and waiting()[:2] == ["mask", "quick"]
    # A quick job cannot be moved into the 3D models' lane.
    queue.move(quick, 5)
    assert waiting()[:2] == ["mask", "quick"]
    queue.cancel(a)
    assert not queue.move(a, 1)


def test_finished_jobs_are_timed(tmp_path):
    queue = JobQueue(tmp_path / "queue.json")
    with queue.turn("3d", "3D model desk", "office") as notes:
        notes["server_s"] = 12.5
    try:
        with queue.turn("light", "mask desk", "office"):
            raise ValueError
    except ValueError:
        pass
    done, failed = timings.read(tmp_path / "timings.jsonl")
    assert (done["kind"], done["run"], done["outcome"], done["server_s"]) == ("3d", "office", "done", 12.5)
    assert done["waited_s"] >= 0 and done["ran_s"] >= 0
    assert (failed["label"], failed["outcome"]) == ("mask desk", "failed")


def test_a_file_belongs_to_the_run_folder_it_is_in(tmp_path):
    assert timings.run_of(tmp_path / "runs" / "20261119-office" / "debug" / "depth.npy") == "20261119-office"
    assert timings.run_of(tmp_path / "lab" / "scene.usda") == ""
