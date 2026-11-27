import json
import os

from aomma import background
import pytest

from aomma.dashboard import (
    experiment_detail, experiment_status, job_owner, list_experiments, newest_experiment, open_actions, record_action,
    record_review, unreviewed,
)


def _write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def test_experiments_are_listed_and_shown_whatever_they_hold_so_far(tmp_path):
    lab = tmp_path / "lab"
    # A finished experiment from before pose trials: its trials carry no "kind" and fewer measurements.
    old = lab / "20261113-213931-office"
    _write(old / "brief.txt", "RULES\n- You have 2 trials. Each one takes minutes.\n\nQUESTION\nCan the desk be made faster?\n")
    _write(old / "trials.json", json.dumps([
        {"trial": 1, "object_id": "desk", "generator": "pixal3d_image_to_glb", "settings": {"upsample_stage.target_resolution": 1024},
         "reason": "fewer voxels", "generation_seconds": 885, "outline_fit": 0.6, "colour_error": 0.06, "picture": "t1_desk.jpg"},
        {"trial": 2, "object_id": "lamp", "generator": "pixal3d_image_to_glb", "settings": {}, "reason": "r", "error": "\n\tFailed to open layer\n"},
    ]))
    _write(old / "t1_desk.jpg", "jpeg")
    _write(old / "desk_1536_vs_1024.jpg", "jpeg")
    _write(old / "report.md", "# Experiment on 20261113-095725-office\n\n## Trials\n\n## The worker's findings\n\nYes.\n<b>not html</b>\n")
    # One just started: a brief, a worker message, no trials yet, and its process still alive.
    new = lab / "20261114-090000-bedroom"
    _write(new / "brief.txt", "- You may generate 1 models with run_trial.\n- You may run 8 pose trials with pose_trial.\n\nQUESTION\nWhich pose?\n")
    _write(new / "log.jsonl", "\n".join(json.dumps(e) for e in [
        {"kind": "worker", "text": "My plan: baselines first."}, {"kind": "result", "tool": "pose_trial", "result": {}},
        {"kind": "worker", "text": "", "calls": []},
    ]) + "\n")
    _write(new / background.PID, str(os.getpid()))
    # A check run by hand, with no brief: not an experiment.
    _write(lab / "20261113-213344-hunyuan-check" / "t1_lamp.jpg", "jpeg")

    listed = list_experiments(lab)
    assert [e["name"] for e in listed] == ["20261114-090000-bedroom", "20261113-213931-office"]
    assert listed[0] | {} == {
        "name": "20261114-090000-bedroom", "question": "Which pose?", "run": None, "photo": "bedroom",
        "state": "running", "budget": {"generate": 1, "pose": 8}, "review": None, "trials": 0,
    }
    assert listed[1]["state"] == "finished" and listed[1]["run"] == "20261113-095725-office" and listed[1]["budget"] == {"generate": 2}
    assert list_experiments(tmp_path / "no-lab") == []

    detail = experiment_detail(old)
    first, second = detail["trials"]
    assert first["kind"] == "generate" and first["generation_seconds"] == 885 and first["hidden_by_wall"] is None
    assert first["picture"]["path"] == "20261113-213931-office/t1_desk.jpg"
    assert second["error"] == "Failed to open layer" and second["picture"] is None and second["outline_fit"] is None
    # Pictures not tied to a trial are shown as extras; the findings stay plain text.
    assert [p["path"] for p in detail["extras"]] == ["20261113-213931-office/desk_1536_vs_1024.jpg"]
    assert detail["findings"] == "Yes.\n<b>not html</b>"

    running = experiment_detail(new)
    assert running["trials"] == [] and running["findings"] is None and running["messages"] == ["My plan: baselines first."]
    # Its process gone and no report written: it stopped part-way.
    _write(new / background.PID, "999999")
    assert experiment_detail(new)["state"] == "stopped"


def test_an_experiment_is_told_in_a_few_lines_and_the_newest_is_found(tmp_path):
    lab = tmp_path / "lab"
    old = lab / "20261113-213931-office"
    _write(old / "brief.txt", "- You may generate 2 models with run_trial.\n\nQUESTION\nCan the desk be made faster?\n")
    _write(old / "trials.json", json.dumps([
        {"trial": 1, "kind": "run_trial", "object_id": "desk", "generator": "pixal3d", "settings": {"upsample_stage.target_resolution": 1024},
         "reason": "r", "generation_seconds": 885, "outline_fit": 0.6, "picture": "t1_desk.jpg"},
        {"trial": 2, "kind": "run_trial", "object_id": "lamp", "generator": "pixal3d", "settings": {}, "reason": "r", "error": "workflow failed"},
    ]))
    _write(old / "t1_desk.jpg", "jpeg")
    _write(old / "report.md", "# Experiment on 20261113-095725-office\n\n## The worker's findings\n\nYes, 3.5 times faster.\n")
    new = lab / "20261114-090000-bedroom"
    _write(new / "brief.txt", "- You may run 8 pose trials with pose_trial.\n\nQUESTION\nWhich pose?\n")
    _write(new / "log.jsonl", json.dumps({"kind": "worker", "text": "My plan: baselines first."}) + "\n")
    _write(new / background.PID, str(os.getpid()))

    assert newest_experiment(lab) == new
    assert experiment_status(old).splitlines() == [
        f"{old}: finished, not reviewed",
        "question: Can the desk be made faster?",
        "budget: 2 generate; trials so far: 2",
        f"  t1 generate desk with pixal3d, upsample_stage.target_resolution = 1024: generation_seconds 885, outline_fit 0.6 [{old / 't1_desk.jpg'}]",
        "  t2 generate lamp with pixal3d, no changes: failed: workflow failed",
        "findings:",
        "Yes, 3.5 times faster.",
    ]
    running = experiment_status(new)
    assert running.startswith(f"{new}: running (process {os.getpid()})")
    assert running.endswith("worker's latest message:\nMy plan: baselines first.")
    _write(new / background.PID, "999999")
    assert f"stopped without a report; see {new / background.LOG}, not reviewed" in experiment_status(new)

    # Ended experiments wait for review, oldest first, until one is recorded.
    assert unreviewed(lab).splitlines() == [
        f"{old}: finished, 2 trials. Can the desk be made faster?",
        f"{new}: stopped, 0 trials. Which pose?",
    ]
    record_review(old, "2. Can the desk be generated faster?", "Speed holds; quality was not really compared.",
                  "Follow-up on slim objects.", open=True)
    assert unreviewed(lab).splitlines() == [f"{new}: stopped, 0 trials. Which pose?"]
    assert experiment_status(old).splitlines()[0] == f"{old}: finished, reviewed, action open"
    assert experiment_status(old).endswith(
        "in 2. Can the desk be generated faster?:\n  verdict: Speed holds; quality was not really compared.\n"
        "  action: Follow-up on slim objects. (still to do)"
    )
    assert experiment_detail(old)["review"]["action"] == "Follow-up on slim objects."
    with pytest.raises(SystemExit):
        record_review(lab, "s", "v", "a", open=False)  # not an experiment

    # A decided action stays listed until what was done about it is recorded.
    assert open_actions(lab).splitlines() == [f"{old} (2. Can the desk be generated faster?): Follow-up on slim objects."]
    record_action(old, "Started the follow-up.", still_open=True)
    assert open_actions(lab).splitlines() == [f"{old} (2. Can the desk be generated faster?): Follow-up on slim objects."]
    record_action(old, "1024 made the default.")
    assert open_actions(lab) == "every reviewed experiment has been acted on"
    assert experiment_status(old).splitlines()[0] == f"{old}: finished, reviewed, acted on"
    today = experiment_detail(old)["review"]["date"]
    assert experiment_status(old).endswith(f"  acted {today}: Started the follow-up.\n  acted {today}: 1024 made the default.")
    with pytest.raises(SystemExit):
        record_action(new, "a")  # not reviewed


def test_a_queued_job_is_put_down_to_its_run_or_its_experiment_and_trial(tmp_path):
    runs, lab = tmp_path / "runs", tmp_path / "lab"
    (runs / "20261114-064019-bookshelf").mkdir(parents=True)
    (lab / "20261114-081407-office" / "20261114-081407-office-t3").mkdir(parents=True)

    assert job_owner("20261114-064019-bookshelf", runs, lab) == {"kind": "run", "name": "20261114-064019-bookshelf"}
    # The worker's own turns carry the experiment's name; a trial's jobs carry its folder's.
    assert job_owner("20261114-081407-office", runs, lab) == {"kind": "experiment", "name": "20261114-081407-office"}
    assert job_owner("20261114-081407-office-t3", runs, lab) == {"kind": "experiment", "name": "20261114-081407-office", "trial": 3}
    # A name that is neither (a one-off check, or a folder since removed) has no owner.
    assert job_owner("lab-ping", runs, lab) is None and job_owner("", runs, lab) is None
    assert job_owner("20261114-081407-office", runs, None) is None


def test_a_photo_that_is_not_a_picture_is_refused_before_anything_is_saved(tmp_path):
    import pytest

    from aomma.dashboard import start_run

    with pytest.raises(ValueError):
        start_run(b"not a picture", tmp_path)
    assert not (tmp_path / "images").exists()


def test_a_name_given_for_a_photo_is_cleaned_up_for_its_file(tmp_path, monkeypatch):
    import io
    import subprocess

    from PIL import Image

    from aomma import dashboard

    seen = []
    monkeypatch.setattr(
        subprocess, "run",
        lambda arguments, **_: seen.append(arguments[-1]) or commands.append(arguments[-2]) or subprocess.CompletedProcess(arguments, 0, "  folder: runs/x-kitchen\n", ""),
    )
    commands = []
    buffer = io.BytesIO()
    Image.new("RGB", (4, 4)).save(buffer, format="JPEG")
    assert dashboard.start_run(buffer.getvalue(), tmp_path, "My kitchen/../!") == "x-kitchen"
    assert seen == ["images/My-kitchen.jpg"]
    # The layout-first build steps, not `aomma run`'s single inventory and models.
    assert commands == ["staged"]
    dashboard.start_run(buffer.getvalue(), tmp_path, "My kitchen")
    assert seen[1] != seen[0]


def test_a_zipped_scan_is_unpacked_under_its_name_and_never_replaces_one(tmp_path):
    import zipfile

    import pytest

    from aomma.dashboard import save_scan

    archive = tmp_path / "scan.zip"
    with zipfile.ZipFile(archive, "w") as zipped:
        zipped.writestr("Scan-1795028251/capture.json", "{}")
        zipped.writestr("Scan-1795028251/shots/shots.json", "[]")
        zipped.writestr("Scan-1795028251/../../outside.txt", "")
        zipped.writestr("Scan-1795028251/._capture.json", "")
        zipped.writestr("__MACOSX/Scan-1795028251/._rooms.json", "")
    scans = tmp_path / "scans"
    assert save_scan(archive, scans) == "Scan-1795028251"
    assert (scans / "Scan-1795028251" / "shots" / "shots.json").is_file()
    assert not (scans / "Scan-1795028251" / "._capture.json").exists()
    assert not (tmp_path / "outside.txt").exists()
    assert save_scan(archive, scans, "Scan-Office/../1795028251") == "Scan-Office-1795028251"
    assert save_scan(archive, scans, "Scan-Office-1795028251").startswith("Scan-Office-1795028251-")
    # Only the scans remain: the unpacking folders are gone.
    assert sorted(path.name for path in scans.iterdir())[0] == "Scan-1795028251"
    assert len(list(scans.iterdir())) == 3

    flat = tmp_path / "flat.zip"
    with zipfile.ZipFile(flat, "w") as zipped:
        zipped.writestr("capture.json", "{}")
        zipped.writestr("rooms.json", "[]")
    assert (scans / save_scan(flat, scans, "Scan-Flat-1") / "rooms.json").is_file()

    empty = tmp_path / "empty.zip"
    with zipfile.ZipFile(empty, "w") as zipped:
        zipped.writestr("notes.txt", "")
    with pytest.raises(ValueError):
        save_scan(empty, scans)
    with pytest.raises(ValueError):
        save_scan(tmp_path / "scans" / "Scan-1795028251" / "capture.json", scans)


def test_a_round_that_has_ended_waits_for_review_like_an_experiment(tmp_path):
    lab = tmp_path / "lab"
    folder = lab / "20261120-capture-0b-noise"
    steps = [{"id": "score", "items": [{"phase": "done"}]}]
    _write(folder / "round.json", json.dumps({"title": "Noise floor", "steps": steps, "finished": 1.0, "outcome": "done"}))
    assert unreviewed(lab) == f"{folder}: round done. Noise floor"
    record_review(folder, "27. Noise floor", "Held.", "Chair to look into.", open=True)
    assert unreviewed(lab) == "every experiment that has ended has been reviewed"
    assert open_actions(lab) == f"{folder} (27. Noise floor): Chair to look into."
    record_action(folder, "Fixed.")
    # A round from before rounds were pinned ran the working tree.
    assert experiment_status(folder).splitlines()[:4] == [
        f"{folder}: round done, reviewed, acted on", "title: Noise floor", "code: working tree", "  score: done",
    ]


def test_labelled_runs_are_listed_by_round_and_one_is_shown_with_its_pictures(tmp_path):
    import numpy as np
    from PIL import Image

    from aomma.dashboard import labels_detail, list_labelled
    from aomma.labels import label_run

    run = tmp_path / "20261122-084312-KitchenScan-006-depth+roomplan"
    (run / "masks").mkdir(parents=True)
    Image.new("RGB", (64, 48)).save(run / "input.jpg")
    depth = np.full((48, 64), 2.0)
    depth[30:] = 1.0  # something a metre in front of the wall, which no mask claims: not grown into
    v, u = np.mgrid[0:48, 0:64].astype(float)
    points = np.stack([(u - 32) / 64 * depth, (v - 24) / 64 * depth, depth], axis=-1).astype(np.float32)
    np.save(run / "points.npy", points)
    (run / "survey.json").write_text(json.dumps({"walls": [{"id": "main_wall"}], "openings": [], "built_ins": []}))
    wall = np.zeros((48, 64), bool)
    wall[:30] = True
    Image.fromarray(wall).save(run / "masks" / "main_wall.png")
    (run / "run.json").write_text(json.dumps({"scan": "scans/Scan-KitchenScan-1795219255", "shot": "006", "condition": "depth+roomplan"}))
    (tmp_path / "20261121-000000-unlabelled").mkdir()  # no label map: not listed
    label_run(run)

    listed = list_labelled(tmp_path)
    assert [row["name"] for row in listed] == [run.name]
    assert listed[0]["batch"] == "20261122-084312"
    assert listed[0]["scan"] == "Scan-KitchenScan-1795219255" and listed[0]["shot"] == "006"
    assert abs(listed[0]["report"]["unknown"] - 18 / 48) < 1e-3

    detail = labels_detail(run)
    assert detail["map"]["path"] == f"{run.name}/labels.png"
    assert detail["how"]["path"] == f"{run.name}/debug/labels_settled.png"
    assert detail["labels"][0]["id"] == "main_wall" and detail["colours"]["wall"]
