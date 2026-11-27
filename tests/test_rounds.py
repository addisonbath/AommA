import json
import subprocess
import os
import sys

import pytest

from aomma import background, rounds

# A plan whose steps are small Python commands: one makes two runs (and prints a stray line, as condition_copies does),
# one works on each of them, one takes them all, one fails.
PLAN = """
title = "test round"
folder = "test"

[vars]
greeting = "hello"

[[step]]
id = "make"
title = "Make runs"
command = "python -c 'import os; [os.makedirs(p, exist_ok=True) or print(p) for p in (\\"runs/a\\", \\"runs/b\\")]; print(\\"depth fitted to the LiDAR: x0.97\\")'"
makes_runs = true
required = true

[[step]]
id = "each"
title = "Touch each"
each = ["make"]
command = "python -c 'import sys, pathlib; pathlib.Path(sys.argv[1], \\"touched\\").write_text(sys.argv[2])' {run} {greeting}"
log = "{run}/each.log"

[[step]]
id = "all"
title = "List them all"
command = "python -c 'import sys; print(sys.argv[1:])' {runs:make} --into {out}"

[[step]]
id = "broken"
title = "Fail"
command = "python -c 'raise SystemExit(3)'"
"""


@pytest.fixture
def plan(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(background, "notify", lambda *a: None)
    (tmp_path / "plan.toml").write_text(PLAN)
    return tmp_path / "plan.toml"


def test_a_round_passes_runs_between_steps_and_carries_on_past_a_failure(plan, tmp_path):
    folder = rounds.prepare(plan.relative_to(tmp_path), pinned=False)
    waiting = json.loads((folder / rounds.STATE).read_text())
    assert [s["id"] for s in waiting["steps"]] == ["make", "each", "all", "broken"]

    state = rounds.run(folder)
    make, each, every, broken = state["steps"]
    # The stray line is not a run.
    assert make["runs"] == ["runs/a", "runs/b"]
    assert [i["id"] for i in each["items"]] == ["runs/a", "runs/b"]
    assert (tmp_path / "runs/a/touched").read_text() == "hello"
    assert (tmp_path / "runs/b/each.log").exists()
    assert f"['runs/a', 'runs/b', '--into', '{folder}']" in (folder / "logs" / "all.log").read_text()
    assert broken["items"][0]["phase"] == "failed" and broken["items"][0]["exit"] == 3
    assert state["outcome"] == "done, 1 failed"


def test_a_round_taken_up_again_runs_only_what_is_not_done(plan, tmp_path):
    folder = rounds.prepare(plan.relative_to(tmp_path), pinned=False)
    rounds.run(folder)
    (tmp_path / "runs/a/touched").unlink()
    failed_at = json.loads((folder / rounds.STATE).read_text())["steps"][3]["items"][0]["started"]
    state = rounds.run(folder)
    assert not (tmp_path / "runs/a/touched").exists()
    # The failed step is left as it was, not run again.
    assert state["steps"][3]["items"][0]["phase"] == "failed"
    assert state["steps"][3]["items"][0]["started"] == failed_at


def test_a_round_taken_up_with_retry_failed_runs_its_failed_steps_again(plan, tmp_path):
    folder = rounds.prepare(plan.relative_to(tmp_path), pinned=False)
    rounds.run(folder)
    failed_at = json.loads((folder / rounds.STATE).read_text())["steps"][3]["items"][0]["started"]
    state = rounds.run(folder, retry_failed=True)
    assert state["steps"][3]["items"][0]["started"] > failed_at


def test_a_required_step_that_fails_stops_the_round(plan, tmp_path):
    plan.write_text(PLAN.replace("for p in (\\\"runs/a\\\", \\\"runs/b\\\")", "for p in ()"))
    folder = rounds.prepare(plan.relative_to(tmp_path), pinned=False)
    with pytest.raises(RuntimeError, match="required step 'make'"):
        rounds.run(folder)
    state = json.loads((folder / rounds.STATE).read_text())
    assert state["steps"][0]["items"][0]["note"] == "printed no run folder"
    assert state["steps"][1]["items"] == [] and state["outcome"].startswith("stopped")


@pytest.mark.parametrize("bad, message", [
    ('each = ["later"]', "not an earlier step"),
    ('command = "aomma score {runs:nowhere}"', "names nothing"),
    ('command = "aomma large {run}"', "names nothing"),  # {run} only in a step run on each run
    ('command = "aomma score x{runs:make}"', "stand alone"),
])
def test_a_plan_is_checked_before_anything_runs(tmp_path, bad, message):
    path = tmp_path / "plan.toml"
    step = 'id = "two"\ncommand = "aomma score {runs:make}"' if bad.startswith("each") else 'id = "two"'
    path.write_text(f'[[step]]\nid = "make"\ncommand = "true"\nmakes_runs = true\n\n[[step]]\n{step}\n{bad}\n')
    with pytest.raises(SystemExit, match=message):
        rounds.load_plan(path)


def test_a_plan_that_builds_boxes_must_score_them_or_say_why_not(tmp_path):
    path = tmp_path / "plan.toml"
    steps = '[[step]]\nid = "make"\ncommand = "true"\nmakes_runs = true\n\n[[step]]\nid = "gate4"\ncommand = "aomma judge-boxes {runs:make}"\n'
    path.write_text(steps)
    with pytest.raises(SystemExit, match="never scores them"):
        rounds.load_plan(path)
    path.write_text('no_score_why = "a probe of the judge alone"\n' + steps)
    rounds.load_plan(path)
    path.write_text(steps + '\n[[step]]\nid = "score"\ncommand = "aomma score {runs:make} --scene boxes"\n')
    rounds.load_plan(path)


def test_the_dashboard_shows_a_round_whose_process_died_as_stopped(plan, tmp_path):
    folder = rounds.prepare(plan.relative_to(tmp_path), pinned=False)
    state = json.loads((folder / rounds.STATE).read_text())
    state.update(started=1.0, pid=os.getpid())
    state["steps"][0]["items"] = [{"id": "", "phase": "live", "log": str(folder / "x.log")}]
    (folder / rounds.STATE).write_text(json.dumps(state))
    (shown,) = rounds.shown_rounds(tmp_path / "lab")
    assert not shown["running"] and shown["outcome"].startswith("stopped")
    assert shown["steps"][0]["items"][0]["phase"] == "failed"

    (folder / background.PID).write_text(str(os.getpid()))
    (folder / "x.log").write_text("first\nlast line\n")
    (shown,) = rounds.shown_rounds(tmp_path / "lab")
    assert shown["running"] and shown["steps"][0]["items"][0]["last_line"] == "last line"


def test_aomma_and_python_are_this_harness(tmp_path):
    argv = rounds.command("aomma score {runs:a} --into {out}", {"out": "lab/x"}, {"a": ["runs/1", "runs/2"]})
    assert argv[:3] == [sys.executable, "-c", "from aomma.cli import main; main()"]
    assert argv[3:] == ["score", "runs/1", "runs/2", "--into", "lab/x"]


def test_a_round_that_ended_long_ago_is_still_listed(plan, tmp_path):
    folder = rounds.prepare(plan.relative_to(tmp_path), pinned=False)
    state = json.loads((folder / rounds.STATE).read_text())
    state.update(started=1.0, finished=2.0, outcome="done")
    (folder / rounds.STATE).write_text(json.dumps(state))
    (shown,) = rounds.shown_rounds(tmp_path / "lab")
    assert not shown["running"] and shown["outcome"] == "done"


def _code_repo(root):
    """A small git repo standing in for the code: a module under src/ and a lab script, both saying "one"."""
    (root / "src").mkdir(parents=True)
    (root / "lab").mkdir()
    (root / "src" / "pinprobe.py").write_text('SAYS = "one"\n')
    (root / "lab" / "probe.py").write_text('import sys, pinprobe; print("script one", pinprobe.SAYS, file=open(sys.argv[1], "w"))\n')
    (root / "NOTES.md").write_text("notes\n")
    git = ["git", "-C", str(root), "-c", "user.name=t", "-c", "user.email=t@t"]
    subprocess.run([*git, "init", "-q"], check=True)
    subprocess.run([*git, "add", "."], check=True)
    subprocess.run([*git, "commit", "-qm", "one"], check=True)
    return git


def test_a_round_refuses_code_not_committed_but_not_other_changes(tmp_path):
    code = tmp_path / "code"
    git = _code_repo(code)
    (code / "NOTES.md").write_text("notes, changed\n")
    commit = rounds.pin(code)
    assert commit == subprocess.run([*git, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()

    (code / "lab" / "new.py").write_text("print()\n")
    with pytest.raises(SystemExit, match="lab/new.py"):
        rounds.pin(code)
    (code / "lab" / "new.py").unlink()
    (code / "src" / "pinprobe.py").write_text('SAYS = "two"\n')
    with pytest.raises(SystemExit, match="src/pinprobe.py"):
        rounds.pin(code)


def test_a_pinned_round_runs_its_commit_whatever_the_working_tree_holds_later(plan, tmp_path):
    code = tmp_path / "code"
    git = _code_repo(code)
    plan.write_text(
        'title = "pinned"\nfolder = "pinned"\n\n'
        '[[step]]\nid = "module"\ncommand = "python -c \'import sys, pinprobe; print(pinprobe.SAYS, file=open(sys.argv[1], \\"w\\"))\' {out}/module.txt"\n\n'
        '[[step]]\nid = "script"\ncommand = "uv run python lab/probe.py {out}/script.txt"\n'
    )
    folder = rounds.prepare(plan.relative_to(tmp_path), code_root=code)
    state = json.loads((folder / rounds.STATE).read_text())
    assert state["code"]["checkout"] == str(code / rounds.CHECKOUTS / state["code"]["commit"][:12])

    # The code moves on, committed and not, before the round's steps come up.
    (code / "src" / "pinprobe.py").write_text('SAYS = "two"\n')
    (code / "lab" / "probe.py").write_text('import sys; print("script two", file=open(sys.argv[1], "w"))\n')
    subprocess.run([*git, "commit", "-qam", "two"], check=True)
    (code / "src" / "pinprobe.py").write_text('SAYS = "three"\n')

    rounds.run(folder)
    assert (folder / "module.txt").read_text() == "one\n"
    assert (folder / "script.txt").read_text() == "script one one\n"


def test_uv_run_and_lab_scripts_in_a_pinned_round(tmp_path):
    checkout = tmp_path / "checkout"
    (checkout / "lab").mkdir(parents=True)
    (checkout / "lab" / "x.py").write_text("")
    argv = rounds.command("uv run python lab/x.py {out}", {"out": "lab/r"}, {}, checkout)
    assert argv == [sys.executable, str(checkout / "lab" / "x.py"), "lab/r"]
    # Unpinned, the script is the working tree's.
    assert rounds.command("python lab/x.py", {}, {}) == [sys.executable, "lab/x.py"]
    # Other uv commands are left alone.
    assert rounds.command("uv sync", {}, {})[:2] == ["uv", "sync"]


def test_a_plan_on_a_held_out_scan_must_say_how_it_uses_it(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    code = tmp_path / "code"
    (code / "lab").mkdir(parents=True)
    (code / "lab" / "held_out.toml").write_text('[[scan]]\nname = "Scan-DiningScan-1795717793"\nadded = "2026-11-26"\n')
    plan = tmp_path / "plan.toml"

    def write(head: str, *commands: str) -> None:
        steps = "".join(f'\n[[step]]\nid = "s{n}"\ncommand = "{c}"\n' for n, c in enumerate(commands))
        plan.write_text(head + steps)

    # Naming the scan, or a run made from it, with no use declared.
    for command in ("aomma ref-shell scans/Scan-DiningScan-1795717793", "aomma judge-boxes runs/20261126-1-DiningScan-005-photo"):
        write("", command)
        with pytest.raises(SystemExit, match="held-out scan Scan-DiningScan"):
            rounds.load_plan(plan, code)
    # References: the capture checks and references only.
    refs = 'held_out = "references"\nheld_out_why = "its references, before any run"\n'
    write(refs, "aomma ref-shell scans/Scan-DiningScan-1795717793", "aomma ref-items scans/Scan-DiningScan-1795717793 --small")
    rounds.load_plan(plan, code)
    write(refs, "aomma ref-shell scans/Scan-DiningScan-1795717793", "aomma import-capture scans/Scan-DiningScan-1795717793")
    with pytest.raises(SystemExit, match="'s1' runs more than"):
        rounds.load_plan(plan, code)
    # Confirm: anything, but on committed code only.
    write('held_out = "confirm"\nheld_out_why = "a committed change, confirmed"\n', "aomma import-capture scans/Scan-DiningScan-1795717793")
    rounds.load_plan(plan, code)
    with pytest.raises(SystemExit, match="without --working-tree"):
        rounds.prepare(plan.relative_to(tmp_path), pinned=False, code_root=code)
    # Other scans need nothing.
    write("", "aomma ref-shell scans/Scan-OfficeScan-1795136900")
    rounds.load_plan(plan, code)


@pytest.mark.lab
def test_the_project_lists_its_held_out_scans_as_they_are_named_in_scans():
    held = rounds.held_out_scans()
    assert held == {"Scan-DiningScan-1795717793": "DiningScan", "Scan-GuestroomScan-1795796884": "GuestroomScan"}
