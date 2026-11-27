import os
import signal
import subprocess
import sys

import pytest

from aomma import background


def test_a_folder_records_who_is_at_work_in_it_and_refuses_a_second(tmp_path):
    pid_file = tmp_path / "run" / background.PID
    assert background.alive(pid_file) is None
    with background.recorded(pid_file):
        assert background.alive(pid_file) == os.getpid()
        # The same process may record itself again (a detached command finds its id already written).
        with background.recorded(pid_file):
            pass
    assert not pid_file.exists()

    other = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        pid_file.write_text(str(other.pid))
        with pytest.raises(SystemExit, match="already running"):
            with background.recorded(pid_file):
                pass
        with pytest.raises(SystemExit, match="already running"):
            background.detach(["jobs", "--once"], tmp_path / "run" / background.LOG, pid_file)
    finally:
        other.kill()
        other.wait()
    # A process id left behind by something that has died does not count.
    assert background.alive(pid_file) is None


def test_stop_ends_the_recorded_process_and_its_group(tmp_path):
    pid_file = tmp_path / background.PID
    assert background.stop(pid_file) is None

    process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True)
    pid_file.write_text(str(process.pid))
    # The stopped process is this test's child, so it lingers until waited for; do not wait long for it to vanish.
    assert background.stop(pid_file, wait_s=0.3) == process.pid
    assert process.wait(timeout=5) in (-signal.SIGTERM, -signal.SIGKILL)
    assert not pid_file.exists()


def test_the_end_of_a_log_is_read_without_the_rest(tmp_path):
    log = tmp_path / background.LOG
    assert background.log_tail(log) == []
    log.write_text("".join(f"line {number}\n" for number in range(5000)) + "x" * 500 + "\n")
    tail = background.log_tail(log, lines=3, width=10)
    assert tail == ["line 4998", "line 4999", "x" * 10]


def test_a_notification_opens_a_file_holding_its_prompt(monkeypatch, tmp_path):
    calls = []
    prompt_file = tmp_path / "next-prompt.txt"
    monkeypatch.setattr(background.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(
        background.subprocess, "run", lambda argv, **kw: calls.append(argv) or subprocess.CompletedProcess(argv, 0)
    )
    monkeypatch.setattr(background.shutil, "which", lambda name: "/opt/homebrew/bin/terminal-notifier")
    background.notify("aomma round finished", 'Review the round lab/x, "A" against B', prompt_file)
    [own] = (tmp_path / "prompts").resolve().iterdir()
    lines = prompt_file.read_text().splitlines()
    assert lines[0].endswith(" · aomma round finished")
    assert lines[1:] == ['Review the round lab/x, "A" against B']
    assert calls == [[
        "/opt/homebrew/bin/terminal-notifier", "-title", "aomma round finished",
        "-message", 'Click to open the prompt to paste after /clear: Review the round lab/x, "A" against B',
        "-open", own.as_uri(),
    ]]
    assert own.read_text() == prompt_file.read_text()


def test_without_terminal_notifier_the_prompt_file_opens_at_once(monkeypatch, tmp_path):
    calls = []
    prompt_file = tmp_path / "next-prompt.txt"
    monkeypatch.setattr(background.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(background.subprocess, "run", lambda argv, **kw: calls.append(argv))
    monkeypatch.setattr(background.shutil, "which", lambda name: None)
    background.notify("aomma round finished", 'Review "A"', prompt_file)
    assert calls[0][0] == "osascript" and 'Review \\"A\\"' in calls[0][2]
    [own] = (tmp_path / "prompts").resolve().iterdir()
    assert calls[1] == ["open", "-t", str(own)]


def test_a_refused_terminal_notifier_falls_back_to_opening_the_file(monkeypatch, tmp_path):
    calls = []
    prompt_file = tmp_path / "next-prompt.txt"
    monkeypatch.setattr(background.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(
        background.subprocess, "run", lambda argv, **kw: calls.append(argv) or subprocess.CompletedProcess(argv, 3)
    )
    monkeypatch.setattr(background.shutil, "which", lambda name: "/opt/homebrew/bin/terminal-notifier")
    background.notify("aomma round finished", "Review", prompt_file)
    assert [argv[0] for argv in calls] == ["/opt/homebrew/bin/terminal-notifier", "osascript", "open"]


def test_the_prompt_file_keeps_earlier_notifications_newest_first(monkeypatch, tmp_path):
    prompt_file = tmp_path / "next-prompt.txt"
    monkeypatch.setattr(background.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(background.subprocess, "run", lambda argv, **kw: subprocess.CompletedProcess(argv, 0))
    monkeypatch.setattr(background.shutil, "which", lambda name: "/opt/homebrew/bin/terminal-notifier")
    background.notify("first job", "Review A", prompt_file)
    background.notify("second job", "Review B", prompt_file)
    lines = prompt_file.read_text().splitlines()
    assert lines[0].endswith(" · second job") and lines[1] == "Review B"
    assert lines[2] == ""
    assert lines[3].endswith(" · first job") and lines[4] == "Review A"
    assert len(lines) == 5
    # Each notification opens a file holding only its own prompt.
    own = sorted((tmp_path / "prompts").iterdir())
    assert [file.read_text().splitlines()[1] for file in own] == ["Review A", "Review B"]
