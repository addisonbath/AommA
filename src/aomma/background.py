"""Commands that outlive the shell that started them.

A run takes hours and the dashboard is meant to stay up, but neither needs a
terminal: started with `--detach`, a command carries on in the background
with its output in a log file. Every long command, detached or not, leaves its
process id in its folder, so that `aomma stop` can find it and a second copy
is not started on the same folder by mistake. One that ends unwatched can say
so with a notification.
"""

import os
import platform
import shutil
import signal
import subprocess
import sys
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

# The files a long command keeps in its folder: its output when detached, and its process id.
LOG, PID = "run.log", "run.pid"


def alive(pid_file: Path) -> int | None:
    """The process id recorded in `pid_file`, if that process is still running."""
    try:
        pid = int(pid_file.read_text())
        os.kill(pid, 0)  # signal 0 only checks that the process exists
    except (OSError, ValueError):
        return None
    return pid


def detach(arguments: list[str], log: Path, pid_file: Path) -> int:
    """Start `aomma <arguments>` in the background, its output added to `log`; returns its process id.

    The new process has a session of its own, so closing the terminal does not
    end it.
    """
    if pid := alive(pid_file):
        raise SystemExit(f"already running here (process {pid})")
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a") as out:
        out.write(f"== {time.strftime('%Y-%m-%d %H:%M:%S')}  aomma {' '.join(arguments)}\n")
        out.flush()
        process = subprocess.Popen(
            [sys.executable, "-c", "from aomma.cli import main; main()", *arguments],
            stdin=subprocess.DEVNULL, stdout=out, stderr=subprocess.STDOUT, start_new_session=True,
            # Output is written as it happens, so the log can be followed.
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
    pid_file.write_text(str(process.pid))
    return process.pid


@contextmanager
def recorded(pid_file: Path):
    """Leave this process's id in `pid_file` while the block runs; refuses if another process is at work there."""
    other = alive(pid_file)
    if other and other != os.getpid():
        raise SystemExit(f"already running here (process {other}); stop it first with `aomma stop`")
    pid_file.parent.mkdir(parents=True, exist_ok=True)
    pid_file.write_text(str(os.getpid()))
    try:
        yield
    finally:
        pid_file.unlink(missing_ok=True)


def stop(pid_file: Path, wait_s: float = 10) -> int | None:
    """End the process recorded in `pid_file`, with whatever it started; returns its id, or None if none was running."""
    pid = alive(pid_file)
    pid_file.unlink(missing_ok=True)
    if pid is None:
        return None
    group = os.getpgid(pid)

    def send(sign: int) -> None:
        try:
            # Its group takes in the Blender it may be running. Never our own group, though.
            os.kill(pid, sign) if group == os.getpgrp() else os.killpg(group, sign)
        except ProcessLookupError:
            pass
        except PermissionError:
            # macOS can refuse a signal to a whole group (seen for one only just started); the process itself still takes it.
            os.kill(pid, sign)

    send(signal.SIGTERM)
    deadline = time.monotonic() + wait_s
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except OSError:
            return pid
        time.sleep(0.2)
    send(signal.SIGKILL)
    return pid


def log_tail(log: Path, lines: int = 40, width: int = 240) -> list[str]:
    """The last `lines` lines of a log, each cut to `width` characters; empty if there is no log."""
    if not log.exists():
        return []
    with log.open("rb") as file:
        file.seek(0, os.SEEK_END)
        file.seek(max(file.tell() - 64_000, 0))
        text = file.read().decode(errors="replace")
    return [line[:width] for line in text.splitlines()[-lines:]]


PROMPT_FILE = Path("runs") / "next-prompt.txt"


def notify(title: str, prompt: str, prompt_file: Path = PROMPT_FILE) -> None:
    """Show a desktop notification of `prompt`, what to ask Claude next, and add it to `prompt_file`:
    a notification cannot be copied from, and the user pastes the prompt after /clear. The file keeps
    every notification, each under a line with its time and title, newest first. Each notification also
    gets a file of its own in a `prompts` folder beside it, and that is the one it opens: clicking an older
    notification then shows its own prompt, not whichever came last, and an editor still holding the list
    open from before cannot show a stale copy or save it back over newer entries. With terminal-notifier,
    clicking the notification opens that file; a notification from osascript would open Script Editor
    instead, so without it the file is opened at once. On a Mac only; elsewhere, or if it cannot be shown,
    nothing happens."""
    if platform.system() != "Darwin":
        return
    prompt_file = prompt_file.resolve()
    now = datetime.now()
    entry = f"{now:%Y-%m-%d %H:%M:%S} · {title}\n{prompt}\n"
    own_file = prompt_file.parent / "prompts" / f"{now:%Y%m%d-%H%M%S-%f}.txt"
    try:
        own_file.parent.mkdir(parents=True, exist_ok=True)
        own_file.write_text(entry)
        earlier = prompt_file.read_text() if prompt_file.exists() else ""
        prompt_file.write_text(entry + ("\n" + earlier if earlier else ""))
    except OSError:
        return
    prompt_file = own_file
    message = f"Click to open the prompt to paste after /clear: {prompt}"
    notifier = shutil.which("terminal-notifier")
    try:
        # terminal-notifier exits non-zero when macOS has not allowed it to notify.
        if notifier and subprocess.run(
            [notifier, "-title", title, "-message", message, "-open", prompt_file.as_uri()],
            capture_output=True, timeout=10,
        ).returncode == 0:
            return
    except (OSError, subprocess.SubprocessError):
        pass
    message = f"Paste after /clear: {prompt}"
    quoted = [text.replace("\\", "\\\\").replace('"', '\\"') for text in (message, title)]
    try:
        subprocess.run(
            ["osascript", "-e", 'display notification "{}" with title "{}"'.format(*quoted)],
            capture_output=True, timeout=10,
        )
        subprocess.run(["open", "-t", str(prompt_file)], capture_output=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        pass
