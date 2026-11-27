"""A round: commands run one after another, unwatched, each on the runs an earlier one made.

An experiment over several runs (import the shots, build them, copy them
under another condition, score, review, score again) used to be a shell
script in lab/. Its plan lived only in the script, so neither the queue nor
the dashboard knew what was still to come, and runs passed from step to step
as printed text, where one stray line became a run.

A round is written as a plan (TOML, see lab/*.toml) and run from a folder of
its own in lab/, where `round.json` holds the plan and how every step stands.
The dashboard shows it from the start. Runs pass between steps as data: a step
that makes runs gives those its output names that are run folders, and
nothing else. A round that stops can be taken up again where it stopped.

A round runs the code it was started on, not whatever the working tree holds
by the time a step comes up: it is pinned to the commit checked out when it
was prepared, and every step runs that commit's package, settings, workflows
and lab scripts from a checkout of its own (`cache/code/<commit>`), in the
project's folder, so runs, refs and scans are the project's. The code must be
committed first (`--working-tree` runs the working tree instead, as rounds
did before). The installed packages are the project's .venv either way.

A plan:

    title = "What the round is for"
    folder = "capture-0b"          # lab/<date>-capture-0b
    then = "What to ask once it ends; {out} is the round's folder"

    [vars]                         # used in commands as {scan}
    scan = "scans/Scan-..."

    [[step]]
    id = "photo-a"
    title = "Import photo A"
    command = "aomma import-capture {scan}"
    makes_runs = true              # its runs are the run folders it prints
    required = true                # the round stops if it fails

    [[step]]
    id = "build-a"
    title = "Build photo A"
    each = ["photo-a"]             # once for each run of these steps, as {run}; these are its runs too
    command = "aomma large {run} --foreground"
    log = "{run}/stage2.log"       # default: the round's logs/ folder

A word `{runs:<id>}` in a command stands for every run of that step.
"""

import json
import os
import re
import shlex
import subprocess
import sys
import time
import tomllib
from datetime import datetime
from pathlib import Path

from aomma import background
from aomma.config import CODE_ROOT

STATE = "round.json"
# Waiting, running (live), finished either way, or skipped because an earlier step gave it no runs.
_PHASES = ("waiting", "live", "done", "failed", "skipped")
_WORD = re.compile(r"\{([a-z_]+(?::[\w-]+)?)\}")
# What a round's commands run on: the package, its settings and workflows, the lab's scripts, and what they need installed.
CODE_PATHS = ("src", "config", "workflows", "lab/*.py", "pyproject.toml", "uv.lock")
# Where each pinned commit is checked out, under the code's own folder (cache/ is not tracked).
CHECKOUTS = Path("cache") / "code"
# Scans held out from development, and the ways a plan may use one (see that file).
HELD_OUT = Path("lab") / "held_out.toml"
HELD_OUT_USES = ("references", "confirm")
# The commands a "references" plan may run on a held-out scan: they check the capture or build its references.
_REFERENCE_COMMANDS = ("scan-check", "pose-check", "ref-shell", "ref-large", "ref-items", "ref-small")


def held_out_scans(code_root: Path = CODE_ROOT) -> dict[str, str]:
    """Each held-out scan's folder name, with the short name its runs carry (Scan-DiningScan-17... -> DiningScan)."""
    path = code_root / HELD_OUT
    if not path.exists():
        return {}
    return {scan["name"]: scan["name"].split("-")[1] for scan in tomllib.loads(path.read_text()).get("scan", [])}


def _check_held_out(plan: dict, where: Path, code_root: Path) -> None:
    """Refuse a plan that names a held-out scan, or a run made from it, without saying how it uses it."""
    def names(text: str) -> list[str]:
        return [name for name, short in held_out_scans(code_root).items() if name in text or short in text]

    touched = names(json.dumps(plan))
    if not touched:
        return
    use = plan.get("held_out")
    if use not in HELD_OUT_USES or not plan.get("held_out_why"):
        raise SystemExit(
            f"{where} names the held-out scan {', '.join(touched)} ({HELD_OUT}); say how it is used with "
            f"held_out = \"references\" or \"confirm\" and held_out_why = \"...\""
        )
    if use == "references":
        for step in plan["step"]:
            words = shlex.split(step["command"])
            if names(step["command"]) and not (words[:1] == ["aomma"] and words[1:2] and words[1] in _REFERENCE_COMMANDS):
                raise SystemExit(f"{where}: step {step['id']!r} runs more than a capture check or a reference on "
                                 f"{', '.join(touched)}; a \"references\" plan runs only {', '.join(_REFERENCE_COMMANDS)}")


def _git(code_root: Path, *arguments: str) -> str:
    return subprocess.run(["git", "-C", str(code_root), *arguments], capture_output=True, text=True, check=True).stdout


def pin(code_root: Path = CODE_ROOT) -> str:
    """The commit a new round runs: the code's HEAD, refused while the code has changes not committed."""
    changed = _git(code_root, "status", "--porcelain", "--untracked-files=all", "--", *CODE_PATHS).splitlines()
    if changed:
        raise SystemExit(
            "a round runs committed code; commit these first, or run the working tree with --working-tree:\n  "
            + "\n  ".join(changed)
        )
    return _git(code_root, "rev-parse", "HEAD").strip()


def checkout(commit: str, code_root: Path = CODE_ROOT) -> Path:
    """A checkout of `commit` for rounds to run from, made the first time it is asked for and shared after."""
    folder = code_root / CHECKOUTS / commit[:12]
    if not (folder / ".git").exists():
        folder.parent.mkdir(parents=True, exist_ok=True)
        added = subprocess.run(["git", "-C", str(code_root), "worktree", "add", "--detach", str(folder), commit],
                               capture_output=True, text=True)
        # Another round may have made it meanwhile.
        if added.returncode and not (folder / ".git").exists():
            raise SystemExit(f"could not check out {commit[:12]} into {folder}: {added.stderr.strip()}")
    return folder


def load_plan(path: Path, code_root: Path = CODE_ROOT) -> dict:
    """Read a plan and check it through: every step's id, what it runs on, every word in its commands, held-out scans."""
    plan = tomllib.loads(path.read_text())
    steps = plan.get("step", [])
    if not steps:
        raise SystemExit(f"{path}: a plan needs at least one [[step]]")
    known = {"out", "date", *plan.get("vars", {})}
    seen: set[str] = set()
    for step in steps:
        where = f"{path}: step {step.get('id', '?')!r}"
        if not step.get("id") or not step.get("command"):
            raise SystemExit(f"{where} needs an id and a command")
        if step["id"] in seen:
            raise SystemExit(f"{where} is listed twice")
        for earlier in step.get("each", []):
            if earlier not in seen:
                raise SystemExit(f"{where} runs on {earlier!r}, which is not an earlier step")
        for token in shlex.split(step["command"]):
            if "{runs:" in token and not _WORD.fullmatch(token):
                raise SystemExit(f"{where}: {{runs:...}} must stand alone as a word, in {token!r}")
        for text in (step["command"], step.get("log", "")):
            for word in _WORD.findall(text):
                name, _, of = word.partition(":")
                if name == "runs" and of in seen:
                    continue
                if name == "run" and step.get("each"):
                    continue
                if name not in known or of:
                    raise SystemExit(f"{where}: {{{word}}} names nothing it can use")
        seen.add(step["id"])
    _check_held_out(plan, path, code_root)
    _check_scored(plan, path)
    return plan


_BUILDS_SCENE = re.compile(r"\baomma (boxes|judge-boxes|judge-models)\b")


def _check_scored(plan: dict, where: Path) -> None:
    """A round that builds or judges a scene also scores it end to end (`aomma score`), so that a gain at one gate
    that costs the whole scene shows; or says why not (`no_score_why`)."""
    commands = [step["command"] for step in plan["step"]]
    if any(_BUILDS_SCENE.search(c) for c in commands) and not any("aomma score" in c for c in commands) \
            and not plan.get("no_score_why"):
        raise SystemExit(f"{where}: builds or judges boxes or models but never scores them: add an `aomma score` step "
                         f"(--scene boxes for a box layout), or no_score_why = \"...\"")


def prepare(plan_path: Path, root: Path = Path("lab"), pinned: bool = True, code_root: Path = CODE_ROOT) -> Path:
    """Make a round's folder for a plan, with every step waiting; returns the folder.

    Pinned, the round runs the commit checked out now, from its own checkout; otherwise the working tree.
    """
    plan = load_plan(plan_path, code_root)
    if plan.get("held_out") == "confirm" and not pinned:
        raise SystemExit(f"{plan_path} confirms on a held-out scan, which needs committed code: run it without --working-tree")
    code = None
    if pinned:
        commit = pin(code_root)
        code = {"commit": commit, "checkout": str(checkout(commit, code_root))}
    folder = root / f"{datetime.now():%Y%m%d}-{plan.get('folder', plan_path.stem)}"
    if (folder / STATE).exists():
        folder = root / f"{datetime.now():%Y%m%d-%H%M%S}-{plan.get('folder', plan_path.stem)}"
    folder.mkdir(parents=True)
    state = {
        "title": plan.get("title", plan_path.stem),
        "plan_file": str(plan_path),
        "plan": plan,
        "folder": str(folder),
        # None: the working tree, as it stands when each step starts.
        "code": code,
        "held_out": plan.get("held_out"),
        "steps": [_new_step(step) for step in plan["step"]],
    }
    _save(folder, state)
    return folder


def _new_step(step: dict) -> dict:
    return {"id": step["id"], "title": step.get("title", step["id"]), "each": bool(step.get("each")), "runs": [], "items": []}


def run(folder: Path, retry_failed: bool = False) -> dict:
    """Run a round's steps in order, taking up any it had not finished; returns its state.

    A step that failed before is left failed unless `retry_failed`: running it again could
    overwrite runs that later steps have already worked on.
    """
    state = json.loads((folder / STATE).read_text())
    plan = state["plan"]
    # A round prepared before rounds were pinned has no "code" and runs the working tree, as it did.
    code = None
    if state.get("code"):
        code = Path(state["code"]["checkout"])
        if not (code / ".git").exists():  # cleared away since: checked out again
            code = checkout(state["code"]["commit"])
    finished = ("done",) if retry_failed else ("done", "failed")
    words = {"out": str(folder), "date": f"{datetime.now():%Y%m%d}", **plan.get("vars", {})}
    state.update(pid=os.getpid(), started=state.get("started") or time.time(), finished=None, outcome=None)
    _save(folder, state)
    try:
        for step, record in zip(plan["step"], state["steps"]):
            runs_of = {s["id"]: s["runs"] for s in state["steps"]}
            if step.get("each"):
                inputs = [run for earlier in step["each"] for run in runs_of[earlier]]
                record["runs"] = inputs
                for run_path in inputs:
                    item = _item(record, run_path)
                    if item["phase"] not in finished:
                        _run_item(folder, state, step, item, {**words, "run": run_path}, runs_of, code)
                if not inputs:
                    record["items"] = [{"id": "", "phase": "skipped", "note": "no runs from " + ", ".join(step["each"])}]
                    _save(folder, state)
            else:
                item = _item(record, "")
                if item["phase"] not in finished:
                    made = _run_item(folder, state, step, item, words, runs_of, code)
                    if step.get("makes_runs") and item["phase"] == "done":
                        record["runs"] = made
                        if not made:
                            item.update(phase="failed", note="printed no run folder")
                    _save(folder, state)
            if step.get("required") and any(item["phase"] == "failed" for item in record["items"]):
                raise RuntimeError(f"required step {step['id']!r} failed")
    except BaseException as error:
        state.update(finished=time.time(), outcome=f"stopped: {error!r}" if not isinstance(error, KeyboardInterrupt) else "stopped")
        _save(folder, state)
        background.notify("aomma round stopped", f"Find out why the round {folder} stopped ({error!r})")
        raise
    failed = sum(item["phase"] == "failed" for record in state["steps"] for item in record["items"])
    state.update(finished=time.time(), outcome=f"done, {failed} failed" if failed else "done")
    _save(folder, state)
    then = plan.get("then", "Review the finished round {out}").replace("{out}", str(folder))
    background.notify("aomma round finished", then)
    return state


def _item(record: dict, run_path: str) -> dict:
    """A step's record for one of its runs ("" for a step run once), added waiting if it is new."""
    item = next((i for i in record["items"] if i["id"] == run_path), None)
    if item is None:
        item = {"id": run_path, "phase": "waiting"}
        record["items"] = [i for i in record["items"] if i["phase"] != "skipped"] + [item]
    return item


def _run_item(folder: Path, state: dict, step: dict, item: dict, words: dict, runs_of: dict,
              code: Path | None = None) -> list[str]:
    """Run one command of a step, its output to its log; returns the run folders it printed.

    With `code`, a checkout of the round's commit, the command runs that checkout's package and scripts.
    """
    argv = command(step["command"], words, runs_of, code)
    env = environment(code)
    name = step["id"] + (f"-{Path(item['id']).name}" if item["id"] else "")
    log = Path(_fill(step["log"], words)) if step.get("log") else folder / "logs" / f"{name}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    item.update(phase="live", started=time.time(), finished=None, log=str(log), note=None)
    _save(folder, state)
    with log.open("a") as out:
        start = out.tell()
        out.write(f"== {time.strftime('%Y-%m-%d %H:%M:%S')}  {shlex.join(argv)}\n")
        out.flush()
        code = subprocess.run(argv, stdin=subprocess.DEVNULL, stdout=out, stderr=subprocess.STDOUT,
                              env=env).returncode
    item.update(phase="done" if code == 0 else "failed", finished=time.time(), exit=code)
    _save(folder, state)
    with log.open() as written:
        written.seek(start)
        lines = written.read().splitlines()[1:]
    return [line.strip() for line in lines if line.strip() and " " not in line.strip() and Path(line.strip()).is_dir()]


def environment(code: Path | None = None) -> dict:
    """The environment a command runs in: output unbuffered and, with `code`, that checkout's package first."""
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    if code:
        env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(code / "src"), os.environ.get("PYTHONPATH")]))
    return env


def command(template: str, words: dict, runs_of: dict, code: Path | None = None) -> list[str]:
    """A step's command as arguments. `{runs:<id>}` alone as a word becomes that step's runs, one argument each.

    Leading `aomma` and `python` (or `uv run` before either) are this harness and this Python. With `code`, a
    checkout of the round's commit, a lab script `python` runs is taken from that checkout.
    """
    argv = []
    for word in shlex.split(template):
        whole = _WORD.fullmatch(word)
        if whole and whole.group(1).startswith("runs:"):
            argv += runs_of[whole.group(1).split(":", 1)[1]]
        else:
            argv.append(_fill(word, words))
    if argv[:2] == ["uv", "run"] and argv[2:3] in (["aomma"], ["python"]):
        argv = argv[2:]
    if argv[0] == "aomma":
        argv[:1] = [sys.executable, "-c", "from aomma.cli import main; main()"]
    elif argv[0] == "python":
        argv[0] = sys.executable
        if code and len(argv) > 1 and argv[1].endswith(".py") and (code / argv[1]).is_file():
            argv[1] = str(code / argv[1])
    return argv


def _fill(text: str, words: dict) -> str:
    return _WORD.sub(lambda m: str(words[m.group(1)]), text)


def _save(folder: Path, state: dict) -> None:
    # Write then rename, so the dashboard never reads a half-written file.
    scratch = folder / (STATE + ".tmp")
    scratch.write_text(json.dumps(state, indent=2))
    scratch.replace(folder / STATE)


def shown_rounds(lab_dir: Path) -> list[dict]:
    """Every round for the dashboard, newest first: the Jobs tab shows those under way, the Rounds tab all of them.

    A round whose process is gone without saying it ended was killed; it is shown as stopped.
    The live command's last line of output comes with it.
    """
    rounds = []
    for path in lab_dir.glob(f"*/{STATE}"):
        try:
            state = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        state.pop("plan", None)
        running = bool(background.alive(path.parent / background.PID)) and not state.get("finished")
        if not running and not state.get("finished"):
            if not state.get("started"):
                continue
            state["outcome"] = "stopped (its process is gone)"
        state["running"] = running
        for record in state["steps"]:
            for item in record["items"]:
                if item["phase"] == "live" and item.get("log"):
                    if running:
                        item["last_line"] = (background.log_tail(Path(item["log"]), 1) or [""])[0]
                    else:
                        item["phase"] = "failed"
        rounds.append(state)
    return sorted(rounds, key=lambda s: -(s.get("started") or 0))
