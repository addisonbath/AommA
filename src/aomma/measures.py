"""Each gate's measure, and the rule that changing it is agreed like changing a bar.

A gate's verdict rests on its bar and on its measure: what is counted, which pixels or items are judged, how the
reference is read. Each bar is agreed after its baseline and before any fix, so that
the fixes don't set their own bar. A measure changed after its failures are seen loosens a gate as much: gate 2 went
from 11 to 13 of 16 on two such changes. So a measure is held to the same rule:

- `lab/measures.toml` names, per gate, the code its measure is (its judge and the reference code it reads), its
  baseline runs, and every agreed version of it, with the pass count on the baseline before and after.
- `aomma measure-change <gate>` judges the baseline with the measure at an earlier commit (by default the last
  recorded) and with the code as it stands, each on copies of the runs, and sets the two side by side: the pass
  counts, all shots and those whose pose passed gate 0, and every verdict that moved.
- `aomma measure-record <folder>` adds a comparison, once agreed, to the ledger.
- tests/test_measures.py fails while a gate's code differs from its last recorded version.

The references on disk (`refs/`) are made by the `ref-*` commands, and are part of the measure too: a gate whose
reference is made again is compared the same way, from the commit the old reference was made with.
"""

import ast
import hashlib
import io
import json
import shutil
import subprocess
import tokenize
from datetime import datetime
from glob import glob
from pathlib import Path

import tomllib

from aomma import rounds
from aomma.config import CODE_ROOT

LEDGER = Path("lab") / "measures.toml"
COMPARISONS = Path("cache") / "measures"
# Functions that only print or draw what was measured; changing them changes no verdict.
_NOT_MEASURE = ("describe", "_save_pictures")


def load(ledger: Path = LEDGER) -> dict[str, dict]:
    """The ledger's gates, each with its `version` list (oldest first, maybe empty)."""
    gates = tomllib.loads((CODE_ROOT / ledger if not ledger.is_absolute() else ledger).read_text())
    for spec in gates.values():
        spec.setdefault("version", [])
    return gates


def fingerprint(files: list[Path]) -> str:
    """A hash of the code in `files` that can change a verdict: not comments, docstrings, layout or `_NOT_MEASURE`."""
    digest = hashlib.sha256()
    for path in files:
        source = path.read_text()
        skipped: list[tuple[int, int]] = []
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in _NOT_MEASURE:
                skipped.append((node.lineno - len(node.decorator_list), node.end_lineno))
            elif isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and node.body:
                first = node.body[0]
                if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
                    skipped.append((first.lineno, first.end_lineno))
        digest.update(path.name.encode())
        for token in tokenize.generate_tokens(io.StringIO(source).readline):
            if token.type in (tokenize.COMMENT, tokenize.NL, tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT):
                continue
            if any(start <= token.start[0] <= end for start, end in skipped):
                continue
            digest.update(f"{token.type}:{token.string}\n".encode())
    return digest.hexdigest()[:16]


def gate_fingerprint(spec: dict, code_root: Path = CODE_ROOT) -> str:
    return fingerprint([code_root / name for name in spec["code"]])


def baseline(spec: dict) -> list[Path]:
    runs = sorted(Path(p) for p in glob(spec["baseline"]))
    if not runs:
        raise SystemExit(f"no runs match the baseline {spec['baseline']!r}")
    return runs


def _clone(run: Path, copy: Path) -> None:
    """A copy of a run to judge, the judge's own output in it removed. On APFS a clone, sharing the files' blocks."""
    copy.parent.mkdir(parents=True, exist_ok=True)
    if subprocess.run(["cp", "-cR", str(run), str(copy)], capture_output=True).returncode:
        shutil.copytree(run, copy, symlinks=True)


def _judge(spec: dict, runs: list[Path], code: Path, folder: Path) -> dict[str, dict]:
    """Judge copies of `runs` with the code checked out at `code`; each run's verdict, pose_off, or why there is none."""
    copies = []
    for run in runs:
        copy = folder / run.name
        _clone(run, copy)
        (copy / spec["writes"]).unlink(missing_ok=True)
        copies.append(copy)
    argv = rounds.command(spec["command"], {}, {}, code) + [str(c) for c in copies]
    with (folder.parent / f"{folder.name}.txt").open("w") as out:
        subprocess.run(argv, stdin=subprocess.DEVNULL, stdout=out, stderr=subprocess.STDOUT, env=rounds.environment(code))
    found = {}
    for run, copy in zip(runs, copies):
        written = copy / spec["writes"]
        if not written.exists():
            found[run.name] = {"ok": None, "pose_off": False, "note": "the judge wrote nothing"}
            continue
        result = json.loads(written.read_text())
        found[run.name] = {"ok": bool(result["verdict"]["ok"]), "pose_off": "pose_off" in result}
    return found


def _count(verdicts: dict[str, dict]) -> dict:
    trusted = [v for v in verdicts.values() if not v["pose_off"]]
    return {"pass": sum(v["ok"] is True for v in verdicts.values()), "of": len(verdicts),
            "pose_good_pass": sum(v["ok"] is True for v in trusted), "pose_good_of": len(trusted)}


def _code_state(code_root: Path, files: list[str]) -> str:
    commit = rounds._git(code_root, "rev-parse", "--short", "HEAD").strip()
    changed = rounds._git(code_root, "status", "--porcelain", "--", *files).strip()
    return commit + (" with changes not committed" if changed else "")


def compare(gate: str, since: str | None = None, runs: list[Path] | None = None, ledger: Path = LEDGER,
            code_root: Path = CODE_ROOT, into: Path | None = None) -> Path:
    """Judge `gate`'s baseline with its measure at `since` and as the code stands; returns the comparison's folder.

    `since` defaults to the gate's last recorded commit; with none recorded, only the code as it stands is judged.
    """
    gates = load(ledger)
    if gate not in gates:
        raise SystemExit(f"no gate {gate!r} in {ledger}; it has {', '.join(gates)}")
    spec = gates[gate]
    since = since or (spec["version"][-1]["commit"] if spec["version"] else None)
    runs = runs or baseline(spec)
    if into is None:
        stamp, n = f"{datetime.now():%Y%m%d-%H%M%S}", 1
        while (into := COMPARISONS / (f"{stamp}-{gate}" + ("" if n == 1 else f"-{n}"))).exists():
            n += 1
    into.mkdir(parents=True)
    before = _judge(spec, runs, rounds.checkout(since, code_root), into / "before") if since else None
    after = _judge(spec, runs, code_root, into / "after")
    shutil.rmtree(into / "after")
    if before is not None:
        shutil.rmtree(into / "before")
    result = {
        "gate": gate, "title": spec.get("title", gate), "at": f"{datetime.now():%Y-%m-%d %H:%M}",
        "since": since, "now": _code_state(code_root, spec["code"]), "fingerprint": gate_fingerprint(spec, code_root),
        "runs": [str(r) for r in runs],
        "before": _count(before) if before is not None else None, "after": _count(after),
        "verdicts": {name: {"before": before[name] if before is not None else None, "after": after[name]} for name in after},
    }
    (into / "compare.json").write_text(json.dumps(result, indent=2))
    (into / "compare.txt").write_text(describe(result))
    return into


def _word(verdict: dict | None) -> str:
    if verdict is None:
        return "-"
    return {True: "PASS", False: "FAIL", None: "NONE"}[verdict["ok"]]


def describe(result: dict) -> str:
    def counts(c: dict) -> str:
        return f"{c['pass']} of {c['of']} (pose good: {c['pose_good_pass']} of {c['pose_good_of']})"

    lines = [f"{result['gate']} ({result['title']}), judged {result['at']} on {len(result['runs'])} runs"]
    if result["before"]:
        lines += [f"  measure at {result['since']}: {counts(result['before'])}",
                  f"  measure now ({result['now']}): {counts(result['after'])}"]
        moved = {n: v for n, v in result["verdicts"].items() if _word(v["before"]) != _word(v["after"])}
        lines.append(f"  {len(moved)} verdicts moved" + (":" if moved else ""))
        lines += [f"    {n}: {_word(v['before'])} -> {_word(v['after'])}" + (" (pose off)" if v["after"]["pose_off"] else "")
                  for n, v in moved.items()]
    else:
        lines.append(f"  measure now ({result['now']}), nothing recorded to compare with: {counts(result['after'])}")
    lines += [f"    {n}: {v['after']['note']}" for n, v in result["verdicts"].items() if v["after"].get("note")]
    return "\n".join(lines) + "\n"


def record(folder: Path, section: str, why: str, ledger: Path = LEDGER, code_root: Path = CODE_ROOT) -> dict:
    """Add an agreed comparison to the ledger as the gate's newest version; the measure's code must be committed."""
    result = json.loads((folder / "compare.json").read_text())
    spec = load(ledger)[result["gate"]]
    if "not committed" in result["now"]:
        raise SystemExit(f"{folder} judged code not committed; commit the measure, compare again, then record")
    if gate_fingerprint(spec, code_root) != result["fingerprint"]:
        raise SystemExit(f"{result['gate']}'s code has changed since {folder} was judged; compare again")
    entry = {
        "date": result["at"][:10], "commit": rounds._git(code_root, "rev-parse", "HEAD").strip(),
        "fingerprint": result["fingerprint"], "section": section, "why": why, "since": result["since"] or "",
        "runs": len(result["runs"]), "comparison": str(folder),
    }
    for side in ("before", "after"):
        for key, value in (result[side] or {}).items():
            entry[f"{side}_{key}"] = value
    path = code_root / ledger if not ledger.is_absolute() else ledger
    # Written by hand: tomllib only reads. A JSON string is a TOML basic string.
    lines = [f"\n[[{result['gate']}.version]]"] + [f"{k} = {json.dumps(v)}" for k, v in entry.items()]
    with path.open("a") as out:
        out.write("\n".join(lines) + "\n")
    return entry
