import json
import subprocess
import tomllib

import pytest

from aomma import measures
from aomma.config import CODE_ROOT


@pytest.mark.lab
def test_each_gate_measure_is_its_last_recorded_version():
    """A gate's measure changed without an agreed comparison fails here."""
    unrecorded = []
    for gate, spec in measures.load().items():
        now = measures.gate_fingerprint(spec)
        last = spec["version"][-1]["fingerprint"] if spec["version"] else None
        if now != last:
            unrecorded.append(f"{gate} ({', '.join(spec['code'])}): {last} recorded, {now} now")
    assert not unrecorded, (
        "a gate's measure changed with no agreed comparison; run `aomma measure-change <gate>`, agree the counts, "
        "then `aomma measure-record <folder> --section ... --why ...`:\n  " + "\n  ".join(unrecorded)
    )


def test_the_fingerprint_follows_what_can_change_a_verdict_only(tmp_path):
    judge = tmp_path / "judge.py"
    base = 'BAR = 0.1\n\n\ndef verdict(r):\n    return r < BAR\n\n\ndef describe(r):\n    return "PASS"\n'
    judge.write_text(base)
    first = measures.fingerprint([judge])
    for same in (
        '"""A judge."""\n' + base,                                      # a docstring
        base.replace("BAR = 0.1", "BAR = 0.1  # agreed on the baseline"),    # a comment
        base.replace('"PASS"', '"PASS, as before"'),                     # what describe prints
        base.replace("\n\n\ndef verdict", "\n\ndef verdict"),            # layout
    ):
        judge.write_text(same)
        assert measures.fingerprint([judge]) == first, same
    for changed in (base.replace("0.1", "0.2"), base.replace("r < BAR", "r <= BAR")):
        judge.write_text(changed)
        assert measures.fingerprint([judge]) != first, changed


# A stand-in for a judge: a lab script that writes each run's verdict, passing runs whose number is under its bar.
JUDGE = '''import json, sys, pathlib
BAR = {bar}
for run in sys.argv[1:]:
    n = int(pathlib.Path(run, "n").read_text())
    result = {{"verdict": {{"ok": n < BAR}}}}
    if n == 0:
        result["pose_off"] = "off"
    pathlib.Path(run, "judge_x.json").write_text(json.dumps(result))
'''


@pytest.fixture
def project(tmp_path, monkeypatch):
    """A code repo with a judge script and a ledger, and four runs numbered 0-3, each with a stale verdict in it."""
    code = tmp_path / "code"
    (code / "lab").mkdir(parents=True)
    (code / "lab" / "judge.py").write_text(JUDGE.format(bar=2))
    (code / "lab" / "measures.toml").write_text(
        '[gx]\ntitle = "test gate"\ncommand = "python lab/judge.py"\nwrites = "judge_x.json"\n'
        'code = ["lab/judge.py"]\nbaseline = "runs/*"\n'
    )
    git = ["git", "-C", str(code), "-c", "user.name=t", "-c", "user.email=t@t"]
    subprocess.run([*git, "init", "-q"], check=True)
    subprocess.run([*git, "add", "."], check=True)
    subprocess.run([*git, "commit", "-qm", "bar 2"], check=True)
    data = tmp_path / "data"
    for n in range(4):
        (data / "runs" / f"r{n}").mkdir(parents=True)
        (data / "runs" / f"r{n}" / "n").write_text(str(n))
        (data / "runs" / f"r{n}" / "judge_x.json").write_text('{"verdict": {"ok": true}}')
    monkeypatch.chdir(data)
    return code, git, data


def test_a_measure_change_is_judged_both_ways_on_copies_and_recorded(project):
    code, git, data = project
    ledger = code / "lab" / "measures.toml"
    first = measures.compare("gx", ledger=ledger, code_root=code)
    result = json.loads((first / "compare.json").read_text())
    # Nothing recorded yet: the measure as it stands only. Run 0 passes but its pose is off.
    assert result["before"] is None
    assert result["after"] == {"pass": 2, "of": 4, "pose_good_pass": 1, "pose_good_of": 3}
    measures.record(first, "1", "first version", ledger=ledger, code_root=code)
    subprocess.run([*git, "commit", "-qam", "recorded"], check=True)

    (code / "lab" / "judge.py").write_text(JUDGE.format(bar=3))
    with pytest.raises(SystemExit, match="not committed"):
        measures.record(measures.compare("gx", ledger=ledger, code_root=code), "2", "x", ledger=ledger, code_root=code)
    subprocess.run([*git, "commit", "-qam", "bar 3"], check=True)
    second = measures.compare("gx", ledger=ledger, code_root=code)
    text = (second / "compare.txt").read_text()
    assert "2 of 4 (pose good: 1 of 3)" in text and "3 of 4 (pose good: 2 of 3)" in text
    assert "1 verdicts moved" in text and "r2: FAIL -> PASS" in text
    measures.record(second, "2", "bar 3", ledger=ledger, code_root=code)

    versions = tomllib.loads(ledger.read_text())["gx"]["version"]
    assert [v["section"] for v in versions] == ["1", "2"]
    assert versions[1]["before_pass"] == 2 and versions[1]["after_pass"] == 3
    assert versions[1]["fingerprint"] == measures.gate_fingerprint(measures.load(ledger)["gx"], code)
    # The runs themselves were never judged: their stale verdicts are as they were.
    assert all((data / "runs" / f"r{n}" / "judge_x.json").read_text() == '{"verdict": {"ok": true}}' for n in range(4))


@pytest.mark.lab
def test_the_project_ledger_names_code_that_exists():
    for gate, spec in measures.load().items():
        for name in spec["code"]:
            assert (CODE_ROOT / name).is_file(), f"{gate}: {name}"
