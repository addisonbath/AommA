import json

from aomma import cli


def test_staged_runs_each_step_once_and_picks_up_where_it_stopped(tmp_path, monkeypatch):
    ran = []
    monkeypatch.setattr(cli, "load_config", lambda _: None)

    def shell(run_dir, *_):
        ran.append("shell")
        (run_dir / "shell.json").write_text("{}")

    def small(*_):
        ran.append("small")
        raise RuntimeError("stopped")

    monkeypatch.setattr(cli, "shell", shell)
    monkeypatch.setattr(cli, "small", small)
    (tmp_path / "large_survey.json").write_text("{}")  # listed already: not listed again
    try:
        cli.staged(tmp_path, tmp_path / "config.toml", None)
    except RuntimeError:
        pass
    assert ran == ["shell", "shell", "small"]
    assert json.loads((tmp_path / "staged.json").read_text())["done"] == ["shell", "large-survey", "shell-again"]


def test_staged_stops_when_no_shell_is_built(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "load_config", lambda _: None)
    monkeypatch.setattr(cli, "shell", lambda *_: None)
    monkeypatch.setattr(cli, "small", lambda *_: (_ for _ in ()).throw(AssertionError("went on")))
    cli.staged(tmp_path, tmp_path / "config.toml", None)
    assert not (tmp_path / "staged.json").exists()
