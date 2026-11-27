import yaml

from aomma import config


def test_local_settings_are_laid_over_the_defaults_key_by_key(tmp_path, monkeypatch):
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "local.yaml").write_text(yaml.safe_dump({
        "vlms": {"local": {"url": "http://gpu-box:11434"}},
        "comfy": {"heavy": ["http://gpu-box:8188"]},
    }))
    monkeypatch.chdir(tmp_path)
    settings = config.load_config()
    assert settings.vlms["local"].url == "http://gpu-box:11434"
    assert settings.vlms["local"].model  # the rest of the entry is the default's
    assert settings.comfy.heavy == ["http://gpu-box:8188"]
    assert settings.comfy.light == ["http://localhost:8189"]


def test_the_defaults_name_no_other_machine(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(config, "CODE_ROOT", tmp_path)  # no local.yaml anywhere
    monkeypatch.setattr(config, "code_path", lambda p: config.Path(__file__).resolve().parents[1] / p)
    settings = config.load_config()
    assert settings.vlms["local"].url == "http://localhost:11434"
    assert settings.comfy.all == ["http://localhost:8189", "http://localhost:8188"]
