import json

from PIL import Image

from aomma.stages import asset


class FakeComfy:
    def __init__(self):
        self.runs = 0

    def upload_image(self, data, name):
        return name

    def run(self, workflow, label="", ticket=None):
        self.runs += 1
        return {"output_model": {"3d": [f"model{self.runs}"]}}

    def fetch(self, ref):
        return ref.encode()


def _workflow(tmp_path):
    path = tmp_path / "wf.json"
    path.write_text(json.dumps({"input_image": {"inputs": {"image": ""}}, "sampler": {"inputs": {"seed": 7}}, "output_model": {"inputs": {}}}))
    return path


def test_same_crop_and_seed_comes_from_cache(tmp_path, monkeypatch):
    monkeypatch.delenv("AOMMA_FRESH_MODELS", raising=False)
    comfy, wf, cache = FakeComfy(), _workflow(tmp_path), tmp_path / "cache"
    crop = Image.new("RGB", (8, 8), (10, 20, 30))
    first = asset.generate_model(comfy, wf, crop, 30.0, "a", cache=cache)
    again = asset.generate_model(comfy, wf, crop.copy(), 30.0, "b", cache=cache)
    assert first == again and comfy.runs == 1
    assert list(cache.glob("*.part")) == [] and len(list(cache.glob("*.glb"))) == 1


def test_another_seed_crop_or_fov_is_generated(tmp_path, monkeypatch):
    monkeypatch.delenv("AOMMA_FRESH_MODELS", raising=False)
    comfy, wf, cache = FakeComfy(), _workflow(tmp_path), tmp_path / "cache"
    crop = Image.new("RGB", (8, 8), (10, 20, 30))
    asset.generate_model(comfy, wf, crop, 30.0, "a", cache=cache)
    asset.generate_model(comfy, wf, crop, 30.0, "a", attempt=1, cache=cache)
    asset.generate_model(comfy, wf, crop, 31.0, "a", cache=cache)
    asset.generate_model(comfy, wf, Image.new("RGB", (8, 8), (10, 20, 31)), 30.0, "a", cache=cache)
    assert comfy.runs == 4


def test_fresh_models_skip_the_cache(tmp_path, monkeypatch):
    comfy, wf, cache = FakeComfy(), _workflow(tmp_path), tmp_path / "cache"
    crop = Image.new("RGB", (8, 8))
    asset.generate_model(comfy, wf, crop, 30.0, "a", cache=cache)
    monkeypatch.setenv("AOMMA_FRESH_MODELS", "1")
    assert asset.generate_model(comfy, wf, crop, 30.0, "a", cache=cache) == b"model2"
