import json

import pytest

from aomma import lab

LIMITS = {
    "upsample_stage.target_resolution": {"value": 1536, "lowest": 1024, "step": 128, "base": 1024, "whole": True},
    "texture_sample.steps": {"value": 12, "lowest": 3.0, "step": None, "base": 1, "whole": True},
}


def test_a_setting_can_only_be_lowered_within_its_range():
    assert lab.check_settings({"upsample_stage.target_resolution": 1280, "texture_sample.steps": 6.0}, LIMITS) == {
        "upsample_stage.target_resolution": 1280, "texture_sample.steps": 6,
    }
    for changes, complaint in [
        ({"structure_sample.seed": 1}, "not a setting"),
        ({"upsample_stage.target_resolution": 2048}, "between 1024 and 1536"),
        ({"upsample_stage.target_resolution": 512}, "between 1024 and 1536"),
        ({"upsample_stage.target_resolution": 1100}, "multiple of 128"),
        ({"texture_sample.steps": 2}, "between 3 and 12"),
        ({"texture_sample.steps": 6.5}, "whole number"),
        ({"texture_sample.steps": "few"}, "must be a number"),
    ]:
        with pytest.raises(ValueError, match=complaint):
            lab.check_settings(changes, LIMITS)


def test_changed_settings_go_into_a_copy_of_the_workflow():
    workflow = {"texture_sample": {"inputs": {"steps": 12, "seed": 43}}}
    changed = lab.with_settings(workflow, {"texture_sample.steps": 6})
    assert changed["texture_sample"]["inputs"] == {"steps": 6, "seed": 43}
    assert workflow["texture_sample"]["inputs"]["steps"] == 12


def test_a_trial_copy_leaves_the_run_alone_and_keeps_its_own_models_when_the_run_replaces_them(tmp_path):
    run = tmp_path / "run"
    for path, text in [
        ("layout.json", "{}"), ("masks/desk.png", "m"), ("agent.jsonl", "x"), ("debug/compare.jpg", "x"),
        ("assets/desk/desk.glb", "old"), ("assets/lamp/lamp.glb", "lamp glb"), ("assets/lamp/lamp.usdc", "lamp"),
        ("assets/lamp/textures/Image_0.png", "lamp texture"), ("assets/floor.usda", "f"),
        ("assets_replaced/desk.1/desk.glb", "older"),
    ]:
        (run / path).parent.mkdir(parents=True, exist_ok=True)
        (run / path).write_text(text)
    trial = tmp_path / "trial"
    lab.copy_run(run, trial, "desk")

    assert sorted(p.name for p in trial.iterdir()) == ["assets", "debug", "layout.json", "masks"]
    # The object under test starts with no model; the others are copied, without the GLBs nothing reads.
    assert list((trial / "assets" / "desk").iterdir()) == [] and list((trial / "debug").iterdir()) == []
    lamp = trial / "assets" / "lamp"
    assert not lamp.is_symlink() and (lamp / "lamp.usdc").read_text() == "lamp"
    assert (lamp / "textures" / "Image_0.png").read_text() == "lamp texture" and not (lamp / "lamp.glb").exists()
    assert (trial / "assets" / "floor.usda").read_text() == "f"
    assert (run / "assets" / "desk" / "desk.glb").read_text() == "old"

    # The run replaces the lamp's model the way a review does: the old folder is moved
    # aside, and for a while there is no model, then a new one lands.
    (run / "assets_replaced" / "lamp.1").parent.mkdir(exist_ok=True)
    (run / "assets" / "lamp").rename(run / "assets_replaced" / "lamp.1")
    assert (lamp / "lamp.usdc").read_text() == "lamp"
    (run / "assets" / "lamp").mkdir()
    (run / "assets" / "lamp" / "lamp.usdc").write_text("new lamp")
    # The trial still reads the model it started with.
    assert (lamp / "lamp.usdc").read_text() == "lamp"

    # A trial that starts while the run has set the lamp's model aside, before the new one lands, copies the one set aside.
    (run / "assets" / "lamp" / "lamp.usdc").unlink()
    (run / "assets_replaced" / "lamp.2").mkdir()
    (run / "assets_replaced" / "lamp.2" / "lamp.usdc").write_text("lamp two")
    later = tmp_path / "later"
    lab.copy_run(run, later, "desk")
    assert (later / "assets" / "lamp" / "lamp.usdc").read_text() == "lamp two"


def test_the_worker_is_shown_results_and_pictures_until_it_finishes():
    def call(name, **arguments):
        return {"role": "assistant", "content": "", "thinking": "hm", "tool_calls": [{"function": {"name": name, "arguments": arguments}}]}

    replies = iter([
        {"role": "assistant", "content": "My plan: lower the resolution."},
        call("run_trial", object_id="desk", settings={"a.b": 1}, reason="is it faster"),
        call("finish", report="Yes."),
    ])
    seen = []

    def act(name, arguments):
        seen.append((name, arguments))
        return ({"trial": 1, "generation_seconds": 600}, b"jpeg", False) if name == "run_trial" else ({}, None, True)

    messages = [{"role": "user", "content": "brief"}]
    log = []
    assert lab.converse(lambda _: next(replies), messages, act, 5, lambda **entry: log.append(entry))
    assert [name for name, _ in seen] == ["run_trial", "finish"]
    # A reply with no tool call is answered with a reminder; a result is followed by its picture.
    assert [m["role"] for m in messages] == ["user", "assistant", "user", "assistant", "tool", "user", "assistant"]
    assert json.loads(messages[4]["content"])["generation_seconds"] == 600 and messages[5]["images"]
    assert all("thinking" not in m for m in messages) and log[1]["thinking"] == "hm"
    # A worker that never finishes is stopped when its turns run out.
    assert not lab.converse(lambda _: {"role": "assistant", "content": "thinking..."}, [], act, 3, lambda **entry: None)


def test_the_report_sets_the_measurements_beside_the_findings():
    sits = {
        "outline_fit": 0.6, "colour_error": 0.07, "hidden_by_wall": 0.98, "off_floor_m": 0.0, "longest_side_m": 0.96,
        "distance_vs_photo": 1.3, "size_vs_estimate": 0.64, "inside_other": 0.0, "inside_which": None,
    }
    models = {"desk": {"description": "a desk", "generation_seconds": 3076, **sits}}
    trials = [
        {"trial": 1, "kind": "run_trial", "object_id": "desk", "generator": "trellis2", "settings": {"upsample_stage.target_resolution": 1024},
         "reason": "r", "generation_seconds": 1200, **sits, "outline_fit": 0.58, "picture": "t1_desk.jpg"},
        {"trial": 2, "kind": "run_trial", "object_id": "desk", "generator": "pixal3d", "settings": {}, "reason": "r", "error": "workflow failed"},
        {"trial": 3, "kind": "pose_trial", "object_id": "desk", "generator": "pixal3d", "settings": {"wall_weight": 1.0, "pull_forward": True},
         "reason": "r", **sits, "hidden_by_wall": 0.01, "off_floor_m": 0.2, "picture": "t3_desk.jpg"},
    ]
    report = lab._report("Faster?", lab.Path("runs/office"), "pixal3d", models, trials, "It is faster.", True)
    assert "| current | | desk | pixal3d | as run | 3076 | 0.6 | 0.07 | 0.98 | 0.0 | 0.96 | 1.3 | 0.64 | 0.0 | None | |" in report
    assert "| 1 | generate | desk | trellis2 | upsample_stage.target_resolution = 1024 | 1200 | 0.58 | 0.07 | 0.98 | 0.0 | 0.96 | 1.3 | 0.64 | 0.0 | None | t1_desk.jpg |" in report
    assert "| 2 | generate | desk | pixal3d | none | failed: workflow failed |" in report
    assert "| 3 | pose | desk | pixal3d | wall_weight = 1.0, pull_forward = True |  | 0.6 | 0.07 | 0.01 | 0.2 | 0.96 | 1.3 | 0.64 | 0.0 | None | t3_desk.jpg |" in report
    assert report.rstrip().endswith("It is faster.")
    assert "did not finish" in lab._report("Faster?", lab.Path("runs/office"), "pixal3d", models, trials, "", False)


def test_pose_settings_are_checked_before_a_trial_is_spent():
    assert lab.check_pose_settings({}) == lab.PoseSettings()
    assert lab.check_pose_settings({"wall_weight": 2, "pull_forward": True}) == lab.PoseSettings(wall_weight=2.0, pull_forward=True)
    assert lab.check_pose_settings({"reseat": "stretch"}).reseat == "stretch"
    assert lab.check_pose_settings({"contact": "trim", "depth_weight": 1, "size_weight": 0.5}) == lab.PoseSettings(
        contact="trim", depth_weight=1.0, size_weight=0.5,
    )
    assert lab.check_pose_settings({"turn_deg": -90, "fit_box": False}) == lab.PoseSettings(turn_deg=-90.0, fit_box=False)
    for changes, complaint in [
        ({"wall_wait": 1}, "not a pose setting"),
        ({"wall_weight": 5}, "between 0 and 3"),
        ({"wall_weight": "high"}, "between 0 and 3"),
        ({"pull_forward": "yes"}, "true or false"),
        ({"reseat": "lift"}, '"none", "drop" or "stretch"'),
        ({"contact": "middle"}, '"lowest" or "trim"'),
        ({"depth_weight": -1}, "depth_weight must be a number between 0 and 3"),
        ({"size_weight": True}, "size_weight must be a number"),
        ({"turn_deg": 270}, "turn_deg must be a number between -180 and 180"),
        ({"fit_box": "no"}, "fit_box must be true or false"),
    ]:
        with pytest.raises(ValueError, match=complaint):
            lab.check_pose_settings(changes)


def test_every_image_to_3d_workflow_in_the_folder_is_a_generator_and_shared_settings_are_listed_once(tmp_path):
    node = {"class_type": "X", "inputs": {}}
    for name, workflow in [
        ("zeta_3d", {"input_image": node, "output_model": node}),
        ("alpha_3d", {"input_image": node, "output_model": node}),
        ("isolate", {"input_image": node, "output_image": node}),
    ]:
        (tmp_path / f"{name}.json").write_text(json.dumps(workflow))
    config = lab.Config(
        vlm="local", vlms={}, comfy={"light": [], "heavy": []},
        image_to_3d_workflow=tmp_path / "zeta_3d.json", repair_workflow=tmp_path / "isolate.json",
    )
    # The one the runs use comes first; a workflow that makes no model is left out.
    assert list(lab.generators(config)) == ["zeta_3d", "alpha_3d"]

    same = lab._describe_settings({"zeta_3d": LIMITS, "alpha_3d": LIMITS})
    assert same.count("texture_sample.steps") == 1 and "Every generator has these settings" in same
    other = {"texture_sample.steps": {**LIMITS["texture_sample.steps"], "value": 8}}
    apart = lab._describe_settings({"zeta_3d": LIMITS, "alpha_3d": other})
    assert "For zeta_3d:" in apart and "For alpha_3d:" in apart and "from 3 to 8" in apart


def test_a_crop_is_set_on_the_background_its_workflow_asks_for(tmp_path):
    import numpy as np
    from PIL import Image

    from aomma.stages.asset import crop_background, crop_object

    node = {"class_type": "X", "inputs": {}}
    (tmp_path / "plain.json").write_text(json.dumps({"input_image": node}))
    (tmp_path / "white.json").write_text(json.dumps({"input_image": node, "input_background": {"class_type": "PrimitiveString", "inputs": {"value": "white"}}}))
    assert crop_background(tmp_path / "plain.json") == "black" and crop_background(tmp_path / "white.json") == "white"

    # A red object at the photo's left edge, with a green neighbour inside its square.
    photo = Image.new("RGB", (100, 100), "blue")
    photo.paste((255, 0, 0), (0, 40, 20, 60))
    photo.paste((0, 255, 0), (22, 40, 30, 60))
    mask = np.zeros((100, 100), dtype=bool)
    mask[40:60, 0:20] = True
    for background, colour in (("black", (0, 0, 0)), ("white", (255, 255, 255))):
        crop, side = crop_object(photo, mask, size=110, background=background)
        assert side == 22 and crop.getpixel((55, 55)) == (255, 0, 0)
        # The neighbour, the rest of the photo and the part of the square beyond the photo's edge are all background.
        assert crop.getpixel((107, 55)) == colour and crop.getpixel((55, 2)) == colour and crop.getpixel((1, 55)) == colour


def test_the_first_failed_trials_do_not_count_against_the_budget():
    trials = [{"kind": "run_trial"}, {"kind": "run_trial", "error": "x"}, {"kind": "pose_trial", "error": "x"}]
    assert lab._spent(trials, "run_trial") == 1 and lab._spent(trials, "pose_trial") == 0
    trials += [{"kind": "run_trial", "error": "x"}, {"kind": "run_trial", "error": "x"}]
    assert lab._spent(trials, "run_trial") == 2  # two failures were free; the third counts
