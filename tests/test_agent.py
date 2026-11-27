import numpy as np
from PIL import Image

from aomma.agent import STAGED_PLANS, Facts, applies, plan
from aomma.stages.critic import Finding, object_view, parse_finding


def test_the_planner_picks_the_first_action_that_applies_is_untried_and_is_affordable():
    facts = {
        "shelf": Facts(on="floor", area_px=9000),
        "poster": Facts(on="wall", area_px=5000),
        "nightstand": Facts(on="floor", area_px=800),
        "pillows": Facts(on="bed", area_px=3000),
        "lamp": Facts(on="nightstand", area_px=400),
    }
    findings = [
        Finding(id="lamp", verdict="wrong_place"),
        Finding(id="nightstand", verdict="wrong_shape"),
        Finding(id="pillows", verdict="several_objects"),
        Finding(id="poster", verdict="wrong_side"),
        Finding(id="shelf", verdict="wrong_shape"),
    ]
    steps = plan(findings, facts, tried=set(), generations_left=6)
    # Biggest first; a misplaced object in plain view has no action.
    assert [(f.id, action) for f, action in steps] == [
        ("shelf", "repair"), ("poster", "flatten"), ("pillows", "relist"), ("nightstand", "repair"),
    ]

    # What was tried is not tried again, and the next choice must fit the budget.
    again = plan(findings, facts, tried={("shelf", "repair"), ("poster", "flatten")}, generations_left=1)
    assert [(f.id, action) for f, action in again] == [("shelf", "regenerate")]
    assert plan(findings, facts, tried=set(), generations_left=0) == [(findings[3], "flatten")]


def test_actions_apply_only_where_they_make_sense():
    assert applies("flatten", Facts(on="wall", area_px=1)) and not applies("flatten", Facts(on="floor", area_px=1))
    assert not applies("flatten", Facts(on="wall", area_px=1, flat=True))
    assert not applies("repair", Facts(on="floor", area_px=1, repaired=True))
    assert not applies("regenerate", Facts(on="floor", area_px=1, flat=True))


def test_critic_replies_are_read_leniently_and_close_ups_line_up():
    assert parse_finding("vase", 'Sure: {"verdict": "wrong_side", "note": "shows a blank back"}').verdict == "wrong_side"
    # A reply that cannot be read, or names no known verdict, is not a complaint.
    assert parse_finding("vase", "it looks fine to me").verdict == "ok"
    assert parse_finding("vase", '{"verdict": "ugly"}').verdict == "ok"

    photo = Image.new("RGB", (200, 100), "white")
    render = Image.new("RGB", (200, 100), "white")
    photo.paste((255, 0, 0), (140, 40, 160, 60))
    render.paste((0, 0, 255), (140, 40, 160, 60))
    mask = np.zeros((100, 200), dtype=bool)
    mask[40:60, 140:160] = True
    view = object_view(photo, render, mask)
    half = view.width // 2
    # The object sits in the middle of each half: red in the photo's, blue in the render's.
    assert view.getpixel((half // 2, view.height // 2 + 12))[:3] == (255, 0, 0)
    assert view.getpixel((half + half // 2, view.height // 2 + 12))[:3] == (0, 0, 255)


def test_the_checks_complain_about_distance_size_and_collisions_one_complaint_per_object():
    from aomma.agent import measured_findings

    depth = {
        "bed": {"ratio": 0.65, "agree": 0.02, "too_near": 0.94, "too_far": 0.03, "pixels": 7080},
        "lamp": {"ratio": 0.96, "agree": 0.77, "too_near": 0.23, "too_far": 0.0, "pixels": 239},
        "cup": {"ratio": 0.5, "agree": 0.0, "too_near": 1.0, "too_far": 0.0, "pixels": 5},  # too few pixels to judge
    }
    sizes = {"bed": 1.4, "desk": 0.5, "lamp": 1.0}
    collisions = [
        {"objects": ["books", "console"], "litres": 2.7, "share_of_smaller": 0.5, "volumes": {"books": 5.4, "console": 120.0}},
        {"objects": ["pillow_a", "pillow_b"], "litres": 0.1, "share_of_smaller": 0.1, "volumes": {"pillow_a": 3.0, "pillow_b": 2.0}},
    ]
    found = {f.id: f for f in measured_findings(depth, sizes, collisions, {"bed", "lamp", "cup", "desk", "books", "console"})}
    # The bigger of two colliding objects is blamed; a small overlap is not a complaint.
    assert found["console"].verdict == "collides" and "books" in found["console"].note
    assert found["desk"].verdict == "wrong_size"
    assert found["bed"].verdict == "wrong_place" and "too near" in found["bed"].note  # 1.4 is within the size range
    assert set(found) == {"console", "desk", "bed"} and all(f.source == "checks" for f in found.values())
    # Objects not being judged this round are left alone.
    assert measured_findings(depth, sizes, collisions, {"lamp"}) == []


def test_a_model_covering_other_objects_is_complained_about_and_fixed_first():
    from aomma.agent import measured_findings

    depth = {
        "slipper": {"ratio": 1.04, "agree": 1.0, "too_near": 0.0, "too_far": 0.0, "pixels": 179, "over_others": 68.4},
        "basket": {"ratio": 0.95, "agree": 0.93, "too_near": 0.04, "too_far": 0.03, "pixels": 1494, "over_others": 0.74},
        "washer": {"ratio": 1.12, "agree": 0.15, "too_near": 0.0, "too_far": 0.85, "pixels": 1389, "hidden_by_others": 0.82},
    }
    found = measured_findings(depth, {}, [], {"slipper", "basket", "washer"})
    assert {(f.id, f.verdict) for f in found} == {("slipper", "covers_others"), ("washer", "wrong_place")}
    # The small slipper is served before the big washer: until it is fixed, the washer cannot be judged.
    facts = {"slipper": Facts(on="floor", area_px=200), "washer": Facts(on="floor", area_px=120000, repaired=True)}
    assert [f.id for f, _ in plan(found, facts, set(), 6)] == ["slipper", "washer"]


def test_a_change_is_kept_by_outline_and_by_measurement():
    from aomma.agent import keep_change, trouble

    floating = {"outline_fit": 0.83, "distance_vs_photo": 0.66, "size_vs_estimate": 0.96, "inside_other": 0.0}
    on_the_floor = {**floating, "outline_fit": 0.82, "distance_vs_photo": 0.97}
    assert trouble(floating) > 0.3 and trouble(on_the_floor) == 0.0
    assert keep_change(floating, on_the_floor)  # same fit, the distance put right
    assert not keep_change(on_the_floor, floating)  # same fit, but now floating: the old check would have kept it
    # A big measured improvement may cost a little fit, not a lot.
    held = {**floating, "outline_fit": 0.75, "distance_vs_photo": 0.97}
    assert keep_change(floating, held) and not keep_change(floating, {**held, "outline_fit": 0.6})


def test_only_a_partly_hidden_objects_pose_is_tried_again():
    assert applies("repose", Facts(on="floor", area_px=1, repaired=True))
    assert not applies("repose", Facts(on="floor", area_px=1))
    assert not applies("repose", Facts(on="floor", area_px=1, repaired=True, reposed=True))
    steps = plan([Finding(id="bed", verdict="wrong_size", source="checks")], {"bed": Facts(on="floor", area_px=9, repaired=True)}, set(), 0)
    assert [action for _, action in steps] == ["repose"]  # costs no generation, so the budget does not stop it


def test_when_the_checks_complain_too_their_free_fix_is_tried_before_the_critics():
    facts = {"bed": Facts(on="floor", area_px=9, repaired=True)}
    findings = [
        Finding(id="bed", verdict="wrong_shape"),
        Finding(id="bed", verdict="wrong_place", source="checks"),
    ]
    steps = plan(findings, facts, set(), 1)
    # The critic's regeneration is kept as a fallback, for if reposing changes nothing.
    assert [(f.source, action) for f, action in steps] == [("checks", "repose"), ("critic", "regenerate")]
    assert [action for _, action in plan(findings, facts, {("bed", "repose")}, 1)] == ["regenerate"]
    # A complaint from the checks alone, with a costly fix, does not jump ahead of the critic's.
    collides = [findings[0], Finding(id="bed", verdict="collides", source="checks")]
    assert [action for _, action in plan(collides, facts, set(), 1)] == ["regenerate"]


def test_undoing_a_new_model_puts_back_the_old_pose_exactly(tmp_path, monkeypatch):
    from aomma import agent, cli

    (tmp_path / "assets" / "bed").mkdir(parents=True)
    (tmp_path / "assets" / "bed" / "bed.usdc").write_text("old model")
    layout = '{"objects": [], "posed with": "old settings"}'
    (tmp_path / "layout.json").write_text(layout)
    (tmp_path / "poses.json").write_text('[{"id": "bed", "iou": 0.83}]')
    facts = iter([{"outline_fit": 0.83}, {"outline_fit": 0.5}])

    def regenerate(run_dir, ids, config_path):
        (run_dir / "assets" / "bed").mkdir()
        (run_dir / "assets" / "bed" / "bed.usdc").write_text("new model")
        (run_dir / "layout.json").write_text('{"objects": [], "posed with": "new settings"}')
        (run_dir / "poses.json").write_text('[{"id": "bed", "iou": 0.5}]')

    rendered = []
    monkeypatch.setattr(cli, "placement_facts", lambda run_dir, object_id: next(facts))
    monkeypatch.setattr(cli, "overrides", lambda run_dir: {})
    monkeypatch.setattr(cli, "set_override", lambda run_dir, object_id, **changes: None)
    monkeypatch.setattr(cli, "assets", regenerate)
    monkeypatch.setattr(cli, "pose", lambda *args: rendered.append("posed again"))
    monkeypatch.setattr(agent.SceneLayout, "model_validate_json", staticmethod(lambda text: text))
    monkeypatch.setattr(cli, "_compose_and_render", lambda run_dir, scene: rendered.append(scene))

    outcome, generated, changed = agent._act(tmp_path, "bed", "regenerate", tmp_path / "config.toml", None)
    assert outcome.startswith("undone") and generated == 1 and changed == []
    assert (tmp_path / "assets" / "bed" / "bed.usdc").read_text() == "old model"
    assert (tmp_path / "layout.json").read_text() == layout and rendered == [layout]
    assert (tmp_path / "poses.json").read_text() == '[{"id": "bed", "iou": 0.83}]'


def test_an_object_behind_a_wall_is_complained_about_first_and_brought_out():
    from aomma.agent import keep_change, measured_findings

    depth = {"desk": {"ratio": 1.35, "agree": 0.31, "too_near": 0.0, "too_far": 0.69, "pixels": 2250}}
    found = measured_findings(depth, {"desk": 0.9}, [], {"desk"}, walled={"desk": 0.69})
    assert [(f.id, f.verdict) for f in found] == [("desk", "behind_wall")]
    assert measured_findings({}, {}, [], {"desk"}, walled={"desk": 0.1}) == []
    steps = plan(found, {"desk": Facts(on="floor", area_px=9, repaired=True)}, set(), 0)
    assert [action for _, action in steps] == ["pull_forward"]
    assert plan(found, {"desk": Facts(on="floor", area_px=9, pulled=True)}, set(), 0) == []
    # With the fit judged as the camera sees it, bringing the desk out is kept.
    hidden = {"outline_fit": 0.01, "distance_vs_photo": 1.5, "size_vs_estimate": 0.9, "hidden_by_wall": 0.69}
    out = {"outline_fit": 0.84, "distance_vs_photo": 1.12, "size_vs_estimate": 0.68, "hidden_by_wall": 0.02}
    assert keep_change(hidden, out) and not keep_change(out, hidden)


def test_a_staged_runs_large_items_are_fixed_by_holding_them_to_the_shell():
    from aomma.agent import STAGED_PLANS

    facts = {
        "cabinet": Facts(on="floor", area_px=900, staged=True),
        "desk": Facts(on="floor", area_px=800, staged=True),
        "fridge": Facts(on="floor", area_px=700, staged=True),
        "shelf": Facts(on="floor", area_px=600, staged=True),
    }
    findings = [
        Finding(id="cabinet", verdict="off_square", source="checks"),
        Finding(id="desk", verdict="left_out", source="checks"),
        Finding(id="fridge", verdict="off_support", source="checks"),
        Finding(id="shelf", verdict="behind_wall", source="checks"),
    ]
    steps = plan(findings, facts, set(), 1, STAGED_PLANS)
    # Posed again and held to the shell, never pulled out of a wall; a model left out is redrawn.
    assert [(f.id, action) for f, action in steps] == [
        ("cabinet", "square"), ("desk", "repair"), ("fridge", "repose"), ("shelf", "repose"),
    ]
    assert not applies("pull_forward", facts["shelf"]) and not applies("square", Facts(on="floor", area_px=1))


def test_a_large_items_trouble_counts_its_support_and_its_squareness():
    from aomma.agent import keep_change, trouble

    square = {"outline_fit": 0.89, "off_support_m": 0.0, "off_square_deg": 0.0}
    turned = {**square, "off_square_deg": 44.1}
    assert trouble(square) == 0.0 and trouble(turned) > 0.7
    assert trouble({**square, "off_support_m": 0.06}) > 0.05
    # A round thing has no squareness to count.
    assert trouble({**square, "off_square_deg": None}) == 0.0
    assert keep_change(turned, {**square, "outline_fit": 0.82})


def test_an_undone_fix_hands_on_to_the_complaints_next_action():
    # What the review asks for again after a fix was undone: the same complaint, with that fix tried.
    facts = {"bed": Facts(on="floor", area_px=9000, staged=True)}
    finding = Finding(id="bed", verdict="wrong_shape")
    assert [a for _, a in plan([finding], facts, tried={("bed", "repair")}, generations_left=2, plans=STAGED_PLANS)] == ["regenerate"]
    # With nothing left to try, or no budget, nothing is handed on.
    assert plan([finding], facts, tried={("bed", "repair"), ("bed", "regenerate")}, generations_left=2, plans=STAGED_PLANS) == []
    assert plan([finding], facts, tried={("bed", "repair")}, generations_left=0, plans=STAGED_PLANS) == []


def test_a_handed_on_fix_waits_for_the_generations_already_queued():
    from aomma.agent import hand_on

    # OfficeScan 007 (depth): two repairs still queued, two generations left; a third must not be planned.
    facts = {"crate": Facts(on="floor", area_px=9000, staged=True)}
    crate = Finding(id="crate", verdict="wrong_shape")
    queued = [(Finding(id="stand", verdict="wrong_shape"), "repair"), (Finding(id="table", verdict="wrong_shape"), "repair")]
    assert hand_on(crate, facts, {("crate", "repair")}, 2, queued, STAGED_PLANS) == []
    assert [a for _, a in hand_on(crate, facts, {("crate", "repair")}, 3, queued, STAGED_PLANS)] == ["regenerate"]


def test_a_fix_that_failed_may_be_tried_again_when_the_review_is_taken_up():
    from aomma.agent import tried_before

    # OfficeScan 006 (depth): jai dropped mid-review, so the repairs it was asked for made nothing.
    log = [
        {"round": 1, "kind": "action", "id": "desk", "action": "repair", "outcome": "undone: outline fit 0.56 -> 0.58", "generated": 1},
        {"round": 1, "kind": "action", "id": "sofa", "action": "repair", "outcome": "failed: no new model was produced; the old one is back", "generated": 0},
        {"round": 1, "kind": "action", "id": "amp", "action": "regenerate", "outcome": "failed: ConnectError", "generated": 0},
        {"round": 1, "kind": "skipped", "id": "amp", "verdict": "wrong_shape", "reason": "repair not needed"},
    ]
    assert tried_before(log) == {("desk", "repair")}
