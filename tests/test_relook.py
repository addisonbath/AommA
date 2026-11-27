import json

import numpy as np
from PIL import Image

from aomma.labels import SETTLED_GROWN
from aomma.stages.relook import ROLES, SCHEMA, add_to_surveys, flags, parse, take_out_earlier


def test_roles_are_the_surveys_and_parsed():
    assert ROLES == ("built_in", "large", "small")
    assert SCHEMA["properties"]["objects"]["items"]["properties"]["role"]["enum"] == list(ROLES)
    text = json.dumps({"objects": [
        {"id": "window", "role": "room", "bbox_2d": [0, 0, 10, 10]},
        {"id": "lamp", "role": "small", "bbox_2d": [0, 0, 10, 10]},
        {"id": "ghost", "role": "furniture", "bbox_2d": [0, 0, 10, 10]},
    ]})
    assert [o["id"] for o in parse(text)] == ["lamp"]


def test_flags_unclaimed_segments_only(tmp_path):
    segments = np.zeros((100, 100), np.int32)
    segments[:20, :20] = 1  # all unknown: flagged
    segments[50:70, 50:70] = 2  # all claimed: not flagged
    segments[0, 99] = 3  # under MIN_SHARE: not read
    labels = np.full((100, 100), 5, np.uint8)
    labels[:20, :10] = 0
    settled = np.zeros((100, 100), np.uint8)
    settled[:20, 10:20] = SETTLED_GROWN  # grown counts as unknown before growth
    labels[0, 99] = 0
    np.savez_compressed(tmp_path / "panoptic.npz", segments=segments, info="{}")
    Image.fromarray(labels).save(tmp_path / "labels.png")
    (tmp_path / "debug").mkdir()
    Image.fromarray(settled).save(tmp_path / "debug" / "labels_settled.png")
    found = flags(tmp_path)
    assert [f["segment"] for f in found] == [1]
    assert found[0]["unknown_before"] == 1.0


def test_run_again_takes_out_earlier_entries(tmp_path):
    (tmp_path / "masks").mkdir()
    (tmp_path / "survey.json").write_text(json.dumps({"room_type": "kitchen", "built_ins": [{"id": "counter"}]}))
    (tmp_path / "small_survey.json").write_text(json.dumps({"objects": [{"id": "cup"}]}))
    entry = {"id": "lamp", "role": "small", "bbox_2d": [1, 2, 3, 4], "description": "", "size_m": [0.1] * 3,
             "on": "floor", "count": 1}
    add_to_surveys(tmp_path, [entry, entry | {"id": "faucet", "role": "built_in"}])
    for oid in ("lamp", "faucet", "cup"):
        Image.fromarray(np.zeros((4, 4), bool)).save(tmp_path / "masks" / f"{oid}.png")
    assert [e["id"] for e in json.loads((tmp_path / "small_survey.json").read_text())["objects"]] == ["cup", "lamp"]

    assert take_out_earlier(tmp_path)
    assert json.loads((tmp_path / "small_survey.json").read_text())["objects"] == [{"id": "cup"}]
    survey = json.loads((tmp_path / "survey.json").read_text())
    assert survey["built_ins"] == [{"id": "counter"}] and survey["room_type"] == "kitchen"
    assert sorted(p.stem for p in (tmp_path / "masks").glob("*.png")) == ["cup"]
    assert not take_out_earlier(tmp_path)


def test_kept_entries_keep_their_kind(tmp_path):
    (tmp_path / "survey.json").write_text(json.dumps({"built_ins": [], "kinds": {"window": "window"}}))
    (tmp_path / "large_survey.json").write_text(json.dumps({"objects": [{"id": "bed"}], "kinds": {"bed": "bed"}}))
    base = {"bbox_2d": [1, 2, 3, 4], "description": "", "size_m": [0.1] * 3, "on": "floor", "count": 1}
    add_to_surveys(tmp_path, [base | {"id": "nightstand", "role": "large", "kind": "nightstand"},
                              base | {"id": "sink", "role": "built_in", "kind": "sink"},
                              base | {"id": "lamp", "role": "small", "kind": "other"}])
    large = json.loads((tmp_path / "large_survey.json").read_text())
    assert large["kinds"] == {"bed": "bed", "nightstand": "nightstand"}
    assert large["objects"][1]["kind"] == "nightstand"
    assert json.loads((tmp_path / "survey.json").read_text())["kinds"] == {"window": "window", "sink": "sink"}
    assert "kinds" not in json.loads((tmp_path / "small_survey.json").read_text())

    assert take_out_earlier(tmp_path)
    assert json.loads((tmp_path / "large_survey.json").read_text())["kinds"] == {"bed": "bed"}
    assert json.loads((tmp_path / "survey.json").read_text())["kinds"] == {"window": "window"}


def test_prompt_names_each_roles_kinds():
    from aomma.stages.large import KINDS as LARGE_KINDS
    from aomma.stages.relook import KINDS, PROMPT
    from aomma.stages.shell import BUILT_IN_KINDS

    assert KINDS["large"] == LARGE_KINDS and KINDS["built_in"] == BUILT_IN_KINDS
    text = PROMPT.format(room_type="kitchen", listed="- nothing", built_in_kinds='"sink"', large_kinds='"nightstand"')
    assert '"kind": for a built-in one of "sink"' in text
