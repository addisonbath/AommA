"""The model bank (lab/model_bank.toml): a scan's draws, and copies of two draws of one shot made together."""

import importlib.util
import tomllib
from pathlib import Path

import pytest

LAB = Path(__file__).parents[1] / "lab"
if not LAB.is_dir():
    pytest.skip("needs the project's private lab/ folder", allow_module_level=True)
spec = importlib.util.spec_from_file_location("condition_copies", LAB / "condition_copies.py")
condition_copies = importlib.util.module_from_spec(spec)
spec.loader.exec_module(condition_copies)

BANK = """
[scans.Scan-BedroomScan-1.draws]
A = ["runs/a-BedroomScan-005", "runs/a-BedroomScan-006"]
C = ["runs/c-BedroomScan-005", "runs/c-BedroomScan-006"]

[scans.Scan-KitchenScan-2.draws]
A = ["runs/a-KitchenScan-005"]
C = ["runs/c-KitchenScan-005"]
"""


@pytest.fixture
def bank(tmp_path):
    path = tmp_path / "bank.toml"
    path.write_text(BANK)
    return path


def test_a_scan_gives_every_draw_and_at_names_one(bank):
    every = condition_copies.from_bank("BedroomScan", bank)
    assert [p.name for p in every] == ["a-BedroomScan-005", "a-BedroomScan-006", "c-BedroomScan-005", "c-BedroomScan-006"]
    assert [p.name for p in condition_copies.from_bank("BedroomScan@C", bank)] == ["c-BedroomScan-005", "c-BedroomScan-006"]
    assert len(condition_copies.from_bank("all@A", bank)) == 3


def test_an_unknown_scan_or_draw_is_refused(bank):
    with pytest.raises(SystemExit):
        condition_copies.from_bank("GarageScan", bank)
    with pytest.raises(SystemExit):
        condition_copies.from_bank("BedroomScan@B", bank)


def test_two_draws_of_one_shot_copied_together_get_different_names():
    stamp = "20261122-090000"
    a = condition_copies.copy_name(Path("runs/20261120-162900-BedroomScan-005"), {}, "depth", stamp)
    c = condition_copies.copy_name(Path("runs/20261121-160509-BedroomScan-005"), {"seed_offset": 100}, "depth", stamp)
    assert a == f"{stamp}-BedroomScan-005-depth"
    assert c == f"{stamp}-BedroomScan-005-s100-depth"
    # A copy of a copy keeps one tag.
    again = condition_copies.copy_name(Path(f"runs/{c}"), {"seed_offset": 100}, "depth+roomplan", stamp)
    assert again == f"{stamp}-BedroomScan-005-s100-depth-depth+roomplan"


def test_the_real_bank_has_both_draws_for_every_scan():
    scans = tomllib.loads((LAB / "model_bank.toml").read_text())["scans"]
    for scan, entry in scans.items():
        assert set(entry["draws"]) == {"A", "C"}, scan
        assert len(entry["draws"]["A"]) == len(entry["draws"]["C"]), scan
