"""Tests marked `lab` exercise the project's own experiment tooling and records in lab/, which only the private
development repo holds; they are skipped where lab/ is absent."""

from pathlib import Path

import pytest

LAB = Path(__file__).resolve().parents[1] / "lab"


def pytest_configure(config):
    config.addinivalue_line("markers", "lab: needs the project's private lab/ folder")


def pytest_collection_modifyitems(config, items):
    if LAB.is_dir():
        return
    skip = pytest.mark.skip(reason="needs the project's private lab/ folder")
    for item in items:
        if "lab" in item.keywords:
            item.add_marker(skip)
