"""
Packaging metadata consistency.
"""

import pathlib
import re

import samatch


def test_version_is_importable_and_matches_pyproject():
    assert samatch.__version__

    pyproject = (
        pathlib.Path(__file__).resolve().parent.parent / "pyproject.toml"
    ).read_text()

    name = re.search(r'^name = "(.+)"', pyproject, re.MULTILINE).group(1)
    version = re.search(r'^version = "(.+)"', pyproject, re.MULTILINE).group(1)

    assert name == "samatch"
    assert samatch.__version__ == version


def test_citation_version_matches_pyproject():
    root = pathlib.Path(__file__).resolve().parent.parent

    pyproject_version = re.search(
        r'^version = "(.+)"', (root / "pyproject.toml").read_text(), re.MULTILINE
    ).group(1)
    citation_version = re.search(
        r"^version: (.+)$", (root / "CITATION.cff").read_text(), re.MULTILINE
    ).group(1)

    assert citation_version.strip() == pyproject_version
