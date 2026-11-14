"""
Packaging metadata consistency.
"""

import pathlib
import re

import samatch


def _pyproject_version(root):
    pyproject = (root / "pyproject.toml").read_text()

    name = re.search(r'^name = "(.+)"', pyproject, re.MULTILINE).group(1)
    version = re.search(r'^version = "(.+)"', pyproject, re.MULTILINE).group(1)

    return name, version


def test_version_is_importable_and_matches_pyproject():
    assert samatch.__version__

    root = pathlib.Path(__file__).resolve().parent.parent
    name, version = _pyproject_version(root)

    assert name == "samatch"

    if samatch.__file__.startswith(str(root)):
        # Running from a source checkout. `importlib.metadata` may resolve to
        # an unrelated samatch installed elsewhere (for example a stale copy
        # in site-packages), so the importable version is not what this
        # checkout declares. The constant that *is* under the checkout's
        # control is the in-code fallback, which a checkout with no samatch
        # installed at all reports; that is what must track pyproject.
        init_source = (root / "samatch" / "__init__.py").read_text()
        fallback = re.search(
            r'__version__ = "([^"]+)"', init_source
        ).group(1)

        assert fallback == version, (
            f"the in-code fallback version {fallback!r} does not match "
            f"pyproject's {version!r}"
        )

    else:
        # Imported from an installation; it must be this checkout's build.
        assert samatch.__version__ == version


def test_citation_version_matches_pyproject():
    root = pathlib.Path(__file__).resolve().parent.parent

    _, pyproject_version = _pyproject_version(root)
    citation_version = re.search(
        r"^version: (.+)$", (root / "CITATION.cff").read_text(), re.MULTILINE
    ).group(1)

    assert citation_version.strip() == pyproject_version
