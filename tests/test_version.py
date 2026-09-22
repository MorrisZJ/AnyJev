"""The version string must not drift from pyproject.toml, which is what PyPI serves."""
import re
from pathlib import Path

import anyjev


def test_version_is_a_release_string():
    assert re.fullmatch(r"\d+\.\d+\.\d+(\.\w+)?", anyjev.__version__), anyjev.__version__


def test_version_matches_pyproject_when_running_from_a_checkout():
    pyproject = Path(__file__).resolve().parent.parent / "pyproject.toml"
    if not pyproject.exists():
        return  # installed without the source tree
    declared = re.search(r'^version = "([^"]+)"', pyproject.read_text(), re.M).group(1)
    assert anyjev.__version__ == declared, (
        f"anyjev.__version__ is {anyjev.__version__} but pyproject.toml declares {declared}; "
        "bump both or the installed package reports the wrong version"
    )
