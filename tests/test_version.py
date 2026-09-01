"""The release workflow owns the version, so main does not carry a number.

No GPU and no model: this is a file and a regex. Until the workflow learned to
write ``wyoming_whisper_trt/VERSION``, somebody had to remember to bump it by
hand, and after 1.1.0 nobody did -- so 1.2.0, 1.2.1 and 1.2.2 each shipped a
package reporting 1.1.0, which is the version the Wyoming info message hands
to Home Assistant.
"""

import re
from pathlib import Path

import pytest

from wyoming_whisper_trt import __version__

_WORKFLOW = Path(__file__).parent.parent / ".github" / "workflows" / "release.yaml"

# Every tag the repository has actually released. The pattern in the workflow
# rejects the branch outright, so validating it against invented examples only
# would prove it accepts what I imagined rather than what we ship.
_RELEASED = [
    "1.0.3",
    "1.0.4",
    "1.0.5",
    "1.0.6",
    "1.0.7",
    "1.0.8",
    "1.0.9",
    "1.0.10",
    "1.0.11",
    "1.0.12",
    "1.0.13",
    "1.1.1",
    "1.1.2",
    "1.1.3",
    "1.1.4",
    "1.1.5",
    "1.1.6",
    "1.2.0",
    "1.2.1",
    "1.2.2",
]


def _release_version_pattern() -> str:
    """The pattern the workflow validates a release branch name with.

    Read out of the workflow rather than copied, so the two cannot drift apart
    silently. It is an ERE fed to ``grep -E``; the constructs used here mean
    the same thing to Python's ``re``.
    """
    text = _WORKFLOW.read_text(encoding="utf-8")
    match = re.search(r"VERSION_PATTERN='([^']+)'", text)
    assert match, f"no VERSION_PATTERN assignment in {_WORKFLOW.name}"
    return match.group(1)


def test_main_carries_a_placeholder_rather_than_a_release_number() -> None:
    """A real number here is a hand-edit the release branch will overwrite."""
    assert __version__ == "0.0.0.dev0"


def test_the_placeholder_is_not_mistaken_for_a_release_version() -> None:
    assert not re.match(_release_version_pattern(), __version__)


@pytest.mark.parametrize("version", _RELEASED)
def test_every_version_released_so_far_is_accepted(version: str) -> None:
    assert re.match(_release_version_pattern(), version)


@pytest.mark.parametrize("version", ["rc1", "1.2", "v1.2.3", "1.2.3.4", ""])
def test_a_branch_that_does_not_name_a_version_is_rejected(version: str) -> None:
    """release/<anything> is pushable; the tag and the package name come from it."""
    assert not re.match(_release_version_pattern(), version)
