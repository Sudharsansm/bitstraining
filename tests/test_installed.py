"""Release guard: when BT_EXPECT_INSTALLED=1 (the release workflow), the tests must be exercising the
installed wheel, not the source checkout. Skipped in normal development."""

import os

import pytest


@pytest.mark.skipif(os.environ.get("BT_EXPECT_INSTALLED") != "1", reason="only enforced in the release workflow")
def test_tests_are_running_against_the_installed_package():
    import bitstraining

    assert "site-packages" in bitstraining.__file__.replace("\\", "/"), bitstraining.__file__
