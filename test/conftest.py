import pytest


@pytest.fixture(scope="session", autouse=True)
def _langdetect_deterministic() -> None:
    """langdetect samples profiles nondeterministically unless seeded."""
    import langdetect

    langdetect.DetectorFactory.seed = 0
