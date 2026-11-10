"""pytest wrapper: `python -m pytest dimuon/pipeline/test_candidates.py -q` runs candidates.selftest()."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import candidates  # noqa: E402


def test_selftest():
    assert candidates.selftest()
