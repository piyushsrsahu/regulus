"""Smoke tests that run the real Streamlit script headlessly via streamlit.testing."""
import pytest
from streamlit.testing.v1 import AppTest

from src import PROJECT_ROOT
from src.db import load_history

APP = str(PROJECT_ROOT / "app.py")


def _app_on_samples() -> AppTest:
    at = AppTest.from_file(APP, default_timeout=120).run()
    assert not at.exception, [e.value for e in at.exception]
    at.radio[0].set_value("Select Bundled Sample").run()
    assert not at.exception, [e.value for e in at.exception]
    return at


@pytest.mark.parametrize("sample, expected", [
    ("sample_kepler_transit.csv", "exoplanet"),
    ("sample_quiet_star.csv", "no_exoplanet"),
    ("sample_variable_star.csv", "no_exoplanet"),
])
def test_app_analyzes_each_sample_without_errors(sample, expected):
    at = _app_on_samples()
    at.selectbox[0].set_value(sample).run()

    assert not at.exception, [e.value for e in at.exception]
    assert not at.error, [e.value for e in at.error]
    assert at.metric, "result metric cards should be rendered"

    latest = load_history()[0]
    assert latest["filename"] == sample
    assert latest["prediction"] == expected
