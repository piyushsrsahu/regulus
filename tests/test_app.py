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


def test_sidebar_exposes_analysis_parameters():
    at = AppTest.from_file(APP, default_timeout=120).run()
    labels = [w.label for w in at.sidebar.slider] + [w.label for w in at.sidebar.number_input]
    assert labels == ["Detrending Window Length", "Min Search Period (days)", "Max Search Period (days)"]


def test_rerun_does_not_duplicate_history():
    """Streamlit reruns the script on every interaction; the same analysis must be logged once."""
    at = _app_on_samples()
    at.selectbox[0].set_value("sample_quiet_star.csv").run()
    n_after_first = len(load_history())
    at.run()
    at.run()
    assert len(load_history()) == n_after_first


def test_changing_parameters_logs_a_new_run():
    at = _app_on_samples()
    n_before = len(load_history())
    at.sidebar.slider[0].set_value(151).run()
    assert not at.exception, [e.value for e in at.exception]
    assert len(load_history()) == n_before + 1
