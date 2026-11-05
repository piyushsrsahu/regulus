"""Unit tests for the SQLite persistence layer (runs against the isolated test DB from conftest)."""
import pytest

from src import PROJECT_ROOT
from src.db import DB_PATH as REAL_DB_PATH, save_analysis, load_history, clear_history
import src.db


def _result(prediction="exoplanet", confidence=0.9, depth=0.01, period=3.5, power=0.008):
    return {"prediction": prediction, "confidence": confidence, "depth": depth,
            "period": period, "bls_power": power}


def test_tests_use_isolated_database():
    """Guard: the autouse fixture must have redirected writes away from the real regulus.db."""
    assert src.db.DB_PATH != REAL_DB_PATH
    assert src.db.DB_PATH != str(PROJECT_ROOT / "regulus.db")


def test_save_and_load_round_trip():
    save_analysis("star_a.csv", _result(confidence=0.875, depth=0.0123, period=4.2, power=0.0071))
    (row,) = load_history()

    assert row["filename"] == "star_a.csv"
    assert row["prediction"] == "exoplanet"
    assert row["confidence"] == pytest.approx(0.875)
    assert row["transit_depth"] == pytest.approx(0.0123)
    assert row["period"] == pytest.approx(4.2)
    assert row["bls_power"] == pytest.approx(0.0071)
    assert row["created_at"]


def test_history_is_newest_first():
    for name in ("first.csv", "second.csv", "third.csv"):
        save_analysis(name, _result())
    assert [r["filename"] for r in load_history()] == ["third.csv", "second.csv", "first.csv"]


def test_clear_history_removes_all_rows():
    save_analysis("x.csv", _result(prediction="no_exoplanet"))
    clear_history()
    assert load_history() == []
