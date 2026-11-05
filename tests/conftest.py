"""Shared pytest fixtures."""
import pytest

import src.db


@pytest.fixture(autouse=True)
def isolated_db(tmp_path, monkeypatch):
    """Point the persistence layer at a throwaway SQLite file so tests never touch regulus.db."""
    db_file = tmp_path / "test_regulus.db"
    monkeypatch.setattr(src.db, "DB_PATH", str(db_file))
    src.db.init_db()
    return db_file
