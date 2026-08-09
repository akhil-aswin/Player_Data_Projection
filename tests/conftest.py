import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# config.py calls load_dotenv() + reads ODDS_API_KEY at import time. Force a
# harmless default so importing modules never depends on a real .env/key
# being present in the environment running the tests.
os.environ.setdefault("ODDS_API_KEY", "test-key")

import pytest


@pytest.fixture
def isolated_db(tmp_path, monkeypatch):
    """Point database.py at a throwaway sqlite file for the duration of a test."""
    import database

    db_path = tmp_path / "test_picks.db"
    monkeypatch.setattr(database, "DB_PATH", str(db_path))
    database.init_db()
    return database
