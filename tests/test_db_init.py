import pytest
from src import ingest


def test_init_db_commit_order(monkeypatch):
    calls = []

    class FakeCursor:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            pass

        def execute(self, sql):
            calls.append(("execute", sql))

    class FakeConn:
        def cursor(self):
            return FakeCursor()

        def commit(self):
            calls.append(("commit", None))

    def fake_execute_values(cur, sql, argslist, template=None):
        calls.append(("execute", sql))

    monkeypatch.setattr(ingest.psycopg2.extras, "execute_values", fake_execute_values)

    row = {col: 1 for col in ingest.SCHEMA_COLUMNS}
    ingest._init_db(FakeConn(), row)

    executed_sql = [c[1] for c in calls if c[0] == "execute"]
    create_idx = next(i for i, s in enumerate(executed_sql) if "CREATE TABLE IF NOT EXISTS btc_weekly" in s)
    insert_idx = next(i for i, s in enumerate(executed_sql) if "INSERT INTO btc_weekly" in s)
    assert create_idx < insert_idx

    commit_indices = [i for i, c in enumerate(calls) if c[0] == "commit"]
    insert_call_index = calls.index(("execute", executed_sql[insert_idx]))
    assert any(idx < insert_call_index for idx in commit_indices)
