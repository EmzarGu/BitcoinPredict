import asyncio
from datetime import datetime, timedelta, timezone

import pytest

from src import backfill, ingest


def test_backfill_runs(monkeypatch):
    calls = []

    async def fake_ingest(week_anchor=None):
        calls.append(week_anchor)

    monkeypatch.setattr(ingest, "ingest_weekly", fake_ingest)

    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    end = start + timedelta(days=14)
    backfill.backfill(start, end)

    expected = [start + timedelta(days=7 * i) for i in range(3)]
    assert calls == expected
