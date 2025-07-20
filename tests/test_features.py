from typing import Any

import numpy as np
import pandas as pd
import pytest

from src import features


def _sample_df(rows: int = 60) -> pd.DataFrame:
    dates = pd.date_range("2022-01-03", periods=rows, freq="W-MON", tz="UTC")
    data = {
        "week_start": dates,
        "close_usd": np.linspace(100, 200, rows),
        "realised_price": np.linspace(90, 180, rows),
        "nupl": np.linspace(0, 1, rows),
        "fed_liq": np.arange(rows, dtype=float),
        "ecb_liq": np.arange(rows, dtype=float),
        "dxy": np.linspace(90, 100, rows),
        "ust10": np.linspace(2, 3, rows),
        "gold_price": np.linspace(1500, 1600, rows),
        "spx_index": np.linspace(3000, 3100, rows),
    }
    return pd.DataFrame(data)


def _patch_csv(monkeypatch: Any, df: pd.DataFrame) -> None:
    def fake_read_csv(path: str) -> pd.DataFrame:
        assert path == "data/btc_weekly_latest.csv"
        return df

    monkeypatch.setattr(pd, "read_csv", fake_read_csv)


def test_feature_columns_present(monkeypatch):
    df = _sample_df()
    monkeypatch.delenv("DATABASE_URL", raising=False)
    _patch_csv(monkeypatch, df)

    feat = features.build_features(lookback_weeks=52)
    for col in features.FEATURE_COLS:
        assert col in feat.columns


def test_target_shift(monkeypatch):
    df = _sample_df(60)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    _patch_csv(monkeypatch, df)

    feat = features.build_features(lookback_weeks=52)
    if feat.empty:
        pytest.skip("no data produced")
    first_idx = feat.index[0]

    orig = df.set_index("week_start")
    target_series = orig["close_usd"].shift(-4) / orig["close_usd"] - 1
    expected = target_series.loc[first_idx]
    assert pytest.approx(feat.loc[first_idx, "Target"], rel=1e-9) == expected


def test_no_nans(monkeypatch):
    df = _sample_df(60)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    _patch_csv(monkeypatch, df)

    feat = features.build_features(lookback_weeks=52)
    assert feat[features.FEATURE_COLS].isna().sum().sum() == 0
