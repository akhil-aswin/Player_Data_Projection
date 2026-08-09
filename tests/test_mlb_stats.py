import datetime

import numpy as np
import pandas as pd
import pytest

from data.mlb_stats import (
    _day_weight,
    add_derived_stats,
    pitcher_weighted_average,
    weighted_recent_average,
)


class TestDayWeight:
    @pytest.mark.parametrize(
        "days_ago,expected",
        [
            (0, 1.0),
            (28, 1.0),
            (29, 0.6),
            (60, 0.6),
            (61, 0.3),
            (90, 0.3),
            (91, 0.1),
            (500, 0.1),
        ],
    )
    def test_tier_boundaries(self, days_ago, expected):
        assert _day_weight(days_ago) == expected


class TestAddDerivedStats:
    def test_sums_hits_runs_rbi(self):
        df = pd.DataFrame({"hits": [2, 1], "runs": [1, 0], "rbi": [3, 2]})
        out = add_derived_stats(df, "hitsRunsRbi")
        assert out["hitsRunsRbi"].tolist() == [6, 3]

    def test_missing_source_column_treated_as_zero(self):
        df = pd.DataFrame({"hits": [2, 1], "runs": [1, 0]})  # no rbi column
        out = add_derived_stats(df, "hitsRunsRbi")
        assert out["hitsRunsRbi"].tolist() == [3, 1]

    def test_non_derived_stat_returns_input_unchanged(self):
        df = pd.DataFrame({"hits": [2, 1]})
        out = add_derived_stats(df, "hits")
        assert out is df


class TestWeightedRecentAverage:
    def test_raises_on_missing_column(self):
        df = pd.DataFrame({"hits": [1, 2, 3]})
        with pytest.raises(ValueError):
            weighted_recent_average(df, "homeRuns")

    def test_raises_on_empty_dataframe(self):
        df = pd.DataFrame(columns=["hits"])
        with pytest.raises(ValueError):
            weighted_recent_average(df, "hits")

    def test_constant_values_give_that_mean_and_zero_std(self):
        df = pd.DataFrame({"hits": [2.0, 2.0, 2.0, 2.0]})
        result = weighted_recent_average(df, "hits", halflife_games=10.0)
        assert result["mean"] == pytest.approx(2.0)
        assert result["std"] == pytest.approx(0.0, abs=1e-9)
        assert result["n_games"] == 4

    def test_recent_games_weighted_more_heavily(self):
        # oldest game far below recent form -> recency-weighted mean should
        # sit closer to the recent value than a plain average would.
        df = pd.DataFrame({"hits": [0.0, 0.0, 0.0, 4.0]})
        result = weighted_recent_average(df, "hits", halflife_games=2.0, baseline_weight=0.0)
        plain_average = 1.0
        assert result["mean"] > plain_average

    def test_matches_manual_weighted_formula(self):
        values = np.array([10.0, 20.0])
        df = pd.DataFrame({"hits": values})
        halflife = 10.0
        baseline_weight = 0.2

        n = len(values)
        games_from_most_recent = (n - 1) - np.arange(n)
        weights = 0.5 ** (games_from_most_recent / halflife)
        recency_mean = np.sum(values * weights) / np.sum(weights)
        baseline_mean = np.mean(values)
        expected_mean = (1 - baseline_weight) * recency_mean + baseline_weight * baseline_mean

        result = weighted_recent_average(df, "hits", halflife_games=halflife, baseline_weight=baseline_weight)
        assert result["mean"] == pytest.approx(expected_mean)


class TestPitcherWeightedAverage:
    def _simple_log(self):
        today = datetime.date(2026, 8, 5)
        return pd.DataFrame({
            "date": pd.to_datetime([today - datetime.timedelta(days=91), today]),
            "opponent": ["Boston Red Sox", "New York Yankees"],
            "is_home": [True, False],
            "strikeOuts": [10.0, 30.0],
        }), today

    def test_raises_on_missing_column(self):
        df = pd.DataFrame({"date": pd.to_datetime(["2026-08-01"]), "opponent": ["X"], "is_home": [True]})
        with pytest.raises(ValueError):
            pitcher_weighted_average(df, "strikeOuts")

    def test_raises_on_empty_dataframe(self):
        df = pd.DataFrame(columns=["date", "opponent", "is_home", "strikeOuts"])
        with pytest.raises(ValueError):
            pitcher_weighted_average(df, "strikeOuts")

    def test_n_games_and_no_ip_context_without_ip_column(self):
        df, today = self._simple_log()
        result = pitcher_weighted_average(df, "strikeOuts", today=today)
        assert result["n_games"] == 2
        assert result["ip_context"] == ""
        assert result["era_context"] == ""
        assert result["pitch_count_adjustments"] == []
        assert len(result["start_breakdown"]) == 2

    def test_day_based_weighting_favors_recent_start(self):
        df, today = self._simple_log()
        result = pitcher_weighted_average(df, "strikeOuts", today=today, baseline_weight=0.2)

        # 91-day-old start (weight 0.1) vs today's start (weight 1.0)
        w_old, w_new = 0.1, 1.0
        recency_mean = (10.0 * w_old + 30.0 * w_new) / (w_old + w_new)
        baseline_mean = (10.0 + 30.0) / 2
        expected_mean = 0.8 * recency_mean + 0.2 * baseline_mean

        assert result["mean"] == pytest.approx(expected_mean)

    def test_pitch_count_scaling_on_return_from_absence(self):
        today = datetime.date(2026, 8, 5)
        df = pd.DataFrame({
            "date": pd.to_datetime([
                today - datetime.timedelta(days=60),
                today - datetime.timedelta(days=25),  # 35-day gap >= IL_GAP_DAYS
            ]),
            "opponent": ["Boston Red Sox", "New York Yankees"],
            "is_home": [True, False],
            "strikeOuts": [8.0, 3.0],
            "numberOfPitches": [95, 40],  # 40 << 0.8 * avg(95,40)=54 -> scaled up
        })

        result = pitcher_weighted_average(df, "strikeOuts", today=today)

        assert len(result["pitch_count_adjustments"]) == 1
        assert "scaled" in result["pitch_count_adjustments"][0]
