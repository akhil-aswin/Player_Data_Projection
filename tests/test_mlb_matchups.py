from unittest.mock import MagicMock

import pytest

import data.mlb_matchups as mlb_matchups
from data.mlb_matchups import (
    get_team_id,
    get_team_k_factor,
    _batter_k_pct,
    get_lineup,
    get_lineup_k_factor,
    get_pitcher_vs_lineup,
    LEAGUE_AVG_K_PCT,
)


def _resp(payload, raises=False):
    resp = MagicMock()
    if raises:
        resp.raise_for_status.side_effect = RuntimeError("http error")
    else:
        resp.raise_for_status.return_value = None
    resp.json.return_value = payload
    return resp


class TestGetTeamId:
    def test_returns_id_of_first_match(self, monkeypatch):
        monkeypatch.setattr(mlb_matchups.statsapi, "lookup_team", lambda name: [{"id": 147}])
        assert get_team_id("Yankees") == 147

    def test_raises_when_no_team_found(self, monkeypatch):
        monkeypatch.setattr(mlb_matchups.statsapi, "lookup_team", lambda name: [])
        with pytest.raises(ValueError):
            get_team_id("Not A Team")


class TestGetTeamKFactor:
    @staticmethod
    def _stats_payload(ks, pas):
        return {"stats": [{"splits": [{"stat": {"strikeOuts": ks, "plateAppearances": pas}}]}]}

    def test_high_k_team_factor_capped_at_upper_bound(self, monkeypatch):
        monkeypatch.setattr(mlb_matchups.statsapi, "lookup_team", lambda name: [{"id": 1}])
        payload = self._stats_payload(ks=500, pas=1000)  # 50% K rate, way above league avg
        monkeypatch.setattr(mlb_matchups.requests, "get", lambda *a, **k: _resp(payload))

        factor, detail = get_team_k_factor("Some Team", 2026)

        assert factor == pytest.approx(1.18)
        assert "capped" in detail

    def test_low_k_team_factor_capped_at_lower_bound(self, monkeypatch):
        monkeypatch.setattr(mlb_matchups.statsapi, "lookup_team", lambda name: [{"id": 1}])
        payload = self._stats_payload(ks=50, pas=1000)  # 5% K rate, well below league avg
        monkeypatch.setattr(mlb_matchups.requests, "get", lambda *a, **k: _resp(payload))

        factor, detail = get_team_k_factor("Some Team", 2026)

        assert factor == pytest.approx(0.82)
        assert "capped" in detail

    def test_no_split_data_returns_neutral(self, monkeypatch):
        monkeypatch.setattr(mlb_matchups.statsapi, "lookup_team", lambda name: [{"id": 1}])
        monkeypatch.setattr(mlb_matchups.requests, "get", lambda *a, **k: _resp({"stats": [{}]}))

        factor, detail = get_team_k_factor("Some Team", 2026)

        assert factor == 1.0
        assert "unavailable" in detail

    def test_lookup_failure_returns_neutral(self, monkeypatch):
        monkeypatch.setattr(
            mlb_matchups.statsapi, "lookup_team",
            lambda name: (_ for _ in ()).throw(ValueError("no team")),
        )

        factor, detail = get_team_k_factor("Not A Team", 2026)

        assert factor == 1.0
        assert "unavailable" in detail


class TestBatterKPct:
    def test_returns_none_below_pa_threshold(self, monkeypatch):
        payload = {"stats": [{"splits": [{"stat": {"strikeOuts": 5, "plateAppearances": 20}}]}]}
        monkeypatch.setattr(mlb_matchups.requests, "get", lambda *a, **k: _resp(payload))
        assert _batter_k_pct(123, 2026) is None

    def test_returns_rate_at_or_above_pa_threshold(self, monkeypatch):
        payload = {"stats": [{"splits": [{"stat": {"strikeOuts": 25, "plateAppearances": 100}}]}]}
        monkeypatch.setattr(mlb_matchups.requests, "get", lambda *a, **k: _resp(payload))
        assert _batter_k_pct(123, 2026) == pytest.approx(0.25)

    def test_request_failure_returns_none(self, monkeypatch):
        def boom(*a, **k):
            raise RuntimeError("network down")

        monkeypatch.setattr(mlb_matchups.requests, "get", boom)
        assert _batter_k_pct(123, 2026) is None


class TestGetLineup:
    def test_returns_players_for_home_team(self, monkeypatch):
        monkeypatch.setattr(mlb_matchups.statsapi, "lookup_team", lambda name: [{"id": 147}])
        payload = {
            "dates": [{
                "games": [{
                    "teams": {"home": {"team": {"id": 147}}},
                    "lineups": {"homePlayers": [{"id": 1, "fullName": "Batter One"}]},
                }]
            }]
        }
        monkeypatch.setattr(mlb_matchups.requests, "get", lambda *a, **k: _resp(payload))

        lineup = get_lineup("Yankees", date_str="2026-08-05")

        assert lineup == [{"id": 1, "fullName": "Batter One"}]

    def test_returns_empty_when_no_games_scheduled(self, monkeypatch):
        monkeypatch.setattr(mlb_matchups.statsapi, "lookup_team", lambda name: [{"id": 147}])
        monkeypatch.setattr(mlb_matchups.requests, "get", lambda *a, **k: _resp({"dates": []}))

        assert get_lineup("Yankees", date_str="2026-08-05") == []

    def test_returns_empty_on_error(self, monkeypatch):
        monkeypatch.setattr(mlb_matchups.statsapi, "lookup_team", lambda name: [{"id": 147}])

        def boom(*a, **k):
            raise RuntimeError("down")

        monkeypatch.setattr(mlb_matchups.requests, "get", boom)
        assert get_lineup("Yankees", date_str="2026-08-05") == []


class TestGetLineupKFactor:
    def test_neutral_when_lineup_not_posted(self, monkeypatch):
        monkeypatch.setattr(mlb_matchups, "get_lineup", lambda team, date_str=None: [])
        factor, detail = get_lineup_k_factor("Yankees", 2026)
        assert factor == 1.0
        assert "not yet posted" in detail

    def test_neutral_when_no_individual_kpct_data(self, monkeypatch):
        monkeypatch.setattr(
            mlb_matchups, "get_lineup",
            lambda team, date_str=None: [{"id": 1, "fullName": "Batter One"}],
        )
        monkeypatch.setattr(mlb_matchups, "_batter_k_pct", lambda pid, season: None)
        factor, detail = get_lineup_k_factor("Yankees", 2026)
        assert factor == 1.0
        assert "unavailable" in detail

    def test_computes_average_kpct_factor(self, monkeypatch):
        monkeypatch.setattr(
            mlb_matchups, "get_lineup",
            lambda team, date_str=None: [
                {"id": 1, "fullName": "Batter One"},
                {"id": 2, "fullName": "Batter Two"},
            ],
        )
        rates = {1: 0.30, 2: 0.20}
        monkeypatch.setattr(mlb_matchups, "_batter_k_pct", lambda pid, season: rates[pid])

        factor, detail = get_lineup_k_factor("Yankees", 2026)

        expected_avg = (0.30 + 0.20) / 2
        assert factor == pytest.approx(expected_avg / LEAGUE_AVG_K_PCT)
        assert "Batter One" in detail and "Batter Two" in detail


class TestGetPitcherVsLineup:
    def test_neutral_when_lineup_not_posted(self, monkeypatch):
        monkeypatch.setattr(mlb_matchups, "get_lineup", lambda team, date_str=None: [])
        factor, detail = get_pitcher_vs_lineup(999, "Yankees", [2026, 2025])
        assert factor == 1.0
        assert "not yet posted" in detail

    def test_uncovered_batters_are_neutral_and_covered_batter_is_capped(self, monkeypatch):
        lineup = [
            {"id": 1, "fullName": "Hot Matchup Batter"},
            {"id": 2, "fullName": "No History Batter"},
        ]
        monkeypatch.setattr(mlb_matchups, "get_lineup", lambda team, date_str=None: lineup)
        monkeypatch.setattr(mlb_matchups, "_batter_k_pct", lambda pid, season: 0.10)

        def fake_get(url, params=None, timeout=None):
            if params.get("opposingPlayerId") == 1:
                # 10 K in 20 BF -> matchup_k_pct 0.5, vs season 0.10 -> raw factor 5.0, capped to 2.0
                return _resp({"stats": [{"splits": [{"stat": {"strikeOuts": 10, "battersFaced": 20}}]}]})
            # batter 2: below the 7-BF minimum sample -> no coverage
            return _resp({"stats": [{"splits": [{"stat": {"strikeOuts": 1, "battersFaced": 3}}]}]})

        monkeypatch.setattr(mlb_matchups.requests, "get", fake_get)

        factor, detail = get_pitcher_vs_lineup(555, "Yankees", [2026, 2025])

        # covered batter capped at x2.0, uncovered batter contributes neutral x1.0
        # raw = (2.0 + 1.0) / 2 = 1.5, capped to overall bound of 1.20
        assert factor == pytest.approx(1.20)
        assert "Hot Matchup Batter" in detail
        assert "1 batter(s) no history" in detail

    def test_no_history_for_any_batter_returns_neutral(self, monkeypatch):
        lineup = [{"id": 1, "fullName": "Batter One"}]
        monkeypatch.setattr(mlb_matchups, "get_lineup", lambda team, date_str=None: lineup)

        def fake_get(url, params=None, timeout=None):
            return _resp({"stats": [{"splits": []}]})

        monkeypatch.setattr(mlb_matchups.requests, "get", fake_get)

        factor, detail = get_pitcher_vs_lineup(555, "Yankees", [2026, 2025])

        assert factor == 1.0
        assert "no pitcher-vs-batter history" in detail
