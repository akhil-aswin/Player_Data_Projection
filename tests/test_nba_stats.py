import numpy as np
import pandas as pd
import pytest

import data.nba_stats as nba_stats
from data.nba_stats import get_recent_games, get_team_abbreviation, get_games_vs_opponent, get_player_id, weighted_recent_average


class TestGetPlayerId:
    def test_returns_id_of_first_match(self, monkeypatch):
        monkeypatch.setattr(
            nba_stats.players, "find_players_by_full_name",
            lambda name: [{"id": 42, "full_name": "Jayson Tatum"}],
        )
        assert get_player_id("Jayson Tatum") == 42

    def test_raises_when_no_match(self, monkeypatch):
        monkeypatch.setattr(nba_stats.players, "find_players_by_full_name", lambda name: [])
        with pytest.raises(ValueError):
            get_player_id("Nobody Real")


class TestGetTeamAbbreviation:
    def test_short_input_returned_uppercased_directly(self):
        assert get_team_abbreviation("lal") == "LAL"

    def test_full_name_resolved_via_teams_lookup(self, monkeypatch):
        monkeypatch.setattr(
            nba_stats.teams, "find_teams_by_full_name",
            lambda name: [{"abbreviation": "BOS"}],
        )
        assert get_team_abbreviation("Boston Celtics") == "BOS"

    def test_falls_back_to_nickname_match(self, monkeypatch):
        monkeypatch.setattr(nba_stats.teams, "find_teams_by_full_name", lambda name: [])
        monkeypatch.setattr(
            nba_stats.teams, "get_teams",
            lambda: [{"nickname": "Knicks", "abbreviation": "NYK"}],
        )
        assert get_team_abbreviation("Knicks") == "NYK"

    def test_raises_when_nothing_matches(self, monkeypatch):
        monkeypatch.setattr(nba_stats.teams, "find_teams_by_full_name", lambda name: [])
        monkeypatch.setattr(nba_stats.teams, "get_teams", lambda: [])
        with pytest.raises(ValueError):
            get_team_abbreviation("Not A Team")


class TestWeightedRecentAverage:
    def test_raises_on_missing_column(self):
        df = pd.DataFrame({"PTS": [10, 20]})
        with pytest.raises(ValueError):
            weighted_recent_average(df, "REB")

    def test_raises_on_empty_dataframe(self):
        with pytest.raises(ValueError):
            weighted_recent_average(pd.DataFrame(), "PTS")

    def test_constant_values_give_zero_std(self):
        df = pd.DataFrame({"PTS": [25.0] * 5})
        result = weighted_recent_average(df, "PTS", halflife_games=5.0)
        assert result["mean"] == pytest.approx(25.0)
        assert result["std"] == pytest.approx(0.0, abs=1e-9)
        assert result["n_games"] == 5

    def test_matches_manual_exponential_decay_formula(self):
        values = np.array([12.0, 30.0])
        df = pd.DataFrame({"PTS": values})
        halflife = 5.0

        n = len(values)
        games_from_most_recent = (n - 1) - np.arange(n)
        weights = 0.5 ** (games_from_most_recent / halflife)
        expected_mean = np.sum(values * weights) / np.sum(weights)

        result = weighted_recent_average(df, "PTS", halflife_games=halflife)
        assert result["mean"] == pytest.approx(expected_mean)


class TestGetRecentGames:
    def test_returns_last_n_rows_from_game_log(self, monkeypatch):
        df = pd.DataFrame({"PTS": list(range(20))})
        monkeypatch.setattr(nba_stats, "get_player_game_log", lambda name, season="2025-26": df)

        result = get_recent_games("Jayson Tatum", n=5, season="2025-26")

        assert len(result) == 5
        assert result["PTS"].tolist() == list(range(15, 20))


class TestGetGamesVsOpponent:
    def test_filters_by_matchup_abbreviation_across_seasons(self, monkeypatch):
        monkeypatch.setattr(nba_stats, "get_team_abbreviation", lambda name: "NYK")

        season_frames = {
            "2025-26": pd.DataFrame({
                "MATCHUP": ["BOS vs. NYK", "BOS vs. LAL"],
                "GAME_DATE": pd.to_datetime(["2026-01-01", "2026-01-05"]),
                "PTS": [30, 20],
            }),
            "2024-25": pd.DataFrame({
                "MATCHUP": ["BOS @ NYK"],
                "GAME_DATE": pd.to_datetime(["2025-01-01"]),
                "PTS": [25],
            }),
        }

        def fake_log(name, season="2025-26"):
            return season_frames.get(season, pd.DataFrame())

        monkeypatch.setattr(nba_stats, "get_player_game_log", fake_log)

        result = get_games_vs_opponent("Jayson Tatum", "Knicks", seasons=["2025-26", "2024-25"])

        assert len(result) == 2
        assert set(result["PTS"]) == {30, 25}

    def test_returns_empty_dataframe_when_no_matchups_found(self, monkeypatch):
        monkeypatch.setattr(nba_stats, "get_team_abbreviation", lambda name: "NYK")
        monkeypatch.setattr(
            nba_stats, "get_player_game_log",
            lambda name, season="2025-26": pd.DataFrame({"MATCHUP": ["BOS vs. LAL"], "GAME_DATE": pd.to_datetime(["2026-01-01"]), "PTS": [10]}),
        )

        result = get_games_vs_opponent("Jayson Tatum", "Knicks", seasons=["2025-26"])

        assert result.empty
