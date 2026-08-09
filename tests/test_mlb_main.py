from scipy.stats import norm, poisson

import mlb_main
from mlb_main import (
    _is_whole_number,
    compute_probs,
    model_prob_over,
    _match_pp_line,
    _normalize_name,
    _build_pitcher_opponent_map,
    _resolve_opponent,
)

import pytest


class TestIsWholeNumber:
    def test_whole_number_line(self):
        assert _is_whole_number(5.0) is True

    def test_half_point_line(self):
        assert _is_whole_number(4.5) is False


class TestComputeProbsNormal:
    def test_matches_scipy_normal_cdf(self):
        mean, std, line = 10.0, 2.0, 8.5
        result = compute_probs(mean, std, line, use_poisson=False)

        expected_over = float(1 - norm.cdf(line, loc=mean, scale=std))
        assert result["p_over"] == pytest.approx(round(expected_over, 4))
        assert result["p_under"] == pytest.approx(round(1 - expected_over, 4))
        assert result["p_push"] == 0.0

    def test_effective_over_rate_equals_p_over_when_no_push(self):
        result = compute_probs(10.0, 2.0, 8.5, use_poisson=False)
        assert result["effective_over_rate"] == result["p_over"]

    def test_zero_std_mean_above_line_is_certain_over(self):
        result = compute_probs(10.0, 0.0, 5.0, use_poisson=False)
        assert result["p_over"] == 1.0
        assert result["p_under"] == 0.0

    def test_zero_std_mean_at_or_below_line_is_certain_under(self):
        result = compute_probs(5.0, 0.0, 5.0, use_poisson=False)
        assert result["p_over"] == 0.0
        assert result["p_under"] == 1.0


class TestComputeProbsPoisson:
    def test_half_point_line_has_no_push(self):
        mean, line = 5.0, 4.5
        result = compute_probs(mean, std=1.0, line=line, use_poisson=True)

        k = 4
        expected_over = float(1 - poisson.cdf(k, mu=mean))
        assert result["p_push"] == 0.0
        assert result["p_over"] == pytest.approx(round(expected_over, 4))
        assert result["p_under"] == pytest.approx(round(1 - expected_over, 4))

    def test_whole_number_line_allows_push_and_probabilities_sum_to_one(self):
        mean, line = 5.0, 5.0
        result = compute_probs(mean, std=1.0, line=line, use_poisson=True)

        expected_over = float(1 - poisson.cdf(int(line), mu=mean))
        expected_push = float(poisson.pmf(int(line), mu=mean))
        expected_under = float(poisson.cdf(int(line) - 1, mu=mean))

        assert result["p_over"] == pytest.approx(round(expected_over, 4))
        assert result["p_push"] == pytest.approx(round(expected_push, 4))
        assert result["p_under"] == pytest.approx(round(expected_under, 4))
        assert result["p_over"] + result["p_push"] + result["p_under"] == pytest.approx(1.0, abs=1e-3)

    def test_effective_over_rate_excludes_push(self):
        result = compute_probs(5.0, std=1.0, line=5.0, use_poisson=True)
        decisive = result["p_over"] + result["p_under"]
        assert result["effective_over_rate"] == pytest.approx(result["p_over"] / decisive)


class TestModelProbOver:
    def test_returns_raw_p_over(self):
        assert model_prob_over(10.0, 2.0, 8.0) == compute_probs(10.0, 2.0, 8.0)["p_over"]

    def test_poisson_flag_forwarded(self):
        assert model_prob_over(5.0, 1.0, 5.0, use_poisson=True) == compute_probs(5.0, 1.0, 5.0, use_poisson=True)["p_over"]


class TestMatchPpLine:
    def test_exact_lowercase_match(self):
        lines = {"aaron judge": 1.5}
        assert _match_pp_line("Aaron Judge", lines) == 1.5

    def test_last_name_fallback_match(self):
        lines = {"a. judge": 1.5}
        assert _match_pp_line("Aaron Judge", lines) == 1.5

    def test_returns_none_when_no_match(self):
        lines = {"shohei ohtani": 0.5}
        assert _match_pp_line("Aaron Judge", lines) is None


class TestNormalizeName:
    def test_strips_accents(self):
        assert _normalize_name("Sánchez") == "sanchez"

    def test_lowercases_and_strips_whitespace(self):
        assert _normalize_name("  Aaron JUDGE  ") == "aaron judge"


class TestBuildPitcherOpponentMap:
    def test_maps_home_and_away_probable_pitchers(self, monkeypatch):
        games = [{
            "home_name": "New York Yankees",
            "away_name": "Boston Red Sox",
            "home_probable_pitcher": "Gerrit Cole",
            "away_probable_pitcher": "Garrett Crochet",
        }]
        monkeypatch.setattr(mlb_main.statsapi, "schedule", lambda date, sportId: games)

        mapping = _build_pitcher_opponent_map()

        assert mapping[_normalize_name("Gerrit Cole")] == "Boston Red Sox"
        assert mapping[_normalize_name("Garrett Crochet")] == "New York Yankees"

    def test_blank_probable_pitchers_are_skipped(self, monkeypatch):
        games = [{
            "home_name": "New York Yankees",
            "away_name": "Boston Red Sox",
            "home_probable_pitcher": "",
            "away_probable_pitcher": "",
        }]
        monkeypatch.setattr(mlb_main.statsapi, "schedule", lambda date, sportId: games)

        assert _build_pitcher_opponent_map() == {}

    def test_schedule_failure_returns_empty_map(self, monkeypatch):
        def boom(date, sportId):
            raise RuntimeError("down")

        monkeypatch.setattr(mlb_main.statsapi, "schedule", boom)
        assert _build_pitcher_opponent_map() == {}


class TestResolveOpponent:
    def test_exact_schedule_match(self):
        schedule_map = {"gerrit cole": "Boston Red Sox"}
        result = _resolve_opponent("Gerrit Cole", schedule_map, "New York Yankees", "Boston Red Sox")
        assert result == "Boston Red Sox"

    def test_last_name_fallback_match(self):
        schedule_map = {"g. cole": "Boston Red Sox"}
        result = _resolve_opponent("Gerrit Cole", schedule_map, "New York Yankees", "Boston Red Sox")
        assert result == "Boston Red Sox"

    def test_falls_back_to_current_team_lookup(self, monkeypatch):
        monkeypatch.setattr(
            mlb_main.statsapi, "lookup_player",
            lambda name: [{"currentTeam": {"name": "New York Yankees"}}],
        )
        result = _resolve_opponent("Some Pitcher", {}, "New York Yankees", "Boston Red Sox")
        assert result == "Boston Red Sox"

    def test_returns_none_when_unresolvable(self, monkeypatch):
        monkeypatch.setattr(mlb_main.statsapi, "lookup_player", lambda name: [])
        result = _resolve_opponent("Unknown Pitcher", {}, "New York Yankees", "Boston Red Sox")
        assert result is None
