import pytest

from analysis.devig import (
    american_to_prob,
    prob_to_american,
    devig_two_way,
    consensus_fair_probability,
)


class TestAmericanToProb:
    def test_positive_odds(self):
        assert american_to_prob(100) == pytest.approx(0.5)
        assert american_to_prob(200) == pytest.approx(1 / 3)

    def test_negative_odds(self):
        assert american_to_prob(-110) == pytest.approx(110 / 210)
        assert american_to_prob(-200) == pytest.approx(2 / 3)

    def test_even_money_matches_both_signs(self):
        # -100 and +100 both describe a 50/50 coin flip
        assert american_to_prob(100) == pytest.approx(american_to_prob(-100))


class TestProbToAmerican:
    def test_favorite_returns_negative_odds(self):
        assert prob_to_american(0.6) < 0

    def test_underdog_returns_positive_odds(self):
        assert prob_to_american(0.4) > 0

    def test_round_trips_through_american_to_prob(self):
        for odds in (-150, -110, 120, 250):
            prob = american_to_prob(odds)
            assert prob_to_american(prob) == pytest.approx(odds, abs=1e-6)


class TestDevigTwoWay:
    def test_removes_vig_so_probabilities_sum_to_one(self):
        result = devig_two_way(-110, -110)
        assert result.fair_prob_over + result.fair_prob_under == pytest.approx(1.0)
        assert result.fair_prob_over == pytest.approx(0.5)

    def test_vig_pct_is_positive_for_standard_juiced_line(self):
        result = devig_two_way(-110, -110)
        assert result.vig_pct > 0
        # -110/-110 is the canonical ~4.76% overround market
        assert result.vig_pct == pytest.approx(4.7619, abs=1e-3)

    def test_asymmetric_odds_favor_the_lower_priced_side(self):
        result = devig_two_way(-150, 130)
        assert result.fair_prob_over > result.fair_prob_under


class TestConsensusFairProbability:
    def test_raises_on_empty_input(self):
        with pytest.raises(ValueError):
            consensus_fair_probability([])

    def test_single_book_passthrough(self):
        book_lines = [{"book": "fanduel", "line": 24.5, "over_odds": -110, "under_odds": -110}]
        result = consensus_fair_probability(book_lines)
        assert result["consensus_line"] == 24.5
        assert result["n_books"] == 1
        assert result["fair_prob_over"] == pytest.approx(0.5)
        assert result["fair_prob_under"] == pytest.approx(0.5)
        assert result["lines_disagreement"] is False

    def test_averages_across_books_on_the_same_line(self):
        book_lines = [
            {"book": "fanduel", "line": 24.5, "over_odds": -120, "under_odds": 100},
            {"book": "draftkings", "line": 24.5, "over_odds": -110, "under_odds": -110},
        ]
        result = consensus_fair_probability(book_lines)
        expected = (
            devig_two_way(-120, 100).fair_prob_over
            + devig_two_way(-110, -110).fair_prob_over
        ) / 2
        assert result["fair_prob_over"] == pytest.approx(expected)
        assert result["n_books"] == 2

    def test_majority_line_wins_when_books_disagree(self):
        book_lines = [
            {"book": "a", "line": 24.5, "over_odds": -110, "under_odds": -110},
            {"book": "b", "line": 24.5, "over_odds": -115, "under_odds": -105},
            {"book": "c", "line": 25.5, "over_odds": -110, "under_odds": -110},
        ]
        result = consensus_fair_probability(book_lines)
        assert result["consensus_line"] == 24.5
        assert result["n_books"] == 2
        assert result["lines_disagreement"] is True
        assert result["all_lines_seen"] == [24.5, 25.5]

    def test_per_book_breakdown_included(self):
        book_lines = [{"book": "fanduel", "line": 24.5, "over_odds": -110, "under_odds": -110}]
        result = consensus_fair_probability(book_lines)
        assert len(result["per_book"]) == 1
        assert "fair_prob_over" in result["per_book"][0]
        assert "vig_pct" in result["per_book"][0]
