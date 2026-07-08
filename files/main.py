"""
End-to-end example:
1. Pull a player's recent games + matchup history vs tonight's opponent
2. Build a simple recency-weighted projection (blended with matchup history)
3. Pull that player's prop odds across books and get the no-vig consensus
4. Compare model probability vs market probability -> edge

This is a BASELINE model — it's deliberately simple (normal-distribution
assumption, no injury/usage adjustments yet) so the pipeline can be validated
end to end before adding the more advanced features (injury-driven usage
redistribution, opponent defensive rating, pace, pos­sion-based rate model).
"""

from scipy.stats import norm

from data.nba_stats import get_recent_games, get_games_vs_opponent, weighted_recent_average
from data.odds_client import OddsAPIClient
from analysis.devig import consensus_fair_probability


def project_stat(player_name: str, opponent: str, stat_col: str = "PTS") -> dict:
    """Blend recent form with matchup history, shrinking matchup weight by sample size."""
    recent = get_recent_games(player_name, n=10)
    recent_stats = weighted_recent_average(recent, stat_col, halflife_games=5)

    matchup = get_games_vs_opponent(player_name, opponent)
    if len(matchup) >= 3:
        matchup_stats = weighted_recent_average(matchup, stat_col, halflife_games=10)
        # empirical-Bayes-style shrinkage: small matchup samples get little weight
        weight_matchup = min(matchup_stats["n_games"] / (matchup_stats["n_games"] + 8), 0.4)
    else:
        matchup_stats = recent_stats
        weight_matchup = 0.0

    blended_mean = (1 - weight_matchup) * recent_stats["mean"] + weight_matchup * matchup_stats["mean"]

    return {
        "mean": blended_mean,
        "std": recent_stats["std"],  # recent-form volatility used as the uncertainty estimate
        "recent_mean": recent_stats["mean"],
        "matchup_mean": matchup_stats["mean"],
        "matchup_weight": weight_matchup,
        "matchup_games": matchup_stats["n_games"],
    }


def model_prob_over(projection: dict, line: float) -> float:
    """P(stat > line), assuming the stat is ~Normal around the projected mean."""
    if projection["std"] == 0:
        return 1.0 if projection["mean"] > line else 0.0
    return 1 - norm.cdf(line, loc=projection["mean"], scale=projection["std"])


def find_edge(player_name: str, opponent: str, event_id: str,
              market: str = "player_points", stat_col: str = "PTS") -> dict:
    projection = project_stat(player_name, opponent, stat_col)

    odds_client = OddsAPIClient()
    props = odds_client.get_player_props(event_id, markets=[market])
    book_lines = odds_client.extract_player_lines(props, player_name, market)
    if not book_lines:
        raise ValueError(f"No odds found for {player_name} in market '{market}'")

    market_data = consensus_fair_probability(book_lines)
    line = market_data["consensus_line"]

    model_p_over = model_prob_over(projection, line)
    market_p_over = market_data["fair_prob_over"]

    return {
        "player": player_name,
        "line": line,
        "model_projection": round(projection["mean"], 1),
        "model_prob_over": round(model_p_over, 3),
        "market_fair_prob_over": round(market_p_over, 3),
        "edge_pct_pts": round((model_p_over - market_p_over) * 100, 1),
        "market_vig_pct": round(market_data["avg_vig_pct"], 1),
        "n_books": market_data["n_books"],
        "matchup_games_used": projection["matchup_games"],
    }


if __name__ == "__main__":
    # Get a real event_id first via OddsAPIClient().get_upcoming_events()
    result = find_edge(
        player_name="Jayson Tatum",
        opponent="Knicks",
        event_id="REPLACE_WITH_REAL_EVENT_ID",
        market="player_points",
        stat_col="PTS",
    )
    for k, v in result.items():
        print(f"{k}: {v}")
