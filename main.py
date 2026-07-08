"""
NBA player prop pipeline.
Pulls regular season game logs, builds a recency-weighted projection,
fetches live odds from The Odds API, de-vigs them, and outputs an edge estimate.
"""

from scipy.stats import norm

from data.nba_stats import get_recent_games, get_games_vs_opponent, weighted_recent_average
from data.odds_client import OddsAPIClient
from analysis.devig import consensus_fair_probability

STAT_MAP = {
    "points": "PTS",
    "rebounds": "REB",
    "assists": "AST",
    "threes": "FG3M",
    "blocks": "BLK",
    "steals": "STL",
    "pts": "PTS",
    "reb": "REB",
    "ast": "AST",
}

# Minimum std per stat to prevent hard 0%/100% on thin samples
MIN_STD = {
    "PTS": 4.0,
    "REB": 2.0,
    "AST": 1.5,
    "FG3M": 1.2,
    "BLK": 0.8,
    "STL": 0.8,
}


def project_stat(player_name: str, opponent: str, stat_col: str = "PTS",
                 season: str = "2025-26") -> dict:
    recent = get_recent_games(player_name, n=10, season=season)
    recent_stats = weighted_recent_average(recent, stat_col, halflife_games=5)

    matchup = get_games_vs_opponent(player_name, opponent)
    if len(matchup) >= 3:
        matchup_stats = weighted_recent_average(matchup, stat_col, halflife_games=10)
        weight_matchup = min(matchup_stats["n_games"] / (matchup_stats["n_games"] + 8), 0.4)
    else:
        matchup_stats = recent_stats
        weight_matchup = 0.0

    blended_mean = (1 - weight_matchup) * recent_stats["mean"] + weight_matchup * matchup_stats["mean"]

    return {
        "mean": blended_mean,
        "std": recent_stats["std"],
        "recent_mean": recent_stats["mean"],
        "matchup_mean": matchup_stats["mean"],
        "matchup_weight": weight_matchup,
        "matchup_games": matchup_stats["n_games"],
    }


def model_prob_over(mean: float, std: float, line: float) -> float:
    if std == 0:
        return 1.0 if mean > line else 0.0
    return 1 - norm.cdf(line, loc=mean, scale=std)


def find_edge(player_name: str, opponent: str, event_id: str,
              market: str = "player_points", stat_col: str = "PTS",
              season: str = "2025-26") -> dict:
    projection = project_stat(player_name, opponent, stat_col, season=season)

    floor = MIN_STD.get(stat_col, 3.0)
    effective_std = max(projection["std"], floor)

    odds_client = OddsAPIClient()
    props = odds_client.get_player_props(event_id, markets=[market])
    book_lines = odds_client.extract_player_lines(props, player_name, market)
    if not book_lines:
        raise ValueError(f"No odds found for {player_name} in market '{market}'")

    market_data = consensus_fair_probability(book_lines)
    line = market_data["consensus_line"]

    model_p_over = model_prob_over(projection["mean"], effective_std, line)
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
    client = OddsAPIClient()

    print("\n=== NBA Player Prop Projector ===\n")
    print("Fetching upcoming NBA events...\n")
    events = client.get_upcoming_events()

    if not events:
        print("No upcoming NBA events found (off-season). Come back in October.")
    else:
        for e in events:
            print(f"  {e['id']}  |  {e['home_team']} vs {e['away_team']}  |  {e['commence_time']}")

        print()
        event_id = input("Event ID: ").strip()
        player = input("Player full name: ").strip()
        opponent = input("Opponent team (e.g. Lakers or LAL): ").strip()
        stat = input("Stat [points/rebounds/assists/threes/blocks/steals]: ").strip() or "points"
        stat_col = STAT_MAP.get(stat.lower(), stat.upper())
        season = input("Season (default: 2025-26): ").strip() or "2025-26"
        market = f"player_{stat.lower().replace(' ', '_')}"

        print(f"\nFetching data for {player}...\n")
        result = find_edge(player, opponent, event_id,
                           market=market, stat_col=stat_col, season=season)

        print(f"  Player:            {result['player']}")
        print(f"  Line:              {result['line']}")
        print(f"  Model projection:  {result['model_projection']}")
        print(f"  Model prob over:   {result['model_prob_over']*100:.1f}%")
        print(f"  Market prob over:  {result['market_fair_prob_over']*100:.1f}%")
        print(f"  Edge:              {result['edge_pct_pts']:+.1f} pp")
        print(f"  Books used:        {result['n_books']}")
        print(f"  Matchup games:     {result['matchup_games_used']}")
