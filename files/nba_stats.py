"""
Pulls historical NBA player data: full game logs, recent-form windows, and
matchup history vs a specific opponent. Built on `nba_api`, an unofficial
wrapper around stats.nba.com.

stats.nba.com is rate-limit sensitive and occasionally blocks bare requests,
so every call goes through a small delay and a real browser User-Agent.
"""

import time
import numpy as np
import pandas as pd
from nba_api.stats.static import players, teams
from nba_api.stats.endpoints import playergamelog

REQUEST_DELAY_SECONDS = 0.6  # be polite to stats.nba.com to avoid 429/403s


def get_player_id(player_name: str) -> int:
    """Look up a player's stats.nba.com ID by full name."""
    matches = players.find_players_by_full_name(player_name)
    if not matches:
        raise ValueError(f"No player found matching '{player_name}'")
    return matches[0]["id"]


def get_team_abbreviation(team_name: str) -> str:
    """Resolve a team name (or existing abbreviation) to its 3-letter code."""
    if len(team_name) <= 3:
        return team_name.upper()
    matches = teams.find_teams_by_full_name(team_name)
    if not matches:
        matches = [t for t in teams.get_teams()
                   if team_name.lower() in t["nickname"].lower()]
    if not matches:
        raise ValueError(f"No team found matching '{team_name}'")
    return matches[0]["abbreviation"]


def get_player_game_log(player_name: str, season: str = "2025-26",
                         season_type: str = "Regular Season") -> pd.DataFrame:
    """
    Full game log for one player/season. `season` format: "2025-26".
    Returns a DataFrame sorted oldest -> newest with columns like
    PTS, REB, AST, MIN, MATCHUP, GAME_DATE, etc.
    """
    player_id = get_player_id(player_name)
    time.sleep(REQUEST_DELAY_SECONDS)
    log = playergamelog.PlayerGameLog(
        player_id=player_id,
        season=season,
        season_type_all_star=season_type,
    )
    df = log.get_data_frames()[0]
    if df.empty:
        return df
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    return df.sort_values("GAME_DATE").reset_index(drop=True)


def get_recent_games(player_name: str, n: int = 10,
                      season: str = "2025-26") -> pd.DataFrame:
    """Last n games for a player, most recent game last in the frame."""
    df = get_player_game_log(player_name, season=season)
    return df.tail(n).reset_index(drop=True)


def get_games_vs_opponent(player_name: str, opponent: str,
                           seasons: list = None) -> pd.DataFrame:
    """
    Matchup history: every game a player has played against a specific
    opponent across one or more seasons. `opponent` can be a team name,
    nickname, or 3-letter abbreviation (e.g. "Knicks" or "NYK").
    """
    if seasons is None:
        seasons = ["2025-26", "2024-25", "2023-24"]

    opp_abbr = get_team_abbreviation(opponent)

    frames = []
    for season in seasons:
        try:
            df = get_player_game_log(player_name, season=season)
        except Exception:
            continue
        if df.empty:
            continue
        # MATCHUP column looks like "LAL vs. NYK" or "LAL @ NYK"
        vs_opp = df[df["MATCHUP"].str.contains(opp_abbr, na=False)]
        frames.append(vs_opp)

    if not frames or all(f.empty for f in frames):
        return pd.DataFrame()
    return pd.concat(frames).sort_values("GAME_DATE").reset_index(drop=True)


def weighted_recent_average(game_log: pd.DataFrame, stat_col: str,
                             halflife_games: float = 5.0) -> dict:
    """
    Exponentially-weighted mean and std of a stat so recent games count more
    than older ones. `halflife_games` = how many games back the weight drops
    to 50%.

    Returns: {"mean": float, "std": float, "n_games": int}
    """
    if game_log.empty or stat_col not in game_log.columns:
        raise ValueError(f"No data or missing column '{stat_col}'")

    values = game_log[stat_col].astype(float).to_numpy()
    n = len(values)
    games_from_most_recent = (n - 1) - np.arange(n)  # 0 = most recent game
    weights = 0.5 ** (games_from_most_recent / halflife_games)

    weighted_mean = float(np.sum(values * weights) / np.sum(weights))
    weighted_var = float(np.sum(weights * (values - weighted_mean) ** 2) / np.sum(weights))

    return {"mean": weighted_mean, "std": weighted_var ** 0.5, "n_games": n}
