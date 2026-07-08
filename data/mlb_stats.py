"""
Pulls MLB player game logs via the MLB Stats API.
Uses the statsapi package for player lookup, and direct requests for game logs
(the statsapi wrapper blocks the season param on gameLog type).
"""

import datetime
import numpy as np
import pandas as pd
import requests
import statsapi

from config import MLB_RECENT_SEASONS

MLB_API_BASE = "https://statsapi.mlb.com/api/v1"

# Weight tiers by how many days ago a start occurred
DAY_WEIGHTS = [
    (28,  1.0),   # 0–28 days:  full weight
    (60,  0.6),   # 28–60 days: moderate
    (90,  0.3),   # 60–90 days: low
    (None, 0.1),  # 90+ days:   minimal (prior season fill-in)
]

# A start is "return from absence" if the gap since the prior start exceeds this
IL_GAP_DAYS = 30

# A return start is considered pitch-count-limited if its pitches are this far
# below the pitcher's own recent average
PITCH_COUNT_LIMIT_RATIO = 0.80


def lookup_player(player_name: str) -> dict:
    results = statsapi.lookup_player(player_name)
    if not results:
        raise ValueError(
            f"No MLB player found matching '{player_name}'.\n"
            "  Tips:\n"
            "  - Use the full name as MLB lists it (e.g. 'Shohei Ohtani', 'Ronald Acuna')\n"
            "  - Drop accents — the API sometimes strips them\n"
            "  - Try last name only to see all matches"
        )
    if len(results) > 1:
        print(f"  Multiple matches for '{player_name}':")
        for p in results:
            print(f"    [{p['id']}] {p['fullName']}")
        print(f"  Using: {results[0]['fullName']}\n")
    return results[0]


def get_player_id(player_name: str) -> int:
    return lookup_player(player_name)["id"]


def get_game_log(player_name: str, season: int,
                 group: str = "hitting") -> pd.DataFrame:
    """
    Per-game stats for one player/season via the MLB Stats API directly.
    Returns a DataFrame sorted oldest -> newest.
    """
    player_id = get_player_id(player_name)
    resp = requests.get(
        f"{MLB_API_BASE}/people/{player_id}/stats",
        params={"stats": "gameLog", "group": group, "season": season, "sportId": 1},
        timeout=10,
    )
    resp.raise_for_status()
    data = resp.json()

    splits = []
    for block in data.get("stats", []):
        if block.get("splits"):
            splits = block["splits"]
            break

    if not splits:
        return pd.DataFrame()

    rows = []
    for s in splits:
        row = {
            "date": s.get("date", ""),
            "opponent": s.get("opponent", {}).get("name", ""),
            "is_home": s.get("isHome", False),
            **s.get("stat", {}),
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


def get_starts_vs_team(player_name: str, opponent: str,
                       seasons: list, group: str = "pitching") -> pd.DataFrame:
    """
    All starts by this pitcher against a specific opponent across the given seasons.
    Includes short outings (pitcher knocked out early) as long as gamesStarted=1.
    """
    frames = []
    for season in seasons:
        try:
            df = get_game_log(player_name, season=season, group=group)
            if df.empty:
                continue
            if group == "pitching" and "gamesStarted" in df.columns:
                df = df[df["gamesStarted"].astype(int) >= 1]
            vs = df[df["opponent"].str.contains(opponent, case=False, na=False)]
            if not vs.empty:
                frames.append(vs)
        except Exception:
            continue

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames).sort_values("date").reset_index(drop=True)


def get_recent_games(player_name: str, n: int = 15,
                     group: str = "hitting") -> tuple:
    """
    Returns (DataFrame, source_label) pulling across seasons until we have
    enough data. For pitchers, filters to starts only (gamesStarted >= 1).
    Always pulls the current season + previous season so thin current-season
    samples (injury return, early-season) are supplemented with prior data.
    The date-based weighting in pitcher_weighted_average handles the
    staleness of prior-season starts automatically.
    """
    frames = []
    seasons_used = []

    for season in MLB_RECENT_SEASONS:
        try:
            df = get_game_log(player_name, season=season, group=group)
            if df.empty:
                continue
            if group == "pitching" and "gamesStarted" in df.columns:
                df = df[df["gamesStarted"].astype(int) >= 1]
            if df.empty:
                continue
            frames.append(df)
            seasons_used.append(str(season))
            # stop once we have 2 seasons of data
            if len(seasons_used) >= 2:
                break
        except ValueError:
            raise
        except Exception:
            continue

    if not frames:
        raise ValueError(f"No MLB game history found for '{player_name}' ({group})")

    combined = pd.concat(frames).sort_values("date").reset_index(drop=True).tail(n)
    source = " + ".join(seasons_used)
    return combined, source


def _day_weight(days_ago: int) -> float:
    for cutoff, w in DAY_WEIGHTS:
        if cutoff is None or days_ago <= cutoff:
            return w
    return DAY_WEIGHTS[-1][1]


def pitcher_weighted_average(game_log: pd.DataFrame, stat_col: str,
                              today: datetime.date = None) -> dict:
    """
    Date-based weighted average for pitcher starts.

    Weight tiers (by calendar days since the start):
      0–28 days  → 1.0
      28–60 days → 0.6
      60–90 days → 0.3
      90+ days   → 0.1

    Pitch-count adjustment: if a start followed a gap of 30+ days AND the
    pitcher threw significantly fewer pitches than his own recent average,
    the stat is scaled up proportionally to estimate full-workload output.
    Flags the adjustment in the returned dict.
    """
    if game_log.empty or stat_col not in game_log.columns:
        available = [c for c in game_log.columns if c not in ("date", "opponent", "is_home")]
        raise ValueError(
            f"Column '{stat_col}' not found.\n"
            f"  Available stat columns: {', '.join(available)}"
        )

    if today is None:
        today = datetime.date.today()

    df = game_log.copy().sort_values("date").reset_index(drop=True)
    today_ts = pd.Timestamp(today)
    df["days_ago"] = (today_ts - df["date"]).dt.days
    df["day_weight"] = df["days_ago"].apply(_day_weight)

    # Pitch-count adjustment for return-from-absence starts
    pitch_col = "numberOfPitches"
    adjustments = []
    if pitch_col in df.columns:
        df[pitch_col] = pd.to_numeric(df[pitch_col], errors="coerce").fillna(0)
        df[stat_col] = df[stat_col].astype(float)
        df["gap_days"] = df["date"].diff().dt.days.fillna(0)

        # avg pitches excluding the start being evaluated
        avg_pitches = df[pitch_col].replace(0, np.nan).mean()

        for idx, row in df.iterrows():
            gap = row["gap_days"]
            pitches = row[pitch_col]
            if (gap >= IL_GAP_DAYS and pitches > 0
                    and avg_pitches > 0
                    and pitches < avg_pitches * PITCH_COUNT_LIMIT_RATIO):
                scale = avg_pitches / pitches
                original = df.at[idx, stat_col]
                df.at[idx, stat_col] = float(original) * scale
                adjustments.append(
                    f"{row['date'].date()}: {gap:.0f}-day gap, "
                    f"{pitches:.0f} pitches vs avg {avg_pitches:.0f} — "
                    f"{stat_col} scaled {original:.1f} → {df.at[idx, stat_col]:.1f}"
                )

    values = df[stat_col].astype(float).to_numpy()
    weights = df["day_weight"].to_numpy()

    weighted_mean = float(np.sum(values * weights) / np.sum(weights))
    weighted_var = float(np.sum(weights * (values - weighted_mean) ** 2) / np.sum(weights))

    return {
        "mean": weighted_mean,
        "std": weighted_var ** 0.5,
        "n_games": len(df),
        "pitch_count_adjustments": adjustments,
        "start_breakdown": [
            f"{row['date'].date()}  {row.get('opponent',''):<25}  "
            f"{stat_col}: {row[stat_col]:.1f}  weight: {row['day_weight']:.1f}  ({row['days_ago']}d ago)"
            for _, row in df.iterrows()
        ],
    }


def weighted_recent_average(game_log: pd.DataFrame, stat_col: str,
                             halflife_games: float = 7.0) -> dict:
    """
    Game-count exponential decay — used for batters (daily players).
    For pitchers use pitcher_weighted_average() instead.
    """
    if game_log.empty or stat_col not in game_log.columns:
        available = [c for c in game_log.columns if c not in ("date", "opponent", "is_home")]
        raise ValueError(
            f"Column '{stat_col}' not found.\n"
            f"  Available stat columns: {', '.join(available)}"
        )

    values = game_log[stat_col].astype(float).to_numpy()
    n = len(values)
    games_from_most_recent = (n - 1) - np.arange(n)
    weights = 0.5 ** (games_from_most_recent / halflife_games)

    weighted_mean = float(np.sum(values * weights) / np.sum(weights))
    weighted_var = float(np.sum(weights * (values - weighted_mean) ** 2) / np.sum(weights))

    return {"mean": weighted_mean, "std": weighted_var ** 0.5, "n_games": n}
