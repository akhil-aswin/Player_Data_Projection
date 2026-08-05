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
                              today: datetime.date = None,
                              baseline_weight: float = 0.20) -> dict:
    """
    Date-based weighted average for pitcher starts.

    Weight tiers (by calendar days since the start):
      0–28 days  → 1.0
      28–60 days → 0.6
      60–90 days → 0.3
      90+ days   → 0.1

    baseline_weight: fraction anchored to the simple unweighted mean so that
        a run of recent bad (or great) starts can't swing the projection too far.

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

    df[stat_col] = df[stat_col].astype(float)

    # ── IP + ERA context (strikeOuts only) ───────────────────────────────────
    # K totals scale with innings pitched. A pitcher yanked at 4 IP with 4 K's
    # is throwing at 9 K/9 — not a bad outing. Normalizing by IP prevents short
    # outings from dragging down the projection.
    # ERA is computed per start and tracked alongside IP to capture "bad outing"
    # variance: high-ERA pitchers get pulled sooner more often, compressing their
    # K ceiling. We apply a mild IP haircut when ERA is above league average,
    # and widen the variance estimate to reflect the higher spread.
    LEAGUE_AVG_ERA = 4.20

    ip_tag = ""
    era_tag = ""
    avg_k_per_ip = None
    avg_ip = None
    pitcher_era = None
    ip_std = None

    # ── ERA + IP context (all pitching stats) ───────────────────────────────
    # Compute avg_ip and pitcher_era once, shared across all stat types.
    # ERA tells us outing length tendency and performance volatility.
    GOOD_ERA_THRESHOLD = 3.20

    def _ip_to_float(ip_str) -> float:
        try:
            parts = str(ip_str).split(".")
            full = int(parts[0])
            outs = int(parts[1]) if len(parts) > 1 else 0
            return full + outs / 3.0
        except Exception:
            return 0.0

    if "inningsPitched" in df.columns:
        df["ip_dec"] = df["inningsPitched"].apply(_ip_to_float)
        valid_ip = df["ip_dec"] > 0

        if valid_ip.sum() >= 3:
            w       = df.loc[valid_ip, "day_weight"].to_numpy()
            ip_vals = df.loc[valid_ip, "ip_dec"].to_numpy()
            avg_ip  = float(np.sum(ip_vals * w) / np.sum(w))
            ip_std  = float(np.std(ip_vals))

            if "earnedRuns" in df.columns:
                df["er_num"] = pd.to_numeric(df["earnedRuns"], errors="coerce").fillna(0)
                df.loc[valid_ip, "era_start"] = (
                    df.loc[valid_ip, "er_num"] * 9 / df.loc[valid_ip, "ip_dec"]
                )
                era_vals    = df.loc[valid_ip, "era_start"].to_numpy()
                pitcher_era = float(np.sum(era_vals * w) / np.sum(w))

                # Symmetric ERA-IP adjustment applied to all stats that benefit
                # from IP normalization (Ks and BBs). ER is excluded — its
                # relationship with IP is circular (high ER caused the short
                # outing), so normalizing ER by adjusted IP would inflate
                # projections for bad starts.
                if pitcher_era > LEAGUE_AVG_ERA:
                    era_penalty = min(0.07, 0.04 * (pitcher_era - LEAGUE_AVG_ERA) / LEAGUE_AVG_ERA)
                    adj_ip = avg_ip * (1 - era_penalty)
                    era_tag = (
                        f"ERA {pitcher_era:.2f} (lg avg {LEAGUE_AVG_ERA}) → "
                        f"IP haircut -{era_penalty*100:.1f}% → expected {adj_ip:.1f} IP"
                    )
                elif pitcher_era < GOOD_ERA_THRESHOLD:
                    era_bonus = min(0.05, 0.03 * (GOOD_ERA_THRESHOLD - pitcher_era) / GOOD_ERA_THRESHOLD)
                    adj_ip = avg_ip * (1 + era_bonus)
                    era_tag = (
                        f"ERA {pitcher_era:.2f} (lg avg {LEAGUE_AVG_ERA}) → "
                        f"IP bonus +{era_bonus*100:.1f}% → expected {adj_ip:.1f} IP"
                    )
                else:
                    adj_ip  = avg_ip
                    era_tag = f"ERA {pitcher_era:.2f} (lg avg {LEAGUE_AVG_ERA}) — no IP adjustment"
            else:
                adj_ip = avg_ip

            # IP normalization for rate-based counting stats: Ks and BBs.
            # Both are truncated by outing length; normalizing to rate × expected_IP
            # removes the bias from short or extra-long starts.
            # ER is intentionally excluded: high ER/IP × short IP would double-penalize
            # an already-bad start, since the short outing was caused by the ER.
            # Walks (BB) inverse note: high-ERA pitchers walk more per inning BUT also
            # get pulled sooner — the IP normalization correctly captures both effects.
            if stat_col in ("strikeOuts", "baseOnBalls"):
                rate_col = f"{stat_col}_per_ip"
                df.loc[valid_ip, rate_col] = (
                    df.loc[valid_ip, stat_col] / df.loc[valid_ip, "ip_dec"]
                )
                rates       = df.loc[valid_ip, rate_col].to_numpy()
                avg_rate    = float(np.sum(rates * w) / np.sum(w))
                df.loc[valid_ip, stat_col] = df.loc[valid_ip, rate_col] * adj_ip
                ip_tag = f"{avg_rate:.2f} {stat_col}/IP × {adj_ip:.1f} avg IP (σ={ip_std:.1f})"

    values = df[stat_col].to_numpy()
    weights = df["day_weight"].to_numpy()

    recency_mean  = float(np.sum(values * weights) / np.sum(weights))
    baseline_mean = float(np.mean(values))
    weighted_mean = (1 - baseline_weight) * recency_mean + baseline_weight * baseline_mean

    # Variance widening for high-ERA pitchers applies to ALL stats:
    # ER, BB, and Ks are all more variable when a pitcher is struggling.
    weighted_var = float(np.sum(weights * (values - weighted_mean) ** 2) / np.sum(weights))
    if pitcher_era and pitcher_era > LEAGUE_AVG_ERA + 1.0:
        era_var_boost = min(0.25, 0.10 * (pitcher_era - LEAGUE_AVG_ERA) / LEAGUE_AVG_ERA)
        weighted_var *= (1 + era_var_boost)

    breakdown = []
    for i, (_, row) in enumerate(df.iterrows()):
        line = (
            f"{row['date'].date()}  {row.get('opponent',''):<25}  "
            f"{stat_col}: {row[stat_col]:.1f}  "
            f"IP: {row.get('ip_dec', '?')}  "
            f"weight: {row['day_weight']:.1f}  ({row['days_ago']}d ago)"
        )
        breakdown.append(line)

    return {
        "mean": weighted_mean,
        "std": weighted_var ** 0.5,
        "n_games": len(df),
        "pitch_count_adjustments": adjustments,
        "ip_context": ip_tag,
        "era_context": era_tag,
        "start_breakdown": breakdown,
    }


def add_derived_stats(game_log: pd.DataFrame, stat_col: str) -> pd.DataFrame:
    """
    Adds computed combination columns to a game log before projection.
    Currently handles hitsRunsRbi (H+R+RBI per game).
    Returns the original df unchanged if no derivation is needed.
    """
    if stat_col == "hitsRunsRbi":
        df = game_log.copy()
        for col in ("hits", "runs", "rbi"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
            else:
                df[col] = 0.0
        df["hitsRunsRbi"] = df["hits"] + df["runs"] + df["rbi"]
        return df
    return game_log


def get_opposing_starter(opponent_team: str, date_str: str = None) -> str | None:
    """
    Returns today's probable starting pitcher name for the given opponent team.
    Returns None if not yet announced or not found.
    """
    if date_str is None:
        date_str = datetime.date.today().strftime("%m/%d/%Y")
    try:
        games = statsapi.schedule(date=date_str, sportId=1)
    except Exception:
        return None

    opp_lower = opponent_team.lower().strip()
    for game in games:
        home = game.get("home_name", "")
        away = game.get("away_name", "")
        home_pitcher = game.get("home_probable_pitcher", "").strip()
        away_pitcher = game.get("away_probable_pitcher", "").strip()

        if opp_lower in home.lower() or home.lower() in opp_lower:
            return home_pitcher or None
        if opp_lower in away.lower() or away.lower() in opp_lower:
            return away_pitcher or None
    return None


def get_batter_game_info(batter_name: str, date_str: str = None) -> dict | None:
    """
    Finds a batter's game today via the MLB schedule (authoritative source),
    bypassing the Odds API event which can mis-assign players to games.

    Matches by team ID first (reliable), falls back to name fuzzy-match.

    Returns:
        {"opponent": str, "opposing_starter": str | None}
        or None if the player/game can't be found.
    """
    if date_str is None:
        date_str = datetime.date.today().strftime("%m/%d/%Y")
    try:
        matches = statsapi.lookup_player(batter_name)
        if not matches:
            return None

        player = matches[0]
        team_info = player.get("currentTeam") or {}
        # currentTeam often has id but no name — use both
        current_team_id   = team_info.get("id")
        current_team_name = (team_info.get("name") or "").strip().lower()

        games = statsapi.schedule(date=date_str, sportId=1)

        for game in games:
            home      = game.get("home_name", "")
            away      = game.get("away_name", "")
            home_id   = game.get("home_id")
            away_id   = game.get("away_id")

            # Prefer team-ID match; fall back to fuzzy name
            if current_team_id:
                batter_is_home = (current_team_id == home_id)
                batter_is_away = (current_team_id == away_id)
            elif current_team_name:
                batter_is_home = (
                    current_team_name in home.lower() or
                    home.lower() in current_team_name
                )
                batter_is_away = (
                    current_team_name in away.lower() or
                    away.lower() in current_team_name
                )
            else:
                continue

            if batter_is_home:
                pitcher = game.get("away_probable_pitcher", "").strip() or None
                return {"opponent": away, "opposing_starter": pitcher}
            if batter_is_away:
                pitcher = game.get("home_probable_pitcher", "").strip() or None
                return {"opponent": home, "opposing_starter": pitcher}
    except Exception:
        return None
    return None


def get_batter_vs_pitcher(batter_name: str, pitcher_name: str,
                           seasons: list, stat_col: str) -> dict:
    """
    Career per-PA rate for a batter vs a specific pitcher across the given seasons.
    Uses the vsPlayer endpoint on the batter's profile.

    Returns:
        rate_per_pa  — career stat / career PA vs this pitcher (None if no data)
        pa           — total plate appearances vs this pitcher
        detail       — human-readable summary
    """
    try:
        batter_id = get_player_id(batter_name)
        pitcher_id = get_player_id(pitcher_name)
    except ValueError:
        return {"rate_per_pa": None, "pa": 0, "detail": f"player lookup failed"}

    total_stat = 0.0
    total_pa = 0

    for season in seasons:
        try:
            resp = requests.get(
                f"{MLB_API_BASE}/people/{batter_id}/stats",
                params={
                    "stats": "vsPlayer", "group": "hitting",
                    "season": season, "opposingPlayerId": pitcher_id, "sportId": 1,
                },
                timeout=8,
            )
            splits = resp.json().get("stats", [{}])[0].get("splits", [])
            if not splits:
                continue
            stat = splits[0]["stat"]
            pa = int(stat.get("plateAppearances", 0))
            val = float(stat.get(stat_col, 0))
            if pa > 0:
                total_pa += pa
                total_stat += val
        except Exception:
            continue

    if total_pa == 0:
        return {"rate_per_pa": None, "pa": 0, "detail": f"no career PA vs {pitcher_name}"}

    rate = total_stat / total_pa
    return {
        "rate_per_pa": rate,
        "pa": total_pa,
        "detail": f"{total_stat:.1f} {stat_col} / {total_pa} PA = {rate:.3f}/PA vs {pitcher_name}",
    }


def weighted_recent_average(game_log: pd.DataFrame, stat_col: str,
                             halflife_games: float = 10.0,
                             baseline_weight: float = 0.20) -> dict:
    """
    Game-count exponential decay — used for batters (daily players).
    For pitchers use pitcher_weighted_average() instead.

    halflife_games: a game this many games ago carries half the weight of the
        most recent game. 10 (was 7) gives older games more say so 5 cold
        games don't collapse the projection.

    baseline_weight: fraction of the final mean that is anchored to the simple
        unweighted mean of all games. Prevents extreme recency from pushing the
        projection far below a player's established baseline.
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

    recency_mean = float(np.sum(values * weights) / np.sum(weights))
    baseline_mean = float(np.mean(values))

    # Blend: mostly recency-driven, but anchored to the overall baseline
    blended_mean = (1 - baseline_weight) * recency_mean + baseline_weight * baseline_mean

    # Variance uses the blended mean for a consistent spread estimate
    weighted_var = float(np.sum(weights * (values - blended_mean) ** 2) / np.sum(weights))

    return {"mean": blended_mean, "std": weighted_var ** 0.5, "n_games": n}
