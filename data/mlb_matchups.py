"""
Opponent-side context for MLB pitcher strikeout projections:
  1. Opposing team's K% vs league average
  2. Today's confirmed lineup individual K% (if posted)
  3. Pitcher career K rate vs each individual batter in the lineup
"""

import datetime
import requests
import statsapi

MLB_API_BASE = "https://statsapi.mlb.com/api/v1"

# Approximate MLB league-average K/PA — update each spring if needed
LEAGUE_AVG_K_PCT = 0.228


def get_team_id(team_name: str) -> int:
    results = statsapi.lookup_team(team_name)
    if not results:
        raise ValueError(f"No MLB team found matching '{team_name}'")
    return results[0]["id"]


def get_team_k_factor(team_name: str, season: int) -> tuple:
    """
    (factor, detail) where factor = team_K_pct / league_avg_K_pct.
    > 1.0 means this team Ks more than average — good for the pitcher.
    Returns (1.0, reason) on any failure so the pipeline can continue.
    """
    try:
        team_id = get_team_id(team_name)
        resp = requests.get(
            f"{MLB_API_BASE}/teams/{team_id}/stats",
            params={"stats": "season", "group": "hitting", "season": season, "sportId": 1},
            timeout=10,
        )
        resp.raise_for_status()
        splits = resp.json().get("stats", [{}])[0].get("splits", [])
        if not splits:
            return 1.0, "team stats unavailable"

        stat = splits[0]["stat"]
        ks = int(stat.get("strikeOuts", 0))
        pas = int(stat.get("plateAppearances", 1))
        team_k_pct = ks / pas if pas else LEAGUE_AVG_K_PCT
        raw_factor = team_k_pct / LEAGUE_AVG_K_PCT
        factor = max(0.82, min(1.18, raw_factor))
        cap_note = f"  [capped from ×{raw_factor:.3f}]" if raw_factor != factor else ""
        return factor, f"{team_name} K%: {team_k_pct*100:.1f}%  league: {LEAGUE_AVG_K_PCT*100:.1f}%  factor: ×{factor:.3f}{cap_note}"
    except Exception as e:
        return 1.0, f"team K% unavailable ({e})"


def _batter_k_pct(player_id: int, season: int) -> float | None:
    """Season K/PA for one batter. Returns None if sample is too small (<50 PA)."""
    try:
        resp = requests.get(
            f"{MLB_API_BASE}/people/{player_id}/stats",
            params={"stats": "season", "group": "hitting", "season": season, "sportId": 1},
            timeout=8,
        )
        splits = resp.json().get("stats", [{}])[0].get("splits", [])
        if not splits:
            return None
        stat = splits[0]["stat"]
        ks = int(stat.get("strikeOuts", 0))
        pas = int(stat.get("plateAppearances", 0))
        return ks / pas if pas >= 50 else None
    except Exception:
        return None


def get_lineup(team_name: str, date_str: str = None) -> list:
    """
    Returns list of {"id": int, "fullName": str} for today's confirmed lineup.
    Returns empty list if lineup hasn't been posted yet.
    """
    if date_str is None:
        date_str = datetime.date.today().isoformat()
    try:
        team_id = get_team_id(team_name)
        resp = requests.get(
            f"{MLB_API_BASE}/schedule",
            params={
                "sportId": 1, "date": date_str, "teamId": team_id,
                "hydrate": "lineups", "gameType": "R",
            },
            timeout=10,
        )
        resp.raise_for_status()
        dates = resp.json().get("dates", [])
        if not dates:
            return []

        game = dates[0]["games"][0]
        lineups = game.get("lineups", {})
        is_home = game["teams"]["home"]["team"]["id"] == team_id
        players = lineups.get("homePlayers" if is_home else "awayPlayers", [])
        return [{"id": p["id"], "fullName": p["fullName"]} for p in players]
    except Exception:
        return []


def get_lineup_k_factor(team_name: str, season: int, date_str: str = None) -> tuple:
    """
    (factor, detail) based on the confirmed lineup's individual K rates.
    Falls back to (1.0, reason) if lineup isn't posted or data is unavailable.
    """
    lineup = get_lineup(team_name, date_str)
    if not lineup:
        return 1.0, "lineup not yet posted — skipping individual K% adjustment"

    k_pcts = []
    for p in lineup:
        k = _batter_k_pct(p["id"], season)
        if k is not None:
            k_pcts.append((p["fullName"], k))

    if not k_pcts:
        return 1.0, "lineup posted but individual K% data unavailable"

    avg_k = sum(k for _, k in k_pcts) / len(k_pcts)
    factor = avg_k / LEAGUE_AVG_K_PCT
    batter_list = ", ".join(f"{name} {k*100:.0f}%" for name, k in k_pcts)
    detail = f"lineup K%: {avg_k*100:.1f}%  factor: ×{factor:.3f}  ({batter_list})"
    return factor, detail


def get_pitcher_vs_lineup(pitcher_id: int, team_name: str,
                           seasons: list, date_str: str = None) -> tuple:
    """
    For each batter in today's lineup, computes how much the pitcher's
    historical K rate vs that batter differs from the batter's own season K rate.

    factor = avg(matchup_K% / batter_season_K%) across the full lineup.
    Batters with no qualifying history (< 7 BF) contribute 1.0 (neutral),
    so all lineup spots are always represented and the factor can't be inflated
    by only counting batters where the pitcher has the most history.

    Capped at ×0.80–×1.20 to prevent single-batter outliers from dominating.
    """
    lineup = get_lineup(team_name, date_str)
    if not lineup:
        return 1.0, "lineup not yet posted — skipping pitcher-vs-batter adjustment"

    current_season = seasons[0]
    covered = []

    for batter in lineup:
        batter_id = batter["id"]
        for season in seasons:
            try:
                resp = requests.get(
                    f"{MLB_API_BASE}/people/{pitcher_id}/stats",
                    params={
                        "stats": "vsPlayer", "group": "pitching",
                        "season": season, "opposingPlayerId": batter_id, "sportId": 1,
                    },
                    timeout=8,
                )
                splits = resp.json().get("stats", [{}])[0].get("splits", [])
                if not splits:
                    continue
                stat = splits[0]["stat"]
                ks = int(stat.get("strikeOuts", 0))
                bf = int(stat.get("battersFaced", 0))
                if bf >= 7:
                    matchup_k_pct = ks / bf
                    # Compare vs batter's own season rate, not league avg —
                    # lineup_factor already captures whether batters are above/below avg
                    season_k = _batter_k_pct(batter_id, current_season) or LEAGUE_AVG_K_PCT
                    # Floor matchup rate at 5% so 0K/nBF doesn't collapse the factor;
                    # cap per-batter factor at ×2.0 / ×0.50 so one low-K outlier
                    # (e.g. Arraez at 4% own rate) can't dominate the lineup average
                    raw_batter_factor = max(matchup_k_pct, 0.05) / season_k
                    batter_factor = max(0.50, min(2.0, raw_batter_factor))
                    covered.append({
                        "name": batter["fullName"],
                        "matchup_k_pct": matchup_k_pct,
                        "season_k_pct": season_k,
                        "raw_batter_factor": raw_batter_factor,
                        "batter_factor": batter_factor,
                        "ks": ks,
                        "bf": bf,
                    })
                    break
            except Exception:
                continue

    lineup_size = len(lineup)
    n_neutral = lineup_size - len(covered)

    if not covered:
        return 1.0, "no pitcher-vs-batter history found (likely first-time matchups)"

    # All lineup spots contribute; uncovered batters are neutral (1.0)
    total_factor = sum(m["batter_factor"] for m in covered) + n_neutral * 1.0
    raw_factor = total_factor / lineup_size

    # Cap: one hot/cold batter can shift the mean at most ±20%
    factor = max(0.80, min(1.20, raw_factor))

    lines = []
    for m in covered:
        capped = m["raw_batter_factor"] != m["batter_factor"]
        cap_note = f" [capped from ×{m['raw_batter_factor']:.2f}]" if capped else ""
        lines.append(
            f"{m['name']}: {m['ks']}K/{m['bf']}BF "
            f"({m['matchup_k_pct']*100:.0f}% vs {m['season_k_pct']*100:.0f}% own rate "
            f"→ ×{m['batter_factor']:.2f}{cap_note})"
        )
    neutral_note = f"{n_neutral} batter(s) no history → ×1.00" if n_neutral else ""
    detail = (
        f"pitcher-vs-batter ({len(covered)}/{lineup_size} covered): "
        f"raw ×{raw_factor:.3f}  capped ×{factor:.3f}\n    "
        + "\n    ".join(lines + ([neutral_note] if neutral_note else []))
    )
    return factor, detail
