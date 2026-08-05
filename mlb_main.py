"""
MLB player prop pipeline.
Pulls game logs from the MLB Stats API, builds a recency-weighted projection
adjusted for opponent context, fetches live odds from The Odds API,
de-vigs them, and outputs an edge estimate.
"""

import math
import time as _time
import unicodedata
import statsapi
import requests as _requests
from scipy.stats import poisson, norm

from config import MLB_SPORT_KEY, MLB_CURRENT_SEASON, MLB_RECENT_SEASONS
from data.mlb_stats import (
    get_recent_games, get_player_id, get_starts_vs_team,
    weighted_recent_average, pitcher_weighted_average,
    get_opposing_starter, get_batter_vs_pitcher, get_batter_game_info,
    add_derived_stats,
)
from data.mlb_matchups import (
    get_team_k_factor,
    get_lineup_k_factor,
    get_pitcher_vs_lineup,
)
from data.odds_client import OddsAPIClient
from analysis.devig import consensus_fair_probability

# ── Historical calibration ────────────────────────────────────────────────────
# Loaded from picks.db; refreshed every hour so a running server picks up new
# resolved picks without a restart. Gracefully returns {} if DB unavailable.

_cal_cache:    dict  = {}
_cal_cache_ts: float = 0.0
_CAL_TTL = 3600        # refresh every hour
_CAL_MIN_SAMPLES = 30  # don't apply until we have this many resolved picks
_CAL_MAX_CORRECTION = 0.30  # cap correction at 30% of raw projection


def _load_calibration() -> dict:
    global _cal_cache, _cal_cache_ts
    if _time.time() - _cal_cache_ts < _CAL_TTL:
        return _cal_cache
    try:
        from database import get_calibration_data
        _cal_cache    = get_calibration_data(min_samples=_CAL_MIN_SAMPLES)
        _cal_cache_ts = _time.time()
    except Exception:
        pass
    return _cal_cache


BATTER_STAT_MAP = {
    "hits": "hits",
    "h": "hits",
    "home runs": "homeRuns",
    "hr": "homeRuns",
    "rbi": "rbi",
    "rbis": "rbi",
    "runs": "runs",
    "r": "runs",
    "total bases": "totalBases",
    "tb": "totalBases",
    "strikeouts": "strikeOuts",
    "k": "strikeOuts",
    "stolen bases": "stolenBases",
    "sb": "stolenBases",
}

PITCHER_STAT_MAP = {
    "strikeouts": "strikeOuts",
    "k": "strikeOuts",
    "hits allowed": "hits",
    "earned runs": "earnedRuns",
    "er": "earnedRuns",
    "walks": "baseOnBalls",
    "bb": "baseOnBalls",
}

ODDS_MARKET_MAP = {
    "hits":        "batter_hits",
    "homeRuns":    "batter_home_runs",
    "rbi":         "batter_rbis",
    "runs":        "batter_runs_scored",
    "totalBases":  "batter_total_bases",
    "stolenBases": "batter_stolen_bases",
    "earnedRuns":  "pitcher_earned_runs",
    "baseOnBalls": "pitcher_walks",
    "hitsRunsRbi": "batter_hits_runs_rbis",
}

MIN_STD = {
    "hits": 0.8, "homeRuns": 0.3, "rbi": 0.8, "runs": 0.7,
    "totalBases": 1.0, "stolenBases": 0.4, "strikeOuts": 1.5,
    "earnedRuns": 1.2, "baseOnBalls": 0.8, "hitsRunsRbi": 1.5,
}

# All integer-count stats benefit from Poisson instead of Normal distribution.
# Normal overestimates/underestimates probability near half-point lines for
# discrete right-skewed stats (e.g. total bases: most games 0–1, spikes to 3–4).
POISSON_STATS = {
    "strikeOuts", "hits", "homeRuns", "totalBases",
    "rbi", "runs", "stolenBases", "baseOnBalls", "earnedRuns",
    "hitsRunsRbi",
}


def _is_whole_number(line: float) -> bool:
    return line == math.floor(line)


def compute_probs(mean: float, std: float, line: float,
                  use_poisson: bool = False) -> dict:
    """
    Returns P(over), P(push), P(under) for a given line.

    Push is only possible on whole-number lines (5.0, 4.0) with Poisson —
    a discrete count can land exactly on the line. Half-point lines (4.5, 5.5)
    can never push, so P(push) = 0.

    Effective win rate accounts for push: P(over) / (P(over) + P(under)).
    This is what to compare against the 50% PrizePicks implied, not raw P(over).
    """
    if use_poisson:
        k = math.floor(line)
        p_over = 1 - poisson.cdf(k, mu=mean)          # P(X >= k+1)
        if _is_whole_number(line):
            p_push = float(poisson.pmf(k, mu=mean))    # P(X = k)
            p_over = 1 - poisson.cdf(k, mu=mean) - p_push  # P(X > k) = P(X >= k+1)
            # re-derive cleanly
            p_over = float(1 - poisson.cdf(int(line), mu=mean))   # P(X >= line+1)
            p_push = float(poisson.pmf(int(line), mu=mean))        # P(X = line)
            p_under = float(poisson.cdf(int(line) - 1, mu=mean))  # P(X <= line-1)
        else:
            p_push = 0.0
            p_over = float(1 - poisson.cdf(k, mu=mean))
            p_under = float(poisson.cdf(k, mu=mean))
    else:
        p_push = 0.0
        if std == 0:
            p_over = 1.0 if mean > line else 0.0
        else:
            p_over = float(1 - norm.cdf(line, loc=mean, scale=std))
        p_under = 1.0 - p_over

    decisive = p_over + p_under  # excludes push
    effective_over_rate = p_over / decisive if decisive > 0 else 0.5

    return {
        "p_over": round(p_over, 4),
        "p_push": round(p_push, 4),
        "p_under": round(p_under, 4),
        "effective_over_rate": round(effective_over_rate, 4),
    }


def model_prob_over(mean: float, std: float, line: float,
                    use_poisson: bool = False) -> float:
    """Convenience wrapper — returns raw P(over) for sportsbook comparisons."""
    return compute_probs(mean, std, line, use_poisson)["p_over"]


def _build_projection(player_name: str, opponent: str,
                      stat_col: str, group: str, fast: bool = False) -> tuple:
    """
    Shared projection logic used by both find_edge and prizepicks_edge.
    Returns (final_mean, base_mean, base_stats, adjustments, data_source).
    """
    n_games = 15 if group == "pitching" else 20
    game_log, data_source = get_recent_games(player_name, n=n_games, group=group)
    game_log = add_derived_stats(game_log, stat_col)

    if group == "pitching":
        base_stats = pitcher_weighted_average(game_log, stat_col)
    else:
        base_stats = weighted_recent_average(game_log, stat_col)

    base_mean = base_stats["mean"]
    adjustments = []

    # IP / ERA context (pitchers only)
    if base_stats.get("ip_context"):
        adjustments.append(f"[IP] {base_stats['ip_context']}")
    if base_stats.get("era_context"):
        adjustments.append(f"[ERA] {base_stats['era_context']}")
    for note in base_stats.get("pitch_count_adjustments", []):
        adjustments.append(f"[pitch-count adj] {note}")
    # Per-start breakdown
    for line in base_stats.get("start_breakdown", []):
        adjustments.append(f"[data] {line}")

    if group == "pitching":
        # Pitchers: blend in history vs this specific team
        vs_team = get_starts_vs_team(player_name, opponent, MLB_RECENT_SEASONS[:2], group=group)
        if not vs_team.empty:
            vs_stats = weighted_recent_average(vs_team, stat_col)
            n = len(vs_team)
            history_weight = min(n / (n + 10), 0.25)
            blended_mean = (1 - history_weight) * base_mean + history_weight * vs_stats["mean"]
            adjustments.append(
                f"vs {opponent} history ({n} start{'s' if n > 1 else ''}, last 2 seasons): "
                f"avg {vs_stats['mean']:.1f} — weight {history_weight:.2f} → {blended_mean:.2f}"
            )
        else:
            blended_mean = base_mean
            adjustments.append(f"vs {opponent} history: no starts found in last 2 seasons")
    else:
        # Batters: look up today's game via the MLB schedule (authoritative),
        # not from the Odds API event which can mis-assign players to games.
        # Blend in career per-PA rate vs the opposing starter, capped at 50%
        # weight because a starter only faces a batter ~50% of PAs.
        AVG_PA_PER_GAME = 4.0
        game_info = get_batter_game_info(player_name)
        if game_info:
            true_opponent      = game_info["opponent"]
            opposing_pitcher   = game_info["opposing_starter"]
            if true_opponent and true_opponent.lower() != opponent.lower():
                adjustments.append(
                    f"[schedule] corrected opponent: {opponent} → {true_opponent}"
                )
            opponent = true_opponent or opponent
        else:
            opposing_pitcher = get_opposing_starter(opponent)

        if opposing_pitcher:
            general_rate = base_mean / AVG_PA_PER_GAME

            if stat_col == "hitsRunsRbi":
                # Sum BvP rates across all three components independently
                sub_hits = get_batter_vs_pitcher(player_name, opposing_pitcher, MLB_RECENT_SEASONS, "hits")
                sub_runs = get_batter_vs_pitcher(player_name, opposing_pitcher, MLB_RECENT_SEASONS, "runs")
                sub_rbi  = get_batter_vs_pitcher(player_name, opposing_pitcher, MLB_RECENT_SEASONS, "rbi")
                sub_valid = [b for b in (sub_hits, sub_runs, sub_rbi)
                             if b["rate_per_pa"] is not None and b["pa"] >= 5]
                if sub_valid:
                    combined_rate  = sum(b["rate_per_pa"] for b in sub_valid)
                    min_pa         = min(b["pa"] for b in sub_valid)
                    matchup_weight = min(min_pa / (min_pa + 20), 0.5)
                    blended_rate   = (1 - matchup_weight) * general_rate + matchup_weight * combined_rate
                    blended_mean   = blended_rate * AVG_PA_PER_GAME
                    adjustments.append(
                        f"vs {opposing_pitcher} H+R+RBI ({len(sub_valid)}/3 components, ≥{min_pa} PA): "
                        f"{combined_rate:.3f}/PA vs {general_rate:.3f}/PA general — "
                        f"weight {matchup_weight:.2f} → {blended_mean:.2f}"
                    )
                else:
                    blended_mean = base_mean
                    adjustments.append(f"vs {opposing_pitcher}: no career H+R+RBI data — using base")
            else:
                bvp = get_batter_vs_pitcher(
                    player_name, opposing_pitcher, MLB_RECENT_SEASONS, stat_col
                )
                if bvp["rate_per_pa"] is not None and bvp["pa"] >= 5:
                    matchup_rate   = bvp["rate_per_pa"]
                    matchup_weight = min(bvp["pa"] / (bvp["pa"] + 20), 0.5)
                    blended_rate   = (1 - matchup_weight) * general_rate + matchup_weight * matchup_rate
                    blended_mean   = blended_rate * AVG_PA_PER_GAME
                    adjustments.append(
                        f"vs {opposing_pitcher} ({bvp['pa']} career PA): "
                        f"{matchup_rate:.3f}/PA vs {general_rate:.3f}/PA general — "
                        f"weight {matchup_weight:.2f} → {blended_mean:.2f}"
                    )
                else:
                    blended_mean = base_mean
                    note = f"({bvp['pa']} PA, need 5+)" if bvp["pa"] > 0 else "(no career PA)"
                    adjustments.append(f"vs {opposing_pitcher} {note} — using base projection")
        else:
            blended_mean = base_mean
            adjustments.append(f"opposing starter not yet announced for {opponent}")

    if group == "pitching" and stat_col == "strikeOuts" and not fast:
        pitcher_id = get_player_id(player_name)
        pvb_factor, pvb_detail = get_pitcher_vs_lineup(pitcher_id, opponent, MLB_RECENT_SEASONS)

        lineup_factor, lineup_detail = get_lineup_k_factor(opponent, MLB_CURRENT_SEASON)

        # When both pvb and lineup are available they measure correlated things:
        # lineup captures batter quality vs all pitchers; pvb captures pitcher-specific
        # history vs those same batters. Applying both at full strength double-stacks.
        # Halve pvb's pull toward 1.0 when lineup is posted so it contributes
        # additional signal without compounding aggressively.
        if lineup_factor != 1.0 and pvb_factor != 1.0:
            pvb_factor_applied = 1.0 + (pvb_factor - 1.0) * 0.5
            adjustments.append(
                f"pitcher-vs-batter: {pvb_detail}\n"
                f"    [halved to ×{pvb_factor_applied:.3f} since lineup K% also applied]"
            )
        else:
            pvb_factor_applied = pvb_factor
            adjustments.append(f"pitcher-vs-batter: {pvb_detail}")

        blended_mean *= pvb_factor_applied
        blended_mean *= lineup_factor
        adjustments.append(f"lineup K%: {lineup_detail}")

        team_factor, team_detail = get_team_k_factor(opponent, MLB_CURRENT_SEASON)
        if lineup_factor == 1.0:
            blended_mean *= team_factor
            adjustments.append(f"team K% (lineup unavailable): {team_detail}")
        else:
            adjustments.append(f"team K% (skipped — lineup factor used): {team_detail}")
    elif group == "pitching" and stat_col == "strikeOuts" and fast:
        team_factor, team_detail = get_team_k_factor(opponent, MLB_CURRENT_SEASON)
        blended_mean *= team_factor
        adjustments.append(f"team K% (fast mode): {team_detail}")

    # ── Historical calibration correction ────────────────────────────────────
    # Apply a bias correction learned from resolved picks in picks.db.
    # bias = avg(projection - actual): positive means we over-project.
    # Only activates once >= 30 resolved picks exist for this stat.
    # Correction is capped at 30% of the raw projection to prevent
    # calibration from dominating on thin or noisy samples.
    cal = _load_calibration()
    if stat_col in cal:
        c    = cal[stat_col]
        bias = c["bias"]                          # positive = over-project
        if abs(bias) >= 0.05:                     # ignore micro-biases < 0.05
            max_correction = blended_mean * _CAL_MAX_CORRECTION
            correction     = max(-max_correction, min(max_correction, bias))
            calibrated     = max(0.0, blended_mean - correction)
            adjustments.append(
                f"[calibration] {c['n']} resolved picks: "
                f"avg proj {c['avg_proj']:.2f} vs avg actual {c['avg_actual']:.2f} "
                f"→ bias {bias:+.3f}, correction {-correction:+.3f} "
                f"→ {blended_mean:.2f} → {calibrated:.2f}  "
                f"(hit rate {c['actual_hit_rate']*100:.0f}% vs "
                f"expected {c['expected_hit_rate']*100:.0f}%)"
            )
            blended_mean = calibrated

    return blended_mean, base_mean, base_stats, adjustments, data_source


def prizepicks_edge(player_name: str, opponent: str, pp_line: float,
                    stat_col: str, event_id: str, group: str = "hitting") -> dict:
    """
    PrizePicks comparison. Pulls sportsbook odds for the true market probability,
    then separately evaluates the model at the PrizePicks line vs 50% implied.
    """
    final_mean, base_mean, base_stats, adjustments, data_source = \
        _build_projection(player_name, opponent, stat_col, group)

    use_poisson = stat_col in POISSON_STATS
    effective_std = max(base_stats["std"], MIN_STD.get(stat_col, 1.0))

    # Sportsbook market probability
    if stat_col == "strikeOuts":
        market = "pitcher_strikeouts" if group == "pitching" else "batter_strikeouts"
    else:
        market = ODDS_MARKET_MAP.get(stat_col)
        if not market:
            raise ValueError(f"No Odds API market mapped for stat '{stat_col}'")

    client = OddsAPIClient()
    props = client.get_player_props(event_id, markets=[market], sport=MLB_SPORT_KEY)
    book_lines = client.extract_player_lines(props, player_name, market)
    if not book_lines:
        raise ValueError(f"No odds found for {player_name} in market '{market}'")

    market_data = consensus_fair_probability(book_lines)
    book_line = market_data["consensus_line"]
    market_fair_prob = market_data["fair_prob_over"]
    book_probs = compute_probs(final_mean, effective_std, book_line, use_poisson=use_poisson)

    # PrizePicks probabilities — push is possible on whole-number lines
    pp_probs = compute_probs(final_mean, effective_std, pp_line, use_poisson=use_poisson)
    pp_edge = round((pp_probs["effective_over_rate"] - 0.5) * 100, 1)

    return {
        "player": player_name,
        "opponent": opponent,
        "stat": stat_col,
        "book_line": book_line,
        "pp_line": pp_line,
        "base_projection": round(base_mean, 2),
        "adjusted_projection": round(final_mean, 2),
        # sportsbook
        "model_prob_at_book_line": book_probs["p_over"],
        "market_fair_prob_over": round(market_fair_prob, 3),
        "market_edge_pp": round((book_probs["p_over"] - market_fair_prob) * 100, 1),
        # prizepicks
        "pp_p_over": pp_probs["p_over"],
        "pp_p_push": pp_probs["p_push"],
        "pp_p_under": pp_probs["p_under"],
        "pp_effective_over_rate": pp_probs["effective_over_rate"],
        "pp_edge_pp": pp_edge,
        "pp_has_push": _is_whole_number(pp_line),
        # meta
        "market_vig_pct": round(market_data["avg_vig_pct"], 1),
        "n_books": market_data["n_books"],
        "games_used": base_stats["n_games"],
        "data_source": data_source,
        "adjustments": adjustments,
        "distribution": "Poisson" if use_poisson else "Normal",
    }


def find_edge(player_name: str, opponent: str, event_id: str,
              stat_col: str, group: str = "hitting") -> dict:
    """
    Full pipeline: project stat (with opponent adjustments) -> fetch odds -> edge.
    group='hitting' for batters, group='pitching' for pitchers.
    """
    final_mean, base_mean, base_stats, adjustments, data_source = \
        _build_projection(player_name, opponent, stat_col, group)

    use_poisson = stat_col in POISSON_STATS
    effective_std = max(base_stats["std"], MIN_STD.get(stat_col, 1.0))

    # --- Odds ---
    if stat_col == "strikeOuts":
        market = "pitcher_strikeouts" if group == "pitching" else "batter_strikeouts"
    else:
        market = ODDS_MARKET_MAP.get(stat_col)
        if not market:
            raise ValueError(f"No Odds API market mapped for stat '{stat_col}'")

    client = OddsAPIClient()
    props = client.get_player_props(event_id, markets=[market], sport=MLB_SPORT_KEY)
    book_lines = client.extract_player_lines(props, player_name, market)
    if not book_lines:
        raise ValueError(f"No odds found for {player_name} in market '{market}'")

    market_data = consensus_fair_probability(book_lines)
    line = market_data["consensus_line"]

    book_probs    = compute_probs(final_mean, effective_std, line, use_poisson=use_poisson)
    model_p_over  = book_probs["p_over"]
    compare_p     = book_probs["effective_over_rate"] if use_poisson else model_p_over
    market_p_over = market_data["fair_prob_over"]

    return {
        "player": player_name,
        "opponent": opponent,
        "stat": stat_col,
        "line": line,
        "base_projection": round(base_mean, 2),
        "adjusted_projection": round(final_mean, 2),
        "model_prob_over": round(model_p_over, 3),
        "market_fair_prob_over": round(market_p_over, 3),
        "edge_pct_pts": round((compare_p - market_p_over) * 100, 1),
        "market_vig_pct": round(market_data["avg_vig_pct"], 1),
        "n_books": market_data["n_books"],
        "games_used": base_stats["n_games"],
        "data_source": data_source,
        "adjustments": adjustments,
        "distribution": "Poisson" if use_poisson else "Normal",
    }


def _print_result(result: dict) -> None:
    print(f"  Player:               {result['player']} vs {result['opponent']}")
    print(f"  Stat:                 {result['stat']}")
    print(f"  Base projection:      {result['base_projection']}  ({result['games_used']} starts, {result['data_source']})")
    print(f"  Adjusted projection:  {result['adjusted_projection']}  ({result['distribution']})")
    print()
    print("  Adjustments applied:")
    for adj in result["adjustments"]:
        print(f"    • {adj}")
    print()

    if "pp_line" in result:
        print(f"  ── Sportsbook (line {result['book_line']}, {result['n_books']} books, vig {result['market_vig_pct']}%) ──")
        print(f"  Model prob over:   {result['model_prob_at_book_line']*100:.1f}%")
        print(f"  Market fair prob:  {result['market_fair_prob_over']*100:.1f}%")
        book_lean = "OVER" if result["market_edge_pp"] > 0 else "UNDER"
        print(f"  Edge vs market:    {result['market_edge_pp']:+.1f} pp  → lean {book_lean}")
        print()
        push_note = " (push possible — whole number line)" if result["pp_has_push"] else ""
        print(f"  ── PrizePicks (line {result['pp_line']}{push_note}) ──")
        print(f"  P(over):           {result['pp_p_over']*100:.1f}%")
        if result["pp_has_push"]:
            print(f"  P(push):           {result['pp_p_push']*100:.1f}%  ← money back")
        print(f"  P(under):          {result['pp_p_under']*100:.1f}%")
        if result["pp_has_push"]:
            print(f"  Effective over %:  {result['pp_effective_over_rate']*100:.1f}%  (excl. push)")
        print(f"  PP implied:        50.0%")
        pp_lean = "OVER" if result["pp_edge_pp"] > 0 else "UNDER"
        print(f"  Edge vs PP:        {result['pp_edge_pp']:+.1f} pp  → lean {pp_lean}")
    else:
        # Standard sportsbook result
        print(f"  Line:              {result['line']}")
        print(f"  Model prob over:   {result['model_prob_over']*100:.1f}%")
        print(f"  Market fair prob:  {result['market_fair_prob_over']*100:.1f}%  ({result['n_books']} books, vig {result['market_vig_pct']}%)")
        lean = "OVER" if result["edge_pct_pts"] > 0 else "UNDER"
        print(f"  Edge:              {result['edge_pct_pts']:+.1f} pp  → lean {lean}")


def _fetch_pp_lines(stat_type: str = "Pitcher Strikeouts") -> dict:
    """
    Pulls today's PrizePicks lines from their projection API.
    Returns {player_name_lower: line_score} for the given stat type.
    Falls back to {} on any error so the batch can continue without PP data.
    """
    try:
        resp = _requests.get(
            "https://api.prizepicks.com/projections",
            params={"league_id": 2, "per_page": 250, "single_stat": "true"},
            headers={
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
                "Accept": "application/json, text/plain, */*",
                "Referer": "https://app.prizepicks.com/",
                "Origin": "https://app.prizepicks.com",
            },
            timeout=10,
        )
        if resp.status_code in (403, 429):
            print("  [PP] API blocked (Cloudflare protection) — PP lines will be estimated from book line")
            return {}
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"  [PP] API unavailable ({e}) — PP lines will be estimated from book line")
        return {}

    # Build player id -> name from included objects
    player_map: dict = {}
    for item in data.get("included", []):
        if item.get("type") == "new_player":
            player_map[item["id"]] = item["attributes"].get("name", "")

    lines: dict = {}
    for proj in data.get("data", []):
        attrs = proj.get("attributes", {})
        if attrs.get("stat_type", "").lower() != stat_type.lower():
            continue
        line = attrs.get("line_score")
        if line is None:
            continue
        pid = proj.get("relationships", {}).get("new_player", {}).get("data", {}).get("id")
        name = player_map.get(pid, "").strip()
        if name:
            lines[name.lower()] = float(line)

    return lines


def _match_pp_line(player_name: str, pp_lines: dict) -> float | None:
    """
    Matches an Odds API player name to a PrizePicks line.
    Tries exact lowercase match first, then last-name match as fallback.
    """
    key = player_name.lower().strip()
    if key in pp_lines:
        return pp_lines[key]
    last = key.split()[-1]
    for pp_name, line in pp_lines.items():
        if pp_name.split()[-1] == last:
            return line
    return None


def _normalize_name(name: str) -> str:
    """Lowercase + strip accents so 'Sánchez' == 'Sanchez' in comparisons."""
    return ''.join(
        c for c in unicodedata.normalize('NFD', name.lower().strip())
        if unicodedata.category(c) != 'Mn'
    )


def _build_pitcher_opponent_map() -> dict:
    """
    Pulls today's MLB schedule from statsapi and returns a dict mapping
    accent-normalized lowercase pitcher name → opponent team name.

    Home pitchers face the away team; away pitchers face the home team.
    Only probable starters are included — blank entries are skipped.
    """
    from datetime import date
    today = date.today().strftime("%m/%d/%Y")
    try:
        games = statsapi.schedule(date=today, sportId=1)
    except Exception:
        return {}

    mapping: dict = {}
    for game in games:
        home = game.get("home_name", "")
        away = game.get("away_name", "")
        home_pitcher = game.get("home_probable_pitcher", "").strip()
        away_pitcher = game.get("away_probable_pitcher", "").strip()
        if home_pitcher:
            mapping[_normalize_name(home_pitcher)] = away
        if away_pitcher:
            mapping[_normalize_name(away_pitcher)] = home
    return mapping


def _resolve_opponent(player_name: str, schedule_map: dict,
                      home_team: str, away_team: str) -> str | None:
    """
    Returns the opponent team name for a pitcher, or None if it can't be resolved.
    Tries the statsapi schedule first (exact, then last-name match), then falls
    back to a currentTeam lookup against the event's home/away teams.
    All comparisons are accent-normalized so 'Sanchez' matches 'Sánchez'.
    """
    name_norm = _normalize_name(player_name)

    # Exact match against schedule (accent-normalized)
    if name_norm in schedule_map:
        return schedule_map[name_norm]

    # Last-name match (handles middle initials, spelling variants)
    last_name = name_norm.split()[-1]
    for sched_name, opponent in schedule_map.items():
        if sched_name.split()[-1] == last_name:
            return opponent

    # Fall back to statsapi currentTeam
    try:
        matches = statsapi.lookup_player(player_name)
        if matches:
            current_team = matches[0].get("currentTeam", {}).get("name", "").strip()
            if current_team:
                home_l = _normalize_name(home_team)
                away_l = _normalize_name(away_team)
                ct_l   = _normalize_name(current_team)
                if ct_l in home_l or home_l in ct_l:
                    return away_team
                if ct_l in away_l or away_l in ct_l:
                    return home_team
    except Exception:
        pass

    return None


def batch_scan(stat_col: str = "strikeOuts", group: str = "pitching",
               pp_assume_round: bool = True) -> list:
    """
    Scans all today's MLB games for pitcher_strikeouts props, runs projections
    for every listed pitcher, and returns results sorted by absolute market edge.

    fast=True is used internally — lineup and pitcher-vs-batter matchup calls are
    skipped since lineups aren't posted at scan time. Team K% is still applied.
    """
    market = "pitcher_strikeouts" if group == "pitching" else "batter_strikeouts"
    client = OddsAPIClient()

    print("Fetching upcoming MLB events...")
    events = client.get_upcoming_events(sport=MLB_SPORT_KEY)
    if not events:
        print("No upcoming MLB events found on this API plan.")
        return []

    print(f"  {len(events)} events found. Pulling {market} props...\n")

    # Collect all pitcher props grouped by player across all events
    all_players: dict = {}  # player_name -> {event_id, home_team, away_team, book_lines, props}

    for event in events:
        eid = event["id"]
        home = event["home_team"]
        away = event["away_team"]
        try:
            props = client.get_player_props(eid, markets=[market], sport=MLB_SPORT_KEY)
        except Exception as e:
            print(f"  [{home} vs {away}] props fetch error: {e}")
            continue

        # Collect all player names that appear in any bookmaker's markets
        players_in_event: set = set()
        for bookmaker in props.get("bookmakers", []):
            for m in bookmaker.get("markets", []):
                if m["key"] != market:
                    continue
                for outcome in m["outcomes"]:
                    name = outcome.get("description", "").strip()
                    if name:
                        players_in_event.add(name)

        for player_name in players_in_event:
            book_lines = client.extract_player_lines(props, player_name, market)
            if book_lines:
                all_players[player_name] = {
                    "event_id": eid,
                    "home_team": home,
                    "away_team": away,
                    "book_lines": book_lines,
                }

    if not all_players:
        print("No pitcher strikeout props found for today's games.")
        return []

    print("  Building today's probable starter map from MLB schedule...")
    schedule_map = _build_pitcher_opponent_map()
    print(f"  {len(schedule_map)} probable starters found in schedule.")

    print(f"  {len(all_players)} pitchers listed. Running projections...\n")

    results = []

    for player_name, event_data in all_players.items():
        opponent = _resolve_opponent(
            player_name, schedule_map,
            event_data["home_team"], event_data["away_team"],
        )
        if opponent is None:
            print(f"  [{player_name}] could not resolve opponent — skipping")
            continue

        try:
            final_mean, base_mean, base_stats, adjustments, _ = \
                _build_projection(player_name, opponent, stat_col, group, fast=True)
        except Exception as e:
            print(f"  [{player_name}] projection error: {e} — skipping")
            continue

        # Market edge
        market_data = consensus_fair_probability(event_data["book_lines"])
        book_line = market_data["consensus_line"]
        market_fair_prob = market_data["fair_prob_over"]
        use_poisson = stat_col in POISSON_STATS
        effective_std = max(base_stats["std"], MIN_STD.get(stat_col, 1.0))

        book_probs = compute_probs(final_mean, effective_std, book_line, use_poisson=use_poisson)
        model_p_over = book_probs["p_over"]
        # For whole-number Poisson lines, push probability is non-trivial and the
        # market doesn't price it — use effective_over_rate (push-excluded) so we
        # compare decisive-outcome probabilities on equal footing.
        # For half-integer lines, effective_over_rate == p_over (no push possible).
        compare_p = book_probs["effective_over_rate"] if use_poisson else model_p_over
        market_edge = round((compare_p - market_fair_prob) * 100, 1)

        pp_line = round(book_line) if pp_assume_round else book_line
        pp_probs = compute_probs(final_mean, effective_std, pp_line, use_poisson=use_poisson)
        pp_edge = round((pp_probs["effective_over_rate"] - 0.5) * 100, 1)
        lean = "OVER" if market_edge > 0 else "UNDER"

        results.append({
            "player": player_name,
            "opponent": opponent,
            "line": book_line,
            "pp_line": pp_line,
            "projection": round(final_mean, 2),
            "model_prob_over": round(model_p_over, 3),
            "market_fair_prob": round(market_fair_prob, 3),
            "market_edge": market_edge,
            "pp_effective_over_rate": round(pp_probs["effective_over_rate"], 3),
            "pp_edge": pp_edge,
            "abs_market_edge": abs(market_edge),
            "n_books": market_data["n_books"],
            "games_used": base_stats["n_games"],
            "lean": lean,
            "adjustments": adjustments,
        })

    # Sort by absolute market edge, highest first
    results.sort(key=lambda x: x["abs_market_edge"], reverse=True)

    # Print ranked table
    print()
    print("  ── RANKED BY EDGE (highest first) ──\n")
    print(f"  {'#':<3} {'Player':<25} {'vs':<22} {'Line':>5}  {'Proj':>5}  {'Mdl%':>6}  {'Mkt%':>6}  {'Edge':>7}  {'PP~':>4}  {'Eff%':>6}  {'PP Edge':>8}  {'Lean'}")
    print(f"  {'-'*3} {'-'*25} {'-'*22} {'-'*5}  {'-'*5}  {'-'*6}  {'-'*6}  {'-'*7}  {'-'*4}  {'-'*6}  {'-'*8}  {'-'*6}")
    for i, r in enumerate(results, 1):
        edge_str = f"{r['market_edge']:+.1f} pp"
        pp_edge_str = f"{r['pp_edge']:+.1f} pp"
        print(
            f"  {i:<3} {r['player']:<25} {r['opponent']:<22} {r['line']:>5.1f}  "
            f"{r['projection']:>5.2f}  {r['model_prob_over']*100:>5.1f}%  "
            f"{r['market_fair_prob']*100:>5.1f}%  {edge_str:>7}  "
            f"{r['pp_line']:>4.0f}  {r['pp_effective_over_rate']*100:>5.1f}%  "
            f"{pp_edge_str:>8}  {r['lean']}"
        )

    print(f"\n  Total: {len(results)} pitchers scanned across {len(events)} games  (PP~ = estimated from book line)")
    return results


if __name__ == "__main__":
    print("\n=== MLB Player Prop Projector ===\n")
    mode = input("Mode [single / batch] (default: single): ").strip().lower() or "single"

    if mode == "batch":
        print("\nBatch scan: pitcher strikeouts for all today's games\n")
        batch_scan(stat_col="strikeOuts", group="pitching", pp_assume_round=True)
        exit(0)

    player = input("Player full name (e.g. Colin Rea): ").strip()
    opponent = input("Opponent team (e.g. Cubs or Chicago Cubs): ").strip()
    role = input("Role [batter / pitcher] (default: batter): ").strip().lower() or "batter"
    group = "pitching" if role == "pitcher" else "hitting"

    stat_map = PITCHER_STAT_MAP if group == "pitching" else BATTER_STAT_MAP
    stat_options = " / ".join(sorted(set(stat_map.keys())))
    stat = input(f"Stat [{stat_options}]: ").strip().lower()
    stat_col = stat_map.get(stat)
    if not stat_col:
        print(f"Unknown stat '{stat}'. Valid: {stat_options}")
        exit(1)

    source = input("Source [sportsbook / prizepicks] (default: sportsbook): ").strip().lower() or "sportsbook"

    client = OddsAPIClient()
    print("\nFetching upcoming MLB events...\n")
    events = client.get_upcoming_events(sport=MLB_SPORT_KEY)
    if not events:
        print("No upcoming MLB events found on this API plan.")
        exit(1)
    for e in events[:20]:
        print(f"  {e['id']}  |  {e['home_team']} vs {e['away_team']}  |  {e['commence_time']}")
    print()
    event_id = input("Event ID: ").strip()

    print(f"\nFetching data for {player} vs {opponent}...\n")

    if source == "prizepicks":
        pp_line = float(input("PrizePicks line (e.g. 4.0): ").strip())
        result = prizepicks_edge(player, opponent, pp_line, stat_col, event_id, group=group)
    else:
        result = find_edge(player, opponent, event_id, stat_col, group=group)

    _print_result(result)
