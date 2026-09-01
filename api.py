"""
FastAPI backend for the MLB props model.
Run with: uvicorn api:app --reload
"""

import asyncio
import contextlib, datetime, io, os, sys, time
from contextlib import asynccontextmanager
import pandas as _pd
import statsapi as _statsapi
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import database as _db

try:
    from zoneinfo import ZoneInfo as _ZoneInfo
    _CDT = _ZoneInfo("America/Chicago")
except ImportError:
    _CDT = datetime.timezone(datetime.timedelta(hours=-5))

# ── Server-side cache ─────────────────────────────────────────────────────────
# Odds API calls are expensive (token-limited). Cache results for CACHE_TTL
# seconds so page reloads don't trigger new API calls.
CACHE_TTL = 1200  # 20 minutes

_cache: dict = {}   # key -> {"data": any, "ts": float, "date": str}

def _cache_get(key: str):
    entry = _cache.get(key)
    if not entry:
        return None
    today = datetime.date.today().isoformat()
    if entry["date"] != today:
        return None                      # stale from yesterday
    if time.time() - entry["ts"] > CACHE_TTL:
        return None                      # TTL expired
    return entry["data"]

def _cache_set(key: str, data):
    _cache[key] = {"data": data, "ts": time.time(), "date": datetime.date.today().isoformat()}

def _cache_clear(key: str = None):
    if key:
        _cache.pop(key, None)
    else:
        _cache.clear()

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

from config import MLB_SPORT_KEY, MLB_RECENT_SEASONS
from data.odds_client import OddsAPIClient
from analysis.devig import consensus_fair_probability
from mlb_main import (
    _build_projection,
    _build_pitcher_opponent_map,
    _resolve_opponent,
    compute_probs,
    prizepicks_edge,
    find_edge,
    MIN_STD,
    POISSON_STATS,
    _fetch_pp_lines,
    _match_pp_line,
)

# PrizePicks display name for each Odds API market key
PP_STAT_TYPES = {
    "pitcher_strikeouts":   "Pitcher Strikeouts",
    "pitcher_earned_runs":  "Pitcher Earned Runs",
    "pitcher_walks":        "Pitcher Walks",
    "batter_hits":          "Hits",
    "batter_home_runs":     "Home Runs",
    "batter_rbis":          "RBIs",
    "batter_runs_scored":   "Runs",
    "batter_total_bases":   "Total Bases",
    "batter_stolen_bases":  "Stolen Bases",
    "batter_hits_runs_rbis":"Hits+Runs+RBIs",
}

def _seconds_until_midnight_cdt() -> float:
    """Seconds from now until 12:05 AM CDT (gives MLB Stats API time to post final boxscores)."""
    now = datetime.datetime.now(tz=_CDT)
    next_run = (now + datetime.timedelta(days=1)).replace(
        hour=0, minute=5, second=0, microsecond=0
    )
    return max((next_run - now).total_seconds(), 0)


def _seconds_until_morning_cdt(hour: int = 10, minute: int = 30) -> float:
    """Seconds from now until the next 10:30 AM CDT (lines are reliably posted by then)."""
    now = datetime.datetime.now(tz=_CDT)
    target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if now >= target:
        target += datetime.timedelta(days=1)
    return max((target - now).total_seconds(), 0)


async def _nightly_resolve_loop():
    """Background task: auto-resolve picks every night at 12:05 AM CDT."""
    while True:
        wait = _seconds_until_midnight_cdt()
        await asyncio.sleep(wait)
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, _do_resolve)
        except Exception:
            pass


_last_auto_project: dict  = {"ran_at": None, "count": 0, "status": "never run"}
_auto_project_running: bool = False


async def _morning_project_loop():
    """Background task: auto-run all market projections every morning at 10:30 AM CDT."""
    while True:
        wait = _seconds_until_morning_cdt()
        await asyncio.sleep(wait)
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, _do_auto_project)
        except Exception:
            pass


def _do_resolve():
    """Synchronous resolve logic — called from the background task and the HTTP endpoint."""
    from data.mlb_stats import get_game_log, add_derived_stats

    unresolved = _db.get_picks(unresolved_only=True, past_only=True)
    resolved = failed = skipped = 0

    for pick in unresolved:
        try:
            target = datetime.date.fromisoformat(pick["date"])
            group  = pick["group_type"]
            stat   = pick["stat_col"]

            df = get_game_log(pick["player"], season=target.year, group=group)
            if df.empty:
                failed += 1
                continue

            df = add_derived_stats(df, stat)
            if stat not in df.columns:
                failed += 1
                continue

            df["date"] = _pd.to_datetime(df["date"])
            game = df[df["date"].dt.date == target]
            if game.empty:
                skipped += 1
                continue

            actual = float(game[stat].iloc[0])
            lean   = pick["lean"]
            line   = pick["line"]
            hit    = 1 if (lean == "OVER" and actual > line) or \
                          (lean == "UNDER" and actual < line) else 0
            _db.resolve_pick(pick["id"], actual, hit)
            resolved += 1
        except Exception:
            failed += 1

    return {"resolved": resolved, "skipped": skipped, "failed": failed}


def _do_auto_project() -> dict:
    """Run all markets and snapshot projections to DB. Called by the morning loop and manually."""
    global _last_auto_project, _auto_project_running
    all_keys = list(MARKET_CONFIGS.keys())
    try:
        results = _silence(_batch_scan, all_keys)
        if results:
            _db.save_picks(results)
            # Also push results into the server cache so the board reflects them immediately
            for mkey in all_keys:
                cache_key = "batch:" + ",".join(sorted(all_keys))
                _cache_set(cache_key, results)
            _cache_clear("players")  # force player list refresh too
        _last_auto_project = {
            "ran_at": datetime.datetime.now(tz=_CDT).isoformat(),
            "count":  len(results),
            "status": "ok",
        }
        return {"projected": len(results)}
    except Exception as e:
        _last_auto_project = {
            "ran_at": datetime.datetime.now(tz=_CDT).isoformat(),
            "count":  0,
            "status": f"error: {e}",
        }
        return {"projected": 0, "error": str(e)}
    finally:
        _auto_project_running = False


@asynccontextmanager
async def lifespan(app):
    resolve_task = asyncio.create_task(_nightly_resolve_loop())
    project_task = asyncio.create_task(_morning_project_loop())
    yield
    resolve_task.cancel()
    project_task.cancel()
    for t in (resolve_task, project_task):
        try:
            await t
        except asyncio.CancelledError:
            pass


app = FastAPI(title="MLB Props Model", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# All Odds API markets mapped to internal stat/group config
MARKET_CONFIGS = {
    "pitcher_strikeouts": {"stat_col": "strikeOuts",   "group": "pitching", "label": "Pitcher K's"},
    "pitcher_earned_runs":{"stat_col": "earnedRuns",   "group": "pitching", "label": "Earned Runs"},
    "pitcher_walks":      {"stat_col": "baseOnBalls",  "group": "pitching", "label": "Walks (P)"},
    "batter_hits":        {"stat_col": "hits",         "group": "hitting",  "label": "Hits"},
    "batter_total_bases": {"stat_col": "totalBases",   "group": "hitting",  "label": "Total Bases"},
    "batter_home_runs":   {"stat_col": "homeRuns",     "group": "hitting",  "label": "Home Runs"},
    "batter_rbis":        {"stat_col": "rbi",          "group": "hitting",  "label": "RBI"},
    "batter_runs_scored": {"stat_col": "runs",         "group": "hitting",  "label": "Runs"},
    "batter_stolen_bases":   {"stat_col": "stolenBases",  "group": "hitting",  "label": "Stolen Bases"},
    "batter_hits_runs_rbis": {"stat_col": "hitsRunsRbi",  "group": "hitting",  "label": "H+R+RBI"},
}

_FRONTEND = os.path.join(os.path.dirname(__file__), "frontend", "index.html")


@app.get("/", response_class=HTMLResponse)
def root():
    with open(_FRONTEND) as f:
        return f.read()


@app.get("/api/events")
def get_events():
    try:
        client = OddsAPIClient()
        events = client.get_upcoming_events(sport=MLB_SPORT_KEY)
        return {"events": [
            {
                "id": e["id"],
                "home_team": e["home_team"],
                "away_team": e["away_team"],
                "commence_time": e["commence_time"],
            }
            for e in events
        ]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/players")
def get_players(refresh: bool = False):
    """
    Returns every player with props today, with their game and resolved opponent.
    Results are cached for 20 minutes — pass ?refresh=true to force a re-fetch.
    """
    cache_key = "players"
    if not refresh:
        cached = _cache_get(cache_key)
        if cached is not None:
            return {"players": cached, "cached": True}

    try:
        client = OddsAPIClient()
        events = _silence(client.get_upcoming_events, sport=MLB_SPORT_KEY)
        if not events:
            return {"players": []}

        schedule_map = _silence(_build_pitcher_opponent_map)
        all_keys = list(MARKET_CONFIGS.keys())
        players: dict = {}

        for event in events:
            eid, home, away = event["id"], event["home_team"], event["away_team"]
            try:
                props = _silence(
                    client.get_player_props, eid, markets=all_keys, sport=MLB_SPORT_KEY
                )
            except Exception:
                continue

            for mkey, cfg in MARKET_CONFIGS.items():
                group = cfg["group"]
                for bookmaker in props.get("bookmakers", []):
                    for m in bookmaker.get("markets", []):
                        if m["key"] != mkey:
                            continue
                        for outcome in m["outcomes"]:
                            name = outcome.get("description", "").strip()
                            if not name or name in players:
                                continue
                            smap = schedule_map if group == "pitching" else {}
                            opponent = _silence(
                                _resolve_opponent, name, smap, home, away
                            )
                            players[name] = {
                                "name":       name,
                                "group":      group,
                                "event_id":   eid,
                                "home_team":  home,
                                "away_team":  away,
                                "opponent":   opponent or "",
                            }

        result = sorted(players.values(), key=lambda x: x["name"])
        _cache_set(cache_key, result)
        return {"players": result, "cached": False}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/resolve-player")
def resolve_player(name: str):
    """
    Looks up a player by name in the MLB Stats API schedule for today.
    Returns their game, opponent, role, and matching Odds API event_id.
    Falls back gracefully if no odds event is found (user can still run
    a projection once lines are posted).
    """
    try:
        matches = _statsapi.lookup_player(name)
        if not matches:
            raise HTTPException(status_code=404, detail=f"No MLB player found for '{name}'")

        player    = matches[0]
        full_name = player.get("fullName", name)
        pos_abbr  = player.get("primaryPosition", {}).get("abbreviation", "")
        group     = "pitching" if pos_abbr == "P" else "hitting"
        team_info = player.get("currentTeam") or {}
        team_id   = team_info.get("id")
        team_name = (team_info.get("name") or "").strip()

        # Find today's game from the MLB schedule
        today = datetime.date.today().strftime("%m/%d/%Y")
        games = _statsapi.schedule(date=today, sportId=1)

        mlb_home = mlb_away = opponent = ""
        for game in games:
            home    = game.get("home_name", "")
            away    = game.get("away_name", "")
            home_id = game.get("home_id")
            away_id = game.get("away_id")

            if team_id and team_id in (home_id, away_id):
                if team_id == home_id:
                    mlb_home, mlb_away, opponent = home, away, away
                else:
                    mlb_home, mlb_away, opponent = home, away, home
                break
            elif team_name:
                ct = team_name.lower()
                if ct in home.lower() or home.lower() in ct:
                    mlb_home, mlb_away, opponent = home, away, away
                    break
                if ct in away.lower() or away.lower() in ct:
                    mlb_home, mlb_away, opponent = home, away, home
                    break

        if not opponent:
            raise HTTPException(
                status_code=404,
                detail=f"{full_name} has no game scheduled today"
            )

        # Try to match to an Odds API event by team name keyword
        client   = OddsAPIClient()
        events   = _silence(client.get_upcoming_events, sport=MLB_SPORT_KEY)
        event_id = ""
        odds_home = mlb_home
        odds_away = mlb_away

        def _team_keyword(t: str) -> str:
            # Use the last word (city-less nickname) for matching: "Astros", "Dodgers"…
            return t.strip().split()[-1].lower()

        kw_home = _team_keyword(mlb_home)
        kw_away = _team_keyword(mlb_away)

        for ev in events:
            eh = ev["home_team"].lower()
            ea = ev["away_team"].lower()
            if kw_home in eh and kw_away in ea:
                event_id  = ev["id"]
                odds_home = ev["home_team"]
                odds_away = ev["away_team"]
                break

        return {
            "name":       full_name,
            "group":      group,
            "event_id":   event_id,
            "home_team":  odds_home,
            "away_team":  odds_away,
            "opponent":   opponent,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/batch")
def api_batch(markets: str = "pitcher_strikeouts", refresh: bool = False):
    keys = [k.strip() for k in markets.split(",") if k.strip() in MARKET_CONFIGS]
    if not keys:
        raise HTTPException(status_code=400, detail="No valid markets specified")
    cache_key = "batch:" + ",".join(sorted(keys))
    if not refresh:
        cached = _cache_get(cache_key)
        if cached is not None:
            return {"results": cached, "count": len(cached), "cached": True}
    try:
        results = _batch_scan(keys)
        _cache_set(cache_key, results)
        # Auto-snapshot today's projections into the tracker DB
        try:
            _db.save_picks(results)
        except Exception:
            pass
        return {"results": results, "count": len(results), "cached": False}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/cache-status")
def cache_status():
    now = time.time()
    today = datetime.date.today().isoformat()
    out = {}
    for key, entry in _cache.items():
        age = int(now - entry["ts"])
        out[key] = {
            "age_seconds": age,
            "expires_in": max(0, CACHE_TTL - age),
            "stale": entry["date"] != today or age > CACHE_TTL,
        }
    return out


@app.post("/api/cache-clear")
def cache_clear():
    _cache_clear()
    return {"cleared": True}


class ProjectRequest(BaseModel):
    player: str
    opponent: str
    event_id: str
    stat_col: str
    group: str
    source: str = "sportsbook"
    pp_line: Optional[float] = None


@app.post("/api/project")
def api_project(body: ProjectRequest):
    try:
        if body.source == "prizepicks" and body.pp_line is not None:
            result = prizepicks_edge(
                body.player, body.opponent, body.pp_line,
                body.stat_col, body.event_id, group=body.group,
            )
        else:
            result = find_edge(
                body.player, body.opponent, body.event_id,
                body.stat_col, group=body.group,
            )
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ── Picks tracker ─────────────────────────────────────────────────────────────

@app.get("/api/picks")
def get_picks(date: str = None):
    """Return all saved picks, optionally filtered to a specific date (YYYY-MM-DD)."""
    return {"picks": _db.get_picks(date=date)}


@app.get("/api/picks/stats")
def picks_stats():
    """Calibration summary: win rate by edge tier and by market."""
    return _db.get_stats()


@app.get("/api/calibration")
def get_calibration():
    """Per-stat projection bias learned from historical picks."""
    return _db.get_calibration_data(min_samples=30)


class SavePicksRequest(BaseModel):
    results: list
    date: Optional[str] = None


@app.post("/api/picks/save")
def save_picks(body: SavePicksRequest):
    """Snapshot the current board results into the database."""
    result = _db.save_picks(body.results, date=body.date)
    return result


class ResolveManualRequest(BaseModel):
    pick_id: int
    actual: float


@app.post("/api/picks/resolve-manual")
def resolve_manual(body: ResolveManualRequest):
    """Manually set the actual value for a pick."""
    ok = _db.manual_resolve(body.pick_id, body.actual)
    if not ok:
        raise HTTPException(status_code=404, detail="Pick not found")
    return {"ok": True}


@app.post("/api/picks/resolve")
def resolve_picks():
    """Auto-resolve unresolved picks from past dates using actual MLB stats."""
    return _do_resolve()


@app.get("/api/auto-project/status")
def auto_project_status():
    """Next scheduled auto-projection run time, running state, and last run summary."""
    wait = _seconds_until_morning_cdt()
    next_run = (datetime.datetime.now(tz=_CDT) + datetime.timedelta(seconds=wait))
    return {
        "next_run_cdt":  next_run.strftime("%Y-%m-%d %H:%M CDT"),
        "seconds_until": int(wait),
        "running":       _auto_project_running,
        "last_run":      _last_auto_project,
    }


@app.post("/api/auto-project/run")
async def run_auto_project():
    """Kick off projection run as a background task — returns immediately."""
    global _auto_project_running
    if _auto_project_running:
        return {"status": "already_running"}
    _auto_project_running = True
    loop = asyncio.get_event_loop()
    loop.run_in_executor(None, _do_auto_project)
    return {"status": "started"}


@app.post("/api/picks/{pick_id}/unresolve")
def unresolve_pick(pick_id: int):
    """Clear the actual/hit values so a pick can be re-entered."""
    _db.unresolve_pick(pick_id)
    return {"ok": True}


@app.post("/api/picks/import")
def import_picks(body: SavePicksRequest):
    """Bulk-import raw pick rows from another DB instance (e.g. local → Railway)."""
    return _db.import_picks(body.results)


@app.delete("/api/picks/{pick_id}")
def delete_pick(pick_id: int):
    _db.delete_pick(pick_id)
    return {"ok": True}


# ── Batch engine ──────────────────────────────────────────────────────────────

def _silence(fn, *args, **kwargs):
    """Call fn suppressing all stdout chatter from the model pipeline."""
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args, **kwargs)


def _batch_scan(market_keys: list, pp_assume_round: bool = True) -> list:
    client = OddsAPIClient()

    events = _silence(client.get_upcoming_events, sport=MLB_SPORT_KEY)
    if not events:
        return []

    schedule_map = _silence(_build_pitcher_opponent_map)

    # Try to pull real PrizePicks lines for each market — returns {} if blocked
    pp_by_market: dict = {}
    for mkey in market_keys:
        stat_type = PP_STAT_TYPES.get(mkey)
        if stat_type:
            pp_by_market[mkey] = _silence(_fetch_pp_lines, stat_type)
        else:
            pp_by_market[mkey] = {}

    # One API call per event fetching ALL requested markets at once
    entries: dict = {}  # (player_name, market_key) -> data

    for event in events:
        eid, home, away = event["id"], event["home_team"], event["away_team"]
        try:
            props = _silence(
                client.get_player_props, eid, markets=market_keys, sport=MLB_SPORT_KEY
            )
        except Exception:
            continue

        for mkey in market_keys:
            cfg = MARKET_CONFIGS[mkey]
            seen: set = set()

            for bookmaker in props.get("bookmakers", []):
                for m in bookmaker.get("markets", []):
                    if m["key"] != mkey:
                        continue
                    for outcome in m["outcomes"]:
                        name = outcome.get("description", "").strip()
                        if name:
                            seen.add(name)

            for player_name in seen:
                book_lines = client.extract_player_lines(props, player_name, mkey)
                if book_lines:
                    entries[(player_name, mkey)] = {
                        "event_id": eid,
                        "home_team": home,
                        "away_team": away,
                        "book_lines": book_lines,
                        **cfg,
                    }

    results = []

    for (player_name, mkey), data in entries.items():
        smap = schedule_map if data["group"] == "pitching" else {}
        opponent = _silence(
            _resolve_opponent, player_name, smap,
            data["home_team"], data["away_team"],
        )
        if not opponent:
            continue

        try:
            final_mean, _, base_stats, adjustments, _ = _silence(
                _build_projection,
                player_name, opponent, data["stat_col"], data["group"], fast=True,
            )
        except Exception:
            continue

        market_data = consensus_fair_probability(data["book_lines"])
        book_line = market_data["consensus_line"]
        market_fair_prob = market_data["fair_prob_over"]

        use_poisson = data["stat_col"] in POISSON_STATS
        eff_std = max(base_stats["std"], MIN_STD.get(data["stat_col"], 1.0))

        book_probs = compute_probs(final_mean, eff_std, book_line, use_poisson=use_poisson)
        market_edge = round((book_probs["p_over"] - market_fair_prob) * 100, 1)

        # Real PP line > estimated round > null (user enters manually)
        real_pp = _match_pp_line(player_name, pp_by_market.get(mkey, {}))
        if real_pp is not None:
            pp_line   = real_pp
            pp_source = "prizepicks"
        else:
            pp_line   = None   # frontend will show editable input
            pp_source = "manual"

        if pp_line is not None:
            pp_probs = compute_probs(final_mean, eff_std, pp_line, use_poisson=use_poisson)
            pp_edge  = round((pp_probs["effective_over_rate"] - 0.5) * 100, 1)
        else:
            pp_edge  = None

        results.append({
            "player":               player_name,
            "opponent":             opponent,
            "event_id":             data["event_id"],
            "home_team":            data["home_team"],
            "away_team":            data["away_team"],
            "market":               data["label"],
            "stat_col":             data["stat_col"],
            "group":                data["group"],
            "line":                 book_line,
            "pp_line":              pp_line,
            "pp_source":            pp_source,
            "projection":           round(final_mean, 2),
            "eff_std":              round(eff_std, 3),
            "use_poisson":          use_poisson,
            "model_prob_over":      round(book_probs["p_over"], 3),
            "market_fair_prob":     round(market_fair_prob, 3),
            "market_edge":          market_edge,
            "pp_edge":              pp_edge,
            "abs_market_edge":      abs(market_edge),
            "n_books":              market_data["n_books"],
            "games_used":           base_stats["n_games"],
            "lean":                 "OVER" if market_edge > 0 else "UNDER",
            "adjustments":          adjustments,
        })

    results.sort(key=lambda x: x["abs_market_edge"], reverse=True)
    return results
