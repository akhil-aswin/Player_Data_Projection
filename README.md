# Player Props Model

Pulls historical player data, pulls sportsbook (and PrizePicks) player-prop
lines, de-vigs the market, projects a stat with opponent/matchup context, and
surfaces the edge between the model and the market. Ships with an MLB web
dashboard (FastAPI + a single-page frontend) and a SQLite picks tracker that
feeds a self-calibrating bias correction back into the projection.

Started as an NBA CLI tool (`main.py`); the MLB pipeline (`mlb_main.py` +
`api.py`) is the actively developed half of the project.

## Setup

```bash
pip install -r requirements.txt
```

Create a `.env` file (or export directly) with:

```bash
ODDS_API_KEY="your_key_here"   # from https://the-odds-api.com
```

## Running the MLB dashboard

```bash
uvicorn api:app --reload
```

Open `http://localhost:8000`. The frontend (`frontend/index.html`) shows a
sortable board of every MLB prop with a market edge, lets you flip between
sportsbook and PrizePicks lines, and tracks picks over time.

## Running the NBA CLI

```bash
python main.py
```

Prompts for an event, player, opponent, and stat, then prints a model
projection vs. the market's no-vig probability.

## Structure

**Shared**
- `config.py` — API keys / sport keys / market lists, loaded from `.env`
- `data/odds_client.py` — pulls player prop odds across books (The Odds API)
- `analysis/devig.py` — American odds → probability, vig removal, multi-book consensus

**NBA**
- `data/nba_stats.py` — game logs, recent form, matchup history (via `nba_api`)
- `main.py` — CLI: projection vs. market, outputs an edge estimate

**MLB**
- `data/mlb_stats.py` — game logs, weighted recent form, batter-vs-pitcher and
  pitcher-vs-team splits (via `MLB-StatsAPI`)
- `data/mlb_matchups.py` — lineup/team strikeout-rate factors
- `mlb_main.py` — the projection pipeline: recency-weighted base stat, matchup
  and opponent-context blending, pitcher-vs-lineup adjustments, Poisson vs.
  Normal distribution selection, and historical-bias calibration
- `api.py` — FastAPI backend: serves the frontend, batch-scans all of today's
  props across markets, and exposes the picks tracker endpoints
- `database.py` — SQLite persistence for saved picks, resolution of actual
  outcomes, and per-stat calibration bias
- `frontend/index.html` — single-page dashboard (vanilla HTML/CSS/JS)

## How the MLB pipeline works

1. **Base projection** — recency-weighted average over a player's last N
   games (pitchers: 15 starts, batters: 20 games), using a half-life decay
   so recent performance counts more.
2. **Matchup context** — pitchers blend in starts vs. the specific opponent;
   batters blend in career rate vs. the day's opposing starter (capped
   weight, since a starter faces a batter for only part of the game).
3. **Strikeout-specific adjustments** — pitcher-vs-lineup history and
   opponent team/lineup strikeout rate multiply the base K projection.
4. **Distribution choice** — discrete counting stats (hits, total bases,
   strikeouts, etc.) use a Poisson model instead of Normal, since they're
   right-skewed and can push on whole-number lines; PrizePicks pushes are
   modeled explicitly.
5. **Calibration correction** — `database.py` tracks every saved pick's
   projection vs. its actual outcome. Once 30+ resolved picks exist for a
   stat, the average bias is fed back into future projections for that stat
   (capped at 30% of the raw projection).
6. **Nightly auto-resolve** — a background task in `api.py` resolves the
   prior day's picks against actual box scores at 12:05 AM Central.

## What this does NOT yet do

1. **Injury/usage adjustment** — no logic for redistributing usage when a
   teammate is out, on either the NBA or MLB side.
2. **Minutes/innings model (NBA)** — assumes a stable minutes distribution;
   no back-to-back, blowout-risk, or foul-trouble logic.
3. **Book weighting** — consensus weights all sportsbooks equally.
4. **NBA calibration loop** — the historical bias-correction feedback loop
   only exists for the MLB pipeline.

## Notes

- `stats.nba.com` (via `nba_api`) and the MLB Stats API are both rate-limit
  sensitive — clients add small delays between calls. If you see 403s on the
  NBA side, check for an `nba_api` update; the endpoint occasionally changes
  required headers.
- The Odds API's player-prop endpoint requires calling per-event (not
  per-league) — `get_upcoming_events()` is called first to get event IDs.
- Free tier of The Odds API has a low monthly request cap — the MLB batch
  scan fetches all markets for an event in one call rather than per-player.
- PrizePicks lines are pulled directly from their public projections API and
  can get Cloudflare-blocked; the pipeline falls back to an estimated line
  from the sportsbook consensus when that happens.
