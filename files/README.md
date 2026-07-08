# NBA Player Props — Base Pipeline

Base pipeline for: pulling historical player data + matchup history, pulling
player prop odds across sportsbooks, de-vigging them, and comparing a simple
model projection against the no-vig market probability.

## Setup

```bash
pip install -r requirements.txt
export ODDS_API_KEY="your_key_here"   # from https://the-odds-api.com
```

## Structure

- `data/nba_stats.py` — game logs, recent form, matchup history vs an opponent
  (via `nba_api`, unofficial stats.nba.com wrapper)
- `data/odds_client.py` — pulls player prop odds across books (via The Odds API)
- `analysis/devig.py` — American odds → probability, vig removal, multi-book consensus
- `main.py` — ties it together: projection vs. market, outputs an edge estimate

## Usage

```python
from data.odds_client import OddsAPIClient
from main import find_edge

# 1. Get a real event_id for tonight's game
events = OddsAPIClient().get_upcoming_events()
print(events)  # find the matchup you want, grab its "id"

# 2. Compare model projection to the market
result = find_edge(
    player_name="Jayson Tatum",
    opponent="Knicks",
    event_id="<event_id_from_step_1>",
    market="player_points",
    stat_col="PTS",
)
print(result)
```

## What this baseline does NOT yet do

This is intentionally the minimal working skeleton. Known gaps to build next,
in priority order:

1. **Injury/usage adjustment** — no logic yet for redistributing usage when a
   teammate is out, or for opponent injuries softening a matchup.
2. **Minutes model** — currently assumes the player's minutes distribution is
   stable; no back-to-back, blowout-risk (via spread), or foul-trouble logic.
3. **Non-normal stat distributions** — points/rebounds/assists are modeled as
   Normal for simplicity. Poisson/negative-binomial fits the discrete,
   right-skewed nature of box-score counting stats better.
4. **Opponent defensive/pace context** — matchup history uses the player's own
   past games vs. an opponent, but doesn't yet fold in the opponent's current
   defensive rating vs. position or current team pace.
5. **Book weighting** — consensus currently weights all books equally; sharper
   books (e.g., lower-limit-but-efficient pricing) could be weighted higher.

## Notes

- `stats.nba.com` (via `nba_api`) is rate-limit sensitive — the client adds a
  small delay between calls. If you start seeing 403s, check for an `nba_api`
  update; the endpoint occasionally changes required headers.
- The Odds API player-prop endpoint requires calling per-event (not
  per-league), so pull `get_upcoming_events()` first to get event IDs.
- Free tier of The Odds API has a low monthly request cap — batch player
  lookups per game rather than one API call per player.
