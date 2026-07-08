"""
Configuration for the NBA props model.
API keys are read from environment variables — never hardcode keys in source.

Set before running:
    export ODDS_API_KEY="your_key_here"

Get a key at https://the-odds-api.com (free tier available, paid tiers for
more requests / more books).
"""
import os

ODDS_API_KEY = os.environ.get("ODDS_API_KEY", "")
ODDS_API_BASE_URL = "https://api.the-odds-api.com/v4"

# NBA sport key for The Odds API
ODDS_API_SPORT_KEY = "basketball_nba"

# Player prop markets available on The Odds API (availability varies by book/plan)
PROP_MARKETS = [
    "player_points",
    "player_rebounds",
    "player_assists",
    "player_threes",
    "player_blocks",
    "player_steals",
    "player_points_rebounds_assists",
]

# Regions determine which sportsbooks are queried: us, us2, uk, eu, au
ODDS_REGIONS = "us"
ODDS_FORMAT = "american"

# Most recently completed NBA season — update this once the new season starts
DEFAULT_SEASON = "2025-26"
RECENT_SEASONS = ["2025-26", "2024-25", "2023-24"]
