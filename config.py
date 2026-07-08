"""
Configuration for the NBA/MLB props model.
API keys are read from environment variables — never hardcode keys in source.

Set before running:
    export ODDS_API_KEY="your_key_here"

Get a key at https://the-odds-api.com
"""
import os
from dotenv import load_dotenv

load_dotenv()

ODDS_API_KEY = os.environ.get("ODDS_API_KEY", "")
ODDS_API_BASE_URL = "https://api.the-odds-api.com/v4"

ODDS_REGIONS = "us"
ODDS_FORMAT = "american"

# Sport keys for The Odds API
NBA_SPORT_KEY = "basketball_nba"
MLB_SPORT_KEY = "baseball_mlb"

# NBA
NBA_DEFAULT_SEASON = "2025-26"
NBA_RECENT_SEASONS = ["2025-26", "2024-25", "2023-24"]

NBA_PROP_MARKETS = [
    "player_points",
    "player_rebounds",
    "player_assists",
    "player_threes",
    "player_blocks",
    "player_steals",
    "player_points_rebounds_assists",
]

# MLB
MLB_CURRENT_SEASON = 2026
MLB_RECENT_SEASONS = [2026, 2025, 2024]

MLB_PROP_MARKETS = [
    "batter_hits",
    "batter_total_bases",
    "batter_rbis",
    "batter_runs_scored",
    "batter_home_runs",
    "batter_strikeouts",
    "pitcher_strikeouts",
    "pitcher_earned_runs",
    "pitcher_hits_allowed",
    "pitcher_walks",
]
