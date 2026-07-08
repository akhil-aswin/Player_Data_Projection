"""
Pulls NBA player prop odds from multiple sportsbooks via The Odds API
(https://the-odds-api.com). Requires ODDS_API_KEY as an environment variable.
"""

import requests
from config import (
    ODDS_API_KEY, ODDS_API_BASE_URL, ODDS_API_SPORT_KEY,
    ODDS_REGIONS, ODDS_FORMAT,
)


class OddsAPIClient:
    def __init__(self, api_key: str = None):
        self.api_key = api_key or ODDS_API_KEY
        if not self.api_key:
            raise ValueError("No Odds API key set. Set the ODDS_API_KEY env variable.")
        self.session = requests.Session()

    def get_upcoming_events(self, sport: str = ODDS_API_SPORT_KEY) -> list:
        """Upcoming NBA games with event IDs — needed before pulling prop odds."""
        url = f"{ODDS_API_BASE_URL}/sports/{sport}/events"
        resp = self.session.get(url, params={"apiKey": self.api_key})
        resp.raise_for_status()
        return resp.json()

    def get_player_props(self, event_id: str, markets: list,
                          sport: str = ODDS_API_SPORT_KEY) -> dict:
        """
        Player prop odds for one game across all books available on your plan.
        `markets` e.g. ["player_points", "player_rebounds"].
        """
        url = f"{ODDS_API_BASE_URL}/sports/{sport}/events/{event_id}/odds"
        params = {
            "apiKey": self.api_key,
            "regions": ODDS_REGIONS,
            "markets": ",".join(markets),
            "oddsFormat": ODDS_FORMAT,
        }
        resp = self.session.get(url, params=params)
        resp.raise_for_status()
        return resp.json()

    @staticmethod
    def extract_player_lines(props_response: dict, player_name: str,
                              market: str) -> list:
        """
        Flattens the nested API response into:
        [{"book": str, "line": float, "over_odds": int, "under_odds": int}, ...]
        for one player/market, one entry per book.
        """
        results = []
        for bookmaker in props_response.get("bookmakers", []):
            book_name = bookmaker["key"]
            for m in bookmaker.get("markets", []):
                if m["key"] != market:
                    continue
                line = over_odds = under_odds = None
                for outcome in m["outcomes"]:
                    if outcome.get("description", "").lower() != player_name.lower():
                        continue
                    line = outcome.get("point")
                    if outcome["name"] == "Over":
                        over_odds = outcome["price"]
                    elif outcome["name"] == "Under":
                        under_odds = outcome["price"]
                if None not in (line, over_odds, under_odds):
                    results.append({
                        "book": book_name,
                        "line": line,
                        "over_odds": over_odds,
                        "under_odds": under_odds,
                    })
        return results
