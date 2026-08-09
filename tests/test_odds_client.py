from unittest.mock import MagicMock

import pytest

import data.odds_client as odds_client
from data.odds_client import OddsAPIClient


class TestConstructor:
    def test_uses_explicit_api_key(self):
        client = OddsAPIClient(api_key="explicit-key")
        assert client.api_key == "explicit-key"

    def test_falls_back_to_module_level_key(self, monkeypatch):
        monkeypatch.setattr(odds_client, "ODDS_API_KEY", "env-key")
        client = OddsAPIClient()
        assert client.api_key == "env-key"

    def test_raises_when_no_key_available(self, monkeypatch):
        monkeypatch.setattr(odds_client, "ODDS_API_KEY", "")
        with pytest.raises(ValueError):
            OddsAPIClient()


def _fake_response(payload):
    resp = MagicMock()
    resp.json.return_value = payload
    resp.raise_for_status.return_value = None
    return resp


class TestGetUpcomingEvents:
    def test_returns_parsed_json_and_hits_expected_url(self):
        client = OddsAPIClient(api_key="k")
        client.session = MagicMock()
        client.session.get.return_value = _fake_response([{"id": "evt1"}])

        events = client.get_upcoming_events(sport="baseball_mlb")

        assert events == [{"id": "evt1"}]
        called_url = client.session.get.call_args.args[0]
        assert "baseball_mlb/events" in called_url
        assert client.session.get.call_args.kwargs["params"]["apiKey"] == "k"

    def test_raises_for_status_propagates(self):
        client = OddsAPIClient(api_key="k")
        client.session = MagicMock()
        resp = MagicMock()
        resp.raise_for_status.side_effect = RuntimeError("boom")
        client.session.get.return_value = resp

        with pytest.raises(RuntimeError):
            client.get_upcoming_events()


class TestGetPlayerProps:
    def test_passes_markets_and_event_in_url(self):
        client = OddsAPIClient(api_key="k")
        client.session = MagicMock()
        client.session.get.return_value = _fake_response({"bookmakers": []})

        client.get_player_props("evt42", markets=["batter_hits", "batter_home_runs"], sport="baseball_mlb")

        called_url = client.session.get.call_args.args[0]
        params = client.session.get.call_args.kwargs["params"]
        assert "baseball_mlb/events/evt42/odds" in called_url
        assert params["markets"] == "batter_hits,batter_home_runs"


class TestExtractPlayerLines:
    def _props(self):
        return {
            "bookmakers": [
                {
                    "key": "fanduel",
                    "markets": [
                        {
                            "key": "batter_hits",
                            "outcomes": [
                                {"description": "Aaron Judge", "name": "Over", "point": 1.5, "price": -120},
                                {"description": "Aaron Judge", "name": "Under", "point": 1.5, "price": 100},
                                {"description": "Juan Soto", "name": "Over", "point": 0.5, "price": -200},
                            ],
                        },
                        {
                            "key": "batter_home_runs",
                            "outcomes": [
                                {"description": "Aaron Judge", "name": "Over", "point": 0.5, "price": 150},
                                {"description": "Aaron Judge", "name": "Under", "point": 0.5, "price": -180},
                            ],
                        },
                    ],
                },
                {
                    "key": "draftkings",
                    "markets": [
                        {
                            "key": "batter_hits",
                            "outcomes": [
                                {"description": "Aaron Judge", "name": "Over", "point": 1.5, "price": -115},
                                {"description": "Aaron Judge", "name": "Under", "point": 1.5, "price": -105},
                            ],
                        }
                    ],
                },
            ]
        }

    def test_extracts_complete_lines_across_books(self):
        lines = OddsAPIClient.extract_player_lines(self._props(), "Aaron Judge", "batter_hits")
        assert lines == [
            {"book": "fanduel", "line": 1.5, "over_odds": -120, "under_odds": 100},
            {"book": "draftkings", "line": 1.5, "over_odds": -115, "under_odds": -105},
        ]

    def test_skips_incomplete_outcomes(self):
        # Juan Soto only has an Over outcome for batter_hits -> excluded
        lines = OddsAPIClient.extract_player_lines(self._props(), "Juan Soto", "batter_hits")
        assert lines == []

    def test_case_insensitive_name_match(self):
        lines = OddsAPIClient.extract_player_lines(self._props(), "aaron JUDGE", "batter_hits")
        assert len(lines) == 2

    def test_filters_by_market_key(self):
        lines = OddsAPIClient.extract_player_lines(self._props(), "Aaron Judge", "batter_home_runs")
        assert len(lines) == 1
        assert lines[0]["book"] == "fanduel"

    def test_unknown_player_returns_empty(self):
        lines = OddsAPIClient.extract_player_lines(self._props(), "Nobody", "batter_hits")
        assert lines == []

    def test_empty_bookmakers_returns_empty(self):
        lines = OddsAPIClient.extract_player_lines({}, "Aaron Judge", "batter_hits")
        assert lines == []
