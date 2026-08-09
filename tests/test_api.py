import pytest
from fastapi.testclient import TestClient

import api


@pytest.fixture
def client(isolated_db, monkeypatch):
    monkeypatch.setattr(api, "_cache", {})
    with TestClient(api.app) as c:
        yield c


def _result(player="Aaron Judge", stat_col="hits", group="hitting", edge=15.0,
            lean="OVER", line=1.5, projection=2.0, model_prob=0.6, market_prob=0.45,
            event_id="evt1", market="Hits"):
    return {
        "player": player,
        "opponent": "Boston Red Sox",
        "event_id": event_id,
        "home_team": "New York Yankees",
        "away_team": "Boston Red Sox",
        "market": market,
        "stat_col": stat_col,
        "group": group,
        "line": line,
        "pp_line": None,
        "pp_source": "manual",
        "projection": projection,
        "eff_std": 1.0,
        "use_poisson": True,
        "model_prob_over": model_prob,
        "market_fair_prob": market_prob,
        "market_edge": edge,
        "pp_edge": None,
        "abs_market_edge": abs(edge),
        "n_books": 3,
        "games_used": 20,
        "lean": lean,
        "adjustments": [],
    }


class TestRoot:
    def test_serves_frontend_html(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]


class TestPicksTracker:
    def test_get_picks_empty_initially(self, client):
        resp = client.get("/api/picks")
        assert resp.status_code == 200
        assert resp.json() == {"picks": []}

    def test_save_then_list_picks(self, client):
        save_resp = client.post("/api/picks/save", json={"results": [_result()], "date": "2026-08-01"})
        assert save_resp.status_code == 200
        assert save_resp.json() == {"saved": 1, "skipped": 0}

        list_resp = client.get("/api/picks", params={"date": "2026-08-01"})
        picks = list_resp.json()["picks"]
        assert len(picks) == 1
        assert picks[0]["player"] == "Aaron Judge"

    def test_saving_same_pick_twice_is_deduped(self, client):
        client.post("/api/picks/save", json={"results": [_result()], "date": "2026-08-01"})
        resp = client.post("/api/picks/save", json={"results": [_result()], "date": "2026-08-01"})
        assert resp.json() == {"saved": 0, "skipped": 1}

    def test_resolve_manual_computes_hit(self, client):
        client.post("/api/picks/save", json={"results": [_result(lean="OVER", line=1.5)], "date": "2026-08-01"})
        pick_id = client.get("/api/picks", params={"date": "2026-08-01"}).json()["picks"][0]["id"]

        resp = client.post("/api/picks/resolve-manual", json={"pick_id": pick_id, "actual": 2.0})

        assert resp.status_code == 200
        assert resp.json() == {"ok": True}
        picks = client.get("/api/picks", params={"date": "2026-08-01"}).json()["picks"]
        assert picks[0]["hit"] == 1
        assert picks[0]["actual"] == 2.0

    def test_resolve_manual_404_for_unknown_pick(self, client):
        resp = client.post("/api/picks/resolve-manual", json={"pick_id": 999, "actual": 1.0})
        assert resp.status_code == 404

    def test_unresolve_pick(self, client):
        client.post("/api/picks/save", json={"results": [_result()], "date": "2026-08-01"})
        pick_id = client.get("/api/picks", params={"date": "2026-08-01"}).json()["picks"][0]["id"]
        client.post("/api/picks/resolve-manual", json={"pick_id": pick_id, "actual": 2.0})

        resp = client.post(f"/api/picks/{pick_id}/unresolve")

        assert resp.status_code == 200
        picks = client.get("/api/picks", params={"date": "2026-08-01"}).json()["picks"]
        assert picks[0]["hit"] is None

    def test_delete_pick(self, client):
        client.post("/api/picks/save", json={"results": [_result()], "date": "2026-08-01"})
        pick_id = client.get("/api/picks", params={"date": "2026-08-01"}).json()["picks"][0]["id"]

        resp = client.delete(f"/api/picks/{pick_id}")

        assert resp.status_code == 200
        assert client.get("/api/picks", params={"date": "2026-08-01"}).json()["picks"] == []

    def test_picks_stats_empty_when_nothing_resolved(self, client):
        resp = client.get("/api/picks/stats")
        assert resp.status_code == 200
        assert resp.json()["overall"].get("total") in (0, None)

    def test_calibration_empty_below_min_samples(self, client):
        client.post("/api/picks/save", json={"results": [_result()], "date": "2026-08-01"})
        pick_id = client.get("/api/picks", params={"date": "2026-08-01"}).json()["picks"][0]["id"]
        client.post("/api/picks/resolve-manual", json={"pick_id": pick_id, "actual": 2.0})

        resp = client.get("/api/calibration")

        assert resp.status_code == 200
        assert resp.json() == {}


class TestCache:
    def test_cache_status_empty_initially(self, client):
        resp = client.get("/api/cache-status")
        assert resp.status_code == 200
        assert resp.json() == {}

    def test_cache_clear_empties_cache(self, client):
        api._cache_set("players", [{"name": "test"}])
        assert client.get("/api/cache-status").json() != {}

        resp = client.post("/api/cache-clear")

        assert resp.json() == {"cleared": True}
        assert client.get("/api/cache-status").json() == {}


class TestProjectEndpoint:
    def test_sportsbook_source_calls_find_edge(self, client, monkeypatch):
        captured = {}

        def fake_find_edge(player, opponent, event_id, stat_col, group):
            captured["args"] = (player, opponent, event_id, stat_col, group)
            return {"player": player, "edge_pct_pts": 10.0}

        monkeypatch.setattr(api, "find_edge", fake_find_edge)

        resp = client.post("/api/project", json={
            "player": "Aaron Judge", "opponent": "Boston Red Sox", "event_id": "evt1",
            "stat_col": "hits", "group": "hitting", "source": "sportsbook",
        })

        assert resp.status_code == 200
        assert resp.json() == {"player": "Aaron Judge", "edge_pct_pts": 10.0}
        assert captured["args"] == ("Aaron Judge", "Boston Red Sox", "evt1", "hits", "hitting")

    def test_prizepicks_source_calls_prizepicks_edge(self, client, monkeypatch):
        def fake_pp_edge(player, opponent, pp_line, stat_col, event_id, group):
            return {"player": player, "pp_line": pp_line}

        monkeypatch.setattr(api, "prizepicks_edge", fake_pp_edge)

        resp = client.post("/api/project", json={
            "player": "Aaron Judge", "opponent": "Boston Red Sox", "event_id": "evt1",
            "stat_col": "hits", "group": "hitting", "source": "prizepicks", "pp_line": 1.5,
        })

        assert resp.status_code == 200
        assert resp.json() == {"player": "Aaron Judge", "pp_line": 1.5}

    def test_error_from_model_returns_400(self, client, monkeypatch):
        def boom(*args, **kwargs):
            raise ValueError("no odds found")

        monkeypatch.setattr(api, "find_edge", boom)

        resp = client.post("/api/project", json={
            "player": "Aaron Judge", "opponent": "Boston Red Sox", "event_id": "evt1",
            "stat_col": "hits", "group": "hitting",
        })

        assert resp.status_code == 400
        assert "no odds found" in resp.json()["detail"]


class TestEventsEndpoint:
    def test_returns_trimmed_event_fields(self, client, monkeypatch):
        class FakeClient:
            def get_upcoming_events(self, sport=None):
                return [{
                    "id": "evt1", "home_team": "New York Yankees", "away_team": "Boston Red Sox",
                    "commence_time": "2026-08-05T23:00:00Z", "extra_field": "ignored",
                }]

        monkeypatch.setattr(api, "OddsAPIClient", FakeClient)

        resp = client.get("/api/events")

        assert resp.status_code == 200
        events = resp.json()["events"]
        assert events == [{
            "id": "evt1", "home_team": "New York Yankees", "away_team": "Boston Red Sox",
            "commence_time": "2026-08-05T23:00:00Z",
        }]

    def test_client_error_returns_500(self, client, monkeypatch):
        class FakeClient:
            def get_upcoming_events(self, sport=None):
                raise RuntimeError("api down")

        monkeypatch.setattr(api, "OddsAPIClient", FakeClient)

        resp = client.get("/api/events")

        assert resp.status_code == 500
