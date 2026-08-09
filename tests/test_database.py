import datetime

import pytest


def _pick(player="Aaron Judge", stat_col="hits", market="batter_hits", edge=12.0,
          lean="OVER", line=1.5, projection=2.1, model_prob=0.62, market_prob=0.5,
          group="hitting", event_id="evt1"):
    return {
        "player": player,
        "opponent": "Boston Red Sox",
        "market": market,
        "stat_col": stat_col,
        "group": group,
        "line": line,
        "projection": projection,
        "model_prob_over": model_prob,
        "market_fair_prob": market_prob,
        "market_edge": edge,
        "lean": lean,
        "event_id": event_id,
    }


class TestSavePicks:
    def test_saves_new_picks(self, isolated_db):
        result = isolated_db.save_picks([_pick()], date="2026-08-01")
        assert result == {"saved": 1, "skipped": 0}
        rows = isolated_db.get_picks(date="2026-08-01")
        assert len(rows) == 1
        assert rows[0]["player"] == "Aaron Judge"
        assert rows[0]["hit"] is None

    def test_defaults_to_todays_date_when_omitted(self, isolated_db):
        isolated_db.save_picks([_pick()])
        today = datetime.date.today().isoformat()
        rows = isolated_db.get_picks(date=today)
        assert len(rows) == 1

    def test_duplicate_date_player_market_is_skipped(self, isolated_db):
        isolated_db.save_picks([_pick()], date="2026-08-01")
        result = isolated_db.save_picks([_pick()], date="2026-08-01")
        assert result == {"saved": 0, "skipped": 1}
        assert len(isolated_db.get_picks(date="2026-08-01")) == 1

    def test_same_player_different_market_both_saved(self, isolated_db):
        isolated_db.save_picks(
            [_pick(market="batter_hits"), _pick(market="batter_home_runs")],
            date="2026-08-01",
        )
        rows = isolated_db.get_picks(date="2026-08-01")
        assert len(rows) == 2

    def test_missing_required_key_counts_as_skipped_not_raised(self, isolated_db):
        bad = _pick()
        del bad["player"]
        result = isolated_db.save_picks([bad], date="2026-08-01")
        assert result == {"saved": 0, "skipped": 1}


class TestResolvePick:
    def test_resolve_pick_sets_actual_and_hit(self, isolated_db):
        isolated_db.save_picks([_pick()], date="2026-08-01")
        pick_id = isolated_db.get_picks(date="2026-08-01")[0]["id"]

        isolated_db.resolve_pick(pick_id, actual=3.0, hit=1)

        row = isolated_db.get_picks(date="2026-08-01")[0]
        assert row["actual"] == 3.0
        assert row["hit"] == 1
        assert row["resolved_at"] is not None


class TestManualResolve:
    def test_over_lean_hit_when_actual_exceeds_line(self, isolated_db):
        isolated_db.save_picks([_pick(lean="OVER", line=1.5)], date="2026-08-01")
        pick_id = isolated_db.get_picks(date="2026-08-01")[0]["id"]

        ok = isolated_db.manual_resolve(pick_id, actual=2.0)

        assert ok is True
        row = isolated_db.get_picks(date="2026-08-01")[0]
        assert row["hit"] == 1

    def test_over_lean_miss_when_actual_below_line(self, isolated_db):
        isolated_db.save_picks([_pick(lean="OVER", line=1.5)], date="2026-08-01")
        pick_id = isolated_db.get_picks(date="2026-08-01")[0]["id"]

        isolated_db.manual_resolve(pick_id, actual=1.0)

        row = isolated_db.get_picks(date="2026-08-01")[0]
        assert row["hit"] == 0

    def test_under_lean_hit_when_actual_below_line(self, isolated_db):
        isolated_db.save_picks([_pick(lean="UNDER", line=1.5)], date="2026-08-01")
        pick_id = isolated_db.get_picks(date="2026-08-01")[0]["id"]

        isolated_db.manual_resolve(pick_id, actual=1.0)

        row = isolated_db.get_picks(date="2026-08-01")[0]
        assert row["hit"] == 1

    def test_returns_false_for_unknown_pick_id(self, isolated_db):
        assert isolated_db.manual_resolve(999, actual=1.0) is False


class TestUnresolveAndDelete:
    def test_unresolve_pick_clears_resolution_fields(self, isolated_db):
        isolated_db.save_picks([_pick()], date="2026-08-01")
        pick_id = isolated_db.get_picks(date="2026-08-01")[0]["id"]
        isolated_db.resolve_pick(pick_id, actual=2.0, hit=1)

        isolated_db.unresolve_pick(pick_id)

        row = isolated_db.get_picks(date="2026-08-01")[0]
        assert row["actual"] is None
        assert row["hit"] is None
        assert row["resolved_at"] is None

    def test_delete_pick_removes_row(self, isolated_db):
        isolated_db.save_picks([_pick()], date="2026-08-01")
        pick_id = isolated_db.get_picks(date="2026-08-01")[0]["id"]

        isolated_db.delete_pick(pick_id)

        assert isolated_db.get_picks(date="2026-08-01") == []


class TestGetPicks:
    def test_filters_unresolved_only(self, isolated_db):
        isolated_db.save_picks(
            [_pick(player="Player A"), _pick(player="Player B")], date="2026-08-01"
        )
        rows = isolated_db.get_picks(date="2026-08-01")
        isolated_db.resolve_pick(rows[0]["id"], actual=1.0, hit=1)

        unresolved = isolated_db.get_picks(unresolved_only=True)
        assert len(unresolved) == 1
        assert unresolved[0]["player"] == rows[1]["player"]

    def test_filters_past_only(self, isolated_db):
        future_date = (datetime.date.today() + datetime.timedelta(days=5)).isoformat()
        past_date = (datetime.date.today() - datetime.timedelta(days=5)).isoformat()
        isolated_db.save_picks([_pick(player="Future")], date=future_date)
        isolated_db.save_picks([_pick(player="Past")], date=past_date)

        past_rows = isolated_db.get_picks(past_only=True)
        names = {r["player"] for r in past_rows}
        assert "Past" in names
        assert "Future" not in names

    def test_orders_by_date_desc_then_edge_desc(self, isolated_db):
        isolated_db.save_picks([_pick(player="Low Edge", edge=5.0)], date="2026-08-01")
        isolated_db.save_picks([_pick(player="High Edge", edge=25.0)], date="2026-08-01")

        rows = isolated_db.get_picks(date="2026-08-01")
        assert [r["player"] for r in rows] == ["High Edge", "Low Edge"]


class TestGetStats:
    def _resolved(self, isolated_db, edge, hit, player):
        isolated_db.save_picks([_pick(player=player, edge=edge)], date="2026-08-01")
        pick_id = isolated_db.get_picks(date="2026-08-01")[0]["id"]
        isolated_db.resolve_pick(pick_id, actual=2.0, hit=hit)

    def test_overall_and_tier_breakdown(self, isolated_db):
        self._resolved(isolated_db, edge=25.0, hit=1, player="A")  # 20pp+
        self._resolved(isolated_db, edge=-17.0, hit=0, player="B")  # 15-20pp
        self._resolved(isolated_db, edge=5.0, hit=1, player="C")  # <10pp

        stats = isolated_db.get_stats()

        assert stats["overall"]["total"] == 3
        assert stats["overall"]["wins"] == 2
        tiers = {t["tier"]: t for t in stats["tiers"]}
        assert tiers["20pp+"]["total"] == 1
        assert tiers["20pp+"]["wins"] == 1
        assert tiers["15-20pp"]["total"] == 1
        assert tiers["15-20pp"]["wins"] == 0
        assert tiers["<10pp"]["total"] == 1

    def test_unresolved_picks_excluded(self, isolated_db):
        isolated_db.save_picks([_pick(player="Unresolved")], date="2026-08-01")
        stats = isolated_db.get_stats()
        assert stats["overall"].get("total") == 0

    def test_by_market_breakdown(self, isolated_db):
        self._resolved(isolated_db, edge=12.0, hit=1, player="A")
        stats = isolated_db.get_stats()
        assert len(stats["by_market"]) == 1
        assert stats["by_market"][0]["market"] == "batter_hits"


class TestGetCalibrationData:
    def test_below_min_samples_is_excluded(self, isolated_db):
        for i in range(5):
            isolated_db.save_picks([_pick(player=f"P{i}")], date="2026-08-01")
        for row in isolated_db.get_picks(date="2026-08-01"):
            isolated_db.resolve_pick(row["id"], actual=2.5, hit=1)

        cal = isolated_db.get_calibration_data(min_samples=30)
        assert cal == {}

    def test_bias_is_average_projection_minus_actual(self, isolated_db):
        # projection=2.1 (see _pick default), actual=1.1 -> bias of +1.0 each time
        for i in range(31):
            isolated_db.save_picks([_pick(player=f"P{i}")], date="2026-08-01")
        for row in isolated_db.get_picks(date="2026-08-01"):
            hit = 1 if row["lean"] == "OVER" else 0
            isolated_db.resolve_pick(row["id"], actual=1.1, hit=hit)

        cal = isolated_db.get_calibration_data(min_samples=30)
        assert "hits" in cal
        assert cal["hits"]["n"] == 31
        assert cal["hits"]["bias"] == pytest.approx(1.0)
