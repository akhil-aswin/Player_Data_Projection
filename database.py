"""
SQLite persistence layer for tracking model projections and their outcomes.

Schema (single table `picks`):
  - One row per (date, player, market) — captured when the board is saved
  - actual / hit columns filled in by /api/picks/resolve after games end
"""

import sqlite3
import datetime
import os

# On Railway: set DB_PATH=/data/picks.db (persistent volume). Defaults to local file.
DB_PATH = os.environ.get("DB_PATH", os.path.join(os.path.dirname(__file__), "picks.db"))


def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with _conn() as conn:
        conn.execute("""
        CREATE TABLE IF NOT EXISTS picks (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            date          TEXT    NOT NULL,
            player        TEXT    NOT NULL,
            opponent      TEXT    NOT NULL,
            market        TEXT    NOT NULL,
            stat_col      TEXT    NOT NULL,
            group_type    TEXT    NOT NULL,
            line          REAL    NOT NULL,
            projection    REAL    NOT NULL,
            model_prob    REAL    NOT NULL,
            market_prob   REAL    NOT NULL,
            edge          REAL    NOT NULL,
            lean          TEXT    NOT NULL,
            event_id      TEXT    DEFAULT '',
            actual        REAL,
            hit           INTEGER,
            resolved_at   TEXT,
            saved_at      TEXT    NOT NULL,
            UNIQUE(date, player, market)
        )
        """)
        conn.commit()


# ── Write ──────────────────────────────────────────────────────────────────────

def save_picks(results: list, date: str = None) -> dict:
    """
    Insert batch results as picks for today. Skips rows that already exist
    (UNIQUE on date+player+market), so calling this multiple times is safe.
    Returns {"saved": int, "skipped": int}.
    """
    if date is None:
        date = datetime.date.today().isoformat()

    saved = skipped = 0
    now = datetime.datetime.utcnow().isoformat()

    with _conn() as conn:
        for r in results:
            try:
                conn.execute("""
                    INSERT OR IGNORE INTO picks
                      (date, player, opponent, market, stat_col, group_type,
                       line, projection, model_prob, market_prob, edge, lean,
                       event_id, saved_at)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """, (
                    date,
                    r["player"],
                    r["opponent"],
                    r.get("market", r.get("stat_col", "")),
                    r["stat_col"],
                    r["group"],
                    r["line"],
                    r["projection"],
                    r["model_prob_over"],
                    r["market_fair_prob"],
                    r["market_edge"],
                    r["lean"],
                    r.get("event_id", ""),
                    now,
                ))
                if conn.execute("SELECT changes()").fetchone()[0]:
                    saved += 1
                else:
                    skipped += 1
            except Exception:
                skipped += 1
        conn.commit()

    return {"saved": saved, "skipped": skipped}


def import_picks(rows: list) -> dict:
    """
    Import raw pick rows from another DB instance (e.g. local → Railway).
    Inserts rows that don't exist yet (UNIQUE on date+player+market).
    For newly inserted rows that are already resolved, also writes actual/hit.
    For rows that already exist in Railway, fills in actual/hit only if still NULL.
    """
    inserted = updated = skipped = 0
    now = datetime.datetime.utcnow().isoformat()

    with _conn() as conn:
        for r in rows:
            try:
                conn.execute("""
                    INSERT OR IGNORE INTO picks
                      (date, player, opponent, market, stat_col, group_type,
                       line, projection, model_prob, market_prob, edge, lean,
                       event_id, actual, hit, resolved_at, saved_at)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """, (
                    r["date"], r["player"], r["opponent"], r["market"],
                    r["stat_col"], r["group_type"], r["line"], r["projection"],
                    r["model_prob"], r["market_prob"], r["edge"], r["lean"],
                    r.get("event_id", ""),
                    r.get("actual"), r.get("hit"), r.get("resolved_at"),
                    r.get("saved_at") or now,
                ))
                if conn.execute("SELECT changes()").fetchone()[0]:
                    inserted += 1
                else:
                    # Row already exists — backfill actual/hit if Railway doesn't have it yet
                    if r.get("actual") is not None:
                        conn.execute("""
                            UPDATE picks SET actual=?, hit=?, resolved_at=?
                            WHERE date=? AND player=? AND market=? AND actual IS NULL
                        """, (r["actual"], r["hit"], r.get("resolved_at"),
                              r["date"], r["player"], r["market"]))
                        if conn.execute("SELECT changes()").fetchone()[0]:
                            updated += 1
                        else:
                            skipped += 1
                    else:
                        skipped += 1
            except Exception:
                skipped += 1
        conn.commit()

    return {"inserted": inserted, "updated": updated, "skipped": skipped}


def resolve_pick(pick_id: int, actual: float, hit: int):
    with _conn() as conn:
        conn.execute(
            "UPDATE picks SET actual=?, hit=?, resolved_at=? WHERE id=?",
            (actual, hit, datetime.datetime.utcnow().isoformat(), pick_id),
        )
        conn.commit()


def manual_resolve(pick_id: int, actual: float):
    """Resolve a pick with a manually entered actual value."""
    with _conn() as conn:
        row = conn.execute("SELECT line, lean FROM picks WHERE id=?", (pick_id,)).fetchone()
        if not row:
            return False
        line, lean = row["line"], row["lean"]
        hit = 1 if (lean == "OVER" and actual > line) or (lean == "UNDER" and actual < line) else 0
        conn.execute(
            "UPDATE picks SET actual=?, hit=?, resolved_at=? WHERE id=?",
            (actual, hit, datetime.datetime.utcnow().isoformat(), pick_id),
        )
        conn.commit()
        return True


def unresolve_pick(pick_id: int):
    with _conn() as conn:
        conn.execute(
            "UPDATE picks SET actual=NULL, hit=NULL, resolved_at=NULL WHERE id=?",
            (pick_id,),
        )
        conn.commit()


def delete_pick(pick_id: int):
    with _conn() as conn:
        conn.execute("DELETE FROM picks WHERE id=?", (pick_id,))
        conn.commit()


def get_calibration_data(min_samples: int = 30) -> dict:
    """
    Compute per-stat projection bias from resolved picks.

    bias = avg(projection - actual)
      > 0  →  model over-projects (correct by subtracting)
      < 0  →  model under-projects (correct by adding)

    Also computes a probability calibration ratio:
      expected_hit_rate = avg model_prob (in the direction of lean)
      actual_hit_rate   = wins / total
      cal_ratio = actual / expected  (< 1 = overconfident, > 1 = underconfident)

    Only returns stats for (stat_col, group_type) pairs with >= min_samples picks.
    """
    with _conn() as conn:
        rows = conn.execute("""
            SELECT
                stat_col,
                group_type,
                COUNT(*)                    AS n,
                AVG(projection - actual)    AS bias,
                AVG(projection)             AS avg_proj,
                AVG(actual)                 AS avg_actual,
                SUM(hit)                    AS wins,
                -- model prob in the lean direction
                AVG(CASE WHEN lean='OVER'  THEN model_prob
                         ELSE 1 - model_prob END) AS avg_model_prob
            FROM picks
            WHERE hit IS NOT NULL
              AND actual IS NOT NULL
            GROUP BY stat_col, group_type
            HAVING COUNT(*) >= ?
            ORDER BY n DESC
        """, (min_samples,)).fetchall()

    result = {}
    for r in rows:
        key = r["stat_col"]
        n   = r["n"]
        actual_hit_rate   = r["wins"] / n if n else 0.5
        expected_hit_rate = r["avg_model_prob"] or 0.5
        cal_ratio = actual_hit_rate / expected_hit_rate if expected_hit_rate else 1.0

        result[key] = {
            "n":                 n,
            "bias":              round(r["bias"], 4),       # projection - actual
            "avg_proj":          round(r["avg_proj"], 3),
            "avg_actual":        round(r["avg_actual"], 3),
            "actual_hit_rate":   round(actual_hit_rate, 3),
            "expected_hit_rate": round(expected_hit_rate, 3),
            "cal_ratio":         round(cal_ratio, 3),       # <1 overconfident
            "group_type":        r["group_type"],
        }
    return result


# ── Read ───────────────────────────────────────────────────────────────────────

def get_picks(date: str = None, unresolved_only: bool = False,
              past_only: bool = False) -> list:
    parts = ["SELECT * FROM picks"]
    conditions = []
    params = []

    if date:
        conditions.append("date = ?")
        params.append(date)
    if unresolved_only:
        conditions.append("hit IS NULL")
    if past_only:
        conditions.append("date <= ?")
        params.append(datetime.date.today().isoformat())

    if conditions:
        parts.append("WHERE " + " AND ".join(conditions))
    parts.append("ORDER BY date DESC, ABS(edge) DESC")

    with _conn() as conn:
        rows = conn.execute(" ".join(parts), params).fetchall()
    return [dict(r) for r in rows]


def get_stats(since: str = None) -> dict:
    """
    Calibration stats broken down by edge tier.
    Returns overall + per-tier win rates for all resolved picks.
    Optional `since` (YYYY-MM-DD) filters to picks on or after that date.
    """
    date_clause = "AND date >= ?" if since else ""
    params = (since,) if since else ()

    with _conn() as conn:
        tiers = conn.execute(f"""
            SELECT
                CASE
                    WHEN ABS(edge) >= 20 THEN '20pp+'
                    WHEN ABS(edge) >= 15 THEN '15-20pp'
                    WHEN ABS(edge) >= 10 THEN '10-15pp'
                    ELSE '<10pp'
                END AS tier,
                COUNT(*)          AS total,
                SUM(hit)          AS wins,
                SUM(1 - hit)      AS losses,
                AVG(ABS(edge))    AS avg_edge,
                MIN(ABS(edge))    AS min_edge
            FROM picks
            WHERE hit IS NOT NULL {date_clause}
            GROUP BY tier
        """, params).fetchall()

        overall = conn.execute(f"""
            SELECT
                COUNT(*)       AS total,
                SUM(hit)       AS wins,
                SUM(1 - hit)   AS losses
            FROM picks
            WHERE hit IS NOT NULL {date_clause}
        """, params).fetchone()

        by_market = conn.execute(f"""
            SELECT
                market,
                COUNT(*)       AS total,
                SUM(hit)       AS wins,
                SUM(1 - hit)   AS losses,
                AVG(ABS(edge)) AS avg_edge
            FROM picks
            WHERE hit IS NOT NULL {date_clause}
            GROUP BY market
            ORDER BY total DESC
        """, params).fetchall()

    return {
        "overall":   dict(overall) if overall else {},
        "tiers":     [dict(r) for r in tiers],
        "by_market": [dict(r) for r in by_market],
    }


# Initialize on import
init_db()
