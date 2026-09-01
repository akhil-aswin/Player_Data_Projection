"""
One-shot migration: copy all rows from local picks.db to Railway.
Uses only existing API endpoints — no new endpoint needed.
Run once: python3 migrate_to_railway.py
"""
import sqlite3, requests, os, sys
from collections import defaultdict

LOCAL_DB    = os.path.join(os.path.dirname(__file__), "picks.db")
RAILWAY_URL = "https://spectacular-surprise-production-a78b.up.railway.app"


def main():
    if not os.path.exists(LOCAL_DB):
        print("Local picks.db not found.")
        sys.exit(1)

    conn = sqlite3.connect(LOCAL_DB)
    conn.row_factory = sqlite3.Row
    rows = [dict(r) for r in conn.execute("SELECT * FROM picks ORDER BY date").fetchall()]
    conn.close()
    print(f"Found {len(rows)} picks locally.")

    # ── Step 1: insert base pick data grouped by date ─────────────────────────
    # Remap local column names → what /api/picks/save expects
    by_date = defaultdict(list)
    for r in rows:
        by_date[r["date"]].append({
            "player":           r["player"],
            "opponent":         r["opponent"],
            "market":           r["market"],
            "stat_col":         r["stat_col"],
            "group":            r["group_type"],
            "line":             r["line"],
            "projection":       r["projection"],
            "model_prob_over":  r["model_prob"],
            "market_fair_prob": r["market_prob"],
            "market_edge":      r["edge"],
            "lean":             r["lean"],
            "event_id":         r.get("event_id", ""),
        })

    saved = skipped = 0
    for date, picks in sorted(by_date.items()):
        resp = requests.post(
            f"{RAILWAY_URL}/api/picks/save",
            json={"results": picks, "date": date},
            timeout=30,
        )
        resp.raise_for_status()
        d = resp.json()
        saved   += d.get("saved", 0)
        skipped += d.get("skipped", 0)

    print(f"Step 1 done — inserted: {saved}, already existed: {skipped}")

    # ── Step 2: resolve picks that have actual values ─────────────────────────
    resolved_local = [r for r in rows if r.get("actual") is not None]
    if not resolved_local:
        print("No resolved picks to migrate.")
        return

    # Fetch all Railway picks by date to get their IDs
    all_dates = sorted({r["date"] for r in rows if r.get("actual") is not None})
    railway_picks = []
    for d in all_dates:
        try:
            r2 = requests.get(f"{RAILWAY_URL}/api/picks", params={"date": d}, timeout=30)
            r2.raise_for_status()
            railway_picks.extend(r2.json().get("picks", []))
        except Exception:
            pass
    # Build lookup: (date, player, market) → {id, actual}
    rw_index = {(p["date"], p["player"], p["market"]): p for p in railway_picks}

    resolved = already_done = failed = 0
    for r in resolved_local:
        key = (r["date"], r["player"], r["market"])
        rw  = rw_index.get(key)
        if not rw:
            failed += 1
            continue
        if rw.get("actual") is not None:
            already_done += 1
            continue
        resp = requests.post(
            f"{RAILWAY_URL}/api/picks/resolve-manual",
            json={"pick_id": rw["id"], "actual": r["actual"]},
            timeout=10,
        )
        if resp.ok:
            resolved += 1
        else:
            failed += 1

    print(f"Step 2 done — resolved: {resolved}, already resolved: {already_done}, failed: {failed}")
    print("Migration complete.")


if __name__ == "__main__":
    main()
