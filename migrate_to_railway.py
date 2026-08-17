"""
One-shot script: copy all rows from the local picks.db to the Railway instance.
Run once: python3 migrate_to_railway.py
"""
import sqlite3, requests, os, sys

LOCAL_DB   = os.path.join(os.path.dirname(__file__), "picks.db")
RAILWAY_URL = "https://spectacular-surprise-production-a78b.up.railway.app"

def main():
    if not os.path.exists(LOCAL_DB):
        print("Local picks.db not found — nothing to migrate.")
        sys.exit(1)

    conn = sqlite3.connect(LOCAL_DB)
    conn.row_factory = sqlite3.Row
    rows = [dict(r) for r in conn.execute("SELECT * FROM picks ORDER BY date").fetchall()]
    conn.close()

    if not rows:
        print("No picks found in local DB.")
        sys.exit(0)

    print(f"Found {len(rows)} picks in local DB. Sending to Railway…")

    resp = requests.post(
        f"{RAILWAY_URL}/api/picks/import",
        json={"results": rows},
        timeout=60,
    )
    resp.raise_for_status()
    result = resp.json()
    print(f"Done — inserted: {result['inserted']}, updated: {result['updated']}, skipped (already matched): {result['skipped']}")

if __name__ == "__main__":
    main()
