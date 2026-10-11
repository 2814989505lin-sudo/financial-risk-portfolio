import os, time, csv, math
from datetime import datetime, timezone
from dateutil.relativedelta import relativedelta
import requests
import pandas as pd

API_KEY   = os.getenv("GUARDIAN_API_KEY", "").strip()
BASE_URL  = "https://content.guardianapis.com/search"

TARGET_COUNT = 4000
PAGE_SIZE    = 200
YEARS_BACK   = 3
ORDER_BY     = "newest"
SLEEP_SEC    = 1.1
OUT_CSV      = "guardian_news_raw.csv"

QUERY = (
    "(bank OR bond OR loan OR credit OR default OR bankruptcy OR downgrade "
    "OR rating OR volatility OR currency OR fx OR \"interest rate\" OR fraud "
    "OR bribery OR laundering OR outage OR \"supply chain\" OR market)"
)

TO_DATE   = datetime.now(timezone.utc).date().isoformat()
FROM_DATE = (datetime.now(timezone.utc).date() - relativedelta(years=YEARS_BACK)).isoformat()

KEEP_TYPES = {"article"}

def fetch_page(page: int):
    if not API_KEY:
        raise RuntimeError("GUARDIAN_API_KEY must be set before collecting news.")
    params = {
        "api-key": API_KEY,
        "q": QUERY,
        "from-date": FROM_DATE,
        "to-date": TO_DATE,
        "order-by": ORDER_BY,
        "page-size": PAGE_SIZE,
        "page": page,
        "show-fields": "bodyText,headline,trailText,byline,thumbnail,publication",
    }
    r = requests.get(BASE_URL, params=params, timeout=30)
    r.raise_for_status()
    return r.json()

def convert(result: dict) -> dict:
    fields = (result.get("fields") or {})
    return {
        "id": result.get("id"),
        "type": result.get("type"),
        "sectionName": result.get("sectionName"),
        "webPublicationDate": result.get("webPublicationDate"),
        "webTitle": result.get("webTitle"),
        "headline": fields.get("headline"),
        "trailText": fields.get("trailText"),
        "byline": fields.get("byline"),
        "bodyText": fields.get("bodyText") or "",
        "webUrl": result.get("webUrl"),
        "apiUrl": result.get("apiUrl"),
        "publication": fields.get("publication"),
        "thumbnail": fields.get("thumbnail"),
    }

def main():
    first = fetch_page(1)
    resp = first.get("response", {}) or {}
    total = resp.get("total", 0)
    pages = resp.get("pages", 1)
    print(f"Total results={total}, API pages={pages}")

    rows, seen = [], set()
    page = 1
    while True:
        data = first if page == 1 else fetch_page(page)
        results = (data.get("response") or {}).get("results", []) or []
        batch = []
        for x in results:
            if KEEP_TYPES and x.get("type") not in KEEP_TYPES:
                continue
            row = convert(x)
            if not row["bodyText"]:
                continue
            if row["id"] in seen:
                continue
            seen.add(row["id"])
            batch.append(row)

        rows.extend(batch)
        print(f"Page {page}: got {len(batch)} usable (cum={len(rows)})")

        if len(rows) >= TARGET_COUNT:
            break

        page_count = (data.get("response") or {}).get("pages", page)
        if page >= page_count:
            print("No more pages from API.")
            break

        page += 1
        time.sleep(SLEEP_SEC)

    if len(rows) > TARGET_COUNT:
        rows = rows[:TARGET_COUNT]

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False, quoting=csv.QUOTE_MINIMAL)
    print(f"Saved CSV: {OUT_CSV} (rows={len(df)})")

if __name__ == "__main__":
    main()
