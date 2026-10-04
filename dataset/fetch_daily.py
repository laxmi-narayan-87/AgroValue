#!/usr/bin/env python3
"""Download and archive the Government of India daily APMC/mandi price feed.

Source resource:
https://data.gov.in/resource/current-daily-price-various-commodities-various-markets-mandi
Resource ID: 9ef84268-d588-465a-a308-a864a43d0070

The API returns a paginated feed. This collector reads pages, groups records by
arrival_date, and writes one immutable CSV per date under dataset/daily/.

Required environment variable:
    DATA_GOV_API_KEY

Usage:
    python dataset/fetch_daily.py
    python dataset/fetch_daily.py --date 2026-10-04

Existing date files are never overwritten.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

RESOURCE_ID = "9ef84268-d588-465a-a308-a864a43d0070"
API_URL = f"https://api.data.gov.in/resource/{RESOURCE_ID}"
ROOT = Path(__file__).resolve().parent
OUT = ROOT / "daily"
PAGE_SIZE = 1000

FIELDS = [
    "state",
    "district",
    "market",
    "commodity",
    "variety",
    "grade",
    "arrival_date",
    "min_price",
    "max_price",
    "modal_price",
]


def fetch_page(api_key: str, offset: int, limit: int = PAGE_SIZE) -> dict:
    params = {
        "api-key": api_key,
        "format": "json",
        "offset": offset,
        "limit": limit,
    }
    request = Request(
        f"{API_URL}?{urlencode(params)}",
        headers={"User-Agent": "AgroValue/1.0"},
    )
    with urlopen(request, timeout=60) as response:
        payload = json.load(response)

    if not isinstance(payload, dict):
        raise RuntimeError("Unexpected API response: expected a JSON object.")

    if "error" in payload:
        raise RuntimeError(f"data.gov.in API error: {payload['error']}")

    return payload


def write_csv(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")

    with tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(records)

    tmp.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Archive the paginated APMC/mandi feed into daily CSV files."
    )
    parser.add_argument(
        "--date",
        help="Only archive this arrival date (YYYY-MM-DD). The API is still paginated.",
    )
    args = parser.parse_args()

    api_key = os.environ.get("DATA_GOV_API_KEY")
    if not api_key:
        raise SystemExit(
            "DATA_GOV_API_KEY is required. Set it in the environment or GitHub Actions secret."
        )

    if args.date:
        import datetime as dt

        try:
            dt.date.fromisoformat(args.date)
        except ValueError as exc:
            raise SystemExit("--date must use YYYY-MM-DD.") from exc

    OUT.mkdir(parents=True, exist_ok=True)

    existing_dates = {
        path.stem for path in OUT.glob("*.csv") if path.stem[:4].isdigit()
    }

    offset = 0
    total_seen = 0
    by_date: dict[str, list[dict]] = {}

    while True:
        payload = fetch_page(api_key, offset)
        records = payload.get("records") or []

        if not isinstance(records, list):
            raise RuntimeError("Unexpected API response: 'records' is not a list.")

        total_seen += len(records)

        for record in records:
            arrival_date = str(record.get("arrival_date", "")).strip()
            if not arrival_date:
                continue
            if args.date and arrival_date != args.date:
                continue

            clean = {field: record.get(field, "") for field in FIELDS}
            by_date.setdefault(arrival_date, []).append(clean)

        reported_total = payload.get("total")
        count = payload.get("count")

        # Prefer the API's reported total when present. Otherwise, a short page
        # is the safest termination signal. This also handles APIs whose effective
        # page size is smaller than the requested limit.
        if reported_total is not None:
            try:
                if offset + len(records) >= int(reported_total):
                    break
            except (TypeError, ValueError):
                pass

        if not records or len(records) < PAGE_SIZE and reported_total is None:
            break

        if count is not None:
            try:
                if int(count) == 0:
                    break
            except (TypeError, ValueError):
                pass

        offset += len(records)

    if not by_date:
        target = f" for {args.date}" if args.date else ""
        raise SystemExit(f"No records found{target}.")

    written = 0
    skipped = 0

    for arrival_date, records in sorted(by_date.items()):
        path = OUT / f"{arrival_date}.csv"

        if arrival_date in existing_dates or path.exists():
            print(f"SKIP {path} (already exists)")
            skipped += 1
            continue

        records.sort(
            key=lambda row: (
                str(row.get("state", "")),
                str(row.get("district", "")),
                str(row.get("market", "")),
                str(row.get("commodity", "")),
                str(row.get("variety", "")),
                str(row.get("grade", "")),
            )
        )
        write_csv(path, records)
        print(f"WROTE {path}: {len(records)} records")
        written += 1

    print(f"Fetched {total_seen} API records; wrote {written} date file(s); skipped {skipped} existing file(s).")


if __name__ == "__main__":
    main()
