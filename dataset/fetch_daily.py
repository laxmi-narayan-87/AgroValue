#!/usr/bin/env python3
"""Download the Government of India daily APMC/mandi price dataset.

The source is data.gov.in resource:
9ef84268-d588-465a-a308-a864a43d0070

Required environment variable:
    DATA_GOV_API_KEY

Usage:
    python dataset/fetch_daily.py
    python dataset/fetch_daily.py --date 2026-10-04

Each source date is written to:
    dataset/daily/YYYY-MM-DD.csv

Existing date files are never overwritten.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

RESOURCE_ID = "9ef84268-d588-465a-a308-a864a43d0070"
API_URL = f"https://api.data.gov.in/resource/{RESOURCE_ID}"
ROOT = Path(__file__).resolve().parent
OUT = ROOT / "daily"

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


def fetch_page(api_key: str, offset: int, limit: int = 1000) -> dict:
    params = {
        "api-key": api_key,
        "format": "json",
        "offset": offset,
        "limit": limit,
    }
    req = Request(f"{API_URL}?{urlencode(params)}", headers={"User-Agent": "AgroValue/1.0"})
    with urlopen(req, timeout=60) as response:
        return json.load(response)


def write_csv(path: Path, records: list[dict]) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(records)
    tmp.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", help="Only store this arrival date (YYYY-MM-DD).")
    args = parser.parse_args()

    api_key = os.environ.get("DATA_GOV_API_KEY")
    if not api_key:
        raise SystemExit("DATA_GOV_API_KEY is required. Get an API key from data.gov.in.")

    OUT.mkdir(parents=True, exist_ok=True)

    offset = 0
    page_size = 1000
    by_date: dict[str, list[dict]] = {}

    while True:
        payload = fetch_page(api_key, offset, page_size)
        records = payload.get("records", [])
        if not records:
            break

        for record in records:
            arrival_date = str(record.get("arrival_date", "")).strip()
            if not arrival_date:
                continue
            if args.date and arrival_date != args.date:
                continue

            clean = {field: record.get(field, "") for field in FIELDS}
            by_date.setdefault(arrival_date, []).append(clean)

        if len(records) < page_size:
            break
        offset += page_size

        # When a specific date is requested, the API can still return many pages.
        # Continue until pagination is exhausted so the date is complete.

    if not by_date:
        raise SystemExit("No records found for the requested date/filter.")

    for arrival_date, records in sorted(by_date.items()):
        path = OUT / f"{arrival_date}.csv"
        if path.exists():
            print(f"SKIP {path} (already exists)")
            continue
        write_csv(path, records)
        print(f"WROTE {path}: {len(records)} records")


if __name__ == "__main__":
    main()
