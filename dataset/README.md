# APMC / Mandi Daily Dataset

This folder archives the **AgroValue agricultural-market (APMC/mandi) dataset**. It is separate from the consumer-price/PMS dataset.

## Source

Government of India Open Government Data (OGD) Platform, using the AGMARKNET daily mandi-price resource.

Resource ID:

`9ef84268-d588-465a-a308-a864a43d0070`

The official catalog describes the resource as **daily** wholesale market data containing minimum, maximum and modal prices.

## Structure

```
dataset/
├── README.md
├── fetch_daily.py
└── daily/
    ├── YYYY-MM-DD.csv
    └── ...
```

Each daily CSV contains:

- `state`
- `district`
- `market`
- `commodity`
- `variety`
- `grade`
- `arrival_date`
- `min_price`
- `max_price`
- `modal_price`

Prices are the mandi wholesale prices reported by the source. Missing values are preserved as missing.

## Collector

The collector uses the API's pagination, reads the feed, groups records by `arrival_date`, and creates one CSV per date. It is intentionally **not** a one-request-per-day historical downloader.

It never overwrites an existing daily file.

### Environment

Set your personal data.gov.in API key:

```bash
export DATA_GOV_API_KEY="YOUR_KEY"
```

Do not commit the key to the repository.

### Run

Archive all dates returned by the current feed:

```bash
python dataset/fetch_daily.py
```

Archive only one arrival date from the feed:

```bash
python dataset/fetch_daily.py --date YYYY-MM-DD
```

The `--date` option still paginates the API because the source feed is paginated; it only writes matching records.

## Automated collection

GitHub Actions can run the collector daily. Add a repository secret named:

`DATA_GOV_API_KEY`

The workflow stores newly returned dates under `dataset/daily/` and commits them without modifying existing files.

## Modeling

`modal_price` is the primary price target for forecasting. Other market and price fields can be used as explanatory variables where the modeling setup supports them.

The original `Agriculture_commodities_dataset.csv` is preserved and is not replaced by this archive.
