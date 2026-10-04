# APMC / Mandi Daily Dataset

This folder is for the **previous AgroValue agricultural-market dataset**, not the newer consumer-price/PMS dataset.

## Source

Government of India Open Government Data (OGD) Platform, Ministry of Agriculture & Farmers Welfare / Directorate of Marketing & Inspection (AGMARKNET).

Resource:
`9ef84268-d588-465a-a308-a864a43d0070`

The official OGD catalog describes this resource as **daily** wholesale prices from various agricultural markets (mandis), including minimum, maximum and modal prices. citeturn2search0turn2search1

## Structure

```
dataset/
├── README.md
├── fetch_daily.py
└── daily/
    ├── YYYY-MM-DD.csv
    ├── YYYY-MM-DD.csv
    └── ...
```

Each daily file contains:

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

Prices follow the AGMARKNET convention of ₹ per quintal. Missing values are kept missing; they are not converted to zero.

## Collection rule

Run:

```bash
export DATA_GOV_API_KEY="YOUR_KEY"
python dataset/fetch_daily.py
```

For one date:

```bash
python dataset/fetch_daily.py --date YYYY-MM-DD
```

The collector creates a new file for each arrival date and **never overwrites an existing date file**.

## Modeling

Use `modal_price` as the primary target for price forecasting, with `min_price`, `max_price`, arrivals, market, commodity, state and district available as explanatory variables.

The existing `Agriculture_commodities_dataset.csv` is preserved as the original historical dataset. It is not replaced.
