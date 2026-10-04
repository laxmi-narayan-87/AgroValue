# Daily Price Dataset

This folder stores one immutable CSV snapshot per collection date for AgroValue.

## Source

Department of Consumer Affairs, Price Monitoring System (PMS), Government of India.

The PMS reports daily retail and wholesale prices for essential commodities. The source states that daily price data are maintained by the NIC cell and collected from reporting centres across India.

## File convention

`daily_prices_YYYY-MM-DD.csv`

Each file represents the All-India daily price snapshot for that date. Existing date files must not be overwritten when a new date is collected.

## Schema

- `date`: observation date
- `commodity`: commodity name
- `category`: analytical commodity category
- `retail_price`: reported all-India average retail price
- `retail_unit`: retail unit as reported by PMS
- `wholesale_price`: reported all-India average wholesale price, when available
- `wholesale_unit`: wholesale unit as reported by PMS
- `source`: source system

## Important modeling rule

Do not directly mix prices with different units. Normalize units before cross-commodity comparisons or model training.

Do not invent missing observations. Missing wholesale values remain blank.

## Current coverage

Initial snapshot: 2026-10-04.

Future daily snapshots should be appended as new files. The historical `monthly_data.csv`, `latest_price_data.csv`, and `price_analysis_data.csv` remain unchanged.
