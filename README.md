# AgroValue

AgroValue is an AI/ML-based agricultural commodity price forecasting project.

## Workflow

```text
Government AGMARKNET daily mandi data
        ↓
Data collection and validation
        ↓
Commodity / market time series
        ↓
Feature engineering and analysis
        ↓
Forecasting models
        ↓
Backtesting and evaluation
        ↓
Streamlit dashboard
```

## Data sources

The repository keeps the original historical dataset and a separate archive pipeline for Government of India AGMARKNET/OGD daily mandi data.

### Original dataset

`Agriculture_commodities_dataset.csv` is preserved as the original historical dataset.

### Daily mandi archive

The collector in `dataset/fetch_daily.py` uses Government of India Open Government Data (OGD) resource:

`9ef84268-d588-465a-a308-a864a43d0070`

Daily files are stored under `dataset/daily/` and contain:

- state
- district
- market
- commodity
- variety
- grade
- arrival_date
- min_price
- max_price
- modal_price

Add your personal `DATA_GOV_API_KEY` as an environment variable or GitHub Actions secret. Never commit credentials.

## Forecasting app

The current Streamlit app is in `app.py`.

It currently works with `monthly_data.csv` and compares:

- Seasonal Naive
- SARIMAX

Evaluation uses a chronological holdout with MAE, RMSE and sMAPE.

## Installation

```bash
pip install -r requirements.txt
```

## Run the dashboard

```bash
streamlit run app.py
```

## Run the daily collector

```bash
python dataset/fetch_daily.py
```

To archive one arrival date:

```bash
python dataset/fetch_daily.py --date YYYY-MM-DD
```

## Automated collection

GitHub Actions workflow:

`.github/workflows/daily-mandi.yml`

The workflow is scheduled daily and requires the repository secret `DATA_GOV_API_KEY`.

## License

MIT
