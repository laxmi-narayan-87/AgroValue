from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX

DATA_PATH = "monthly_data.csv"
FORECAST_HORIZON = 60
TEST_HORIZON = 12
SEASONAL_PERIOD = 12
ORDER = (1, 1, 1)
SEASONAL_ORDER = (1, 1, 0, SEASONAL_PERIOD)

st.set_page_config(page_title="AgroValue", layout="wide")
st.title("AgroValue: Agricultural Commodity Price Forecasting")


@st.cache_data
def load_monthly_data(path: str) -> pd.DataFrame:
    """Load the project's wide monthly dataset and return a clean time-indexed frame."""
    df = pd.read_csv(path)

    if "Commodities" not in df.columns:
        raise ValueError("Expected a 'Commodities' column in the monthly dataset.")

    df = df.set_index("Commodities").T
    df.index = pd.to_datetime(df.index, format="%b-%y", errors="coerce")
    df = df[~df.index.isna()].sort_index()
    df = df.apply(pd.to_numeric, errors="coerce")

    # Do not fabricate missing market prices. Keep only observed values.
    return df


def prepare_series(
    df: pd.DataFrame,
    commodity: str,
    min_observations: int = 24,
) -> pd.Series:
    """Return one commodity series with a continuous monthly index where possible."""
    if commodity not in df.columns:
        raise ValueError(f"Commodity '{commodity}' was not found.")

    series = df[commodity].dropna().astype(float).sort_index()
    series = series[~series.index.duplicated(keep="last")]

    if len(series) < min_observations:
        raise ValueError(
            f"'{commodity}' has only {len(series)} observed months; "
            f"at least {min_observations} are required."
        )

    # Preserve real observations; only the index is normalized to month-end.
    series.index = series.index.to_period("M").to_timestamp("M")
    series = series.groupby(level=0).mean().sort_index()
    return series


def fit_sarimax(series: pd.Series):
    """Fit SARIMAX using only historical observations."""
    model = SARIMAX(
        series,
        order=ORDER,
        seasonal_order=SEASONAL_ORDER,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    return model.fit(disp=False)


def rmse(y_true: pd.Series, y_pred: pd.Series) -> float:
    return float(np.sqrt(np.mean(np.square(y_true - y_pred))))


def mae(y_true: pd.Series, y_pred: pd.Series) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def mape(y_true: pd.Series, y_pred: pd.Series) -> float:
    mask = y_true != 0
    if not mask.any():
        return float("nan")
    return float(
        np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    )


def seasonal_naive_forecast(train: pd.Series, steps: int, season: int = 12) -> pd.Series:
    """Seasonal-naive baseline: repeat the value from the same month in the prior year."""
    if len(train) < season:
        raise ValueError("Not enough observations for a seasonal-naive baseline.")

    values = []
    history = train.tolist()

    for _ in range(steps):
        prediction = history[-season]
        values.append(prediction)
        history.append(prediction)

    future_index = pd.date_range(
        start=train.index[-1] + pd.offsets.MonthEnd(1),
        periods=steps,
        freq="ME",
    )
    return pd.Series(values, index=future_index, name=train.name)


def evaluate_models(series: pd.Series, test_horizon: int = TEST_HORIZON) -> pd.DataFrame:
    """Evaluate SARIMAX against a seasonal-naive baseline on unseen future data."""
    if len(series) <= test_horizon + SEASONAL_PERIOD:
        raise ValueError("Not enough history for a one-year chronological holdout.")

    train = series.iloc[:-test_horizon]
    test = series.iloc[-test_horizon:]

    sarimax_result = fit_sarimax(train)
    sarimax_pred = sarimax_result.get_forecast(steps=test_horizon).predicted_mean
    sarimax_pred.index = test.index

    baseline_pred = seasonal_naive_forecast(train, test_horizon)
    baseline_pred.index = test.index

    rows = []
    for name, pred in [
        ("Seasonal Naive", baseline_pred),
        ("SARIMAX", sarimax_pred),
    ]:
        rows.append(
            {
                "Model": name,
                "MAE": mae(test, pred),
                "RMSE": rmse(test, pred),
                "MAPE (%)": mape(test, pred),
            }
        )

    return pd.DataFrame(rows), train, test, sarimax_pred, baseline_pred


try:
    df = load_monthly_data(DATA_PATH)
except FileNotFoundError:
    st.error(f"Data file not found: {DATA_PATH}")
    st.stop()
except Exception as exc:
    st.error(f"Could not load data: {exc}")
    st.stop()

commodities = sorted(df.columns.tolist())
selected_commodity = st.selectbox("Choose a commodity", commodities)

try:
    price_series = prepare_series(df, selected_commodity)
except Exception as exc:
    st.error(str(exc))
    st.stop()

st.caption(
    f"Observed history: {price_series.index.min():%b %Y} to "
    f"{price_series.index.max():%b %Y} "
    f"({len(price_series)} monthly observations)"
)

with st.expander("Model evaluation on unseen data", expanded=True):
    if st.button("Run backtest"):
        try:
            metrics, train, test, sarimax_pred, baseline_pred = evaluate_models(
                price_series
            )
            st.dataframe(metrics, use_container_width=True, hide_index=True)

            best_row = metrics.sort_values("RMSE").iloc[0]
            st.success(
                f"Best RMSE on the held-out final {TEST_HORIZON} months: "
                f"{best_row['Model']} ({best_row['RMSE']:.2f})"
            )

            fig, ax = plt.subplots(figsize=(11, 5))
            ax.plot(train.index, train, label="Train")
            ax.plot(test.index, test, label="Actual holdout")
            ax.plot(
                sarimax_pred.index,
                sarimax_pred,
                label="SARIMAX forecast",
            )
            ax.plot(
                baseline_pred.index,
                baseline_pred,
                label="Seasonal-naive baseline",
            )
            ax.set_title(f"{selected_commodity}: chronological backtest")
            ax.set_xlabel("Date")
            ax.set_ylabel("Price")
            ax.legend()
            ax.grid(alpha=0.2)
            st.pyplot(fig)
            plt.close(fig)
        except Exception as exc:
            st.error(f"Backtest failed: {exc}")

st.divider()

if st.button("Forecast next 60 months", type="primary"):
    try:
        result = fit_sarimax(price_series)
        forecast_result = result.get_forecast(steps=FORECAST_HORIZON)

        forecast_mean = forecast_result.predicted_mean
        confidence_interval = forecast_result.conf_int()

        future_index = pd.date_range(
            start=price_series.index[-1] + pd.offsets.MonthEnd(1),
            periods=FORECAST_HORIZON,
            freq="ME",
        )
        forecast_mean.index = future_index
        confidence_interval.index = future_index

        forecast_df = pd.DataFrame(
            {
                "Date": future_index,
                "Forecast": forecast_mean.to_numpy(),
                "Lower 95%": confidence_interval.iloc[:, 0].to_numpy(),
                "Upper 95%": confidence_interval.iloc[:, 1].to_numpy(),
            }
        )

        st.subheader(
            f"{selected_commodity} forecast: "
            f"{future_index[0]:%b %Y} – {future_index[-1]:%b %Y}"
        )
        st.dataframe(
            forecast_df.style.format(
                {
                    "Forecast": "{:.2f}",
                    "Lower 95%": "{:.2f}",
                    "Upper 95%": "{:.2f}",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

        fig, ax = plt.subplots(figsize=(11, 5))
        ax.plot(price_series.index, price_series, label="Historical")
        ax.plot(future_index, forecast_mean, label="SARIMAX forecast")
        ax.fill_between(
            future_index,
            confidence_interval.iloc[:, 0],
            confidence_interval.iloc[:, 1],
            alpha=0.2,
            label="95% confidence interval",
        )
        ax.set_title(f"{selected_commodity} price forecast")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        ax.legend()
        ax.grid(alpha=0.2)
        st.pyplot(fig)
        plt.close(fig)

        st.info(
            "The forecast is generated from the historical monthly price series. "
            "It does not claim that future arrivals, weather, production, or policy "
            "variables are known."
        )

    except Exception as exc:
        st.error(f"Forecasting failed: {exc}")
