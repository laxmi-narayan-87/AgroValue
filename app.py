from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from statsmodels.stats.diagnostic import acorr_ljungbox
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
    """Load the wide monthly dataset and return a clean time-indexed frame."""
    df = pd.read_csv(path)

    if "Commodities" not in df.columns:
        raise ValueError("Expected a 'Commodities' column in the monthly dataset.")

    df = df.set_index("Commodities").T
    df.index = pd.to_datetime(df.index, format="%b-%y", errors="coerce")
    df = df[~df.index.isna()].sort_index()
    df = df.apply(pd.to_numeric, errors="coerce")
    df.index = df.index.to_period("M").to_timestamp("M")

    if df.index.duplicated().any():
        df = df.groupby(level=0).mean()

    return df


def prepare_series(
    df: pd.DataFrame,
    commodity: str,
    min_observations: int = 24,
) -> pd.Series:
    """Return one commodity series and reject gaps in the monthly timeline."""
    if commodity not in df.columns:
        raise ValueError(f"Commodity '{commodity}' was not found.")

    series = df[commodity].dropna().astype(float).sort_index()
    series = series[~series.index.duplicated(keep="last")]

    if len(series) < min_observations:
        raise ValueError(
            f"'{commodity}' has only {len(series)} observed months; "
            f"at least {min_observations} are required."
        )

    expected = pd.date_range(
        series.index.min(),
        series.index.max(),
        freq="ME",
    )
    missing = expected.difference(series.index)
    if len(missing):
        examples = ", ".join(d.strftime("%b %Y") for d in missing[:5])
        suffix = " ..." if len(missing) > 5 else ""
        raise ValueError(
            f"'{commodity}' has {len(missing)} missing month(s) "
            f"({examples}{suffix}). Fill/repair the source data before forecasting."
        )

    series = series.reindex(expected)
    series.name = commodity
    return series


def fit_sarimax(series: pd.Series):
    """Fit the configured SARIMAX model using historical observations only."""
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


def smape(y_true: pd.Series, y_pred: pd.Series) -> float:
    denominator = np.abs(y_true) + np.abs(y_pred)
    mask = denominator != 0
    if not mask.any():
        return float("nan")
    return float(
        np.mean(2 * np.abs(y_pred[mask] - y_true[mask]) / denominator[mask]) * 100
    )


def seasonal_naive_forecast(
    train: pd.Series,
    steps: int,
    season: int = SEASONAL_PERIOD,
) -> pd.Series:
    """Repeat the value from the same month in the previous year."""
    if len(train) < season:
        raise ValueError("Not enough observations for a seasonal-naive baseline.")

    history = train.tolist()
    predictions = []

    for _ in range(steps):
        prediction = history[-season]
        predictions.append(prediction)
        history.append(prediction)

    future_index = pd.date_range(
        start=train.index[-1] + pd.offsets.MonthEnd(1),
        periods=steps,
        freq="ME",
    )
    return pd.Series(predictions, index=future_index, name=train.name)


def evaluate_models(
    series: pd.Series,
    test_horizon: int = TEST_HORIZON,
):
    """Compare SARIMAX with a seasonal-naive baseline on the final holdout."""
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
                "sMAPE (%)": smape(test, pred),
            }
        )

    metrics = pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True)
    return metrics, train, test, sarimax_pred, baseline_pred


def forecast_best_model(series: pd.Series):
    """Select the lower-RMSE model on the holdout, then refit/use it on full history."""
    metrics, train, test, sarimax_pred, baseline_pred = evaluate_models(series)
    best_model = str(metrics.iloc[0]["Model"])

    future_index = pd.date_range(
        start=series.index[-1] + pd.offsets.MonthEnd(1),
        periods=FORECAST_HORIZON,
        freq="ME",
    )

    if best_model == "SARIMAX":
        result = fit_sarimax(series)
        forecast_result = result.get_forecast(steps=FORECAST_HORIZON)
        forecast_mean = forecast_result.predicted_mean
        confidence_interval = forecast_result.conf_int()
        forecast_mean.index = future_index
        confidence_interval.index = future_index
        return (
            best_model,
            metrics,
            forecast_mean,
            confidence_interval,
            train,
            test,
            sarimax_pred,
            baseline_pred,
        )

    forecast_mean = seasonal_naive_forecast(series, FORECAST_HORIZON)
    forecast_mean.index = future_index
    return (
        best_model,
        metrics,
        forecast_mean,
        None,
        train,
        test,
        sarimax_pred,
        baseline_pred,
    )


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

            best_row = metrics.iloc[0]
            st.success(
                f"Best RMSE on the held-out final {TEST_HORIZON} months: "
                f"{best_row['Model']} ({best_row['RMSE']:.2f})"
            )

            fig, ax = plt.subplots(figsize=(11, 5))
            ax.plot(train.index, train, label="Train")
            ax.plot(test.index, test, label="Actual holdout")
            ax.plot(sarimax_pred.index, sarimax_pred, label="SARIMAX forecast")
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

            residuals = test - (
                sarimax_pred if metrics.iloc[metrics["Model"].eq("SARIMAX").idxmax()]["RMSE"]
                <= metrics.iloc[metrics["Model"].eq("Seasonal Naive").idxmax()]["RMSE"]
                else baseline_pred
            )
            lb = acorr_ljungbox(residuals, lags=[min(6, len(residuals) // 2)], return_df=True)
            st.caption(
                f"Ljung–Box p-value: {lb['lb_pvalue'].iloc[0]:.4f}. "
                "A low value suggests remaining autocorrelation."
            )
        except Exception as exc:
            st.error(f"Backtest failed: {exc}")

st.divider()

if st.button("Forecast next 60 months", type="primary"):
    try:
        (
            best_model,
            metrics,
            forecast_mean,
            confidence_interval,
            _,
            _,
            _,
            _,
        ) = forecast_best_model(price_series)

        st.success(f"Selected forecasting model: {best_model}")

        forecast_df = pd.DataFrame(
            {
                "Date": forecast_mean.index,
                "Forecast": forecast_mean.to_numpy(),
            }
        )

        if confidence_interval is not None:
            forecast_df["Lower 95%"] = confidence_interval.iloc[:, 0].to_numpy()
            forecast_df["Upper 95%"] = confidence_interval.iloc[:, 1].to_numpy()

        st.subheader(
            f"{selected_commodity} forecast: "
            f"{forecast_mean.index[0]:%b %Y} – "
            f"{forecast_mean.index[-1]:%b %Y}"
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
        ax.plot(forecast_mean.index, forecast_mean, label=f"{best_model} forecast")

        if confidence_interval is not None:
            ax.fill_between(
                confidence_interval.index,
                confidence_interval.iloc[:, 0],
                confidence_interval.iloc[:, 1],
                alpha=0.2,
                label="Approx. 95% model interval",
            )

        ax.set_title(f"{selected_commodity} price forecast")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        ax.legend()
        ax.grid(alpha=0.2)
        st.pyplot(fig)
        plt.close(fig)

        if confidence_interval is None:
            st.info(
                "The seasonal-naive model does not provide a model-based confidence "
                "interval in this implementation."
            )
        else:
            st.info(
                "The interval is an approximate model-based 95% interval. "
                "It does not capture future shocks such as weather, supply changes, "
                "policy changes, or unexpected market events."
            )

    except Exception as exc:
        st.error(f"Forecasting failed: {exc}")
