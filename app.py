import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
import joblib
import matplotlib.pyplot as plt

# Load the ML model
try:
    model = joblib.load('model.pkl')  # Load the machine learning model
    st.success("ML Model loaded successfully.")
except FileNotFoundError:
    st.error("ML model file not found! Please upload the model file.")
    model = None  # Fallback in case the model can't be loaded
#except Exception as e:
 #   st.error(f"Error loading the ML model: {e}")
  #  model = None

# Title for the app
st.title("AgroValue: Price Prediction for Agricultural Commodities")

# CSV File Path
file_path = "monthly_data.csv"  

# Load data from CSV
try:
    df = pd.read_csv(file_path)
    st.success("CSV Data loaded successfully.")
except FileNotFoundError:
    st.error(f"CSV file not found: {file_path}")
    df = None
except Exception as e:
    st.error(f"Error loading CSV data: {e}")
    df = None

if df is not None:
    # Data preprocessing for SARIMAX forecasting
    df.set_index('Commodities', inplace=True)
    df = df.T
    df.index = pd.date_range(start='2014-01', periods=len(df), freq='M')
    df = df.ffill()  # Forward fill to handle missing data

    # Get commodity list from CSV data
    commodities = df.columns.tolist()

    # Select a commodity for forecasting
    selected_commodity = st.selectbox("Choose a Commodity for SARIMAX Forecast", commodities)

    # SARIMAX Forecasting
    if st.button("Submit"):
        data = df[selected_commodity]

        # Train SARIMAX model
        model_sarimax = SARIMAX(data, order=(1, 1, 1), seasonal_order=(1, 1, 0, 12))
        sarimax_model = model_sarimax.fit(disp=False)

        # Forecast for 5 years (60 months)
        forecast = sarimax_model.get_forecast(steps=12 * 5)  # 5 years forecast
        forecasted_values = forecast.predicted_mean

        # Create forecast DataFrame
        forecast_years = pd.date_range(start='2025-01', periods=12 * 5, freq='M')
        forecast_df = pd.DataFrame({'Year': forecast_years, f'{selected_commodity}_Price_Forecast': forecasted_values})

        # Display forecast
        st.write(f"### {selected_commodity} Price Forecast (2025-2029)")
        st.write(forecast_df)

        # Plot actual vs forecasted values
        plt.figure(figsize=(10, 6))
        plt.plot(data, label=f'Actual {selected_commodity} Prices')
        plt.plot(forecast_years, forecasted_values, label=f'Forecasted {selected_commodity} Prices', color='orange')
        plt.title(f'{selected_commodity} Price Forecast (2025-2029)')
        plt.xlabel('Year')
        plt.ylabel('Price')
        plt.legend()
        st.pyplot(plt)

        # Calculate and display RMSE for training
        train_rmse = np.sqrt(((data - sarimax_model.fittedvalues) ** 2).mean())
        st.write(f"Training RMSE: {train_rmse:.4f}")

# Input fields for machine learning model prediction (adjust based on your model's inputs)
if model is not None:
    st.write("### Machine Learning Model Prediction")
    feature1 = st.number_input("Enter Feature 1 for ML Prediction", value=0.0)
    feature2 = st.number_input("Enter Feature 2 for ML Prediction", value=0.0)

    # Convert input features to numpy array
    features = np.array([[feature1, feature2]])  # Adjust shape to fit your model

    # Predict using ML model
    if st.button('Predict with ML Model'):
        try:
            prediction = model.predict(features)
            st.success(f"Predicted Price using ML Model: {prediction[0]}")
        except Exception as e:
            st.error(f"Error making prediction: {e}")
