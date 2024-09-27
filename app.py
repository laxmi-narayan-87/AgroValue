import streamlit as st
import joblib
import numpy as np

st.title("AgroValue: Price Prediction for Agricultural Commodities")

# Load the model
try:
    model = joblib.load('model.pkl')
except Exception as e:
    st.error(f"Error loading the model: {e}")

# Input fields for features
feature1 = st.number_input("Enter Feature 1", value=0.0)
feature2 = st.number_input("Enter Feature 2", value=0.0)
# Add more inputs based on your model's features

features = np.array([[feature1, feature2]])  # Adjust based on your input shape

if st.button('Predict'):
    try:
        prediction = model.predict(features)
        st.success(f"Predicted Price: {prediction[0]}")
    except Exception as e:
        st.error(f"Error making prediction: {e}")
