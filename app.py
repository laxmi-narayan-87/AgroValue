import streamlit as st
import joblib
import numpy as np

# Title for the app
st.title("AgroValue: Price Prediction for Agricultural Commodities")

# Try loading the model
try:
    # Load the model with joblib
    model = joblib.load('model.pkl')
    st.success("Model loaded successfully.")
except FileNotFoundError:
    st.error("Model file not found! Please upload the model file.")
except Exception as e:
    st.error(f"Error loading the model: {e}")
    model = None  # Set model to None if loading fails

# Input fields for features (adjust based on your model's input requirements)
feature1 = st.number_input("Enter Feature 1", value=0.0)
feature2 = st.number_input("Enter Feature 2", value=0.0)
# Add more inputs if your model requires more features

# Combine features into a single array (adjust shape based on your model)
features = np.array([[feature1, feature2]])

# Predict button
if st.button('Predict'):
    if model is not None:
        try:
            prediction = model.predict(features)
            st.success(f"Predicted Price: {prediction[0]}")
        except Exception as e:
            st.error(f"Error making prediction: {e}")
    else:
        st.error("Model not loaded, cannot make predictions.")
