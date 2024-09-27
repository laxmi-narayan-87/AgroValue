import streamlit as st
import requests
import json

# Flask API URL
FLASK_API_URL = "http://127.0.0.1:5000/predict"

st.title('Price Prediction of Agri-Horticultural Commodities')

# Input fields for features (adjust according to your model's input features)
feature1 = st.number_input('Enter Feature 1')
feature2 = st.number_input('Enter Feature 2')
# Add more input fields depending on the number of features

features = [feature1, feature2]  # Add all the features here

if st.button('Predict'):
    # Prepare data in the same structure as Flask expects
    input_data = {'features': features}
    
    # Send POST request to Flask API
    response = requests.post(FLASK_API_URL, json=input_data)
    
    if response.status_code == 200:
        result = response.json()
        st.success(f"Prediction: {result['prediction']}")
    else:
        st.error(f"Error: {response.status_code}")
