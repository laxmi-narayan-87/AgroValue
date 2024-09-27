import streamlit as st
import pickle  
import numpy as np

st.title("AgroValue: Price Prediction for Agricultural Commodities")

try:
    with open('model.pkl', 'rb') as file:
        model = pickle.load(file)  
except Exception as e:
    st.error(f"Error loading the model: {e}")

feature1 = st.number_input("Enter Feature 1", value=0.0)
feature2 = st.number_input("Enter Feature 2", value=0.0)

features = np.array([[feature1, feature2]])
if st.button('Predict'):
    try:
        prediction = model.predict(features)
        st.success(f"Predicted Price: {prediction[0]}")
    except Exception as e:
        st.error(f"Error making prediction: {e}")
