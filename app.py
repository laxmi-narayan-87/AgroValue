import pickle  
from flask import Flask, request, jsonify
import numpy as np

app = Flask(__name__)

# Load the model
try:
    with open('model.pkl', 'rb') as file:
        model = pickle.load(file)  
except Exception as e:
    print(f"Error loading the model: {e}")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    features = np.array(data['features']).reshape(1, -1)
    prediction = model.predict(features)
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(debug=True)
