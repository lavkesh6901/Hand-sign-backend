from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import base64
import time
import os
from keras.models import load_model

app = Flask(__name__)
CORS(app)

# Load model and labels (relative paths)
model = load_model("models/landmark_model.keras")
with open("models/labels.txt", "r") as f:
    labels = [line.strip().split(maxsplit=1)[1] for line in f]

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get landmark array from frontend (JSON)
        data = request.get_json()
        landmark_data = data.get("landmarks")

        if landmark_data and isinstance(landmark_data, list) and len(landmark_data) == 42:
            prediction = model.predict(np.array([landmark_data]))
            label_index = np.argmax(prediction)
            confidence = float(np.max(prediction))
            label = labels[label_index]
            return jsonify({'prediction': label, 'confidence': confidence})

        return jsonify({'prediction': 'Invalid input', 'confidence': 0.0})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host="0.0.0.0", port=port)
