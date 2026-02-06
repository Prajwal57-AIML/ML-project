import pickle
import numpy as np
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)


# Load model and scaler

regmodel = pickle.load(open('regmodel.pkl', 'rb'))
scaler = pickle.load(open('scaling.pkl', 'rb'))


# Home page

@app.route('/')
def home():
    return render_template('home.html')


# Prediction using Postman / API

@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.json['data']

    # Convert input to numpy array
    input_data = np.array(list(data.values())).reshape(1, -1)

    # Scale input
    scaled_data = scaler.transform(input_data)

    # Predict
    prediction = regmodel.predict(scaled_data)[0]

    return jsonify({'prediction': float(prediction)})


# Prediction using HTML form

@app.route('/predict', methods=['POST'])
def predict():
    # Get form values
    data = [float(x) for x in request.form.values()]

    # Reshape & scale
    final_input = scaler.transform(np.array(data).reshape(1, -1))

    # Predict
    output = regmodel.predict(final_input)[0]

    return render_template(
        'home.html',
        prediction_text=f'The House Price Prediction is {round(output, 2)}'
    )


# Run Flask app

if __name__ == "__main__":
    app.run(debug=True)
