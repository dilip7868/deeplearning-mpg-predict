import numpy as np
from flask import Flask, render_template, request
from tensorflow.keras import models

app = Flask(__name__)

# Load the trained Keras model
model = models.load_model("dnn_model.keras")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract features from form
        cylinders = float(request.form['cylinders'])
        displacement = float(request.form['displacement'])
        horsepower = float(request.form['horsepower'])
        weight = float(request.form['weight'])
        acceleration = float(request.form['acceleration'])
        model_year = float(request.form['model_year'])
        origin = request.form['origin'].lower()

        # Map origin to one-hot encoding columns (Europe, Japan, USA)
        europe = 1.0 if origin == 'europe' else 0.0
        japan = 1.0 if origin == 'japan' else 0.0
        usa = 1.0 if origin == 'usa' else 0.0

        # Combine into features array matching the model's 9 inputs:
        # [Cylinders, Displacement, Horsepower, Weight, Acceleration, Model Year, Europe, Japan, USA]
        features = np.array([[
            cylinders,
            displacement,
            horsepower,
            weight,
            acceleration,
            model_year,
            europe,
            japan,
            usa
        ]], dtype=np.float32)

        # Predict MPG using the neural network model
        prediction = model.predict(features)
        output = float(prediction[0][0])
        output_rounded = round(output, 2)

        # Formulate response
        prediction_text = f"Predicted Fuel Efficiency: {output_rounded} MPG"
        
        # Render page with result and past values for better UX
        return render_template(
            'index.html',
            prediction_text=prediction_text,
            cylinders=int(cylinders),
            displacement=displacement,
            horsepower=horsepower,
            weight=weight,
            acceleration=acceleration,
            model_year=int(model_year),
            origin=origin
        )
    except Exception as e:
        error_msg = f"Error in prediction: {str(e)}"
        return render_template('index.html', error_text=error_msg)

if __name__ == '__main__':
    app.run(debug=True)
