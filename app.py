from flask import Flask, request, render_template
import os
import numpy as np
import pandas as pd
from src.pipeline.predict_pipeline import CustomData, PredictPipeline
import pickle

# Initialize Flask application
application = Flask(__name__)
app = application

# File paths
model_path = os.path.join('artifacts', 'model.pkl')
preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')

# Check and load model and preprocessor during startup
model = None
preprocessor = None

try:
    if os.path.exists(model_path):
        with open(model_path, 'rb') as model_file:
            model = pickle.load(model_file)
        print("Model loaded successfully.")
    else:
        print(f"Error: Model file not found at {model_path}")

    if os.path.exists(preprocessor_path):
        with open(preprocessor_path, 'rb') as preprocessor_file:
            preprocessor = pickle.load(preprocessor_file)
        print("Preprocessor loaded successfully.")
    else:
        print(f"Error: Preprocessor file not found at {preprocessor_path}")
except Exception as e:
    print(f"Error loading model or preprocessor: {e}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html', results=None)
    else:
        try:
            # Collect and preprocess input data
            data = CustomData(
                Make=request.form.get('Make').lower(),
                Model=request.form.get('Model').lower(),
                Year=int(request.form.get('Year')),
                Mileage=int(request.form.get('Mileage')),
                Fuel_Type=request.form.get('Fuel_Type').lower(),
                Engine_Size=float(request.form.get('Engine_Size')),
                Transmission=request.form.get('Transmission').lower(),
                Body_Type=request.form.get('Body_Type').lower(),
                Color=request.form.get('Color').lower(),
                Owner_History=request.form.get('Owner_History').lower(),
                Age=int(request.form.get('Age')),
            )

            pred_df = data.get_data_as_data_frame()
            print('The data to be predicted:\n', pred_df)

            # Make prediction
            predict_pipeline = PredictPipeline()
            print('Predict pipeline initialized.')
            log_results = predict_pipeline.predict(pred_df)
            print('Log result:', log_results)
            results = np.expm1(log_results)  # Convert log results to actual prices
            print('The predicted result is:', results)

            return render_template('home.html', results=results[0])
        except Exception as e:
            print(f"Error during prediction: {e}")
            return render_template('home.html', results="Error occurred during prediction.")

if __name__ == "__main__":
    # Run the application
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=True)
