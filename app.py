from flask import Flask, request, render_template, session
import numpy as np
import pandas as pd
import os
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)
app = application

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html', results=None)
    else:
        data = CustomData(
            Make=request.form.get('Make'),
            Model=request.form.get('Model'),
            Year=int(request.form.get('Year')),
            Mileage=int(request.form.get('Mileage')),
            Fuel_Type=request.form.get('Fuel_Type'),
            Engine_Size=float(request.form.get('Engine_Size')),
            Transmission=request.form.get('Transmission'),
            Body_Type=request.form.get('Body_Type'),
            Color=request.form.get('Color'),
            Owner_History=request.form.get('Owner_History'),
            Age=int(request.form.get('Age')),
        )
        
        pred_df = data.get_data_as_data_frame()
        print('The data to be predicted: ', pred_df.T)
        
        predict_pipeline = PredictPipeline()
        print('Got the predict pipeline')
        log_results = predict_pipeline.predict(pred_df)
        print('Log result: ', log_results)
        results = np.expm1(log_results)
        print('The result is: ', results)
        return render_template('home.html', results=results[0])

if __name__ == "__main__":
    #app.run(host="0.0.0.0")
    #app.run(host="0.0.0.0", debug=True)
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)), debug=True)