from flask import Flask, request, render_template
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load the model and preprocessor using joblib
model = joblib.load('models/model.pkl')
preprocessor = joblib.load('models/preprocessor.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    year = int(request.form['year'])
    mileage = int(request.form['mileage'])
    brand = request.form['brand']
    model_name = request.form['model']

    input_data = pd.DataFrame({
        'year': [year],
        'mileage': [mileage],
        'brand': [brand],
        'model': [model_name]
    })

    # Preprocess the input data
    processed_data = preprocessor.transform(input_data)
    
    # Make the prediction
    prediction = model.predict(processed_data)
    
    return render_template('index.html', prediction=prediction[0])

if __name__ == "__main__":
    app.run("0.0.0.0", port=8080)
