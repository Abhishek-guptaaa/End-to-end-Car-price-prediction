import os
import joblib
import pandas as pd
from car.exception import CustomException
from car.logger import logging

def load_preprocessor():
    preprocessor_path = os.path.join('models', 'preprocessor.pkl')
    if os.path.exists(preprocessor_path):
        return joblib.load(preprocessor_path)
    else:
        raise FileNotFoundError("Preprocessor file not found")

def load_model():
    model_path = os.path.join('models', 'model.pkl')
    if os.path.exists(model_path):
        return joblib.load(model_path)
    else:
        raise FileNotFoundError("Model file not found")

def predict(data):
    try:
        # Load preprocessor and model
        preprocessor = load_preprocessor()
        model = load_model()

        # Preprocess the input data
        data_transformed = preprocessor.transform(data)

        # Make predictions
        predictions = model.predict(data_transformed)

        return predictions

    except Exception as e:
        raise CustomException(e)

def get_user_input():
    # Example: Collect user input
    year = int(input("Enter year: "))
    mileage = float(input("Enter mileage: "))
    brand = input("Enter brand: ")
    model = input("Enter model: ")

    # Create a DataFrame for input data
    input_data = pd.DataFrame({
        'year': [year],
        'mileage': [mileage],
        'brand': [brand],
        'model': [model]
    })
    return input_data

if __name__ == "__main__":
    try:
        # Get user input
        user_data = get_user_input()

        # Predict
        predictions = predict(user_data)

        print("Predictions:", predictions)

    except Exception as e:
        logging.error(f"Error in prediction: {e}")
        raise CustomException(e)
