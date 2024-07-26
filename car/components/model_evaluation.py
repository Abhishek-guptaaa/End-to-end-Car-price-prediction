import os
import sys
import joblib
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from car.config import Config
from car.logger import logging
from car.exception import CustomException

class ModelEvaluation:
    def __init__(self):
        self.model_file_path = os.path.join("models", "model.pkl")
        self.preprocessor_file_path = os.path.join("models", "preprocessor.pkl")

    def load_object(self, file_path):
        try:
            return joblib.load(file_path)
        except Exception as e:
            raise CustomException(f"Error loading object from {file_path}: {str(e)}")

    def evaluate(self):
        try:
            # Load the model and preprocessor
            model = self.load_object(self.model_file_path)
            preprocessor = self.load_object(self.preprocessor_file_path)

            # Load the test data
            X_test = pd.read_csv(Config.X_TEST_PATH)
            y_test = pd.read_csv(Config.CLEANED_DATA_PATH)["price"]

            # Transform the test data
            X_test_transformed = preprocessor.transform(X_test)

            # Predict on the test data
            y_pred = model.predict(X_test_transformed)

            # Calculate metrics
            rmse = mean_squared_error(y_test, y_pred, squared=False)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            logging.info(f"Model Evaluation Results - RMSE: {rmse}, MAE: {mae}, R^2: {r2}")

            return {
                "rmse": rmse,
                "mae": mae,
                "r2": r2
            }

        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    try:
        evaluator = ModelEvaluation()
        results = evaluator.evaluate()
        print("Model Evaluation Results:", results)
    except Exception as e:
        logging.error(f"Error in model evaluation: {e}")
        raise CustomException(e, sys)
