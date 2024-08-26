import os
import sys
import joblib
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from car.config import Config
from car.logger import logging
from car.exception import CustomException

class ModelEvaluation:
    def __init__(self):
        self.model_file_path = Config.MODEL_PATH
        self.preprocessor_file_path = Config.PREPROCESSOR_PATH

    def load_object(self, file_path):
        try:
            return joblib.load(file_path)
        except Exception as e:
            raise CustomException(f"Error loading object from {file_path}: {str(e)}")

    def evaluate(self):
        try:
            # Load the model
            model = self.load_object(self.model_file_path)

            # Load the test data
            X_test_path = Config.X_TEST_TRANSFORMED_PATH
            y_test_path = Config.Y_TEST_PATH

            X_test = pd.read_csv(X_test_path)
            y_test = pd.read_csv(y_test_path)

            # Debugging: Print or log the first few rows of X_test
            logging.info(f"First few rows of X_test: \n{X_test.head()}")
            logging.info(f"First few rows of y_test: \n{y_test.head()}")

            # Ensure that X_test is numeric and check data types
            logging.info(f"Data types in X_test: \n{X_test.dtypes}")

            # Predict on the test data
            y_pred = model.predict(X_test)

            # Calculate metrics
            rmse = root_mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            logging.info(f"Model Evaluation Results - RMSE: {rmse}, MAE: {mae}, R^2: {r2}")

            return {
                "rmse": rmse,
                "mae": mae,
                "r2": r2
            }

        except Exception as e:
            logging.error(f"Error occurred in model evaluation: {e}")
            raise CustomException(e, sys)

if __name__ == "__main__":
    try:
        evaluator = ModelEvaluation()
        results = evaluator.evaluate()
        print("Model Evaluation Results:", results)
    except Exception as e:
        logging.error(f"Error in model evaluation: {e}")
        raise CustomException(e, sys)
