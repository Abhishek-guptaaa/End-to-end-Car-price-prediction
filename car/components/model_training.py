import os
import sys
import numpy as np
from dataclasses import dataclass
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import AdaBoostRegressor,GradientBoostingRegressor,RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
import joblib
from car.exception import CustomException
from car.logger import logging

@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("models", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def eval_metrics(self, actual, pred):
        rmse = np.sqrt(mean_squared_error(actual, pred))
        mae = mean_absolute_error(actual, pred)
        r2 = r2_score(actual, pred)
        return rmse, mae, r2

    def initiate_model_trainer(self, X_train_transformed, X_test_transformed, y_train, y_test):
        try:
            logging.info("Training and evaluating models")

            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }

            params = {
                "Decision Tree": {
                    'criterion': ['mse', 'friedman_mse', 'mae', 'poisson'],
                },
                "Random Forest": {
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                "Gradient Boosting": {
                    'learning_rate': [0.1, 0.01, 0.05, 0.001],
                    'subsample': [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                "Linear Regression": {},
                "XGBRegressor": {
                    'learning_rate': [0.1, 0.01, 0.05, 0.001],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                "AdaBoost Regressor": {
                    'learning_rate': [0.1, 0.01, 0.5, 0.001],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                }
            }

            model_report = self.evaluate_models(X_train_transformed, y_train, X_test_transformed, y_test, models, params)

            # Get best model score and name from the report
            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            logging.info(f"Best model: {best_model_name}, Score: {best_model_score}")

            if best_model_score < 0.6:
                raise CustomException("No best model found")

            # Save the best model
            self.save_object(self.model_trainer_config.trained_model_file_path, best_model)

            # Evaluate on test set
            y_pred = best_model.predict(X_test_transformed)
            r2 = r2_score(y_test, y_pred)
            logging.info(f"Test set R^2 score: {r2}")

            return best_model, r2

        except Exception as e:
            raise CustomException(e, sys)

    def evaluate_models(self, X_train, y_train, X_test, y_test, models, params):
        model_report = {}
        for name, model in models.items():
            try:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                score = r2_score(y_test, y_pred)
                model_report[name] = score
                logging.info(f"Model: {name}, R^2 score: {score}")
            except Exception as e:
                logging.error(f"Error evaluating model {name}: {str(e)}")
        return model_report

    def save_object(self, file_path, obj):
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            joblib.dump(obj, file_path)
            logging.info(f"Saved object to {file_path}")
        except Exception as e:
            logging.error(f"Error saving object to {file_path}: {str(e)}")
