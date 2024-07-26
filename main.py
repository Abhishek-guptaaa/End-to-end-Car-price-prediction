import os
import sys
import pandas as pd
from car.config import Config
from car.logger import logging
from car.components.data_ingestion import DataIngestion
from car.components.data_cleaning import DataCleaning
from car.components.model_training import ModelTrainer
from car.components.model_evaluation import ModelEvaluation
from car.exception import CustomException

def main():
    try:
        logging.info("Starting model training and evaluation")

        # Data Ingestion
        data_ingestion = DataIngestion()
        raw_data_path = data_ingestion.initiate_data_ingestion()

        # Data Cleaning
        data_cleaning = DataCleaning()
        cleaned_data_path = data_cleaning.initiate_data_cleaning(Config.RAW_DATA_PATH)

        # Data Transformation
        X_train_transformed, X_test_transformed, y_train, y_test = data_cleaning.initiate_data_transformation(cleaned_data_path)

        # Save the test data for later evaluation
        X_test_transformed_path = Config.X_TEST_TRANSFORMED_PATH
        y_test_path = Config.Y_TEST_PATH
        pd.DataFrame(X_test_transformed).to_csv(X_test_transformed_path, index=False)
        pd.DataFrame(y_test).to_csv(y_test_path, index=False)

        # Model Training
        model_trainer = ModelTrainer()
        best_model, best_model_score = model_trainer.initiate_model_trainer(X_train_transformed, X_test_transformed, y_train, y_test)
        logging.info(f"Best model trained with R^2 score: {best_model_score}")



    except Exception as e:
        logging.error(f"Error in main script: {e}")
        raise CustomException(e, sys)

if __name__ == "__main__":
    main()



