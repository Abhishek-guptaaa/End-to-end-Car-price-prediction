import os
import sys
import pandas as pd
from car.config import Config
from car.logger import logging
from car.components.data_ingestion import DataIngestion
from car.components.data_cleaning import DataCleaning
from car.exception import CustomException

def main():
    try:
        logging.info("Starting NLP project")

        # Data Ingestion
        data_ingestion = DataIngestion()
        raw_data_path = data_ingestion.initiate_data_ingestion()

        # Data Cleaning
        data_cleaning = DataCleaning()
        cleaned_data_path = data_cleaning.initiate_data_cleaning(Config.RAW_DATA_PATH)

        # Data Transformation
        X_train_transformed, X_test_transformed, y_train, y_test = data_cleaning.initiate_data_transformation(cleaned_data_path)

        logging.info("Project completed successfully")

    except Exception as e:
        logging.error(f"Error in main script: {e}")
        raise CustomException(e, sys)

if __name__ == "__main__":
    main()
