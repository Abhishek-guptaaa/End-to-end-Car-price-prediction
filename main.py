import os
import sys
from car.components.data_ingestion import DataIngestion
from car.logger import logging
from car.exception import CustomException

if __name__ == "__main__":
        try:
            data_ingestion=DataIngestion()
            data_ingestion.initiate_data_ingestion()
        except Exception as e:
            raise CustomException(e, sys)
        
    
    
