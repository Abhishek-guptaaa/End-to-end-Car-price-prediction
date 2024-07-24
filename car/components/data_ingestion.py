import os
import sys
from car.exception import CustomException
from car.logger import logging
import pandas as pd
from mysql import read_sql_data

from dataclasses import dataclass


@dataclass
class DataIngestionConfig:
    raw_data_path:str=os.path.join('artifacts','raw.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()

    def initiate_data_ingestion(self):
        try:
            ##reading the data from mysql
            df = read_sql_data()
            logging.info("Reading completed mysql database")

            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path),exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)
            logging.info("Data Ingestion is completed")

            return df
        except Exception as e:
            raise CustomException(e,sys)



        