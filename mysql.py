import os
import sys
import pymysql
import pandas as pd
from dotenv import load_dotenv
from car.logger import logging
from car.exception import CustomException

# Load environment variables
load_dotenv()

# Get environment variables
host = os.getenv("HOST")
user = os.getenv("USER")
password = os.getenv("PASSWORD")
db = os.getenv("DB")
def read_sql_data():
    logging.info("Reading SQL database started")
    try:
        mydb=pymysql.connect(
            host=host,
            user=user,
            password=password,
            db=db
        )
        logging.info("Connection Established",mydb)
        df=pd.read_sql_query('Select * from car',mydb)
        print(df.head())

        return df



    except Exception as ex:
        raise CustomException(ex)
