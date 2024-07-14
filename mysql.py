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
    """Read data from SQL database."""
    try:
        # Establish database connection
        mydb = pymysql.connect(
            host=host,
            user=user,
            password=password,
            db=db
        )
        logging.info("Connection established successfully.")
        
        # Read data from SQL
        df = pd.read_sql_query('SELECT * FROM car data', mydb)
        print(df.head())
        return df
        
    except Exception as e:
        raise CustomException(e, sys)
    
    finally:
        # Ensure the database connection is closed
        if 'mydb' in locals() and mydb.open:
            mydb.close()
            logging.info("Database connection closed.")

if __name__ == "__main__":
    read_sql_data()
