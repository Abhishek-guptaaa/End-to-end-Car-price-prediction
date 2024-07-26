from pymongo import MongoClient
from dotenv import load_dotenv
import os
import pandas as pd
from car.logger import logging  # Ensure your logging setup is correctly imported

def read_mongo_data():
    """Read data from MongoDB collection and convert to DataFrame."""
    client = None  # Initialize client to None
    try:
        # Load environment variables from .env file
        load_dotenv()
        
        # Get MongoDB URI from .env file
        mongo_uri = os.getenv("MONGO_URI")
        
        if not mongo_uri:
            logging.error("MongoDB URI not found in .env file")
            return None
        
        # Create a MongoDB client
        client = MongoClient(mongo_uri)
        
        # Access the database
        db = client.get_database("car")  # Replace with your database name
        
        # Access the collection
        collection = db["car_sells"]  # Replace with your collection name
        
        # Fetch all documents
        documents = list(collection.find())  # collection.find() Yeh method MongoDB collection me se saare documents ko retrieve karta hai. 
        
        # Log data collection completion
        logging.info("Data collection completed successfully!")
        
        # Convert documents to DataFrame
        df = pd.DataFrame(documents)
        
        return df
    
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        return None
    
    finally:
        # Close the MongoDB client connection if it was created
        if client is not None:
            client.close()

# Example usage of the function
if __name__ == "__main__":
    df = read_mongo_data()
    if df is not None:
        print(df.head())  # Print the first few rows of the DataFrame