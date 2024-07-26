import os
import sys
import joblib
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from car.exception import CustomException
from car.logger import logging
from car.config import Config

class DataCleaning:
    def __init__(self):
        self.preprocessor_obj_file_path = os.path.join('models', 'preprocessor.pkl')

    def fill_null_values(self, df):
        df['brand'] = df['brand'].fillna(df['brand'].mode()[0])
        df['model'] = df['model'].fillna(df['model'].mode()[0])
        return df

    def remove_duplicates(self, df):
        df = df.drop_duplicates()
        return df

    def remove_outliers(self, df, column):
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

    def clean_data(self, df):
        # Fill null values
        df = self.fill_null_values(df)

        # Remove duplicates
        df = self.remove_duplicates(df)

        # Remove outliers
        for col in ['mileage', 'year', 'price']:
            df = self.remove_outliers(df, col)
        
        return df

    def get_data_transformer_object(self):
        numerical_columns = ["year", "mileage"]
        categorical_columns = ['brand', 'model']

        # Numerical pipeline
        num_pipeline = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy='median')),  # Handling missing values
            ("scaler", StandardScaler()),  # Feature scaling
            ("poly", PolynomialFeatures(degree=2, include_bias=False))  # Polynomial features
        ])
        
        # Categorical pipeline
        cat_pipeline = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),  # Handling missing values
            ("one_hot_encoder", OneHotEncoder()),  # One-hot encoding
            ("scaler", StandardScaler(with_mean=False))  # Feature scaling
        ])

        logging.info(f"Categorical Columns: {categorical_columns}")
        logging.info(f"Numerical Columns: {numerical_columns}")

        preprocessor = ColumnTransformer(
            transformers=[
                ("num_pipeline", num_pipeline, numerical_columns),
                ("cat_pipeline", cat_pipeline, categorical_columns)
            ]
        )
        return preprocessor

    def initiate_data_cleaning(self, raw_data_path):
        try:
            # Load data
            data = pd.read_csv(raw_data_path)

            # Clean data
            data = self.clean_data(data)

            # Save cleaned data
            cleaned_data_path = Config.CLEANED_DATA_PATH
            data.to_csv(cleaned_data_path, index=False)

            return cleaned_data_path
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, cleaned_data_path):
        try:
            # Load cleaned data
            data = pd.read_csv(cleaned_data_path)

            # Separate features and target
            X = data.drop(columns=['price'])
            y = data['price']

            # Split your data into train and test sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Get the preprocessor
            preprocessor = self.get_data_transformer_object()

            # Fit and transform the training data
            X_train_transformed = preprocessor.fit_transform(X_train)

            # Transform the test data
            X_test_transformed = preprocessor.transform(X_test)

            # Save the preprocessor
            os.makedirs('models', exist_ok=True)
            joblib.dump(preprocessor, self.preprocessor_obj_file_path)

            logging.info("Data transformation complete and preprocessor saved.")

            return X_train_transformed, X_test_transformed, y_train, y_test
        except Exception as e:
            raise CustomException(e, sys)
