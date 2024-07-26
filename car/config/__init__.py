class Config:
    RAW_DATA_PATH = 'notebook/raw.csv'
    CLEANED_DATA_PATH = 'notebook/cleaned_data.csv'
    PREPROCESSOR_PATH='models/preprocessor.pkl'
    MODEL_PATH = 'models/model.pkl'
    LOG_PATH = 'logs/logs.log'
    X_TEST_TRANSFORMED_PATH = 'notebook/X_test_transformed.csv'
    Y_TEST_PATH = 'notebook/y_test.csv'
    
    DROP_COLUMNS = ['id']
    AXIS = 1
    INPLACE = True

    APP_HOST = '0.0.0.0'
    APP_PORT = 8080

    TARGET_COLUMN = 'price'


