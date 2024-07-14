import os
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)

project_name='car'
list_of_files=[
    f"{project_name}/components/__init__.py",
    f"{project_name}/components/data_ingestion.py",
    f"{project_name}/components/data_cleaning.py",
    f"{project_name}/components/feature_engineering.py",
    f"{project_name}/components/model_training.py",
    f"{project_name}/components/model_evaluation.py",
    f"{project_name}/pipelines/__init__.py",
    f"{project_name}/pipelines/training_pipeline.py",
    f"{project_name}/pipelines/prediction_pipeline.py",
    f"{project_name}/config/__init__.py",
    f"{project_name}/config/config.py",
    f"{project_name}/logger/__init__.py",
    f"{project_name}/exception/__init__.py",
    "setup.py",
    "main.py",
    "app.py",
    "mysql.py",
    "Dockerfile",
    "requirements.txt"

]

for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)

    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Creating directory: {filedir} for file: {filename}")

    if not os.path.exists(filepath) or os.path.getsize(filepath) == 0:
        with open(filepath, 'w') as w:
            pass
        logging.info(f"Creating empty file: {filepath}")
    else:
        logging.info(f"File already exists and is not empty: {filepath}")