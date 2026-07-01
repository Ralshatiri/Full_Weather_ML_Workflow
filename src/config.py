import os
from dotenv import load_dotenv
from sqlalchemy import create_engine


# Dataset 
CSV_PATH = os.getenv("CSV_path","data/raw/saudi_weather_data.csv")

# Model 
MODEL_DIR = os.getenv("MODEL_DIR","model")
MODEL_PATH = os.getenv(
    "Model_path",
    os.path.join(MODEL_DIR,"xgboost_model.joblib")
)

load_dotenv()

DB_CONN = "postgresql://{}:{}@{}:{}/{}".format(
    os.getenv("POSTGRES_USER"),
    os.getenv("POSTGRES_PASSWORD"),
    os.getenv("DB_HOST"),
    os.getenv("DB_PORT"),
    os.getenv("POSTGRES_DB")
)