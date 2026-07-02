import os
from dotenv import load_dotenv
from sqlalchemy import create_engine

load_dotenv()


# ---------- S3 ----------
S3_BUCKET = os.getenv("MODEL_S3_BUCKET")
RAW_DATA_S3_KEY = os.getenv("RAW_DATA_S3_KEY")
MODEL_S3_KEY = os.getenv("MODEL_S3_KEY")



# Model 
MODEL_DIR = os.getenv("MODEL_DIR","model")
MODEL_PATH = os.getenv(
    "Model_path",
    os.path.join(MODEL_DIR,"xgboost_model.joblib")
)



DB_CONN = "postgresql://{}:{}@{}:{}/{}".format(
    os.getenv("POSTGRES_USER"),
    os.getenv("POSTGRES_PASSWORD"),
    os.getenv("DB_HOST"),
    os.getenv("DB_PORT"),
    os.getenv("POSTGRES_DB")
)