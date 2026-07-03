import os
from dotenv import load_dotenv

load_dotenv()

# ---------- S3 ----------
S3_BUCKET = os.getenv("MODEL_S3_BUCKET")
RAW_DATA_S3_KEY = os.getenv("RAW_DATA_S3_KEY")
MODEL_S3_KEY = os.getenv("MODEL_S3_KEY")

# ---------- Model ----------
MODEL_DIR = os.getenv("MODEL_DIR", "model")
MODEL_PATH = os.getenv(
    "Model_path",
    os.path.join(MODEL_DIR, "xgboost_model.joblib")
)

# ---------- Database ----------
POSTGRES_USER = os.getenv("POSTGRES_USER")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
POSTGRES_DB = os.getenv("POSTGRES_DB")

required_vars = {
    "POSTGRES_USER": POSTGRES_USER,
    "POSTGRES_PASSWORD": POSTGRES_PASSWORD,
    "DB_HOST": DB_HOST,
    "DB_PORT": DB_PORT,
    "POSTGRES_DB": POSTGRES_DB,
}

missing = [name for name, value in required_vars.items() if not value]

if missing:
    raise ValueError(f"Missing required environment variables: {missing}")

DB_CONN = (
    f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}"
    f"@{DB_HOST}:{DB_PORT}/{POSTGRES_DB}"
)

# ---------- Redis ----------
REDIS_HOST = os.getenv("REDIS_HOST")
REDIS_PORT = int(os.getenv("REDIS_PORT"))
REDIS_DB = int(os.getenv("REDIS_DB"))
REDIS_TTL_SECONDS = int(os.getenv("REDIS_TTL_SECONDS"))

# ---------- RabbitMQ ----------
RABBITMQ_HOST = os.getenv("RABBITMQ_HOST")
RABBITMQ_PORT = int(os.getenv("RABBITMQ_PORT"))
RABBITMQ_USER = os.getenv("RABBITMQ_USER")
RABBITMQ_PASSWORD = os.getenv("RABBITMQ_PASSWORD")
RABBITMQ_QUEUE = os.getenv("RABBITMQ_QUEUE")
JOB_TTL_SECONDS = int(os.getenv("JOB_TTL_SECONDS"))