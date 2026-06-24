import os
from dotenv import load_dotenv
import pandas as pd
from sqlalchemy import create_engine

load_dotenv()



DB_CONN = "postgresql://{}:{}@{}:{}/{}".format(
    os.getenv("POSTGRES_USER"),
    os.getenv("POSTGRES_PASSWORD"),
    os.getenv("DB_HOST"),
    os.getenv("DB_PORT"),
    os.getenv("POSTGRES_DB")
)

def ingest():
    print('Reading data from csv.........')
    data = pd.read_csv("data/raw/saudi_weather_data.csv")
    print(f"Loaded {len(data)} rows")

    engine = create_engine(DB_CONN)

    data["time"] = pd.to_datetime(data["time"])
    print("Inserting data into raw_weather...")

    data.to_sql(
        name="raw_weather",
        con=engine,
        if_exists="append",
        index=False
    )
    print("Ingestion completed successfully.")

if __name__=="__main__":
    ingest()



