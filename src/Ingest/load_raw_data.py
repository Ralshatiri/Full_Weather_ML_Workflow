import os
from dotenv import load_dotenv
import pandas as pd
from sqlalchemy import create_engine, text
from src.config import CSV_PATH , DB_CONN

load_dotenv()




def ingest():

    try:
        print('Reading data from csv.........')
        data = pd.read_csv(CSV_PATH)
        print(f"Loaded {len(data)} rows")

        engine = create_engine(DB_CONN)

        with engine.begin() as conn:
            conn.execute(text("TRUNCATE TABLE raw_weather RESTART IDENTITY;"))

        data["time"] = pd.to_datetime(data["time"])
        print("Inserting data into raw_weather...")

        data.to_sql(
            name="raw_weather",
            con=engine,
            if_exists="append",
            index=False
        )
        print("Ingestion completed successfully.")

    except Exception as e:
        print(f"Error handling ingestion {str(e)}")

if __name__=="__main__":
    ingest()