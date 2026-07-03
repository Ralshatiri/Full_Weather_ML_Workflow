import os 
from dotenv import load_dotenv
import pandas as pd
from sqlalchemy import create_engine, text
from src.Preprocess.preprocessing import preprocess_weather
from src.config import DB_CONN

load_dotenv()



def build_processed_data():
    try:
        engine = create_engine(DB_CONN)

        raw_df = pd.read_sql("select * from raw_weather",engine)
        processed_df = preprocess_weather(raw_df)

 # Clear old processed data before inserting the new processed data
        with engine.begin() as conn:
            conn.execute(text("TRUNCATE TABLE processed_weather RESTART IDENTITY;"))

        print("Cleared old processed_weather data.")

        processed_df.to_sql(
            name="processed_weather",
            con=engine,
            if_exists="append",
            index=False
        )
        
        print(f"Inserted {len(processed_df)} rows into processed_weather")
    except Exception as e:
        raise RuntimeError(
            f"Failed to insert processed data {str(e)}"
        )
if __name__ == "__main__":
    build_processed_data()