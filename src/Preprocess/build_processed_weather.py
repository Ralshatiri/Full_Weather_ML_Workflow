"""
Build the processed daily weather table for the recursive LSTM model.

The module retrieves raw daily weather data from PostgreSQL, applies the
 preprocessing , clears the previous processed data,
and inserts the newly processed observations.
"""

import pandas as pd
from sqlalchemy import create_engine, text

from src.Preprocess.preprocessing import preprocess_weather
from src.config import DB_CONN


def build_processed_data() -> pd.DataFrame:
    """
    Retrieve, preprocess, and store daily weather data.

    The function reads observations from ``raw_weather``, processes them for
    the univariate recursive LSTM, clears the previous contents of
    ``processed_weather``, and inserts the newly processed rows.
    """
    engine = create_engine(DB_CONN)

    try:
        raw_df = pd.read_sql(
            """
            SELECT *
            FROM raw_weather
            ORDER BY city, time
            """,
            engine,
        )

        if raw_df.empty:
            raise ValueError(
                "The raw_weather table is empty. "
                "Ingest raw data before running preprocessing."
            )

        print(f"Retrieved {len(raw_df)} rows from raw_weather.")

        processed_df = preprocess_weather(raw_df)

        if processed_df.empty:
            raise ValueError(
                "Preprocessing returned no data. "
                "The processed_weather table was not cleared."
            )

        # Clear old processed data before inserting the new processed data.
        with engine.begin() as connection:
            connection.execute(
                text(
                    "TRUNCATE TABLE processed_weather "
                    "RESTART IDENTITY;"
                )
            )

        print("Cleared old processed_weather data.")

        processed_df.to_sql(
            name="processed_weather",
            con=engine,
            if_exists="append",
            index=False,
        )

        print(
            f"Inserted {len(processed_df)} rows "
            "into processed_weather."
        )

        return processed_df

    except Exception as error:
        raise RuntimeError(
            f"Failed to build processed weather data: {error}"
        ) from error

    finally:
        engine.dispose()


if __name__ == "__main__":
    build_processed_data()