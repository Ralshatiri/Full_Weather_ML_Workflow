import pandas as pd

HORIZON = 24 * 30

def clean_data(df):
    """
    Remove duplicate rows and rows missing temperature_2m.
    """
    df = df.copy()

    df = df.drop_duplicates()
    df = df.dropna(subset=["temperature_2m"])

    return df


def drop_processed_columns(df):
    """
    Drop unnecessary columns while building the processed_weather table.
    """
    df = df.copy()

    columns_to_drop = [
        "id",
        "dew_point_2m",
        "pressure_msl",
        "apparent_temperature",
        "rain",
        "cloud_cover_high",
        "cloud_cover_low",
        "cloud_cover_mid",
        "latitude",
        "longitude"
    ]

    df = df.drop(columns=columns_to_drop, errors="ignore")

    print("Dropped unnecessary processed columns!")

    return df


def prepare_datetime_and_sort(df):
    """
    Convert time column to datetime and sort by city/time.

    Sorting is required before creating:
    - future target
    - lag features
    - rolling features
    """
    df = df.copy()

    df["time"] = pd.to_datetime(df["time"])

    df = df.sort_values(["city", "time"]).reset_index(drop=True)

    return df


def create_forecasting_target(df, horizon=HORIZON):
    """
    Create the forecasting target.

    target:
        temperature_2m 24 hours in the future

    target_time:
        time 24 hours in the future

    The shift is done inside each city separately to avoid mixing cities.
    """
    df = df.copy()

    original_rows = len(df)

    df["target"] = (
        df.groupby("city")["temperature_2m"]
        .shift(-horizon)
    )

    df["target_time"] = (
        df.groupby("city")["time"]
        .shift(-horizon)
    )

    rows_before_drop = len(df)

    df = df.dropna(subset=["target", "target_time"]).reset_index(drop=True)

    rows_after_drop = len(df)

    print("Original rows:", original_rows)
    print("Rows before target drop:", rows_before_drop)
    print("Rows after target drop:", rows_after_drop)
    print("Rows removed because of missing target:", rows_before_drop - rows_after_drop)
    print("Number of cities:", df["city"].nunique())

    return df


def feature_engineering(df):
    df = df.copy()

    df = df.sort_values(["city", "time"]).reset_index(drop=True)

    # Calendar features from current time t
    df["month"] = df["time"].dt.month
    df["day"] = df["time"].dt.day
    df["hour"] = df["time"].dt.hour

    # Temperature 720 hours before current time t
    df["temperature_lag_24"] = (
        df.groupby("city")["temperature_2m"]
        .shift(24)
    )

    # Rolling mean of previous 720 hours
    # shift(1) means current row temperature is not included
    df["temperature_rolling_mean_24"] = (
        df.groupby("city")["temperature_2m"]
        .transform(lambda x: x.shift(1).rolling(24, min_periods=24).mean())
    )

    # Rolling standard deviation of previous 24 hours
    df["temperature_rolling_std_24"] = (
        df.groupby("city")["temperature_2m"]
        .transform(lambda x: x.shift(1).rolling(24, min_periods=24).std())
    )

    df["temperature_lag_720"] = (
        df.groupby("city")["temperature_2m"].shift(720)
    )
    # Drop rows that do not have enough past history
    df = df.dropna(subset=[
        "temperature_lag_24",
        "temperature_rolling_mean_24",
        "temperature_rolling_std_24",
        "temperature_lag_720"
    ]).reset_index(drop=True)

    return df




def preprocess_weather(df):
    """
    Preprocessing used when building the processed_weather table.
    """
    df = clean_data(df)
    df = drop_processed_columns(df)

    return df