import pandas as pd

target = "temperature_2m"

def clean_data(df):
    
    df = df.copy()
    df = df.drop_duplicates()
    df = df.dropna(subset=[target])

    return df

def add_time_features(df):
    
    df = df.copy()
    df["month"] = df["time"].dt.month
    df["day"] = df["time"].dt.day
    df["hour"] = df["time"].dt.hour

    return df

def drop_columns(df):

    df = df.copy()

    columns_to_drop = [
        "id",
        "time",
        "apparent_temperature",
        "rain",
        "cloud_cover_high",
        "cloud_cover_low",
        "cloud_cover_mid"
    ]

    df = df.drop(columns=columns_to_drop)

    return df

def preprocess_weather(df):
    df = clean_data(df)
    df = add_time_features(df)
    df = drop_columns(df)

    return df

