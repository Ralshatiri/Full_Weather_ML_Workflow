
import numpy as np
import pandas as pd


TARGET_COLUMN = "temperature_2m_mean"
SEQUENCE_LENGTH = 30

REQUIRED_COLUMNS = [
    "time",
    "city",
    TARGET_COLUMN,
    "latitude",
    "longitude",
]

PROCESSED_COLUMNS = [
    "time",
    "city",
    TARGET_COLUMN,
    "doy_sin",
    "doy_cos",
    "latitude",
    "longitude",
]


def validate_required_columns(df: pd.DataFrame) -> None:
    """
    Verify that the raw data contains the columns required by the LSTM.

    The univariate recursive model requires only the observation date,
    city, and daily mean temperature..

    """
    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Raw weather data is missing columns: {missing_columns}"
        )


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and chronologically arrange daily weather observations.

    This function:
    - Converts ``time`` to datetime.
    - Converts daily mean temperature to numeric values.
    - Cleans city names.
    - Removes rows missing essential values.
    - Removes duplicate city-date observations.
    - Sorts observations by city and time.

    If multiple rows have the same city and date, the last row is retained.

    Parameters
    ----------
    df : pandas.DataFrame
        Raw weather data retrieved from PostgreSQL.

    Returns
    -------
    pandas.DataFrame
        Cleaned and chronologically sorted weather data.
    """
    validate_required_columns(df)

    cleaned_df = df.copy()

    cleaned_df["time"] = pd.to_datetime(
        cleaned_df["time"],
        errors="coerce",
    )

    cleaned_df["latitude"] = pd.to_numeric(
    cleaned_df["latitude"],
    errors="coerce",
)

    cleaned_df["longitude"] = pd.to_numeric(
        cleaned_df["longitude"],
        errors="coerce",
    )

    cleaned_df[TARGET_COLUMN] = pd.to_numeric(
        cleaned_df[TARGET_COLUMN],
        errors="coerce",
    )

    cleaned_df["city"] = (
        cleaned_df["city"]
        .astype("string")
        .str.strip()
        .replace("", pd.NA)
    )

    cleaned_df = cleaned_df.dropna(
        subset=REQUIRED_COLUMNS
    )

    cleaned_df = cleaned_df.drop_duplicates(
        subset=["city", "time"],
        keep="last",
    )

    cleaned_df = cleaned_df.sort_values(
        by=["city", "time"]
    ).reset_index(drop=True)

    if cleaned_df.empty:
        raise ValueError(
            "No valid rows remain after cleaning raw weather data."
        )

    return cleaned_df


def validate_daily_frequency(df: pd.DataFrame) -> None:
    """
    Verify that observations are consecutive within each city.

    An LSTM sequence assumes that adjacent rows represent adjacent days.
    For example, a 30-row sequence must represent 30 consecutive days.
    Missing dates could otherwise create an incorrect sequence.
    """
    gaps_by_city = {}

    for city, city_df in df.groupby("city", sort=False):
        dates = city_df["time"].sort_values()
        differences = dates.diff().dropna()

        invalid_intervals = differences[
            differences != pd.Timedelta(days=1)
        ]

        if not invalid_intervals.empty:
            gaps_by_city[str(city)] = len(invalid_intervals)

    if gaps_by_city:
        raise ValueError(
            "Non-consecutive daily observations were detected. "
            f"Invalid intervals by city: {gaps_by_city}"
        )


def validate_city_history(
    df: pd.DataFrame,
    sequence_length: int = SEQUENCE_LENGTH,
) -> None:
    """
    Confirm that every city can produce at least one training sequence.

    A ``t+1`` recursive model with a 30-day window needs at least 31 rows:
    30 input observations and one next-day target.

    Parameters
    ----------
    df : pandas.DataFrame
        Cleaned daily weather data.
    sequence_length : int, default=30
        Number of historical days used in one model input window.
    """
    if sequence_length <= 0:
        raise ValueError(
            "sequence_length must be greater than zero."
        )

    minimum_rows = sequence_length + 1
    rows_per_city = df.groupby("city").size()

    insufficient_cities = rows_per_city[
        rows_per_city < minimum_rows
    ].to_dict()

    if insufficient_cities:
        raise ValueError(
            f"Each city needs at least {minimum_rows} rows. "
            f"Insufficient city histories: {insufficient_cities}"
        )


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add cyclical day-of-year features.

    Day of year is represented with sine and cosine values so the model
    understands the cyclical relationship between December and January.
    """
    if "time" not in df.columns:
        raise ValueError(
            "Cannot create calendar features without 'time'."
        )

    if not pd.api.types.is_datetime64_any_dtype(df["time"]):
        raise ValueError(
            "The 'time' column must be datetime before "
            "creating calendar features."
        )

    featured_df = df.copy()

    day_of_year = featured_df["time"].dt.dayofyear

    featured_df["doy_sin"] = np.sin(
        2 * np.pi * day_of_year / 365.25
    )

    featured_df["doy_cos"] = np.cos(
        2 * np.pi * day_of_year / 365.25
    )

    return featured_df


def drop_processed_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove raw columns that are not used by the recursive LSTM.

    The raw table preserves all downloaded weather variables. The univariate
    recursive model uses only daily mean temperature, cyclical calendar
    features, and location coordinates.

    City remains as a text column in ``processed_weather``. It is converted
    to one-hot columns later during training.
    """
    missing_columns = [
        column
        for column in PROCESSED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Cannot build processed data. Missing columns: "
            f"{missing_columns}"
        )

    processed_df = df[PROCESSED_COLUMNS].copy()

    print(
        "Removed raw columns that are not used by "
        "the univariate recursive LSTM."
    )

    return processed_df


def preprocess_weather(
    df: pd.DataFrame,
    sequence_length: int = SEQUENCE_LENGTH,
) -> pd.DataFrame:
    """

    The function:
    1. Cleans raw daily weather observations.
    2. Confirms that dates are consecutive within each city.
    3. Confirms that every city has enough history for LSTM sequences.
    4. Creates cyclical day-of-year features.
    5. Removes columns not used by the recursive model.

    """
    processed_df = clean_data(df)

    validate_daily_frequency(processed_df)
    validate_city_history(
        processed_df,
        sequence_length=sequence_length,
    )

    processed_df = add_calendar_features(processed_df)
    processed_df = drop_processed_columns(processed_df)

    print("Weather preprocessing completed successfully.")
    print(f"Processed rows: {len(processed_df)}")
    print(f"Number of cities: {processed_df['city'].nunique()}")
    print(
        "Date range:",
        processed_df["time"].min().date(),
        "to",
        processed_df["time"].max().date(),
    )

    return processed_df