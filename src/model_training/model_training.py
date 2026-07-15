"""
Train the univariate LSTM used for recursive temperature forecasting.

The model receives 30 days of:
- Scaled daily mean temperature.
- Scaled cyclical day-of-year features.
- city one-hot features.
"""
from pathlib import Path

import boto3
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sqlalchemy import create_engine
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.preprocessing import  StandardScaler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Dropout, Input, LSTM
from tensorflow.keras.models import Sequential

from src.config import (
    DB_CONN,
    MODEL_PATH,
    MODEL_METADATA_PATH,
    S3_BUCKET,
    MODEL_S3_KEY,
    MODEL_METADATA_S3_KEY,
    UPLOAD_ARTIFACTS_TO_S3,
)



RANDOM_SEED = 42

TARGET_COLUMN = "temperature_2m_mean"
SEQUENCE_LENGTH = 30

TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15

NUMERIC_COLUMNS = [
    TARGET_COLUMN,
    "doy_sin",
    "doy_cos",
    "latitude",
    "longitude",
]

EPOCHS = 60
BATCH_SIZE = 64
EARLY_STOPPING_PATIENCE = 8


np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


def get_data() -> pd.DataFrame:
    """
    Retrieve  daily data from ``processed_weather``.

    The processed table is expected to contain the observation date, city,
    daily mean temperature, and cyclical day-of-year features.

    """
    required_columns = [
        "time",
        "city",
        TARGET_COLUMN,
        "doy_sin",
        "doy_cos",
        "latitude",
        "longitude",
    ]

    engine = create_engine(DB_CONN)

    try:
        query = """
            SELECT
                time,
                city,
                temperature_2m_mean,
                doy_sin,
                doy_cos,
                latitude,
                longitude
            FROM processed_weather
            ORDER BY city, time
        """

        df = pd.read_sql(query, engine)

    except Exception as error:
        raise RuntimeError(
            f"Failed to read processed weather data: {error}"
        ) from error

    finally:
        engine.dispose()

    if df.empty:
        raise ValueError(
            "The processed_weather table is empty."
        )

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Processed data is missing columns: {missing_columns}"
        )

    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values(["city", "time"]).reset_index(drop=True)

    print(f"Loaded {len(df)} processed rows.")
    print(f"Cities: {df['city'].nunique()}")
    print(
        "Date range:",
        df["time"].min().date(),
        "to",
        df["time"].max().date(),
    )

    return df


def split_train_validation_test(
    df: pd.DataFrame,
    sequence_length: int = SEQUENCE_LENGTH,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split every city chronologically into train, validation, and test periods.

    Each city contributes its earliest 70% of observations to training, the
    following 15% to validation, and the final 15% to testing.



    Parameters
    ----------
    df : pandas.DataFrame
        Processed daily observations sorted by city and time.
    sequence_length : int, default=30
        Number of historical days used in each input sequence.

    Returns
    -------
    tuple[pandas.DataFrame, pandas.DataFrame, pandas.DataFrame]
        Training, validation, and testing DataFrames.
    """
    train_parts = []
    validation_parts = []
    test_parts = []

    for city, city_df in df.groupby("city", sort=False):
        city_df = (
            city_df
            .sort_values("time")
            .reset_index(drop=True)
        )

        number_of_rows = len(city_df)

        train_end = int(number_of_rows * TRAIN_RATIO)
        validation_end = int(
            number_of_rows
            * (TRAIN_RATIO + VALIDATION_RATIO)
        )

        if train_end <= sequence_length:
            raise ValueError(
                f"City '{city}' does not have enough training rows."
            )

        if validation_end - train_end < 1:
            raise ValueError(
                f"City '{city}' does not have enough validation rows."
            )

        if number_of_rows - validation_end < 1:
            raise ValueError(
                f"City '{city}' does not have enough test rows."
            )

        train_part = city_df.iloc[:train_end].copy()

        validation_part = city_df.iloc[
            train_end - sequence_length:validation_end
        ].copy()

        test_part = city_df.iloc[
            validation_end - sequence_length:
        ].copy()

        # These columns identify which rows are actual prediction targets.
        # The overlapping history rows must not become validation/test labels.
        train_part["is_target_period"] = True

        validation_part["is_target_period"] = (
            validation_part["time"]
            >= city_df.iloc[train_end]["time"]
        )

        test_part["is_target_period"] = (
            test_part["time"]
            >= city_df.iloc[validation_end]["time"]
        )

        train_parts.append(train_part)
        validation_parts.append(validation_part)
        test_parts.append(test_part)

    train_df = pd.concat(
        train_parts,
        ignore_index=True,
    )

    validation_df = pd.concat(
        validation_parts,
        ignore_index=True,
    )

    test_df = pd.concat(
        test_parts,
        ignore_index=True,
    )

    print(f"Training rows: {len(train_df)}")
    print(f"Validation rows including history: {len(validation_df)}")
    print(f"Test rows including history: {len(test_df)}")

    return train_df, validation_df, test_df




def fit_numeric_scaler(
    train_df: pd.DataFrame,
) -> StandardScaler:
    """
    Fit a standard scaler on numeric training features only.
    """
    scaler = StandardScaler()
    scaler.fit(train_df[NUMERIC_COLUMNS])

    return scaler


def transform_features(
    df: pd.DataFrame,
    scaler: StandardScaler,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Scale temperature, calendar features, and coordinates.

    City remains unmodified because it is used only to separate each
    location's time series.
    """
    transformed_df = df.copy()

    transformed_df[NUMERIC_COLUMNS] = scaler.transform(
        df[NUMERIC_COLUMNS]
    )

    feature_columns = NUMERIC_COLUMNS.copy()

    return transformed_df, feature_columns



def create_sequences(
    df: pd.DataFrame,
    feature_columns: list[str],
    sequence_length: int = SEQUENCE_LENGTH,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Create 30-day input windows and next-day temperature targets.

    For each city, a sequence contains ``sequence_length`` consecutive rows.
    Its target is the temperature on the immediately following day, making
    this a dedicated ``t+1`` model suitable for recursive forecasting.

    Targets are returned in scaled form because the model is trained to
    predict scaled temperature. Only rows marked as belonging to the current
    split's target period are used as labels.
    """
    all_sequences = []
    all_targets = []

    for _, city_df in df.groupby("city", sort=False):
        city_df = (
            city_df
            .sort_values("time")
            .reset_index(drop=True)
        )

        feature_values = city_df[
            feature_columns
        ].to_numpy(dtype=np.float32)

        target_values = city_df[
            TARGET_COLUMN
        ].to_numpy(dtype=np.float32)

        target_period = city_df[
            "is_target_period"
        ].to_numpy(dtype=bool)

        for target_index in range(
            sequence_length,
            len(city_df),
        ):
            if not target_period[target_index]:
                continue

            sequence_start = target_index - sequence_length

            all_sequences.append(
                feature_values[
                    sequence_start:target_index
                ]
            )

            all_targets.append(
                target_values[target_index]
            )

    if not all_sequences:
        raise ValueError(
            "No LSTM sequences could be created."
        )

    X = np.asarray(
        all_sequences,
        dtype=np.float32,
    )

    y = np.asarray(
        all_targets,
        dtype=np.float32,
    )

    return X, y


def build_lstm_model(
    sequence_length: int,
    number_of_features: int,
) -> Sequential:
    """
    Build and compile the recursive base LSTM.

    Two  LSTM layers learn patterns from the preceding 30 days. The
    first layer returns a sequence for the second LSTM layer. The final dense
    layer predicts one scaled next-day temperature.
    """
    model = Sequential(
        [
            Input(
                shape=(
                    sequence_length,
                    number_of_features,
                )
            ),
            LSTM(
                64,
                return_sequences=True,
            ),
            Dropout(0.2),
            LSTM(32),
            Dropout(0.2),
            Dense(
                16,
                activation="relu",
            ),
            Dense(1),
        ]
    )

    model.compile(
        optimizer="adam",
        loss="mse",
        metrics=["mae"],
    )

    return model


def evaluate_model(
    model: Sequential,
    X: np.ndarray,
    y_scaled: np.ndarray,
    scaler: StandardScaler,
    split_name: str,
) -> dict[str, float]:
    """
    Evaluate next-day predictions in real degrees Celsius.

    The model produces scaled temperature values. Predictions and labels are
    converted back to degrees Celsius before MAE, RMSE, and R² are calculated.

    """
    target_index = NUMERIC_COLUMNS.index(
        TARGET_COLUMN
    )

    temperature_mean = scaler.mean_[target_index]
    temperature_scale = scaler.scale_[target_index]

    predictions_scaled = (
        model.predict(X, verbose=0)
        .reshape(-1)
    )

    predictions_celsius = (
        predictions_scaled * temperature_scale
        + temperature_mean
    )

    actual_celsius = (
        y_scaled * temperature_scale
        + temperature_mean
    )

    mae = mean_absolute_error(
        actual_celsius,
        predictions_celsius,
    )

    rmse = np.sqrt(
        mean_squared_error(
            actual_celsius,
            predictions_celsius,
        )
    )

    r2 = r2_score(
        actual_celsius,
        predictions_celsius,
    )

    metrics = {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "R2": float(r2),
    }

    print(f"{split_name} RMSE: {rmse:.4f}")
    print(f"{split_name} MAE : {mae:.4f}")
    print(f"{split_name} R²  : {r2:.4f}")
    print("-" * 30)

    return metrics


def save_artifacts(
    model: Sequential,
    scaler: StandardScaler,
    feature_columns: list[str],
) -> tuple[Path, Path]:
    """
    Save the LSTM and its required preprocessing metadata.

    The Keras model and preprocessing objects are saved separately. Both are
    required later because recursive inference must reproduce the same scaling,
    city encoding, and feature order used during training.
    """
    try:
        model_path = Path(MODEL_PATH)
        metadata_path = Path(MODEL_METADATA_PATH)

        model_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        metadata_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        model.save(model_path)

        target_index = NUMERIC_COLUMNS.index(
            TARGET_COLUMN
        )

        metadata = {
            "scaler": scaler,
            "feature_columns": feature_columns,
            "numeric_columns": NUMERIC_COLUMNS,
            "target_column": TARGET_COLUMN,
            "sequence_length": SEQUENCE_LENGTH,
            "temperature_mean": float(
                scaler.mean_[target_index]
            ),
            "temperature_scale": float(
                scaler.scale_[target_index]
            ),
        }

        joblib.dump(
            metadata,
            metadata_path,
        )

        print(f"Saved LSTM model to {model_path}")
        print(f"Saved metadata to {metadata_path}")

        return model_path, metadata_path

    except Exception as error:
        raise RuntimeError(
            f"Failed to save model artifacts: {error}"
        ) from error


def upload_artifact_to_s3(
    local_path: Path,
    s3_key: str,
) -> None:
    """Upload one model artifact to the configured S3 bucket."""
    if not S3_BUCKET:
        raise ValueError(
            "MODEL_S3_BUCKET must be set in the environment."
        )

    try:
        s3_client = boto3.client("s3")

        s3_client.upload_file(
            Filename=str(local_path),
            Bucket=S3_BUCKET,
            Key=s3_key,
        )

        print(
            f"Uploaded {local_path.name} "
            f"to s3://{S3_BUCKET}/{s3_key}"
        )

    except Exception as error:
        raise RuntimeError(
            f"Failed to upload {local_path} to S3: {error}"
        ) from error


def model_training() -> dict[str, dict[str, float]]:
    """
    Run the complete recursive LSTM training workflow.

    This function:
    1. Loads unscaled processed data.
    2. Splits every city chronologically.
    3. Fits the city encoder and numeric scaler on training data only.
    4. Transforms every split.
    5. Creates 30-day-to-next-day sequences.
    6. Trains one dedicated t+1 LSTM.
    7. Evaluates one-step validation and test performance.
    8. Saves the model, scaler, encoder, and feature configuration.
    9. Uploads both artifacts to S3.
    """
    df = get_data()

    train_df, validation_df, test_df = (
        split_train_validation_test(df)
    )

    scaler = fit_numeric_scaler(train_df)

    transformed_train, feature_columns = (
        transform_features(
            train_df,
            scaler,
        )
    )

    transformed_validation, validation_features = (
        transform_features(
            validation_df,
            scaler,
        )
    )

    transformed_test, test_features = (
        transform_features(
            test_df,
            scaler,
        )
    )

    if feature_columns != validation_features:
        raise ValueError(
            "Training and validation feature orders do not match."
        )

    if feature_columns != test_features:
        raise ValueError(
            "Training and test feature orders do not match."
        )

    X_train, y_train = create_sequences(
        transformed_train,
        feature_columns,
    )

    X_validation, y_validation = create_sequences(
        transformed_validation,
        feature_columns,
    )

    X_test, y_test = create_sequences(
        transformed_test,
        feature_columns,
    )

    print(f"X_train shape: {X_train.shape}")
    print(f"X_validation shape: {X_validation.shape}")
    print(f"X_test shape: {X_test.shape}")

    model = build_lstm_model(
        sequence_length=SEQUENCE_LENGTH,
        number_of_features=X_train.shape[2],
    )

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=EARLY_STOPPING_PATIENCE,
        restore_best_weights=True,
    )

    model.fit(
        X_train,
        y_train,
        validation_data=(
            X_validation,
            y_validation,
        ),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stopping],
        verbose=1,
        shuffle=False,
    )

    validation_metrics = evaluate_model(
        model,
        X_validation,
        y_validation,
        scaler,
        "Validation",
    )

    test_metrics = evaluate_model(
        model,
        X_test,
        y_test,
        scaler,
        "Test",
    )

    model_path, metadata_path = save_artifacts(
        model,
        scaler,
        feature_columns,
    )

    if UPLOAD_ARTIFACTS_TO_S3:
        upload_artifact_to_s3(
            model_path,
            MODEL_S3_KEY,
        )

        upload_artifact_to_s3(
            metadata_path,
            MODEL_METADATA_S3_KEY,
        )

        print("Model artifacts uploaded to S3.")

    else:
        print(
            "Local testing: model artifacts were saved locally "
            "and S3 upload was skipped."
        )

    return {
        "validation": validation_metrics,
        "test": test_metrics,
    }


if __name__ == "__main__":
    model_training()