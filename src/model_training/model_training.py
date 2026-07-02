import os
from fastapi import Path
import pandas as pd
import joblib
from sqlalchemy import create_engine
from dotenv import load_dotenv
boto3

from xgboost import XGBRegressor
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from src.Preprocess.preprocessing import prepare_datetime_and_sort, create_forecasting_target,feature_engineering
from src.config import MODEL_DIR,MODEL_PATH,DB_CONN

load_dotenv()


def get_data():
    try:
        engine = create_engine(DB_CONN)
        query = "SELECT * FROM processed_weather"
        df = pd.read_sql(query, engine)
        return df
    except Exception as e:
        raise RuntimeError(
            f"Failed to read processed data"
        )



def split_train_test(df):
  train_df = df[df["target_time"]<"2019-01-01"].copy()

  test_df = df[
      (df["target_time"]>="2019-01-01") &
      (df["target_time"]<"2020-01-01")
  ].copy()

  print("Train rows before feature engineering:", len(train_df))
  print("Test rows before feature engineering:", len(test_df))

  return train_df, test_df

def prepare_features_and_target(train_df, test_df):
    """
    Drop unused columns and separate X and y.
    """

    train_df = train_df.sort_values("target_time").reset_index(drop=True)
    test_df = test_df.sort_values("target_time").reset_index(drop=True)

    columns_to_drop = [
        "id",
        "target",
        "target_time",
        "temperature_2m",
        "time",
    ]

    X_train = train_df.drop(columns=columns_to_drop, errors="ignore")
    y_train = train_df["target"]

    X_test = test_df.drop(columns=columns_to_drop, errors="ignore")
    y_test = test_df["target"]

    print("Features:", X_train.columns.tolist())
    print("Train shape:", X_train.shape)
    print("Test shape:", X_test.shape)

    return X_train, X_test, y_train, y_test

def upload_model_to_s3(local_model_path: str):
    bucket = os.getenv("MODEL_S3_BUCKET")
    key = os.getenv("MODEL_S3_KEY")

    if not bucket or not key:
        raise ValueError("MODEL_S3_BUCKET and MODEL_S3_KEY must be set in .env")

    s3 = boto3.client("s3")

    s3.upload_file(
        Filename=local_model_path,
        Bucket=bucket,
        Key=key
    )

    print(f"Model uploaded to s3://{bucket}/{key}")


def model_training():

    df = get_data()
    df = prepare_datetime_and_sort(df)
    df = create_forecasting_target(df)

    train_df, test_df = split_train_test(df)

    train_df = feature_engineering(train_df)
    test_df = feature_engineering(test_df)

    X_train, X_test, y_train, y_test = prepare_features_and_target(
        train_df,
        test_df
    )


    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), ['city'])
        ],
        remainder="passthrough"
    )

    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", XGBRegressor(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,   
            objective="reg:squarederror",
            eval_metric="rmse",     
            random_state=42,
          
        ))
    ])

    model.fit(X_train, y_train)

    try:
        Path("model").mkdir(parents=True, exist_ok=True)
        local_model_path = "model/xgboost_model.joblib"
        joblib.dump(model, local_model_path)
        upload_model_to_s3(local_model_path)

    except Exception as e:
        raise RuntimeError(
            f"failed to save model {str(e)}"
        )


    for name, X, y_true in [
        ("Train", X_train, y_train),
        ("Test", X_test, y_test)
    ]:
        y_pred = model.predict(X)

        rmse = root_mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        print(f"{name} RMSE: {rmse:.4f}")
        print(f"{name} MAE : {mae:.4f}")
        print(f"{name} R²  : {r2:.4f}")
        print("-" * 30)
