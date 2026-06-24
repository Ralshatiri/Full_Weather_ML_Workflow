import os
from dotenv import load_dotenv
import pandas as pd
from sqlalchemy import create_engine

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


load_dotenv()


DB_CONN = "postgresql+psycopg2://{}:{}@{}:{}/{}".format(
    os.getenv("POSTGRES_USER"),
    os.getenv("POSTGRES_PASSWORD"),
    os.getenv("DB_HOST"),
    os.getenv("DB_PORT"),
    os.getenv("POSTGRES_DB")
)

def get_data():
    engine = create_engine(DB_CONN)

    query ="""
        SELECT *
        FROM processed_weather
"""
    df = pd.read_sql(query,engine)
    return df

def split_data():

    df = get_data()
    df = df.drop(columns="id")

    y = df['temperature_2m'] 
    X = df.drop("temperature_2m",axis=1)


    X_train,X_test,y_train,y_test = train_test_split(
        X,y,
        test_size=0.2,
        random_state=42
    )
    return X_train,X_test,y_train,y_test


def model_training():

    X_train,X_test,y_train,y_test = split_data()
    categorical_features = ["city"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat",OneHotEncoder(handle_unknown="ignore"),categorical_features)
        ],
        remainder="passthrough"
    )


    model =  Pipeline(
        steps=[
            ("preprocessor",preprocessor),(
                "regressor", RandomForestRegressor(
                n_estimators=100,
                max_features="sqrt",
                random_state=42
                )
            )
        ]
    )

    model.fit(X_train,y_train)

    y_pred = model.predict(X_test)
    test_rmse = root_mean_squared_error(y_test,y_pred)
    test_r2  = r2_score(y_test,y_pred)

    y_pred_train = model.predict(X_train)
    train_rmse = root_mean_squared_error(y_train, y_pred_train)
    train_r2 = r2_score(y_train, y_pred_train)

    print(f"Train RMSE: {train_rmse:.4f}")
    print(f"Test RMSE: {test_rmse:.4f}")
    print(f"Train R2: {train_r2:.4f}")
    print(f"Test R2: {test_r2:.4f}")