-- =====================
-- RAW TABLE
-- =====================
CREATE TABLE IF NOT EXISTS raw_weather (
    id SERIAL PRIMARY KEY,
    time TIMESTAMP,
    temperature_2m FLOAT,
    relative_humidity_2m INT,
    dew_point_2m FLOAT,
    apparent_temperature FLOAT,
    precipitation FLOAT,
    rain FLOAT,
    weather_code INT,
    pressure_msl FLOAT,
    surface_pressure FLOAT,
    cloud_cover INT,
    cloud_cover_low INT,
    cloud_cover_mid INT,
    cloud_cover_high INT,
    wind_speed_10m FLOAT,
    wind_direction_10m INT,
    wind_gusts_10m FLOAT,
    city VARCHAR(100),
    latitude FLOAT,
    longitude FLOAT
);

-- =====================
-- PROCESSED TABLE
-- =====================
CREATE TABLE IF NOT EXISTS processed_weather (
    id SERIAL PRIMARY KEY,
    temperature_2m FLOAT,
    time TIMESTAMP,
    relative_humidity_2m FLOAT,
    precipitation FLOAT,
    weather_code INT,
    surface_pressure FLOAT,
    cloud_cover FLOAT,
    wind_speed_10m FLOAT,
    wind_direction_10m FLOAT,
    wind_gusts_10m FLOAT,
    city VARCHAR(100)

);

-- =====================
-- PREDICTIONS TABLE
-- =====================

CREATE TABLE IF NOT EXISTS predictions (
    id SERIAL PRIMARY KEY,

    city TEXT NOT NULL,
    month INTEGER,
    day INTEGER,
    hour INTEGER,
    relative_humidity_2m DOUBLE PRECISION,
    precipitation DOUBLE PRECISION,
    weather_code INTEGER,
    surface_pressure DOUBLE PRECISION,
    cloud_cover DOUBLE PRECISION,
    wind_speed_10m DOUBLE PRECISION,
    wind_direction_10m DOUBLE PRECISION,
    wind_gusts_10m DOUBLE PRECISION,
    temperature_lag_24 DOUBLE PRECISION,
    temperature_rolling_mean_24 DOUBLE PRECISION,
    temperature_rolling_std_24 DOUBLE PRECISION,

    -- Model output
    predicted_temperature DOUBLE PRECISION,

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
