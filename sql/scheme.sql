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
    month INT,
    day INT,
    hour INT,
    relative_humidity_2m FLOAT,
    dew_point_2m FLOAT,
    precipitation FLOAT,
    weather_code INT,
    pressure_msl FLOAT,
    surface_pressure FLOAT,
    cloud_cover FLOAT,
    wind_speed_10m FLOAT,
    wind_direction_10m FLOAT,
    wind_gusts_10m FLOAT,
    city VARCHAR(100),
    latitude FLOAT,
    longitude FLOAT
);

