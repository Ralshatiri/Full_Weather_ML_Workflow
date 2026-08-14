-- =====================
-- RAW TABLE
-- =====================
CREATE TABLE IF NOT EXISTS raw_weather (
    id BIGSERIAL PRIMARY KEY,
    time DATE NOT NULL,
    weather_code INTEGER,
    temperature_2m_mean DOUBLE PRECISION NOT NULL,
    temperature_2m_max DOUBLE PRECISION,
    temperature_2m_min DOUBLE PRECISION,
    apparent_temperature_mean DOUBLE PRECISION,
    apparent_temperature_max DOUBLE PRECISION,
    apparent_temperature_min DOUBLE PRECISION,
    precipitation_sum DOUBLE PRECISION,
    rain_sum DOUBLE PRECISION,
    precipitation_hours DOUBLE PRECISION,
    wind_speed_10m_mean DOUBLE PRECISION,
    wind_speed_10m_max DOUBLE PRECISION,
    wind_gusts_10m_mean DOUBLE PRECISION,
    wind_gusts_10m_max DOUBLE PRECISION,
    wind_direction_10m_dominant INTEGER,
    relative_humidity_2m_mean DOUBLE PRECISION,
    relative_humidity_2m_max DOUBLE PRECISION,
    relative_humidity_2m_min DOUBLE PRECISION,
    dew_point_2m_mean DOUBLE PRECISION,
    pressure_msl_mean DOUBLE PRECISION,
    surface_pressure_mean DOUBLE PRECISION,
    cloud_cover_mean DOUBLE PRECISION,
    city VARCHAR(100) NOT NULL,
    latitude DOUBLE PRECISION,
    longitude DOUBLE PRECISION

);



-- =====================
-- PROCESSED TABLE
-- =====================
CREATE TABLE IF NOT EXISTS processed_weather (
    id BIGSERIAL PRIMARY KEY,
    time DATE NOT NULL,
    city VARCHAR(100) NOT NULL,
    temperature_2m_mean DOUBLE PRECISION NOT NULL,
    doy_sin DOUBLE PRECISION NOT NULL,
    doy_cos DOUBLE PRECISION NOT NULL,
    latitude DOUBLE PRECISION NOT NULL,
    longitude DOUBLE PRECISION NOT NULL

);


-- =====================
-- PREDICTIONS TABLE
-- =====================

CREATE TABLE IF NOT EXISTS predictions (
    id BIGSERIAL PRIMARY KEY,
    job_id UUID NOT NULL,
    city VARCHAR(100) NOT NULL,
    forecast_origin DATE NOT NULL,
    forecast_date DATE NOT NULL,
    forecast_step INTEGER NOT NULL CHECK (forecast_step > 0),
    predicted_temperature DOUBLE PRECISION NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP

);
