# Weather Temperature Prediction ML Project

## Project Overview

This project builds a complete machine learning workflow to predict temperature using weather observations from multiple cities in Saudi Arabia.

The workflow includes data ingestion, raw and processed database tables, preprocessing, model training, evaluation, and storing predictions in the database. The project is containerized using Docker to make the environment reproducible.

---

## Project Objective

The main goal is to predict the temperature using weather-related features such as city, humidity, wind speed, wind gusts, rainfall, and time-based variables.

This project demonstrates a full ML pipeline, including:

- Loading raw weather data
- Storing raw data in PostgreSQL
- Cleaning and processing the data
- Training a ranodm forest regressor model
- Evaluating model performance
- Saving prediction results
- Running the full workflow using Docker

---

## Repository Structure

```text
data/
└── raw/
    └── raw weather dataset

sql/
└── database schema and table creation scripts

src/
└── source code for data ingestion, preprocessing, and model training

.gitignore
Dockerfile
docker-compose.yml
requirements.txt
README.md
