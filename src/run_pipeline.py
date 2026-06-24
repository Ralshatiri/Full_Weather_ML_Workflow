from src.Ingest.load_raw_data import ingest
from src.Preprocess.build_processed_weather import build_processed_data
from src.model_training.model_training import model_training

def run_pipeline():
    ingest()
    build_processed_data()
    model_training()

if __name__ == "__main__":
    run_pipeline()
