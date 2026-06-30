from src.Ingest.load_raw_data import ingest
from src.Preprocess.build_processed_weather import build_processed_data
from src.model_training.model_training import model_training

def run_pipeline():
    try:
        ingest()
        build_processed_data()
        model_training()
    except Exception as e:
        print(f"Pipeline failed: {e}")
        raise

if __name__ == "__main__":
    run_pipeline()
