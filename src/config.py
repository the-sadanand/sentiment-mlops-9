from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "imdb.csv"

MODEL_NAME = "sentiment_model"
EXPERIMENT_NAME = "sentiment-analysis"
