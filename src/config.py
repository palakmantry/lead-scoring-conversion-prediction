from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
PREDICTIONS_DIR = PROJECT_ROOT / "predictions"

DATA_URL = "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip"
RAW_DATA_FILENAME = "bank_marketing_full.csv"
RAW_DATA_PATH = RAW_DIR / RAW_DATA_FILENAME

TARGET_COL = "y"
LEAKAGE_COLUMNS = ["duration"]

TRAIN_FRACTION = 0.70
VALID_FRACTION = 0.15
TEST_FRACTION = 0.15

RANDOM_STATE = 42

# Business assumptions for thresholding / contact policy
CONTACT_COST = 80
CONVERSION_VALUE = 4500

CONTACT_SHARE_GRID = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]