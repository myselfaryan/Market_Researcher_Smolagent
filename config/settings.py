import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Base directory
BASE_DIR = Path(__file__).resolve().parent.parent

# API Keys (load from environment variables for security)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SERP_API_KEY = os.getenv("SERP_API_KEY")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")

# Data paths
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
REPORT_DIR = DATA_DIR / "reports"

# Ensure directories exist
for dir_path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, REPORT_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Web scraping settings
REQUEST_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
}
REQUEST_TIMEOUT = 30  # seconds
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds

# Analysis settings
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
SENTIMENT_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"
MIN_TOKEN_LENGTH = 4
MAX_DOCUMENTS = 100

# Reporting settings
DEFAULT_REPORT_FORMAT = "markdown"
CHART_DPI = 300
DEFAULT_CHART_STYLE = "seaborn-darkgrid"

# API settings
API_HOST = "0.0.0.0"
API_PORT = 8000
API_WORKERS = 4

# Logging settings
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"