"""Centralized paths and configuration settings for the Energy Monitoring Application."""
from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
SRC_DIR = BASE_DIR / "src"
UI_DIR = BASE_DIR / "ui"

# Ensure data directory exists
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Ollama / Local LLM settings
OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_OLLAMA_MODEL = "llama3.2:3b"

# Energy Monitoring Default Parameters
RESAMPLE_FREQUENCY = "15min"  # Common industrial interval
PEAK_HOUR_START = 9
PEAK_HOUR_END = 18
OUTAGE_THRESHOLD_KW = 0.05
ANOMALY_ZSCORE_THRESHOLD = 3.0

