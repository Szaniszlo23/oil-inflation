"""
utils.py - shared helpers used by every stage.

Paths, config loading, the API key, the latest raw snapshot, and unit
constants. Chart style gets added here when charts.py is written.
"""
import os
import re
from pathlib import Path

import yaml
from dotenv import load_dotenv

# --- Paths -------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config.yaml"
INTERVENTIONS_PATH = ROOT / "interventions.csv"
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "outputs" / "results"
FIGURES = ROOT / "outputs" / "figures"
TABLES = ROOT / "outputs" / "tables"

# --- Units -------------------------------------------------------------------
LITRES_PER_BARREL = 158.987294928  # 42 US gallons x 3.785411784 litres
LITRES_PER_KL = 1000               # Oil Bulletin prices and taxes are per 1000 litres


def ensure_dirs():
    """Create the data and output folders if they don't exist yet."""
    for path in (DATA_RAW, DATA_PROCESSED, RESULTS, FIGURES, TABLES):
        path.mkdir(parents=True, exist_ok=True)


def load_config():
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_secret(name):
    """Read a secret (e.g. FRED_API_KEY) from .env in the project root."""
    load_dotenv(ROOT / ".env")
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"{name} is missing - add it to {ROOT / '.env'}")
    return value


def latest_snapshot():
    """Newest complete raw snapshot (data/raw/YYYY-MM-DD), or None."""
    if not DATA_RAW.exists():
        return None
    dated = [p for p in DATA_RAW.iterdir()
             if p.is_dir() and re.fullmatch(r"\d{4}-\d{2}-\d{2}", p.name)]
    return max(dated, default=None)