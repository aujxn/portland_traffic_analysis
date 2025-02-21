from pathlib import Path

PACKAGE_DIR = Path(__file__).parent
DATA_DIR = PACKAGE_DIR / "data"
PLOTS_DIR = PACKAGE_DIR / "plots"

DATA_DIR.mkdir(exist_ok=True)
PLOTS_DIR.mkdir(exist_ok=True)

DATA_SERVER_URL = "https://traffic.aujxn.dev/datasets/"
CHECKSUMS_URL = DATA_SERVER_URL + "checksums.txt"
DATA_FILE = DATA_DIR / "ATR_data.pq"
META_FILE = DATA_DIR / "ATR_metadata.csv"

DATA_FILES = [
    DATA_FILE,
    META_FILE,
]
