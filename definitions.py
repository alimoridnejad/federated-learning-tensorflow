from pathlib import Path

# project root directory
PROJECT_ROOT = Path(__file__).parent

# base directory (parent directory of PROJECT_ROOT)
BASE_DIR = PROJECT_ROOT.parent

# configuration file path
CONFIG_PATH = PROJECT_ROOT / "config.yaml"

# data directory
DATA_DIR = BASE_DIR / "data" / "mslr_web10k" / "Fold1"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# tensorflow records directory
TF_RECORDS_DIR = PROJECT_ROOT / "tf_records"
TF_RECORDS_DIR.mkdir(parents=True, exist_ok=True)

# output directory
OUTPUT_DIR = PROJECT_ROOT / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# model checkpoint directory
CKPT_DIR = OUTPUT_DIR / "ckpt"
CKPT_DIR.mkdir(parents=True, exist_ok=True)

# log directory
LOG_DIR = OUTPUT_DIR / "log"
LOG_DIR.mkdir(parents=True, exist_ok=True)

