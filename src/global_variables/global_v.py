from pathlib import Path

import torch

DATA_DIR = Path("data/local_data")
EXT_DATA_DIR = Path("data/external_data")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
