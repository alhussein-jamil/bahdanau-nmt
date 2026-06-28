import contextlib
from pathlib import Path

import torch

DATA_DIR = Path("data/local_data")
EXT_DATA_DIR = Path("data/external_data")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def maybe_autocast():
    """Use mixed precision on CUDA only; CPU autocast breaks LSTM on some CI runners."""
    if DEVICE == "cuda":
        return torch.autocast("cuda")
    return contextlib.nullcontext()
