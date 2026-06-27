"""Pytest configuration."""

import os

# Disable oneDNN before torch initializes (avoids LSTM failures on some CI CPUs).
os.environ.setdefault("TORCH_USE_ONEDNN", "0")
