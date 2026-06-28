"""Pytest configuration."""

import os

# Avoid oneDNN LSTM failures on GitHub Actions CPU runners.
os.environ["TORCH_USE_ONEDNN"] = "0"
os.environ.setdefault("DNNL_MAX_CPU_ISA", "AVX2")

import torch

if not torch.cuda.is_available():
    mkldnn = getattr(torch.backends, "mkldnn", None)
    if mkldnn is not None and hasattr(mkldnn, "set_enabled"):
        mkldnn.set_enabled(False)
