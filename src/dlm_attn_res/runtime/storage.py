"""Persistent storage locations for model weights and benchmark datasets."""

import os
from pathlib import Path


STORAGE_ROOT = Path(
    os.environ.get("DLM_STORAGE_ROOT", "/extra_disk_1/vasilievpavel/dlm-attn-res")
)
HF_HOME = STORAGE_ROOT / "huggingface"
DATASETS_HOME = STORAGE_ROOT / "datasets"


def configure_storage() -> None:
    """Route Hugging Face downloads away from the system disk.

    Environment variables supplied by a caller still take precedence.
    This function must run before importing `transformers` or `datasets`.
    """
    HF_HOME.mkdir(parents=True, exist_ok=True)
    DATASETS_HOME.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(HF_HOME))
    os.environ.setdefault("HF_HUB_CACHE", str(HF_HOME / "hub"))
    os.environ.setdefault("HF_DATASETS_CACHE", str(DATASETS_HOME))
