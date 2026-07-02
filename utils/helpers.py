"""
utils/helpers.py
--------------------------------

Shared helper utilities for CashYar data generation.
"""

from pathlib import Path


def ensure_data_dir(path: str = "data") -> Path:
    data_dir = Path(path)
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


def print_banner(title: str):
    line = "=" * 60
    print()
    print(line)
    print(title)
    print(line)
    print()
