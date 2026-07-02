"""
utils/distributions.py
--------------------------------

Spending distribution helpers used by the transaction generator.
"""

import numpy as np


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def normal_amount(mean: float, std: float, minimum: float, maximum: float) -> float:
    return round(clamp(np.random.normal(mean, std), minimum, maximum), 2)
