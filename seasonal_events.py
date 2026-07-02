"""
seasonal_events.py
--------------------------------

Defines seasonal events that influence students' financial behavior.
"""

from dataclasses import dataclass
from datetime import date
from typing import Dict, Optional


@dataclass
class SeasonalEvent:
    name: str
    start: date
    end: date
    transaction_multiplier: float
    category_modifier: Dict[str, float]


SEASONAL_EVENTS = [
    SeasonalEvent(
        name="Ramadan",
        start=date(2026, 2, 17),
        end=date(2026, 3, 18),
        transaction_multiplier=1.15,
        category_modifier={
            "Restaurants": 1.40,
            "Coffee": 0.60,
            "Groceries": 1.50,
            "Charity": 2.50,
            "Entertainment": 0.70,
            "Travel": 0.80,
        },
    ),
    SeasonalEvent(
        name="Eid",
        start=date(2026, 3, 19),
        end=date(2026, 3, 25),
        transaction_multiplier=1.45,
        category_modifier={
            "Shopping": 1.70,
            "Clothing": 2.20,
            "Restaurants": 1.80,
            "Gifts": 2.30,
            "Travel": 1.40,
            "Beauty": 1.50,
        },
    ),
    SeasonalEvent(
        name="Midterms",
        start=date(2026, 4, 20),
        end=date(2026, 5, 5),
        transaction_multiplier=0.80,
        category_modifier={
            "Coffee": 1.40,
            "Education": 1.60,
            "Books": 1.80,
            "Entertainment": 0.50,
            "Shopping": 0.60,
        },
    ),
    SeasonalEvent(
        name="Finals",
        start=date(2026, 6, 5),
        end=date(2026, 6, 25),
        transaction_multiplier=0.75,
        category_modifier={
            "Coffee": 1.50,
            "Education": 1.80,
            "Books": 1.70,
            "Restaurants": 0.80,
            "Entertainment": 0.40,
        },
    ),
    SeasonalEvent(
        name="Summer",
        start=date(2026, 7, 1),
        end=date(2026, 8, 31),
        transaction_multiplier=1.25,
        category_modifier={
            "Travel": 2.50,
            "Entertainment": 1.80,
            "Restaurants": 1.40,
            "Shopping": 1.30,
            "Transportation": 0.70,
        },
    ),
    SeasonalEvent(
        name="National Day",
        start=date(2026, 9, 20),
        end=date(2026, 9, 24),
        transaction_multiplier=1.20,
        category_modifier={
            "Shopping": 1.60,
            "Restaurants": 1.40,
            "Entertainment": 1.50,
            "Travel": 1.20,
        },
    ),
    SeasonalEvent(
        name="White Friday",
        start=date(2026, 11, 24),
        end=date(2026, 11, 30),
        transaction_multiplier=1.50,
        category_modifier={
            "Electronics": 2.50,
            "Shopping": 2.30,
            "Clothing": 1.90,
            "Beauty": 1.70,
        },
    ),
]


def get_active_event(current_date: date) -> Optional[SeasonalEvent]:
    """Return the active event on a given date, or None."""
    for event in SEASONAL_EVENTS:
        if event.start <= current_date <= event.end:
            return event
    return None


def is_event_day(current_date: date) -> bool:
    return get_active_event(current_date) is not None
