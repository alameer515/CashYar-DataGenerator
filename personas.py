"""
personas.py
----------------------------

Defines realistic financial personas for CashYar users.
"""

from dataclasses import dataclass
from typing import Dict


@dataclass
class Persona:
    name: str
    spending_ratio: float
    planned_purchase_probability: float
    online_purchase_probability: float
    weekend_multiplier: float
    night_multiplier: float
    saving_priority: float
    category_weights: Dict[str, int]


PERSONAS = {
    "Disciplined": Persona(
        name="Disciplined",
        spending_ratio=0.55,
        planned_purchase_probability=0.95,
        online_purchase_probability=0.35,
        weekend_multiplier=1.05,
        night_multiplier=0.90,
        saving_priority=0.95,
        category_weights={
            "Groceries": 18,
            "Transportation": 18,
            "Education": 15,
            "Books": 12,
            "Coffee": 8,
            "Restaurants": 8,
            "Shopping": 5,
            "Entertainment": 4,
            "Electronics": 5,
            "Healthcare": 5,
            "Bills": 2,
        },
    ),
    "Balanced": Persona(
        name="Balanced",
        spending_ratio=0.72,
        planned_purchase_probability=0.75,
        online_purchase_probability=0.45,
        weekend_multiplier=1.15,
        night_multiplier=1.05,
        saving_priority=0.70,
        category_weights={
            "Coffee": 12,
            "Restaurants": 14,
            "Transportation": 15,
            "Groceries": 15,
            "Shopping": 10,
            "Entertainment": 8,
            "Education": 8,
            "Books": 5,
            "Electronics": 5,
            "Healthcare": 5,
            "Bills": 3,
        },
    ),
    "Coffee Lover": Persona(
        name="Coffee Lover",
        spending_ratio=0.82,
        planned_purchase_probability=0.60,
        online_purchase_probability=0.25,
        weekend_multiplier=1.20,
        night_multiplier=1.10,
        saving_priority=0.55,
        category_weights={
            "Coffee": 35,
            "Restaurants": 18,
            "Transportation": 15,
            "Groceries": 10,
            "Entertainment": 8,
            "Shopping": 5,
            "Education": 4,
            "Books": 2,
            "Electronics": 3,
        },
    ),
    "Shopaholic": Persona(
        name="Shopaholic",
        spending_ratio=1.05,
        planned_purchase_probability=0.30,
        online_purchase_probability=0.80,
        weekend_multiplier=1.40,
        night_multiplier=1.30,
        saving_priority=0.25,
        category_weights={
            "Shopping": 35,
            "Clothing": 20,
            "Beauty": 12,
            "Restaurants": 8,
            "Coffee": 6,
            "Entertainment": 6,
            "Electronics": 5,
            "Transportation": 4,
            "Groceries": 4,
        },
    ),
    "Tech Enthusiast": Persona(
        name="Tech Enthusiast",
        spending_ratio=0.92,
        planned_purchase_probability=0.75,
        online_purchase_probability=0.85,
        weekend_multiplier=1.10,
        night_multiplier=1.20,
        saving_priority=0.45,
        category_weights={
            "Electronics": 30,
            "Subscriptions": 15,
            "Shopping": 12,
            "Coffee": 10,
            "Restaurants": 10,
            "Entertainment": 8,
            "Education": 8,
            "Transportation": 5,
            "Books": 2,
        },
    ),
    "Budget Student": Persona(
        name="Budget Student",
        spending_ratio=0.50,
        planned_purchase_probability=0.92,
        online_purchase_probability=0.30,
        weekend_multiplier=0.95,
        night_multiplier=0.90,
        saving_priority=0.90,
        category_weights={
            "Transportation": 22,
            "Groceries": 22,
            "Education": 18,
            "Books": 12,
            "Coffee": 8,
            "Restaurants": 6,
            "Healthcare": 6,
            "Bills": 6,
        },
    ),
    "Social Spender": Persona(
        name="Social Spender",
        spending_ratio=0.95,
        planned_purchase_probability=0.45,
        online_purchase_probability=0.35,
        weekend_multiplier=1.60,
        night_multiplier=1.35,
        saving_priority=0.40,
        category_weights={
            "Restaurants": 25,
            "Coffee": 18,
            "Entertainment": 18,
            "Shopping": 10,
            "Transportation": 10,
            "Travel": 8,
            "Clothing": 6,
            "Groceries": 5,
        },
    ),
    "Impulsive": Persona(
        name="Impulsive",
        spending_ratio=1.00,
        planned_purchase_probability=0.20,
        online_purchase_probability=0.70,
        weekend_multiplier=1.35,
        night_multiplier=1.40,
        saving_priority=0.20,
        category_weights={
            "Shopping": 22,
            "Electronics": 18,
            "Restaurants": 15,
            "Entertainment": 12,
            "Coffee": 10,
            "Clothing": 10,
            "Beauty": 5,
            "Transportation": 4,
            "Groceries": 4,
        },
    ),
    "Overspender": Persona(
        name="Overspender",
        spending_ratio=1.25,
        planned_purchase_probability=0.15,
        online_purchase_probability=0.75,
        weekend_multiplier=1.50,
        night_multiplier=1.45,
        saving_priority=0.10,
        category_weights={
            "Shopping": 22,
            "Restaurants": 18,
            "Electronics": 18,
            "Entertainment": 15,
            "Coffee": 10,
            "Clothing": 8,
            "Travel": 5,
            "Beauty": 4,
        },
    ),
    "Improving": Persona(
        name="Improving",
        spending_ratio=0.78,
        planned_purchase_probability=0.65,
        online_purchase_probability=0.45,
        weekend_multiplier=1.10,
        night_multiplier=1.00,
        saving_priority=0.60,
        category_weights={
            "Groceries": 18,
            "Transportation": 16,
            "Coffee": 12,
            "Restaurants": 12,
            "Shopping": 10,
            "Education": 10,
            "Entertainment": 8,
            "Books": 5,
            "Electronics": 5,
            "Healthcare": 4,
        },
    ),
}


def get_persona(name: str) -> Persona:
    """Return the Persona object by name."""
    return PERSONAS[name]
