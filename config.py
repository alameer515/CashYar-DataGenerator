"""
CashYar Synthetic Dataset Generator
-----------------------------------

Global configuration for the dataset generation process.
"""

# ============================================================
# DATASET SIZE
# ============================================================

NUM_USERS = 5000
TRANSACTIONS_PER_USER_MIN = 80
TRANSACTIONS_PER_USER_MAX = 120

START_YEAR = 2026

# ============================================================
# UNIVERSITIES
# ============================================================

UNIVERSITIES = [
    "King Saud University",
    "Princess Nourah University",
    "King Abdulaziz University",
    "Imam Mohammad Ibn Saud Islamic University",
    "Prince Sattam bin Abdulaziz University",
    "King Fahd University of Petroleum and Minerals",
    "Qassim University",
    "Taibah University",
    "Umm Al-Qura University",
    "King Khalid University",
]

# ============================================================
# MAJORS
# ============================================================

MAJORS = [
    "Computer Science",
    "Software Engineering",
    "Information Systems",
    "Cybersecurity",
    "Artificial Intelligence",
    "Business Administration",
    "Accounting",
    "Finance",
    "Medicine",
    "Nursing",
    "Law",
    "Mechanical Engineering",
    "Electrical Engineering",
]

# ============================================================
# CITIES
# ============================================================

CITIES = [
    "Riyadh",
    "Jeddah",
    "Dammam",
    "Makkah",
    "Madinah",
    "Abha",
    "Tabuk",
    "Qassim",
]

# ============================================================
# FINANCIAL GOALS
# ============================================================

GOALS = {
    "Laptop": 5000,
    "Phone": 4500,
    "Travel": 4000,
    "Emergency Fund": 6000,
    "Car": 25000,
    "Gaming PC": 8000,
    "Investment": 10000,
}

# ============================================================
# PAYMENT METHODS
# ============================================================

PAYMENT_METHODS = [
    "Cash",
    "Debit Card",
    "Credit Card",
    "Apple Pay",
    "STC Pay",
]

# ============================================================
# TRANSACTION CATEGORIES
# ============================================================

CATEGORIES = {
    "Coffee": (12, 32),
    "Restaurants": (30, 140),
    "Fast Food": (18, 60),
    "Groceries": (30, 220),
    "Transportation": (8, 45),
    "Fuel": (40, 180),
    "Shopping": (80, 600),
    "Electronics": (250, 4000),
    "Entertainment": (20, 250),
    "Education": (50, 500),
    "Books": (30, 180),
    "Healthcare": (40, 350),
    "Bills": (80, 450),
    "Subscriptions": (15, 90),
    "Clothing": (60, 700),
    "Sports": (40, 500),
    "Beauty": (40, 450),
    "Travel": (200, 2500),
    "Charity": (20, 500),
    "Gifts": (50, 800),
}

# ============================================================
# LIVING STATUS
# ============================================================

LIVING_STATUS = [
    "With Family",
    "Dorm",
    "Shared Apartment",
    "Alone",
]

# ============================================================
# PERSONALITIES
# ============================================================

PERSONALITIES = [
    "Disciplined",
    "Balanced",
    "Coffee Lover",
    "Shopaholic",
    "Tech Enthusiast",
    "Social Spender",
    "Budget Student",
    "Impulsive",
    "Overspender",
    "Improving",
]
