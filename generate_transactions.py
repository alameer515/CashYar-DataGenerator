"""
generate_transactions.py
-------------------------------------------------

Generate realistic financial transactions for CashYar users.
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from config import *
from personas import PERSONAS
from seasonal_events import get_active_event

MERCHANTS = {
    "Coffee": ["Starbucks", "Barn's", "Dunkin", "Half Million", "Tim Hortons"],
    "Restaurants": ["AlBaik", "Shawarmer", "Kudu", "Herfy", "McDonald's", "Burger King"],
    "Fast Food": ["KFC", "Pizza Hut", "Dominos", "Hardees"],
    "Groceries": ["Panda", "Danube", "Tamimi", "Lulu", "Carrefour"],
    "Shopping": ["Amazon", "Noon", "Centrepoint", "Jarir Marketplace"],
    "Electronics": ["Jarir", "Extra", "Amazon", "Noon"],
    "Transportation": ["Uber", "Careem", "Taxi"],
    "Fuel": ["Aramco", "Aldrees", "SASCO"],
    "Entertainment": ["VOX Cinema", "AMC", "Winter Wonderland", "Boulevard"],
    "Education": ["Udemy", "Coursera", "University Bookstore"],
    "Books": ["Jarir", "Amazon Books"],
    "Healthcare": ["Nahdi", "Al-Dawaa"],
    "Bills": ["STC", "Mobily", "Saudi Electricity"],
    "Subscriptions": ["Netflix", "Spotify", "YouTube Premium", "Shahid"],
    "Travel": ["Flynas", "Saudia", "Booking"],
    "Beauty": ["Sephora", "Nice One"],
    "Sports": ["Sun & Sand", "Decathlon"],
    "Charity": ["Ehsan", "AlBir"],
    "Gifts": ["Floward", "Amazon"],
    "Clothing": ["Zara", "H&M", "Namshi"],
}


class TransactionGenerator:
    def __init__(self):
        random.seed(42)
        np.random.seed(42)

        self.users = pd.read_csv("data/users.csv")
        self.transactions = []
        self.transaction_id = 1
        self.user_state = {}

    def random_datetime(self):
        start = datetime(START_YEAR, 1, 1)
        end = datetime(START_YEAR, 12, 31)
        delta = end - start
        random_day = random.randint(0, delta.days)
        date = start + timedelta(days=random_day)
        hour = random.randint(7, 23)
        minute = random.randint(0, 59)
        return date.replace(hour=hour, minute=minute)

    def payment_method(self):
        return random.choice(PAYMENT_METHODS)

    def planned_purchase(self, persona):
        return np.random.rand() < persona.planned_purchase_probability

    def online_purchase(self, persona):
        return np.random.rand() < persona.online_purchase_probability

    def is_weekend(self, date):
        return date.weekday() in [4, 5]

    def active_event(self, date):
        return get_active_event(date.date())

    def merchant(self, category):
        return random.choice(MERCHANTS.get(category, ["Unknown"]))

    def choose_category(self, persona, current_date):
        categories = list(persona.category_weights.keys())
        weights = list(persona.category_weights.values())

        event = self.active_event(current_date)
        if event:
            adjusted = []
            for cat, weight in zip(categories, weights):
                multiplier = event.category_modifier.get(cat, 1.0)
                adjusted.append(weight * multiplier)
            weights = adjusted

        return random.choices(population=categories, weights=weights, k=1)[0]

    def income_multiplier(self, income):
        if income < 1000:
            return 0.75
        if income < 2000:
            return 0.90
        if income < 3000:
            return 1.00
        if income < 5000:
            return 1.15
        return 1.35

    def personality_multiplier(self, persona):
        return persona.spending_ratio

    def weekend_multiplier(self, persona, current_date):
        if self.is_weekend(current_date):
            return persona.weekend_multiplier
        return 1.0

    def night_multiplier(self, persona, current_date):
        if current_date.hour >= 20:
            return persona.night_multiplier
        return 1.0

    def event_multiplier(self, current_date):
        event = self.active_event(current_date)
        if event:
            return event.transaction_multiplier
        return 1.0

    def generate_amount(self, category, income, persona, current_date):
        minimum, maximum = CATEGORIES[category]

        if category == "Coffee":
            amount = np.random.normal(20, 4)
        elif category == "Transportation":
            amount = np.random.normal(18, 6)
        elif category == "Restaurants":
            amount = np.random.normal(55, 18)
        elif category == "Fast Food":
            amount = np.random.normal(32, 10)
        elif category == "Groceries":
            amount = np.random.normal(120, 35)
        elif category == "Shopping":
            amount = np.random.normal(220, 90)
        elif category == "Entertainment":
            amount = np.random.normal(90, 35)
        elif category == "Education":
            amount = np.random.normal(180, 70)
        elif category == "Books":
            amount = np.random.normal(95, 35)
        elif category == "Healthcare":
            amount = np.random.normal(140, 50)
        elif category == "Bills":
            amount = np.random.normal(180, 60)
        elif category == "Subscriptions":
            amount = np.random.normal(45, 10)
        elif category == "Clothing":
            amount = np.random.normal(260, 120)
        elif category == "Beauty":
            amount = np.random.normal(170, 70)
        elif category == "Sports":
            amount = np.random.normal(180, 70)
        elif category == "Travel":
            amount = np.random.normal(900, 450)
        elif category == "Electronics":
            amount = np.random.normal(1800, 700)
        elif category == "Charity":
            amount = np.random.normal(120, 60)
        elif category == "Gifts":
            amount = np.random.normal(220, 100)
        else:
            amount = random.uniform(minimum, maximum)

        amount *= self.income_multiplier(income)
        amount *= self.personality_multiplier(persona)
        amount *= self.weekend_multiplier(persona, current_date)
        amount *= self.night_multiplier(persona, current_date)
        amount *= self.event_multiplier(current_date)

        amount = max(minimum, amount)
        amount = min(maximum, amount)
        return round(amount, 2)

    def purchase_location(self, online):
        if online:
            return "Online"
        return random.choice(
            ["Mall", "Campus", "Restaurant", "Gas Station", "Supermarket"]
        )

    def season(self, date):
        month = date.month
        if month in [12, 1, 2]:
            return "Winter"
        if month in [3, 4, 5]:
            return "Spring"
        if month in [6, 7, 8]:
            return "Summer"
        return "Autumn"

    def initialize_user(self, user):
        self.user_state[user.user_id] = {
            "monthly_budget": user.monthly_budget,
            "remaining_budget": user.monthly_budget,
            "monthly_income": user.total_income,
            "monthly_spending": 0,
            "monthly_saving": user.total_income,
            "financial_score": 80,
            "goal_progress": 0,
            "consecutive_good_months": 0,
            "overspending_count": 0,
            "transactions": 0,
        }

    def state(self, user_id):
        return self.user_state[user_id]

    def update_budget(self, user_id, amount):
        state = self.state(user_id)
        state["monthly_spending"] += amount
        state["remaining_budget"] -= amount
        state["monthly_saving"] = max(state["monthly_income"] - state["monthly_spending"], 0)
        state["transactions"] += 1

    def budget_adherence(self, user_id):
        state = self.state(user_id)
        ratio = state["monthly_spending"] / max(state["monthly_budget"], 1)
        return max(0, round(1 - ratio, 2))

    def update_financial_score(self, user_id):
        state = self.state(user_id)
        score = 100
        spending_ratio = state["monthly_spending"] / max(state["monthly_income"], 1)

        if spending_ratio > 1:
            score -= 40
        elif spending_ratio > 0.90:
            score -= 25
        elif spending_ratio > 0.75:
            score -= 15
        elif spending_ratio > 0.60:
            score -= 8

        if state["transactions"] > 120:
            score -= 5

        score = max(0, score)
        state["financial_score"] = score
        return score

    def update_goal_progress(self, user, user_id):
        state = self.state(user_id)
        progress = state["monthly_saving"] / max(user.goal_amount, 1)
        progress = min(progress, 1)
        state["goal_progress"] = round(progress, 3)
        return state["goal_progress"]

    def financial_risk(self, user_id):
        score = self.state(user_id)["financial_score"]
        if score >= 80:
            return "Low"
        if score >= 60:
            return "Medium"
        return "High"

    def will_reach_goal(self, user_id):
        progress = self.state(user_id)["goal_progress"]
        if progress >= 0.80:
            return "Yes"
        return "No"

    def recommendation(self, user_id):
        score = self.state(user_id)["financial_score"]
        if score >= 90:
            return "Maintain Current Habits"
        if score >= 75:
            return "Increase Monthly Savings"
        if score >= 60:
            return "Reduce Restaurant Spending"
        if score >= 40:
            return "Reduce Shopping"
        return "High Financial Risk"

    def generate_transaction(self, user):
        if user.user_id not in self.user_state:
            self.initialize_user(user)

        current_date = self.random_datetime()
        persona = PERSONAS[user.financial_personality]
        category = self.choose_category(persona, current_date)
        merchant = self.merchant(category)
        amount = self.generate_amount(
            category=category,
            income=user.total_income,
            persona=persona,
            current_date=current_date,
        )
        planned = self.planned_purchase(persona)
        online = self.online_purchase(persona)
        payment_method = self.payment_method()
        location = self.purchase_location(online)
        season = self.season(current_date)
        event = self.active_event(current_date)
        event_name = event.name if event else "None"

        self.update_budget(user.user_id, amount)
        self.update_financial_score(user.user_id)
        self.update_goal_progress(user, user.user_id)

        state = self.state(user.user_id)

        transaction = {
            "transaction_id": self.transaction_id,
            "user_id": user.user_id,
            "datetime": current_date,
            "category": category,
            "merchant": merchant,
            "amount": amount,
            "payment_method": payment_method,
            "planned_purchase": planned,
            "online_purchase": online,
            "purchase_location": location,
            "season": season,
            "event": event_name,
            "day_of_week": current_date.strftime("%A"),
            "hour": current_date.hour,
            "remaining_budget": round(state["remaining_budget"], 2),
            "monthly_spending": round(state["monthly_spending"], 2),
            "monthly_saving": round(state["monthly_saving"], 2),
            "goal_progress": state["goal_progress"],
            "financial_score": state["financial_score"],
            "financial_risk": self.financial_risk(user.user_id),
            "recommended_action": self.recommendation(user.user_id),
        }

        self.transaction_id += 1
        return transaction

    def generate_user_transactions(self, user):
        transactions = []
        n_transactions = random.randint(
            TRANSACTIONS_PER_USER_MIN,
            TRANSACTIONS_PER_USER_MAX,
        )
        for _ in range(n_transactions):
            transactions.append(self.generate_transaction(user))
        return transactions

    def generate_dataset(self):
        print()
        print("Generating Transactions...")
        print()

        for _, user in self.users.iterrows():
            user_transactions = self.generate_user_transactions(user)
            self.transactions.extend(user_transactions)

        transactions = pd.DataFrame(self.transactions)
        transactions = transactions.sort_values(["user_id", "datetime"]).reset_index(drop=True)
        return transactions

    def save(self, dataframe):
        dataframe.to_csv("data/transactions.csv", index=False)
        print()
        print("=" * 60)
        print("Transactions Generated Successfully")
        print(f"Rows : {len(dataframe):,}")
        print("=" * 60)


if __name__ == "__main__":
    generator = TransactionGenerator()
    df = generator.generate_dataset()
    generator.save(df)
    print()
    print(df.head())
