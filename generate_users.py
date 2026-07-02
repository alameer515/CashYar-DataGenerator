"""
generate_users.py
------------------------------------

Generate realistic university students for CashYar Synthetic Dataset.
"""

import random
import uuid

import pandas as pd

from config import *


class UserGenerator:
    def __init__(self):
        random.seed(42)

    def generate_age(self):
        return random.randint(18, 25)

    def generate_gender(self):
        return random.choice(["Male", "Female"])

    def generate_study_year(self):
        return random.randint(1, 5)

    def generate_university(self):
        return random.choice(UNIVERSITIES)

    def generate_major(self):
        return random.choice(MAJORS)

    def generate_city(self):
        return random.choice(CITIES)

    def generate_living_status(self):
        return random.choices(
            population=LIVING_STATUS,
            weights=[55, 20, 15, 10],
            k=1,
        )[0]

    def generate_scholarship(self):
        return random.choices(
            [0, 990, 1000, 1500],
            weights=[10, 40, 35, 15],
            k=1,
        )[0]

    def generate_part_time_income(self):
        return random.choices(
            [0, 500, 800, 1200, 1800, 2500],
            weights=[45, 15, 15, 15, 8, 2],
            k=1,
        )[0]

    def generate_other_income(self):
        return random.choices(
            [0, 0, 200, 400, 700],
            weights=[55, 15, 15, 10, 5],
            k=1,
        )[0]

    def generate_goal(self):
        goal = random.choice(list(GOALS.keys()))
        amount = GOALS[goal]
        return goal, amount

    def generate_personality(self):
        return random.choices(
            PERSONALITIES,
            weights=[15, 25, 10, 8, 8, 10, 10, 6, 3, 5],
            k=1,
        )[0]

    def generate_budget(self, income):
        return round(income * random.uniform(0.75, 0.95), 2)

    def generate_user(self, user_id):
        scholarship = self.generate_scholarship()
        part_time = self.generate_part_time_income()
        other_income = self.generate_other_income()
        total_income = scholarship + part_time + other_income
        goal, goal_amount = self.generate_goal()

        return {
            "user_id": user_id,
            "uuid": str(uuid.uuid4()),
            "age": self.generate_age(),
            "gender": self.generate_gender(),
            "city": self.generate_city(),
            "university": self.generate_university(),
            "major": self.generate_major(),
            "study_year": self.generate_study_year(),
            "living_status": self.generate_living_status(),
            "scholarship": scholarship,
            "part_time_income": part_time,
            "other_income": other_income,
            "total_income": total_income,
            "monthly_budget": self.generate_budget(total_income),
            "financial_goal": goal,
            "goal_amount": goal_amount,
            "target_months": random.choice([3, 6, 9, 12]),
            "financial_personality": self.generate_personality(),
        }

    def generate_dataset(self):
        users = []
        for user_id in range(1, NUM_USERS + 1):
            users.append(self.generate_user(user_id))
        return pd.DataFrame(users)


if __name__ == "__main__":
    generator = UserGenerator()
    users = generator.generate_dataset()
    users.to_csv("data/users.csv", index=False)
    print(users.head())
    print()
    print(f"{len(users)} users generated successfully.")
