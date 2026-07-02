"""
generate_behavior_summary.py
------------------------------------

Generate monthly behavioral summaries from transactions and users.
"""

import pandas as pd

from recommendation import enrich_summary_row


class BehaviorSummaryGenerator:
    def __init__(self):
        self.users = pd.read_csv("data/users.csv")
        self.transactions = pd.read_csv("data/transactions.csv", parse_dates=["datetime"])

    def _category_ratio(self, group: pd.DataFrame, category: str) -> float:
        total = group["amount"].sum()
        if total <= 0:
            return 0.0
        return round(group.loc[group["category"] == category, "amount"].sum() / total, 3)

    def _weekend_ratio(self, group: pd.DataFrame) -> float:
        total = group["amount"].sum()
        if total <= 0:
            return 0.0
        weekend_days = {"Friday", "Saturday"}
        weekend_spending = group.loc[group["day_of_week"].isin(weekend_days), "amount"].sum()
        return round(weekend_spending / total, 3)

    def _planned_ratio(self, group: pd.DataFrame) -> float:
        if len(group) == 0:
            return 0.0
        return round(group["planned_purchase"].astype(bool).mean(), 3)

    def _online_ratio(self, group: pd.DataFrame) -> float:
        if len(group) == 0:
            return 0.0
        return round(group["online_purchase"].astype(bool).mean(), 3)

    def _spending_trend(self, current_spending: float, previous_spending: float) -> str:
        if previous_spending <= 0:
            return "Stable"
        change = (current_spending - previous_spending) / previous_spending
        if change > 0.10:
            return "Increasing"
        if change < -0.10:
            return "Decreasing"
        return "Stable"

    def generate_dataset(self) -> pd.DataFrame:
        self.transactions["month"] = self.transactions["datetime"].dt.month
        summaries = []
        previous_spending = {}

        for user_id, user_tx in self.transactions.groupby("user_id"):
            user = self.users.loc[self.users["user_id"] == user_id].iloc[0]

            for month, month_tx in user_tx.groupby("month"):
                total_spending = round(month_tx["amount"].sum(), 2)
                total_income = user["total_income"]
                total_saving = round(total_income - total_spending, 2)
                saving_rate = round(total_saving / max(total_income, 1), 3)
                budget_adherence = round(
                    max(0, 1 - (total_spending / max(user["monthly_budget"], 1))),
                    3,
                )
                financial_score = round(month_tx["financial_score"].iloc[-1], 2)
                goal_progress = round(month_tx["goal_progress"].iloc[-1], 3)
                prev = previous_spending.get(user_id, total_spending)

                row = {
                    "user_id": user_id,
                    "month": int(month),
                    "total_income": total_income,
                    "total_spending": total_spending,
                    "total_saving": total_saving,
                    "saving_rate": saving_rate,
                    "restaurant_ratio": self._category_ratio(month_tx, "Restaurants"),
                    "coffee_ratio": self._category_ratio(month_tx, "Coffee"),
                    "shopping_ratio": self._category_ratio(month_tx, "Shopping"),
                    "transport_ratio": self._category_ratio(month_tx, "Transportation"),
                    "electronics_ratio": self._category_ratio(month_tx, "Electronics"),
                    "entertainment_ratio": self._category_ratio(month_tx, "Entertainment"),
                    "weekend_spending_ratio": self._weekend_ratio(month_tx),
                    "planned_purchase_ratio": self._planned_ratio(month_tx),
                    "online_purchase_ratio": self._online_ratio(month_tx),
                    "budget_adherence": budget_adherence,
                    "financial_score": financial_score,
                    "goal_progress": goal_progress,
                    "consecutive_budget_success": int(budget_adherence >= 0.0),
                    "spending_trend": self._spending_trend(total_spending, prev),
                    "financial_personality": user["financial_personality"],
                    "target_months": user["target_months"],
                }

                summaries.append(enrich_summary_row(row))
                previous_spending[user_id] = total_spending

        return pd.DataFrame(summaries)

    def save(self, dataframe: pd.DataFrame):
        dataframe.to_csv("data/behavioral_summary.csv", index=False)
        print()
        print("=" * 60)
        print("Behavioral Summaries Generated Successfully")
        print(f"Rows : {len(dataframe):,}")
        print("=" * 60)


if __name__ == "__main__":
    generator = BehaviorSummaryGenerator()
    df = generator.generate_dataset()
    generator.save(df)
    print()
    print(df.head())
