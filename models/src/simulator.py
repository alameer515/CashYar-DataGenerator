"""
What-if simulation engine for CashYar AI.

This module allows changing user behavior before prediction:
- Reduce coffee spending
- Reduce restaurant spending
- Increase savings
- Decrease shopping
- Increase income
- Change monthly budget
"""

from __future__ import annotations

from copy import deepcopy
from typing import Dict, List, Optional

import pandas as pd

from predict import CashYarPredictor
from utils import json_ready, logger


class CashYarSimulator:
    """
    Scenario simulator for re-running predictions after behavioral changes.
    """

    def __init__(self) -> None:
        self.predictor = CashYarPredictor()

    @staticmethod
    def _adjust_category_spending(
        transactions: pd.DataFrame,
        category_name: str,
        reduction_percent: float = 0.0,
        increase_percent: float = 0.0,
    ) -> pd.DataFrame:
        """
        Adjust all transactions in a category by a percentage.
        """
        df = transactions.copy()
        mask = df["category"].astype(str).str.lower() == category_name.lower()

        factor = 1.0
        if reduction_percent:
            factor *= max(0.0, 1.0 - reduction_percent)
        if increase_percent:
            factor *= max(0.0, 1.0 + increase_percent)

        df.loc[mask, "amount"] = df.loc[mask, "amount"] * factor
        return df

    @staticmethod
    def _increase_income_in_users(users_df: pd.DataFrame, user_id: str, amount: float) -> pd.DataFrame:
        """
        Increase a user's total income.
        """
        users = users_df.copy()
        mask = users["user_id"].astype(str) == str(user_id)
        users.loc[mask, "total_income"] = pd.to_numeric(users.loc[mask, "total_income"], errors="coerce").fillna(0.0) + amount
        return users

    @staticmethod
    def _change_budget_in_users(users_df: pd.DataFrame, user_id: str, new_budget: float) -> pd.DataFrame:
        """
        Change a user's monthly budget.
        """
        users = users_df.copy()
        mask = users["user_id"].astype(str) == str(user_id)
        users.loc[mask, "monthly_budget"] = new_budget
        return users

    def _predict_with_custom_users(
        self,
        user_id: str,
        current_day: int,
        transactions_df: pd.DataFrame,
        users_df: pd.DataFrame,
    ) -> Dict[str, object]:
        """
        Run prediction using custom users dataframe for simulation.
        """
        original_users = self.predictor.users_df
        try:
            self.predictor.users_df = users_df
            return self.predictor.predict(
                user_id=user_id,
                current_day=current_day,
                current_month_transactions=transactions_df,
            )
        finally:
            self.predictor.users_df = original_users

    def simulate(
        self,
        user_id: str,
        current_day: int,
        current_month_transactions: pd.DataFrame,
        actions: Dict[str, float],
    ) -> Dict[str, object]:
        """
        Run a before/after what-if simulation.

        Supported actions:
        {
            "reduce_coffee_spending": 0.40,
            "reduce_restaurant_spending": 0.25,
            "decrease_shopping_spending": 0.20,
            "increase_income": 500,
            "change_monthly_budget": 2200
        }
        """
        baseline_prediction = self.predictor.predict(
            user_id=user_id,
            current_day=current_day,
            current_month_transactions=current_month_transactions,
        )

        simulated_transactions = current_month_transactions.copy()
        simulated_users = self.predictor.users_df.copy()

        if "reduce_coffee_spending" in actions:
            simulated_transactions = self._adjust_category_spending(
                simulated_transactions, "coffee", reduction_percent=float(actions["reduce_coffee_spending"])
            )

        if "reduce_restaurant_spending" in actions:
            simulated_transactions = self._adjust_category_spending(
                simulated_transactions, "restaurant", reduction_percent=float(actions["reduce_restaurant_spending"])
            )

        if "decrease_shopping_spending" in actions:
            simulated_transactions = self._adjust_category_spending(
                simulated_transactions, "shopping", reduction_percent=float(actions["decrease_shopping_spending"])
            )

        if "increase_electronics_spending" in actions:
            simulated_transactions = self._adjust_category_spending(
                simulated_transactions, "electronics", increase_percent=float(actions["increase_electronics_spending"])
            )

        if "increase_income" in actions:
            simulated_users = self._increase_income_in_users(
                simulated_users, user_id, float(actions["increase_income"])
            )

        if "change_monthly_budget" in actions:
            simulated_users = self._change_budget_in_users(
                simulated_users, user_id, float(actions["change_monthly_budget"])
            )

        new_prediction = self._predict_with_custom_users(
            user_id=user_id,
            current_day=current_day,
            transactions_df=simulated_transactions,
            users_df=simulated_users,
        )

        diff_spending = round(
            new_prediction["predicted_monthly_spending"] - baseline_prediction["predicted_monthly_spending"], 2
        )
        diff_savings = round(
            new_prediction["predicted_monthly_savings"] - baseline_prediction["predicted_monthly_savings"], 2
        )
        diff_goal_delay = int(
            new_prediction["goal_delay_days"] - baseline_prediction["goal_delay_days"]
        )

        old_runout = baseline_prediction["predicted_budget_runout_day"]
        new_runout = new_prediction["predicted_budget_runout_day"]

        if old_runout is None or new_runout is None:
            diff_runout = None
        else:
            diff_runout = int(new_runout - old_runout)

        result = {
            "actions": actions,
            "old_prediction": baseline_prediction,
            "new_prediction": new_prediction,
            "difference": {
                "predicted_spending_change": diff_spending,
                "predicted_savings_change": diff_savings,
                "goal_delay_days_change": diff_goal_delay,
                "budget_runout_day_change": diff_runout,
                "summary": {
                    "spending_message": f"Predicted spending {'decreased' if diff_spending < 0 else 'increased'} by {abs(diff_spending):.2f} SAR",
                    "savings_message": f"Savings {'increased' if diff_savings > 0 else 'decreased'} by {abs(diff_savings):.2f} SAR",
                    "goal_message": f"Goal reached {abs(diff_goal_delay)} days {'earlier' if diff_goal_delay < 0 else 'later' if diff_goal_delay > 0 else 'with no timing change'}",
                    "budget_message": (
                        f"Budget lasts {abs(diff_runout)} additional days"
                        if diff_runout is not None and diff_runout > 0
                        else f"Budget runs out {abs(diff_runout)} days earlier"
                        if diff_runout is not None and diff_runout < 0
                        else "Budget runout day unchanged"
                    ),
                },
            },
        }

        return json_ready(result)