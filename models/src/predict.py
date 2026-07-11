"""
Prediction API for CashYar AI.

This module loads trained models and provides a clean interface for:
- spending prediction
- savings prediction
- overspending probability
- runout day estimate
- goal completion probability
- goal delay estimation
- explainability outputs
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from feature_engineering import build_inference_row, load_raw_data, make_model_matrix
from utils import (
    OVERSPEND_MODEL_FILE,
    SAVINGS_MODEL_FILE,
    SPENDING_MODEL_FILE,
    TRAINING_ARTIFACTS_FILE,
    confidence_from_probability,
    json_ready,
    load_joblib,
    logger,
    risk_level_from_probability,
    safe_divide,
    top_n_feature_importance,
    to_float,
)


class CashYarPredictor:
    """
    Main predictor interface for application integration.
    """

    def __init__(self) -> None:
        self.spending_model = load_joblib(SPENDING_MODEL_FILE)
        self.overspend_model = load_joblib(OVERSPEND_MODEL_FILE)
        self.savings_model = load_joblib(SAVINGS_MODEL_FILE)
        self.artifacts = load_joblib(TRAINING_ARTIFACTS_FILE)

        self.model_feature_names: List[str] = self.artifacts["model_feature_names"]
        self.base_feature_columns: List[str] = self.artifacts["base_feature_columns"]
        self.training_metrics: Dict[str, Dict[str, object]] = self.artifacts["training_metrics"]

        bundle = load_raw_data()
        self.users_df = bundle.users
        self.behavior_df = bundle.behavior

    def _build_model_input(
        self,
        user_id: str,
        current_day: int,
        current_month_transactions: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Build model-aligned feature matrix for one user.
        """
        row_df = build_inference_row(
            user_id=user_id,
            current_day=current_day,
            current_month_transactions=current_month_transactions,
            users_df=self.users_df,
            behavior_df=self.behavior_df,
        )

        X, _ = make_model_matrix(
            row_df,
            feature_columns=[c for c in row_df.columns if c not in {"user_id", "month"}],
            fit_columns=self.model_feature_names,
        )
        return row_df, X

    def _estimate_budget_runout_day(
        self,
        current_spent: float,
        predicted_monthly_spending: float,
        monthly_budget: float,
        current_day: int,
        days_in_month: int,
    ) -> Optional[int]:
        """
        Estimate the day in the month when budget will run out.
        """
        if monthly_budget <= 0:
            return None

        if current_spent >= monthly_budget:
            return int(current_day)

        remaining_budget = monthly_budget - current_spent
        remaining_days = max(1, days_in_month - current_day)
        future_expected_spend = max(0.0, predicted_monthly_spending - current_spent)
        future_daily_rate = safe_divide(future_expected_spend, remaining_days, default=0.0)

        if future_daily_rate <= 0:
            return None

        days_until_runout = int(np.ceil(remaining_budget / future_daily_rate))
        return min(days_in_month, current_day + days_until_runout)

    def _estimate_goal_completion_probability(
        self,
        predicted_monthly_savings: float,
        goal_amount: float,
        current_goal_progress: float,
        target_months: float,
    ) -> float:
        """
        Estimate probability of reaching the goal from savings pace.
        """
        if goal_amount <= 0 or target_months <= 0:
            return 0.5

        current_saved_amount = max(0.0, current_goal_progress)
        remaining_goal = max(0.0, goal_amount - current_saved_amount)
        required_monthly_saving = remaining_goal / target_months

        if required_monthly_saving <= 0:
            return 1.0

        ratio = predicted_monthly_savings / required_monthly_saving
        probability = 1 / (1 + np.exp(-4 * (ratio - 1)))
        return float(np.clip(probability, 0.0, 1.0))

    def _estimate_goal_delay_days(
        self,
        predicted_monthly_savings: float,
        goal_amount: float,
        current_goal_progress: float,
        target_months: float,
    ) -> int:
        """
        Estimate delay in days relative to target pace.
        """
        if goal_amount <= 0 or target_months <= 0:
            return 0

        current_saved_amount = max(0.0, current_goal_progress)
        remaining_goal = max(0.0, goal_amount - current_saved_amount)

        required_monthly_saving = remaining_goal / target_months
        if required_monthly_saving <= 0:
            return 0

        if predicted_monthly_savings <= 0:
            return int(target_months * 30)

        predicted_months_needed = remaining_goal / predicted_monthly_savings
        delay_months = max(0.0, predicted_months_needed - target_months)
        return int(round(delay_months * 30))

    def _collect_feature_importance(self) -> Dict[str, List[Dict[str, float]]]:
        """
        Return model explainability information.
        """
        return {
            "spending_model": top_n_feature_importance(
                self.model_feature_names,
                self.spending_model.feature_importances_,
                top_n=8,
            ),
            "savings_model": top_n_feature_importance(
                self.model_feature_names,
                self.savings_model.feature_importances_,
                top_n=8,
            ),
            "overspend_model": top_n_feature_importance(
                self.model_feature_names,
                self.overspend_model.feature_importances_,
                top_n=8,
            ),
        }

    def predict(
        self,
        user_id: str,
        current_day: int,
        current_month_transactions: pd.DataFrame,
    ) -> Dict[str, object]:
        """
        Main prediction function.

        Returns a JSON-friendly dictionary with:
        - predicted_monthly_spending
        - predicted_monthly_savings
        - overspend_probability
        - predicted_budget_runout_day
        - goal_completion_probability
        - goal_delay_days
        - feature_importance
        - risk_level
        - confidence
        """
        row_df, X = self._build_model_input(user_id, current_day, current_month_transactions)

        predicted_monthly_spending = float(self.spending_model.predict(X)[0])
        predicted_monthly_savings = float(self.savings_model.predict(X)[0])
        overspend_probability = float(self.overspend_model.predict_proba(X)[0][1])

        current_spent = to_float(row_df.iloc[0].get("current_spent_so_far", 0.0))
        monthly_budget = to_float(row_df.iloc[0].get("monthly_budget", 0.0))
        days_in_month = int(to_float(row_df.iloc[0].get("days_in_month", 30), 30))
        goal_amount = to_float(row_df.iloc[0].get("goal_amount", 0.0))
        current_goal_progress = to_float(row_df.iloc[0].get("goal_progress", 0.0))
        target_months = to_float(row_df.iloc[0].get("target_months", 0.0))

        budget_runout_day = self._estimate_budget_runout_day(
            current_spent=current_spent,
            predicted_monthly_spending=predicted_monthly_spending,
            monthly_budget=monthly_budget,
            current_day=current_day,
            days_in_month=days_in_month,
        )

        goal_completion_probability = self._estimate_goal_completion_probability(
            predicted_monthly_savings=predicted_monthly_savings,
            goal_amount=goal_amount,
            current_goal_progress=current_goal_progress,
            target_months=target_months,
        )

        goal_delay_days = self._estimate_goal_delay_days(
            predicted_monthly_savings=predicted_monthly_savings,
            goal_amount=goal_amount,
            current_goal_progress=current_goal_progress,
            target_months=target_months,
        )

        feature_importance = self._collect_feature_importance()
        risk_level = risk_level_from_probability(overspend_probability)
        confidence = confidence_from_probability(overspend_probability)

        result = {
            "user_id": str(user_id),
            "current_day": int(current_day),
            "predicted_monthly_spending": round(max(0.0, predicted_monthly_spending), 2),
            "predicted_monthly_savings": round(predicted_monthly_savings, 2),
            "overspend_probability": round(overspend_probability, 4),
            "predicted_budget_runout_day": budget_runout_day,
            "goal_completion_probability": round(goal_completion_probability, 4),
            "goal_delay_days": int(goal_delay_days),
            "feature_importance": feature_importance,
            "risk_level": risk_level,
            "confidence": confidence,
            "model_metrics": self.training_metrics,
        }

        return json_ready(result)