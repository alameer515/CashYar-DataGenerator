"""
Feature engineering for CashYar AI.

This module:
- Loads and validates raw CSV data
- Cleans and normalizes fields
- Builds monthly training rows from transaction-level data
- Builds real-time inference features for a user/current month
- Produces model-ready feature matrices with one-hot encoding
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from utils import (
    BEHAVIOR_FILE,
    DEFAULT_RANDOM_STATE,
    TRANSACTIONS_FILE,
    USERS_FILE,
    clip_non_negative,
    days_in_month_from_timestamp,
    logger,
    month_key,
    normalize_text,
    safe_divide,
    to_float,
    to_int,
    validate_required_columns,
)


USERS_REQUIRED_COLUMNS = [
    "user_id",
    "age",
    "gender",
    "city",
    "university",
    "major",
    "study_year",
    "living_status",
    "scholarship",
    "part_time_income",
    "other_income",
    "total_income",
    "monthly_budget",
    "financial_goal",
    "goal_amount",
    "target_months",
    "financial_personality",
]

TRANSACTIONS_REQUIRED_COLUMNS = [
    "transaction_id",
    "user_id",
    "datetime",
    "category",
    "merchant",
    "amount",
    "payment_method",
    "planned_purchase",
    "online_purchase",
    "purchase_location",
    "season",
    "event",
    "day_of_week",
    "hour",
    "remaining_budget",
    "monthly_spending",
    "monthly_saving",
    "goal_progress",
    "financial_score",
    "financial_risk",
    "recommended_action",
]

BEHAVIOR_REQUIRED_COLUMNS = [
    "user_id",
    "month",
    "total_income",
    "total_spending",
    "total_saving",
    "saving_rate",
    "restaurant_ratio",
    "coffee_ratio",
    "shopping_ratio",
    "transport_ratio",
    "electronics_ratio",
    "entertainment_ratio",
    "weekend_spending_ratio",
    "planned_purchase_ratio",
    "online_purchase_ratio",
    "budget_adherence",
    "financial_score",
    "goal_progress",
    "consecutive_budget_success",
    "spending_trend",
    "financial_personality",
    "target_months",
    "financial_risk",
    "will_reach_goal",
    "behavior_label",
    "recommended_action",
]


NUMERIC_USER_COLUMNS = [
    "age",
    "study_year",
    "scholarship",
    "part_time_income",
    "other_income",
    "total_income",
    "monthly_budget",
    "goal_amount",
    "target_months",
]

CATEGORICAL_USER_COLUMNS = [
    "gender",
    "city",
    "university",
    "major",
    "living_status",
    "financial_goal",
    "financial_personality",
]

NUMERIC_BEHAVIOR_COLUMNS = [
    "total_income",
    "total_spending",
    "total_saving",
    "saving_rate",
    "restaurant_ratio",
    "coffee_ratio",
    "shopping_ratio",
    "transport_ratio",
    "electronics_ratio",
    "entertainment_ratio",
    "weekend_spending_ratio",
    "planned_purchase_ratio",
    "online_purchase_ratio",
    "budget_adherence",
    "financial_score",
    "goal_progress",
    "consecutive_budget_success",
    "target_months",
]

CATEGORICAL_BEHAVIOR_COLUMNS = [
    "spending_trend",
    "financial_personality",
    "financial_risk",
    "behavior_label",
    "recommended_action",
]


@dataclass
class DatasetBundle:
    users: pd.DataFrame
    transactions: pd.DataFrame
    behavior: pd.DataFrame


def load_raw_data(
    users_path: str | None = None,
    transactions_path: str | None = None,
    behavior_path: str | None = None,
) -> DatasetBundle:
    """
    Load all CSV files from disk and validate their basic structure.
    """
    users = pd.read_csv(users_path or USERS_FILE)
    transactions = pd.read_csv(transactions_path or TRANSACTIONS_FILE)
    behavior = pd.read_csv(behavior_path or BEHAVIOR_FILE)

    validate_required_columns(users, USERS_REQUIRED_COLUMNS, "users.csv")
    validate_required_columns(transactions, TRANSACTIONS_REQUIRED_COLUMNS, "transactions.csv")
    validate_required_columns(behavior, BEHAVIOR_REQUIRED_COLUMNS, "behavioral_summary.csv")

    return DatasetBundle(users=users, transactions=transactions, behavior=behavior)


def clean_users(users: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and normalize user-level data.
    """
    df = users.copy()

    for col in NUMERIC_USER_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in CATEGORICAL_USER_COLUMNS:
        df[col] = df[col].fillna("unknown").astype(str).str.strip()

    df["user_id"] = df["user_id"].astype(str)

    numeric_fill_cols = [
        "age",
        "study_year",
        "scholarship",
        "part_time_income",
        "other_income",
        "total_income",
        "monthly_budget",
        "goal_amount",
        "target_months",
    ]
    for col in numeric_fill_cols:
        df[col] = df[col].fillna(df[col].median())

    return df


def clean_behavior(behavior: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and normalize behavioral summary data.
    """
    df = behavior.copy()
    df["user_id"] = df["user_id"].astype(str)
    df["month"] = df["month"].astype(str)

    for col in NUMERIC_BEHAVIOR_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in CATEGORICAL_BEHAVIOR_COLUMNS:
        df[col] = df[col].fillna("unknown").astype(str).str.strip()

    for col in NUMERIC_BEHAVIOR_COLUMNS:
        df[col] = df[col].fillna(df[col].median())

    if "will_reach_goal" in df.columns:
        df["will_reach_goal"] = (
            df["will_reach_goal"]
            .astype(str)
            .str.lower()
            .map({"true": 1, "false": 0, "1": 1, "0": 0, "yes": 1, "no": 0})
            .fillna(0)
            .astype(int)
        )

    return df


def clean_transactions(transactions: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and normalize transaction-level data.
    """
    df = transactions.copy()
    df["user_id"] = df["user_id"].astype(str)
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0.0)

    bool_like_cols = ["planned_purchase", "online_purchase"]
    for col in bool_like_cols:
        df[col] = (
            df[col]
            .astype(str)
            .str.lower()
            .map({"true": 1, "false": 0, "1": 1, "0": 0, "yes": 1, "no": 0})
            .fillna(0)
            .astype(int)
        )

    numeric_cols = [
        "remaining_budget",
        "monthly_spending",
        "monthly_saving",
        "goal_progress",
        "financial_score",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    cat_cols = [
        "category",
        "merchant",
        "payment_method",
        "purchase_location",
        "season",
        "event",
        "day_of_week",
        "financial_risk",
        "recommended_action",
    ]
    for col in cat_cols:
        df[col] = df[col].fillna("unknown").astype(str).str.strip()

    df["hour"] = pd.to_numeric(df["hour"], errors="coerce")
    df["hour"] = df["hour"].fillna(df["datetime"].dt.hour).fillna(12)

    df = df.dropna(subset=["datetime"]).sort_values(["user_id", "datetime"]).reset_index(drop=True)
    df["month"] = df["datetime"].dt.strftime("%Y-%m")
    df["day"] = df["datetime"].dt.day
    df["is_weekend"] = df["day_of_week"].str.lower().isin(["friday", "saturday", "sat", "fri"]).astype(int)

    return df


def aggregate_transaction_features(group: pd.DataFrame) -> Dict[str, float]:
    """
    Build rich monthly features from a user's transactions in one month.
    """
    group = group.sort_values("datetime").copy()

    total_spent = group["amount"].sum()
    transaction_count = len(group)
    active_days = group["day"].nunique()
    days_in_month = days_in_month_from_timestamp(group["datetime"].max())
    first_day = int(group["day"].min())
    last_day = int(group["day"].max())
    observed_span_days = max(1, last_day - first_day + 1)

    daily_spend = group.groupby("day")["amount"].sum().reindex(range(1, days_in_month + 1), fill_value=0.0)

    first_half = daily_spend.loc[1:max(1, days_in_month // 2)].mean()
    second_half = daily_spend.loc[(days_in_month // 2) + 1:days_in_month].mean()

    week1 = daily_spend.loc[1:7].mean() if len(daily_spend.loc[1:7]) > 0 else 0.0
    week2 = daily_spend.loc[8:14].mean() if len(daily_spend.loc[8:14]) > 0 else 0.0
    week3 = daily_spend.loc[15:21].mean() if len(daily_spend.loc[15:21]) > 0 else 0.0
    week4 = daily_spend.loc[22:days_in_month].mean() if len(daily_spend.loc[22:days_in_month]) > 0 else 0.0

    avg_daily_spending = safe_divide(total_spent, active_days)
    calendar_daily_spending = safe_divide(total_spent, days_in_month)
    spending_velocity = safe_divide(total_spent, observed_span_days)
    spending_acceleration = second_half - first_half

    latest_remaining_budget = pd.to_numeric(group["remaining_budget"], errors="coerce").dropna()
    latest_remaining_budget = float(latest_remaining_budget.iloc[-1]) if len(latest_remaining_budget) else np.nan

    latest_monthly_spending = pd.to_numeric(group["monthly_spending"], errors="coerce").dropna()
    latest_monthly_spending = float(latest_monthly_spending.iloc[-1]) if len(latest_monthly_spending) else total_spent

    latest_monthly_saving = pd.to_numeric(group["monthly_saving"], errors="coerce").dropna()
    latest_monthly_saving = float(latest_monthly_saving.iloc[-1]) if len(latest_monthly_saving) else np.nan

    latest_goal_progress = pd.to_numeric(group["goal_progress"], errors="coerce").dropna()
    latest_goal_progress = float(latest_goal_progress.iloc[-1]) if len(latest_goal_progress) else np.nan

    latest_financial_score = pd.to_numeric(group["financial_score"], errors="coerce").dropna()
    latest_financial_score = float(latest_financial_score.iloc[-1]) if len(latest_financial_score) else np.nan

    category_spend = group.groupby(group["category"].str.lower())["amount"].sum().to_dict()

    restaurant_spend = category_spend.get("restaurant", 0.0)
    coffee_spend = category_spend.get("coffee", 0.0)
    shopping_spend = category_spend.get("shopping", 0.0)
    transport_spend = category_spend.get("transport", 0.0)
    electronics_spend = category_spend.get("electronics", 0.0)
    entertainment_spend = category_spend.get("entertainment", 0.0)

    morning_spend = group.loc[group["hour"].between(5, 11), "amount"].sum()
    afternoon_spend = group.loc[group["hour"].between(12, 17), "amount"].sum()
    evening_spend = group.loc[group["hour"].between(18, 23), "amount"].sum()
    night_spend = group.loc[group["hour"].between(0, 4), "amount"].sum()

    online_spend = group.loc[group["online_purchase"] == 1, "amount"].sum()
    planned_spend = group.loc[group["planned_purchase"] == 1, "amount"].sum()
    weekend_spend = group.loc[group["is_weekend"] == 1, "amount"].sum()

    high_value_threshold = group["amount"].quantile(0.9) if transaction_count > 0 else 0.0
    high_value_tx_count = int((group["amount"] >= high_value_threshold).sum()) if high_value_threshold > 0 else 0

    merchant_diversity = group["merchant"].nunique()
    category_diversity = group["category"].nunique()
    payment_diversity = group["payment_method"].nunique()

    return {
        "transaction_count": transaction_count,
        "active_days": active_days,
        "days_in_month": days_in_month,
        "observed_span_days": observed_span_days,
        "avg_daily_spending": avg_daily_spending,
        "calendar_daily_spending": calendar_daily_spending,
        "spending_velocity": spending_velocity,
        "spending_acceleration": spending_acceleration,
        "week1_average": week1,
        "week2_average": week2,
        "week3_average": week3,
        "week4_average": week4,
        "week12_change": week2 - week1,
        "week23_change": week3 - week2,
        "week34_change": week4 - week3,
        "remaining_budget_tx": latest_remaining_budget,
        "monthly_spending_tx": latest_monthly_spending,
        "monthly_saving_tx": latest_monthly_saving,
        "goal_progress_tx": latest_goal_progress,
        "financial_score_tx": latest_financial_score,
        "restaurant_ratio_tx": safe_divide(restaurant_spend, total_spent),
        "coffee_ratio_tx": safe_divide(coffee_spend, total_spent),
        "shopping_ratio_tx": safe_divide(shopping_spend, total_spent),
        "transport_ratio_tx": safe_divide(transport_spend, total_spent),
        "electronics_ratio_tx": safe_divide(electronics_spend, total_spent),
        "entertainment_ratio_tx": safe_divide(entertainment_spend, total_spent),
        "online_purchase_ratio_tx": safe_divide(online_spend, total_spent),
        "planned_purchase_ratio_tx": safe_divide(planned_spend, total_spent),
        "weekend_spending_ratio_tx": safe_divide(weekend_spend, total_spent),
        "morning_spending_ratio": safe_divide(morning_spend, total_spent),
        "afternoon_spending_ratio": safe_divide(afternoon_spend, total_spent),
        "evening_spending_ratio": safe_divide(evening_spend, total_spent),
        "night_spending_ratio": safe_divide(night_spend, total_spent),
        "avg_transaction_amount": safe_divide(total_spent, transaction_count),
        "max_transaction_amount": float(group["amount"].max()) if transaction_count > 0 else 0.0,
        "std_transaction_amount": float(group["amount"].std(ddof=0)) if transaction_count > 1 else 0.0,
        "high_value_tx_count": high_value_tx_count,
        "merchant_diversity": merchant_diversity,
        "category_diversity": category_diversity,
        "payment_diversity": payment_diversity,
    }


def build_training_dataset(bundle: DatasetBundle) -> pd.DataFrame:
    """
    Build a monthly row-level modeling dataset by combining:
    - user demographics and financial profile
    - monthly transaction aggregates
    - monthly behavioral summary
    """
    users = clean_users(bundle.users)
    transactions = clean_transactions(bundle.transactions)
    behavior = clean_behavior(bundle.behavior)

    monthly_rows: List[Dict[str, object]] = []

    grouped = transactions.groupby(["user_id", "month"])
    for (user_id, month), tx_group in grouped:
        user_match = users.loc[users["user_id"] == str(user_id)]
        behavior_match = behavior.loc[
            (behavior["user_id"] == str(user_id)) & (behavior["month"] == str(month))
        ]

        if user_match.empty:
            continue

        user_row = user_match.iloc[0].to_dict()
        tx_features = aggregate_transaction_features(tx_group)

        base_row: Dict[str, object] = {
            "user_id": str(user_id),
            "month": str(month),
            **user_row,
            **tx_features,
        }

        if not behavior_match.empty:
            behavior_row = behavior_match.iloc[0].to_dict()
            for key, value in behavior_row.items():
                if key not in {"user_id", "month"}:
                    base_row[key] = value

        # Fallback targets when behavioral summary is missing.
        total_income = to_float(base_row.get("total_income", 0.0))
        monthly_budget = to_float(base_row.get("monthly_budget", 0.0))
        total_spending = to_float(base_row.get("total_spending", tx_group["amount"].sum()))
        total_saving = to_float(base_row.get("total_saving", total_income - total_spending))

        base_row["target_monthly_spending"] = total_spending
        base_row["target_monthly_savings"] = total_saving
        base_row["target_overspend"] = int(total_spending > monthly_budget) if monthly_budget > 0 else 0

        monthly_rows.append(base_row)

    dataset = pd.DataFrame(monthly_rows)

    if dataset.empty:
        raise ValueError("Training dataset is empty after merging sources.")

    dataset = finalize_engineered_features(dataset)
    return dataset


def finalize_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add final engineered columns used by the models.
    """
    data = df.copy()

    numeric_candidates = [
        "total_income",
        "monthly_budget",
        "goal_amount",
        "target_months",
        "total_spending",
        "total_saving",
        "saving_rate",
        "budget_adherence",
        "financial_score",
        "goal_progress",
        "consecutive_budget_success",
        "target_monthly_spending",
        "target_monthly_savings",
        "remaining_budget_tx",
        "monthly_spending_tx",
        "monthly_saving_tx",
        "goal_progress_tx",
        "financial_score_tx",
    ]
    for col in numeric_candidates:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")

    data["remaining_budget"] = data["monthly_budget"] - data["target_monthly_spending"]
    data["budget_used_percentage"] = np.where(
        data["monthly_budget"] > 0,
        data["target_monthly_spending"] / data["monthly_budget"],
        0.0,
    )
    data["budget_remaining_percentage"] = np.where(
        data["monthly_budget"] > 0,
        data["remaining_budget"] / data["monthly_budget"],
        0.0,
    )
    data["income_used_percentage"] = np.where(
        data["total_income"] > 0,
        data["target_monthly_spending"] / data["total_income"],
        0.0,
    )
    data["saving_rate_computed"] = np.where(
        data["total_income"] > 0,
        data["target_monthly_savings"] / data["total_income"],
        0.0,
    )
    data["goal_gap_amount"] = np.maximum(
        0.0,
        data["goal_amount"].fillna(0.0) - data["target_monthly_savings"].fillna(0.0)
    )
    data["goal_monthly_requirement"] = np.where(
        data["target_months"].fillna(0) > 0,
        data["goal_amount"].fillna(0.0) / data["target_months"].replace(0, np.nan),
        0.0,
    )
    data["goal_affordability_ratio"] = np.where(
        data["goal_monthly_requirement"] > 0,
        data["target_monthly_savings"] / data["goal_monthly_requirement"],
        0.0,
    )
    data["spending_to_saving_ratio"] = np.where(
        data["target_monthly_savings"].abs() > 0,
        data["target_monthly_spending"] / data["target_monthly_savings"].replace(0, np.nan).abs(),
        data["target_monthly_spending"],
    )
    data["income_minus_budget"] = data["total_income"] - data["monthly_budget"]
    data["budget_buffer_ratio"] = np.where(
        data["total_income"] > 0,
        (data["total_income"] - data["monthly_budget"]) / data["total_income"],
        0.0,
    )

    categorical_cols = [
        "gender",
        "city",
        "university",
        "major",
        "living_status",
        "financial_goal",
        "financial_personality",
        "spending_trend",
        "financial_risk",
        "behavior_label",
        "recommended_action",
    ]
    for col in categorical_cols:
        if col in data.columns:
            data[col] = data[col].fillna("unknown").astype(str).str.strip()

    data = data.replace([np.inf, -np.inf], np.nan)

    numeric_cols = data.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        data[col] = data[col].fillna(data[col].median())

    object_cols = data.select_dtypes(include=["object"]).columns
    for col in object_cols:
        data[col] = data[col].fillna("unknown")

    return data


def get_feature_columns(dataset: pd.DataFrame) -> List[str]:
    """
    Return the base feature columns before one-hot encoding.
    """
    excluded = {
        "user_id",
        "month",
        "target_monthly_spending",
        "target_monthly_savings",
        "target_overspend",
        "uuid",
    }
    return [c for c in dataset.columns if c not in excluded]


def make_model_matrix(
    dataset: pd.DataFrame,
    feature_columns: Optional[List[str]] = None,
    fit_columns: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Convert a dataset into a one-hot encoded model matrix.

    If fit_columns is provided, align to that exact set of columns.
    """
    feature_columns = feature_columns or get_feature_columns(dataset)
    X = dataset[feature_columns].copy()

    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    X = pd.get_dummies(X, columns=cat_cols, drop_first=False)

    if fit_columns is not None:
        for col in fit_columns:
            if col not in X.columns:
                X[col] = 0
        X = X[fit_columns]
        return X, fit_columns

    feature_names = X.columns.tolist()
    return X, feature_names


def build_inference_row(
    user_id: str,
    current_day: int,
    current_month_transactions: pd.DataFrame,
    users_df: pd.DataFrame,
    behavior_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build one real-time inference row for a single user and current month.
    """
    users = clean_users(users_df)
    behavior = clean_behavior(behavior_df)
    tx = clean_transactions(current_month_transactions)

    user_row_match = users.loc[users["user_id"] == str(user_id)]
    if user_row_match.empty:
        raise ValueError(f"user_id '{user_id}' not found in users.csv")

    user_row = user_row_match.iloc[0].to_dict()
    tx_user = tx.loc[tx["user_id"] == str(user_id)].copy()

    if tx_user.empty:
        synthetic_ts = pd.Timestamp.now().replace(day=max(1, current_day), hour=12, minute=0, second=0)
        tx_user = pd.DataFrame(
            [
                {
                    "transaction_id": "synthetic-0",
                    "user_id": str(user_id),
                    "datetime": synthetic_ts,
                    "category": "unknown",
                    "merchant": "unknown",
                    "amount": 0.0,
                    "payment_method": "unknown",
                    "planned_purchase": 0,
                    "online_purchase": 0,
                    "purchase_location": "unknown",
                    "season": "unknown",
                    "event": "none",
                    "day_of_week": synthetic_ts.day_name(),
                    "hour": synthetic_ts.hour,
                    "remaining_budget": user_row.get("monthly_budget", 0.0),
                    "monthly_spending": 0.0,
                    "monthly_saving": user_row.get("total_income", 0.0),
                    "goal_progress": 0.0,
                    "financial_score": 50.0,
                    "financial_risk": "low",
                    "recommended_action": "maintain_budget",
                }
            ]
        )
        tx_user = clean_transactions(tx_user)

    inferred_month = tx_user["month"].iloc[0]
    tx_features = aggregate_transaction_features(tx_user)

    latest_behavior = behavior.loc[behavior["user_id"] == str(user_id)].sort_values("month")
    if not latest_behavior.empty:
        behavior_row = latest_behavior.iloc[-1].to_dict()
    else:
        behavior_row = {}

    current_spent = float(tx_user["amount"].sum())
    monthly_budget = to_float(user_row.get("monthly_budget", 0.0))
    total_income = to_float(user_row.get("total_income", 0.0))
    days_in_month = int(tx_features["days_in_month"])
    days_remaining = max(0, days_in_month - int(current_day))
    avg_so_far = safe_divide(current_spent, max(1, current_day))
    projected_spending = avg_so_far * days_in_month
    projected_saving = total_income - projected_spending

    row = {
        "user_id": str(user_id),
        "month": inferred_month,
        **user_row,
        **tx_features,
        **behavior_row,
        "current_day": int(current_day),
        "days_remaining": days_remaining,
        "current_spent_so_far": current_spent,
        "projected_spending_so_far_rate": projected_spending,
        "projected_saving_so_far_rate": projected_saving,
        "remaining_budget_live": monthly_budget - current_spent,
        "budget_used_percentage_live": safe_divide(current_spent, monthly_budget),
        "income_used_percentage_live": safe_divide(current_spent, total_income),
        "saving_rate_live": safe_divide(total_income - current_spent, total_income),
    }

    row_df = pd.DataFrame([row])
    row_df = finalize_inference_features(row_df)
    return row_df


def finalize_inference_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add inference-time engineered features aligned with training logic.
    """
    data = df.copy()

    numeric_candidates = data.columns
    for col in numeric_candidates:
        if col not in {"user_id", "month", "gender", "city", "university", "major", "living_status",
                       "financial_goal", "financial_personality", "spending_trend", "financial_risk",
                       "behavior_label", "recommended_action"}:
            data[col] = pd.to_numeric(data[col], errors="ignore")

    current_spent = pd.to_numeric(data.get("current_spent_so_far", 0.0), errors="coerce").fillna(0.0)
    monthly_budget = pd.to_numeric(data.get("monthly_budget", 0.0), errors="coerce").fillna(0.0)
    total_income = pd.to_numeric(data.get("total_income", 0.0), errors="coerce").fillna(0.0)
    target_months = pd.to_numeric(data.get("target_months", 0.0), errors="coerce").fillna(0.0)
    goal_amount = pd.to_numeric(data.get("goal_amount", 0.0), errors="coerce").fillna(0.0)

    data["remaining_budget"] = monthly_budget - current_spent
    data["budget_used_percentage"] = np.where(monthly_budget > 0, current_spent / monthly_budget, 0.0)
    data["budget_remaining_percentage"] = np.where(monthly_budget > 0, data["remaining_budget"] / monthly_budget, 0.0)
    data["income_used_percentage"] = np.where(total_income > 0, current_spent / total_income, 0.0)
    data["goal_monthly_requirement"] = np.where(target_months > 0, goal_amount / target_months.replace(0, np.nan), 0.0)
    data["goal_gap_amount"] = np.maximum(0.0, goal_amount - (total_income - current_spent))
    data["budget_buffer_ratio"] = np.where(total_income > 0, (total_income - monthly_budget) / total_income, 0.0)
    data["income_minus_budget"] = total_income - monthly_budget
    data["saving_rate_computed"] = np.where(total_income > 0, (total_income - current_spent) / total_income, 0.0)

    data = data.replace([np.inf, -np.inf], np.nan)

    numeric_cols = data.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        data[col] = data[col].fillna(data[col].median())

    object_cols = data.select_dtypes(include=["object"]).columns
    for col in object_cols:
        data[col] = data[col].fillna("unknown")

    return data