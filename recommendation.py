"""
recommendation.py
--------------------------------

AI label generation for CashYar behavioral summaries.
"""

from typing import Dict


def get_financial_risk(financial_score: float, saving_rate: float) -> str:
    if financial_score >= 80 and saving_rate >= 0.15:
        return "Low"
    if financial_score >= 60 and saving_rate >= 0.05:
        return "Medium"
    return "High"


def get_will_reach_goal(
    goal_progress: float,
    saving_rate: float,
    target_months: int,
    months_elapsed: int,
) -> str:
    months_remaining = max(target_months - months_elapsed, 1)
    projected = goal_progress + (saving_rate * months_remaining)
    if goal_progress >= 0.80 or projected >= 1.0:
        return "Yes"
    return "No"


def get_behavior_label(
    saving_rate: float,
    financial_score: float,
    personality: str,
) -> str:
    if saving_rate >= 0.25 and financial_score >= 80:
        return "Disciplined"
    if saving_rate >= 0.10 and financial_score >= 65:
        return "Balanced"
    if personality == "Improving" and saving_rate >= 0.08:
        return "Recovering"
    if saving_rate < 0.0 or financial_score < 40:
        return "Overspender"
    if saving_rate < 0.05 or financial_score < 55:
        return "Impulsive"
    return "Balanced"


def get_recommended_action(
    financial_score: float,
    coffee_ratio: float,
    shopping_ratio: float,
    restaurant_ratio: float,
    saving_rate: float,
) -> str:
    if saving_rate < 0:
        return "Emergency Spending Alert"
    if coffee_ratio > 0.18:
        return "Reduce Coffee Spending"
    if shopping_ratio > 0.30:
        return "Reduce Shopping"
    if restaurant_ratio > 0.25:
        return "Reduce Restaurant Spending"
    if financial_score >= 85 and saving_rate >= 0.15:
        return "Maintain Current Habits"
    if saving_rate < 0.10:
        return "Increase Savings"
    return "Increase Monthly Savings"


def enrich_summary_row(row: Dict) -> Dict:
    """Add AI labels to a monthly summary row."""
    row["financial_risk"] = get_financial_risk(
        row["financial_score"],
        row["saving_rate"],
    )
    row["will_reach_goal"] = get_will_reach_goal(
        row["goal_progress"],
        row["saving_rate"],
        row["target_months"],
        row["month"],
    )
    row["behavior_label"] = get_behavior_label(
        row["saving_rate"],
        row["financial_score"],
        row["financial_personality"],
    )
    row["recommended_action"] = get_recommended_action(
        row["financial_score"],
        row["coffee_ratio"],
        row["shopping_ratio"],
        row["restaurant_ratio"],
        row["saving_rate"],
    )
    return row
