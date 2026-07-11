"""
Rule-based recommendation engine for CashYar AI.

This module does not use an LLM.
It returns structured recommendation objects based on:
- prediction outputs
- risk level
- overspending probability
- important features
"""

from __future__ import annotations

from typing import Dict, List


class RecommendationEngine:
    """
    Generate structured, explainable financial recommendations.
    """

    def __init__(self) -> None:
        self.category_keywords = {
            "coffee": "Reduce coffee purchases by approximately 30%.",
            "restaurant": "Reduce restaurant spending and shift toward planned meals.",
            "shopping": "Limit non-essential shopping purchases this month.",
            "electronics": "Delay electronics purchases unless they are necessary.",
            "entertainment": "Cap entertainment spending for the remaining days of the month.",
            "transport": "Optimize transport costs by combining trips or using lower-cost options.",
            "planned_purchase": "Increase planned purchases and avoid impulsive transactions.",
            "online_purchase": "Reduce online purchases, especially convenience spending.",
            "weekend_spending": "Set a lower weekend spending cap.",
            "budget_adherence": "Follow your budget more closely during the remaining days of the month.",
            "financial_score": "Improve consistency in spending decisions to increase your financial score.",
        }

    def _extract_reasons(self, feature_importance: Dict[str, List[Dict[str, float]]]) -> List[str]:
        """
        Convert top model features into human-readable reasons.
        """
        reasons = []
        overspend_features = feature_importance.get("overspend_model", [])

        for item in overspend_features[:5]:
            feature_name = item["feature"].lower()

            if "coffee" in feature_name:
                reasons.append("Coffee spending is unusually influential in the overspending prediction.")
            elif "restaurant" in feature_name:
                reasons.append("Restaurant spending is contributing strongly to the current spending risk.")
            elif "shopping" in feature_name:
                reasons.append("Shopping behavior is increasing the likelihood of budget pressure.")
            elif "budget_adherence" in feature_name:
                reasons.append("Budget adherence is weaker than expected for a stable month.")
            elif "financial_score" in feature_name:
                reasons.append("The current financial score suggests moderate-to-high spending risk.")
            elif "online_purchase" in feature_name:
                reasons.append("Online purchases are associated with increased overspending risk.")
            elif "weekend" in feature_name:
                reasons.append("Weekend spending is one of the strongest drivers of the prediction.")
            elif "planned_purchase" in feature_name:
                reasons.append("A low level of planned purchasing may be increasing impulsive spending.")
            elif "remaining_budget" in feature_name:
                reasons.append("Remaining budget is already tight relative to the current point in the month.")
            elif "spending_velocity" in feature_name or "avg_daily_spending" in feature_name:
                reasons.append("The current daily spending pace is higher than ideal.")
            else:
                reasons.append(f"{item['feature']} is one of the strongest predictive factors right now.")

        return reasons[:4]

    def _recommendation_from_features(
        self,
        feature_importance: Dict[str, List[Dict[str, float]]],
    ) -> List[Dict[str, str]]:
        """
        Map important features to rule-based recommendations.
        """
        recommendations: List[Dict[str, str]] = []
        overspend_features = feature_importance.get("overspend_model", [])

        for item in overspend_features[:6]:
            feature_name = item["feature"].lower()
            recommendation_text = None
            expected_impact = None

            if "coffee" in feature_name:
                recommendation_text = self.category_keywords["coffee"]
                expected_impact = "Expected monthly impact: lower repeated discretionary spending."
            elif "restaurant" in feature_name:
                recommendation_text = self.category_keywords["restaurant"]
                expected_impact = "Expected monthly impact: reduce high-frequency food expenses."
            elif "shopping" in feature_name:
                recommendation_text = self.category_keywords["shopping"]
                expected_impact = "Expected monthly impact: preserve budget for essentials and goals."
            elif "electronics" in feature_name:
                recommendation_text = self.category_keywords["electronics"]
                expected_impact = "Expected monthly impact: avoid large spikes in spending."
            elif "entertainment" in feature_name:
                recommendation_text = self.category_keywords["entertainment"]
                expected_impact = "Expected monthly impact: improve end-of-month savings."
            elif "transport" in feature_name:
                recommendation_text = self.category_keywords["transport"]
                expected_impact = "Expected monthly impact: lower recurring operational costs."
            elif "online_purchase" in feature_name:
                recommendation_text = self.category_keywords["online_purchase"]
                expected_impact = "Expected monthly impact: reduce convenience-driven overspending."
            elif "planned_purchase" in feature_name:
                recommendation_text = self.category_keywords["planned_purchase"]
                expected_impact = "Expected monthly impact: improve purchase control and budget stability."
            elif "weekend" in feature_name:
                recommendation_text = self.category_keywords["weekend_spending"]
                expected_impact = "Expected monthly impact: reduce concentrated spending bursts."
            elif "budget_adherence" in feature_name:
                recommendation_text = self.category_keywords["budget_adherence"]
                expected_impact = "Expected monthly impact: lower probability of overspending."
            elif "financial_score" in feature_name:
                recommendation_text = self.category_keywords["financial_score"]
                expected_impact = "Expected monthly impact: stronger overall financial behavior pattern."

            if recommendation_text:
                recommendations.append(
                    {
                        "type": "action",
                        "feature": item["feature"],
                        "importance": f"{item['importance']:.4f}",
                        "recommendation": recommendation_text,
                        "expected_impact": expected_impact,
                    }
                )

        return recommendations[:4]

    def generate(
        self,
        prediction_results: Dict[str, object],
        top_feature_importance: Dict[str, List[Dict[str, float]]],
    ) -> Dict[str, object]:
        """
        Generate a structured recommendation object.
        """
        overspend_probability = float(prediction_results.get("overspend_probability", 0.0))
        risk_level = str(prediction_results.get("risk_level", "low"))
        predicted_spending = float(prediction_results.get("predicted_monthly_spending", 0.0))
        predicted_savings = float(prediction_results.get("predicted_monthly_savings", 0.0))
        goal_delay_days = int(prediction_results.get("goal_delay_days", 0))

        reasons = self._extract_reasons(top_feature_importance)
        recommendations = self._recommendation_from_features(top_feature_importance)

        if risk_level == "high":
            headline = "High financial risk this month."
        elif risk_level == "medium":
            headline = "Moderate financial risk this month."
        else:
            headline = "Financial risk is currently manageable."

        return {
            "risk": risk_level.capitalize(),
            "headline": headline,
            "overspend_probability_percent": round(overspend_probability * 100, 1),
            "predicted_monthly_spending": round(predicted_spending, 2),
            "predicted_monthly_savings": round(predicted_savings, 2),
            "goal_delay_days": goal_delay_days,
            "reasons": reasons,
            "recommendations": recommendations,
        }