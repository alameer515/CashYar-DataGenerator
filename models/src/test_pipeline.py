
import pandas as pd
from pathlib import Path
from predict import CashYarPredictor
from clustering import predict_cluster
from simulator import CashYarSimulator
from recommendation_engine import RecommendationEngine

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR     = PROJECT_ROOT / "data"

# ── Load real data ─────────────────────────────────────────────────────────
users        = pd.read_csv(DATA_DIR / "users.csv")
transactions = pd.read_csv(DATA_DIR / "transactions.csv")
transactions['datetime'] = pd.to_datetime(transactions['datetime'])

# Pick a test user
TEST_USER_ID  = 1
TEST_DAY      = 15
TEST_MONTH    = "2026-01"

# Get that user's transactions for that month
current_month_tx = transactions[
    (transactions['user_id'] == TEST_USER_ID) &
    (transactions['datetime'].dt.to_period('M').astype(str) == TEST_MONTH) &
    (transactions['datetime'].dt.day <= TEST_DAY)
]

print(f"Testing with User {TEST_USER_ID}, Day {TEST_DAY} of {TEST_MONTH}")
print(f"Transactions so far this month: {len(current_month_tx)}")
print(f"Spent so far: {current_month_tx['amount'].sum():.2f} SAR")

# ── Test 1: Prediction ─────────────────────────────────────────────────────
print("\n" + "="*50)
print("TEST 1: CashYarPredictor")
print("="*50)

predictor = CashYarPredictor()
prediction = predictor.predict(TEST_USER_ID, TEST_DAY, current_month_tx)

print(f"Predicted monthly spending   : {prediction['predicted_monthly_spending']:.2f} SAR")
print(f"Predicted monthly savings    : {prediction['predicted_monthly_savings']:.2f} SAR")
print(f"Overspend probability        : {prediction['overspend_probability']*100:.1f}%")
print(f"Budget runout day            : Day {prediction['predicted_budget_runout_day']}")
print(f"Goal completion probability  : {prediction['goal_completion_probability']*100:.1f}%")
print(f"Risk level                   : {prediction['risk_level']}")

# ── Test 2: Clustering ─────────────────────────────────────────────────────
print("\n" + "="*50)
print("TEST 2: Clustering (Financial Persona)")
print("="*50)

behavior = pd.read_csv(DATA_DIR / "behavioral_summary.csv")
user_behavior = behavior[behavior['user_id'] == TEST_USER_ID].iloc[0].to_dict()
cluster_result = predict_cluster(user_behavior)

print(f"Cluster ID   : {cluster_result['cluster_id']}")
print(f"Persona      : {cluster_result['cluster_name']}")

# ── Test 3: Simulator ──────────────────────────────────────────────────────
print("\n" + "="*50)
print("TEST 3: What-If Simulator")
print("="*50)

simulator = CashYarSimulator()
simulation = simulator.simulate(
    user_id=TEST_USER_ID,
    current_day=TEST_DAY,
    current_month_transactions=current_month_tx,
    action="reduce_coffee_spending",
    action_value=0.4  # reduce by 40%
)

print(f"Before: {simulation['old_prediction']['predicted_monthly_spending']:.2f} SAR/month")
print(f"After : {simulation['new_prediction']['predicted_monthly_spending']:.2f} SAR/month")
print(f"Change: {simulation['difference']['predicted_spending_change']:.2f} SAR")
print(f"Goal  : {simulation['difference']['goal_delay_days_change']} days earlier/later")

# ── Test 4: Recommendations ────────────────────────────────────────────────
print("\n" + "="*50)
print("TEST 4: Recommendation Engine")
print("="*50)

rec_engine = RecommendationEngine()
recommendations = rec_engine.generate(
    prediction_results=prediction,
    top_feature_importance=prediction["feature_importance"]
)
print(f"Risk     : {recommendations['risk']}")
print(f"Headline : {recommendations['headline']}")
print(f"\nReasons:")
for r in recommendations['reasons']:
    print(f"  - {r}")
print(f"\nRecommendations:")
for r in recommendations['recommendations']:
    print(f"  - {r['recommendation']} (impact: {r['expected_impact']})")

print("\n✅ ALL TESTS PASSED — Ready for API integration")