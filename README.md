# CashYar Data Generator

Synthetic fintech dataset generator for the **CashYar** project — a smart financial assistant that helps university students build better money habits through behavioral learning rather than traditional lessons.[1][2]

This generator produces the three core CSV datasets used by the CashYar AI pipeline: user profiles, transaction histories, and monthly behavioral summaries designed for downstream prediction, simulation, explainability, and recommendation workflows.[3][4][5]

## Project purpose

CashYar is designed as an educational financial assistant for students, so the generated data aims to represent realistic student spending behavior, budget pressure, savings progress, and decision patterns instead of only storing raw expenses.[1][3]

The generated outputs are structured to support later modules such as monthly spending prediction, overspending probability estimation, savings forecasting, what-if simulation, and rule-based recommendation generation.[4][5]

## Project structure

```text
CashYar-DataGenerator/
├── config.py
├── personas.py
├── seasonal_events.py
├── generate_users.py
├── generate_transactions.py
├── generate_behavior_summary.py
├── recommendation.py
├── main.py
├── utils/
│   ├── helpers.py
│   └── distributions.py
└── data/
    ├── users.csv
    ├── transactions.csv
    └── behavioral_summary.csv
```

## Generated datasets

| File | Approx. rows | Purpose |
|------|-------------:|---------|
| `users.csv` | 5,000 | Student demographic, income, budget, and financial profile records. |
| `transactions.csv` | ~500,000 | Realistic transaction-level spending logs across categories, merchants, timing, and purchase context. |
| `behavioral_summary.csv` | ~60,000 | Monthly AI-ready summaries containing savings, category ratios, budget adherence, risk labels, and recommendation fields. |

## Schema overview

### `users.csv`

Contains the base student profile used to condition behavior generation and train downstream models.

```text
user_id
uuid
age
gender
city
university
major
study_year
living_status
scholarship
part_time_income
other_income
total_income
monthly_budget
financial_goal
goal_amount
target_months
financial_personality
```

### `transactions.csv`

Contains transaction-level behavioral history for each student, including context needed for feature engineering and temporal analysis.

```text
transaction_id
user_id
datetime
category
merchant
amount
payment_method
planned_purchase
online_purchase
purchase_location
season
event
day_of_week
hour
remaining_budget
monthly_spending
monthly_saving
goal_progress
financial_score
financial_risk
recommended_action
```

### `behavioral_summary.csv`

Contains monthly behavior snapshots used directly by the AI engine for training targets, behavioral labels, and explainability-ready features.

```text
user_id
month
total_income
total_spending
total_saving
saving_rate
restaurant_ratio
coffee_ratio
shopping_ratio
transport_ratio
electronics_ratio
entertainment_ratio
weekend_spending_ratio
planned_purchase_ratio
online_purchase_ratio
budget_adherence
financial_score
goal_progress
consecutive_budget_success
spending_trend
financial_personality
target_months
financial_risk
will_reach_goal
behavior_label
recommended_action
```

## Key features

- 10 financial personas such as disciplined, shopaholic, coffee lover, and goal-driven profiles that influence income use, savings behavior, and spending style.
- Saudi-specific seasonality and events such as Ramadan, Eid, exam periods, and White Friday to create realistic demand shifts and spending spikes.
- Realistic merchant and category behavior including restaurants, coffee, shopping, transport, electronics, and entertainment transactions.
- Built-in AI labels such as `will_reach_goal`, `financial_risk`, `behavior_label`, and `recommended_action` for model training and evaluation.
- Monthly budget tracking with rolling financial signals such as remaining budget, monthly spending, monthly saving, and financial score.
- Data suitable for downstream Random Forest regression and classification pipelines that persist models with `joblib`, use `predict_proba` for overspending probability, and expose feature importance for explainability.[6][5][4]

## How this connects to the AI engine

The generated data was designed to plug directly into the CashYar AI prediction engine, which uses the dataset for three core prediction tasks: monthly spending prediction, overspending prediction, and monthly savings prediction.[4][5]

It also supports higher-level product features such as budget runout estimation, goal completion probability, feature-importance explanations, structured recommendations, and what-if simulations like reducing coffee spending or changing the monthly budget.[4][5]

## Quick start

Install dependencies and run the full pipeline:

```bash
pip install -r requirements.txt
python main.py
```

This generates all three output files inside the `data/` directory in one run.[1][3]

## Run individual modules

Use these commands when you want to generate one stage at a time:

```bash
python generate_users.py
python generate_transactions.py
python generate_behavior_summary.py
```

A staged workflow is useful for debugging generation quality, validating intermediate outputs, or regenerating a single dataset after logic changes.[3]

## Configuration

Edit `config.py` to control dataset scale and generation intensity:

```python
NUM_USERS = 5000
TRANSACTIONS_PER_USER_MIN = 80
TRANSACTIONS_PER_USER_MAX = 120
```

Typical configuration changes include increasing the number of users for model training volume, widening transaction ranges for denser histories, or tuning persona and seasonal probabilities for realism.[3]

## Downstream use cases

The generated dataset is intended to support:

- Feature engineering for spending velocity, budget usage, saving rate, goal progress, and category-level ratios.
- Regression targets such as predicted monthly spending and predicted monthly savings.
- Classification targets such as overspending probability and goal completion likelihood.
- Explainability workflows based on important drivers like coffee ratio, shopping ratio, budget adherence, and financial score.[4]
- Rule-based recommendation generation without requiring an LLM at the data layer.

## Example workflow

1. Generate synthetic users with realistic student financial profiles.
2. Generate transaction histories shaped by persona, time, merchant, and seasonal context.
3. Aggregate each user-month into behavioral summaries.
4. Feed the CSV outputs into the CashYar AI training pipeline.
5. Train prediction models and expose results to the app for dashboards, advice, and what-if simulations.[6][5][4]

## Notes

Keep this README focused on setup, dataset structure, and integration, and move any deep implementation details to separate documentation files if the repository grows larger.[1][3]