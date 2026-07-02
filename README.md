# CashYar Data Generator

Synthetic fintech dataset generator for the **CashYar** project — a smart financial assistant that teaches university students money management through behavioral learning (not traditional lessons).

## Project Structure

```
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

## Quick Start

```bash
pip install -r requirements.txt
python main.py
```

## Dataset Output

| Table | Approx. Rows | Description |
|-------|-------------:|-------------|
| users | 5,000 | Saudi university student profiles |
| transactions | ~500,000 | Realistic spending transactions |
| behavioral_summary | 60,000 | Monthly AI-ready behavior summaries |

## Features

- 10 financial personas (Disciplined, Shopaholic, Coffee Lover, etc.)
- Saudi seasonal events (Ramadan, Eid, exams, White Friday)
- Realistic merchant database (AlBaik, Starbucks, Jarir, etc.)
- AI labels: `will_reach_goal`, `financial_risk`, `behavior_label`, `recommended_action`
- Monthly budget engine with financial score tracking

## Run Individual Modules

```bash
python generate_users.py
python generate_transactions.py
python generate_behavior_summary.py
```

## Configuration

Edit `config.py` to change dataset size:

```python
NUM_USERS = 5000
TRANSACTIONS_PER_USER_MIN = 80
TRANSACTIONS_PER_USER_MAX = 120
```
