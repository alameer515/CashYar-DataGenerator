from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "models" / "src"))

from predict import CashYarPredictor
from clustering import predict_cluster
from simulator import CashYarSimulator

app = FastAPI(title="CashYar AI API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
transactions = pd.read_csv(DATA_DIR / "transactions.csv")
behavior = pd.read_csv(DATA_DIR / "behavioral_summary.csv")

transactions["datetime"] = pd.to_datetime(transactions["datetime"])
transactions["user_id"] = transactions["user_id"].astype(str)
behavior["user_id"] = behavior["user_id"].astype(str)

predictor = CashYarPredictor()
simulator = CashYarSimulator()


class SimulateRequest(BaseModel):
    user_id: int
    current_day: int
    month: str
    action: str
    action_value: float


@app.get("/")
def root():
    return {"status": "CashYar AI API is running"}


@app.get("/predict/{user_id}")
def predict(user_id: int, day: int = 15, month: str = "2026-01"):
    try:
        user_id_str = str(user_id)

        current_tx = transactions[
            (transactions["user_id"] == user_id_str) &
            (transactions["datetime"].dt.to_period("M").astype(str) == month) &
            (transactions["datetime"].dt.day <= day)
        ].copy()

        if current_tx.empty:
            raise HTTPException(status_code=404, detail="No transactions found")

        return predictor.predict(user_id_str, day, current_tx)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/persona/{user_id}")
def get_persona(user_id: int):
    try:
        user_id_str = str(user_id)

        user_rows = behavior[behavior["user_id"] == user_id_str]
        if user_rows.empty:
            raise HTTPException(status_code=404, detail="No behavioral summary found")

        user_behavior = user_rows.iloc[0].to_dict()
        return predict_cluster(user_behavior)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/simulate")
def simulate(req: SimulateRequest):
    try:
        user_id_str = str(req.user_id)

        current_tx = transactions[
            (transactions["user_id"] == user_id_str) &
            (transactions["datetime"].dt.to_period("M").astype(str) == req.month) &
            (transactions["datetime"].dt.day <= req.current_day)
        ].copy()

        if current_tx.empty:
            raise HTTPException(status_code=404, detail="No transactions found for simulation")

        result = simulator.simulate(
            user_id=user_id_str,
            current_day=req.current_day,
            current_month_transactions=current_tx,
            actions={req.action: req.action_value}
        )
        return result

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/summary/{user_id}")
def get_summary(user_id: int, month: str = "2026-01", day: int = 15):
    try:
        user_id_str = str(user_id)

        current_tx = transactions[
            (transactions["user_id"] == user_id_str) &
            (transactions["datetime"].dt.to_period("M").astype(str) == month) &
            (transactions["datetime"].dt.day <= day)
        ].copy()

        if current_tx.empty:
            raise HTTPException(status_code=404, detail="No transactions found")

        user_rows = behavior[behavior["user_id"] == user_id_str]
        if user_rows.empty:
            raise HTTPException(status_code=404, detail="No behavioral summary found")

        prediction = predictor.predict(user_id_str, day, current_tx)
        persona = predict_cluster(user_rows.iloc[0].to_dict())

        return {
            "user_id": user_id_str,
            "month": month,
            "day": day,
            "prediction": prediction,
            "persona": persona,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))