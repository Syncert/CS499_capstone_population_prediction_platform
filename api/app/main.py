from fastapi import FastAPI
from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.engine import create_engine
import os

DATABASE_URL = os.getenv("DATABASE_URL")
engine = create_engine(DATABASE_URL, pool_pre_ping=True)

app = FastAPI(title="Population Prediction API", version="0.1.0")

@app.get("/health")
def health():
    with engine.connect() as conn:
        r = conn.execute(text("SELECT 1")).scalar()
    return {"status": "ok", "db": bool(r == 1)}

class PredictRequest(BaseModel):
    geography: str
    start_year: int
    end_year: int
    model: str | None = None  # e.g., "linear" | "prophet" | "dqn"

@app.post("/predict")
def predict(req: PredictRequest):
    # stub response for now; you’ll wire in real models later
    return {
        "geography": req.geography,
        "years": list(range(req.start_year, req.end_year + 1)),
        "model": req.model or "linear",
        "forecast": [None] * (req.end_year - req.start_year + 1)
    }