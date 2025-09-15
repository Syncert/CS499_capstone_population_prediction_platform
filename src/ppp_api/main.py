from fastapi import FastAPI
from pydantic import BaseModel
from sqlalchemy import text
from ppp_common.db import get_engine

app = FastAPI(title="Population Prediction API", version="0.1.0")

@app.get("/health")
def health():
    with get_engine().connect() as conn:
        r = conn.execute(text("SELECT 1")).scalar()
    return {"status": "ok", "db": bool(r == 1)}

class PredictRequest(BaseModel):
    geography: str
    start_year: int
    end_year: int
    model: str | None = None

@app.post("/predict")
def predict(req: PredictRequest):
    return {
        "geography": req.geography,
        "years": list(range(req.start_year, req.end_year + 1)),
        "model": req.model or "linear",
        "forecast": [None] * (req.end_year - req.start_year + 1),
    }

def run():
    import uvicorn
    uvicorn.run("ppp_api.main:app", host="0.0.0.0", port=8000, reload=False)
