from __future__ import annotations
import os
import time
import pickle
from pathlib import Path
from typing import Optional, List, Dict, Any, Literal
import pandas as pd
import jwt
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field
from sqlalchemy import text
from sqlalchemy.engine import Engine
from ppp_common.orm import engine as shared_engine
from prophet.serialize import model_from_json

# -----------------------
# Config
# -----------------------
API_TITLE = "Population Prediction API"
API_VERSION = "0.2.0"

API_USERNAME = os.getenv("API_USERNAME", "admin")
API_PASSWORD = os.getenv("API_PASSWORD", "changeme")
API_JWT_SECRET = os.getenv("API_JWT_SECRET", "super-secret-key")
API_JWT_TTL_SECONDS = int(os.getenv("API_JWT_TTL_SECONDS", "3600"))

MODELS_DIR = Path(os.getenv("MODELS_DIR", "models"))
MODEL_PATHS = {
    "linear": lambda geo: MODELS_DIR / "linear" / f"{geo}.pkl",
    "ridge":  lambda geo: MODELS_DIR / "ridge" / f"{geo}.pkl",
    "xgb":    lambda geo: MODELS_DIR / "xgb" / f"{geo}.pkl",
    "prophet":lambda geo: MODELS_DIR / "prophet" / f"{geo}.json",
}

IDENT_COLS = {"geo_code", "year"}  # keep these out of feature matrix

# -----------------------
# App / Security
# -----------------------
app = FastAPI(title=API_TITLE, version=API_VERSION)
bearer = HTTPBearer()

MODEL_REGISTRY: Dict[str, Any] = {}
DB_ENGINE: Optional[Engine] = None

# -----------------------
# Schemas (keep your old PredictRequest shape)
# -----------------------
class LoginRequest(BaseModel):
    username: str
    password: str

class TokenResponse(BaseModel):
    access_token: str
    token_type: Literal["bearer"] = "bearer"
    expires_in: int

class PredictRequest(BaseModel):
    geography: str = Field(..., description="FIPS or 'US'")
    start_year: int
    end_year: int
    model: Optional[Literal["linear", "ridge", "xgb", "prophet"]] = None

# Optional explicit request if you want a list-based variant later:
class PredictRequestExplicit(BaseModel):
    geo_code: str
    years: List[int]
    model: Literal["linear", "ridge", "xgb"] = "linear"

class PredictResponse(BaseModel):
    geography: str
    model: str
    years: List[int]
    forecast: List[float]
    features_used: List[str]

# -----------------------
# Helpers
# -----------------------
def _issue_jwt(username: str) -> str:
    now = int(time.time())
    payload = {"sub": username, "iat": now, "exp": now + API_JWT_TTL_SECONDS}
    return jwt.encode(payload, API_JWT_SECRET, algorithm="HS256")

def _require_auth(creds: HTTPAuthorizationCredentials = Depends(bearer)) -> str:
    token = creds.credentials
    try:
        decoded = jwt.decode(token, API_JWT_SECRET, algorithms=["HS256"])
        return decoded["sub"]
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")

def _get_engine() -> Engine:
    global DB_ENGINE
    if DB_ENGINE is None:
        DB_ENGINE = shared_engine
    return DB_ENGINE


# in ppp_api.main
def _load_models() -> None:
    MODEL_REGISTRY.clear()

    # sklearn/xgb by geo
    for family in ("linear", "ridge", "xgb"):
        d = MODELS_DIR / family
        if not d.exists():
            continue
        for p in d.glob("*.pkl"):
            stem = p.stem  # e.g., "10" or "linear_10"
            if "_" in stem:
                model, geo = stem.split("_", 1)  # ("linear", "10")
            else:
                model, geo = family, stem        # ("linear", "10")
            key = f"{model}_{geo}"               # "linear_10"
            try:
                MODEL_REGISTRY[key] = pickle.load(p.open("rb"))
            except Exception as e:
                print(f"[WARN] load fail {p}: {e}")

    # prophet by geo
    proph_dir = MODELS_DIR / "prophet"
    if proph_dir.exists():
        for p in proph_dir.glob("prophet_model_*.json"):
            geo = p.stem.split("prophet_model_")[-1]
            try:
                MODEL_REGISTRY[f"prophet_{geo}"] = {
                    "type":"prophet_model",
                    "model": model_from_json(p.read_text("utf-8"))
                }
            except Exception as e:
                print(f"[WARN] load fail {p}: {e}")


def _fetch_features_window(geo_code: str, start_year: int, end_year: int) -> pd.DataFrame:
    q = text("""
        select * from ml.feature_matrix
        where geo_code = :geo and year between :y0 and :y1
        order by year
    """)
    eng = _get_engine()
    return pd.read_sql_query(q, eng, params={"geo": geo_code, "y0": start_year, "y1": end_year})

def _predict(model_name: str, X: pd.DataFrame) -> List[float]:
    m = MODEL_REGISTRY.get(model_name)
    if m is None:
        raise HTTPException(status_code=400, detail=f"Model '{model_name}' is not loaded on server")
    y = m.predict(X)  # scikit/xgboost style
    return [float(v) for v in y]

# -----------------------
# Startup
# -----------------------
@app.on_event("startup")
def _startup() -> None:
    _load_models()
    try:
        with _get_engine().connect() as conn:
            conn.execute(text("select 1"))
    except Exception as e:
        print(f"[WARN] DB not reachable on startup: {e}")

# -----------------------
# Routes
# -----------------------
@app.post("/login", response_model=TokenResponse, tags=["auth"])
def login(req: LoginRequest) -> TokenResponse:
    if not (req.username == API_USERNAME and req.password == API_PASSWORD):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")
    token = _issue_jwt(req.username)
    return TokenResponse(access_token=token, expires_in=API_JWT_TTL_SECONDS)

@app.get("/health", tags=["ops"])
def health():
    with _get_engine().connect() as conn:
        r = conn.execute(text("SELECT 1")).scalar()
    return {"status": "ok", "db": bool(r == 1), "models_loaded": list(MODEL_REGISTRY.keys())}

@app.get("/metrics", dependencies=[Depends(_require_auth)], tags=["models"])
def metrics():
    # Keep your lightweight DB sanity metric; you can expand to read models/metrics.csv later
    with _get_engine().connect() as conn:
        n = conn.execute(text("SELECT count(*) FROM core.population_observations")).scalar_one()
    return {"population_rows": int(n)}

@app.post("/predict", response_model=PredictResponse, dependencies=[Depends(_require_auth)], tags=["models"])
def predict(req: PredictRequest):
    geo = req.geography
    model_name = (req.model or "linear").lower()
    years = list(range(int(req.start_year), int(req.end_year) + 1))

    # ----------------------------
    # Prophet: per-geo model (JSON) + regressors from feature_matrix
    # ----------------------------
    if model_name == "prophet":
        key = f"prophet_{geo}"
        reg = MODEL_REGISTRY.get(key)
        if reg is None:
            raise HTTPException(status_code=404, detail=f"No Prophet model for geo={geo}. Expected models/prophet/prophet_model_{geo}.json")

        model = reg["model"]  # from model_from_json

        # Prophet expects ds and any extra regressors used at fit time
        try:
            required_regs = list(getattr(model, "extra_regressors", {}).keys())
        except Exception:
            required_regs = []

        future = pd.DataFrame({"ds": pd.to_datetime([f"{y}-01-01" for y in years])})
        future["year"] = future["ds"].dt.year

        if required_regs:
            df_feat = _fetch_features_window(geo, req.start_year, req.end_year)
            missing_in_feat = [c for c in required_regs if c not in df_feat.columns]
            if missing_in_feat:
                raise HTTPException(
                    status_code=400,
                    detail=f"prophet prediction failed: regressors missing from feature_matrix: {missing_in_feat}"
                )

            grp = (df_feat.groupby("year", as_index=False)[required_regs]
                         .mean(numeric_only=True))
            future = future.merge(grp, on="year", how="left")

            # light imputations for stability
            future[required_regs] = future[required_regs].ffill().bfill()
            for c in required_regs:
                if future[c].isna().any():
                    future[c].fillna(float(df_feat[c].mean()), inplace=True)

        future = future.drop(columns=["year"])

        try:
            fcst = model.predict(future)
            yhat = [float(v) for v in fcst["yhat"].tolist()]
            return PredictResponse(
                geography=geo,
                model="prophet",
                years=years,
                forecast=yhat,
                features_used=required_regs,
            )
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"prophet prediction failed: {e}")

    # ----------------------------
    # Sklearn/XGB: per-geo estimator (pickle) + feature_matrix window
    # ----------------------------
    df = _fetch_features_window(geo, req.start_year, req.end_year)
    if df.empty:
        raise HTTPException(status_code=404, detail="No feature rows found for selection")

    IDENT_COLS = {"geo_code", "year"}
    TARGETish  = {"population", "y", "target", "label"}

    features = [c for c in df.columns if c not in (IDENT_COLS | TARGETish)]
    features = [c for c in features if pd.api.types.is_numeric_dtype(df[c])]
    if not features:
        raise HTTPException(status_code=400, detail="No numeric feature columns available")

    # coerce to numeric to avoid object/Decimal
    X_all = df[features].apply(pd.to_numeric, errors="coerce").fillna(0.0)

    key = f"{model_name}_{geo}"
    m = MODEL_REGISTRY.get(key)
    if m is None:
        raise HTTPException(status_code=404, detail=f"No {model_name} model for geo={geo}. Expected models/{model_name}/{geo}.pkl")

    # unwrap dict artifacts: {"model": estimator, "features": [...], "rename_map": {...}}
    estimator = m.get("model") if isinstance(m, dict) else m
    if estimator is None or not hasattr(estimator, "predict"):
        raise HTTPException(status_code=400, detail=f"model '{model_name}' for geo={geo} is not a valid predictor")

    saved_feats = (m.get("features") if isinstance(m, dict) else None) \
                  or (list(getattr(estimator, "feature_names_in_", [])) or None)

    rename_map = m.get("rename_map") if isinstance(m, dict) else None
    if rename_map:
        X_all = X_all.rename(columns=rename_map)

    if saved_feats:
        missing = [c for c in saved_feats if c not in X_all.columns]
        extra   = [c for c in X_all.columns if c not in saved_feats]
        if missing:
            raise HTTPException(
                status_code=400,
                detail=f"model '{model_name}' for geo={geo} missing required features: {missing}; extra provided: {extra}"
            )
        X = X_all.reindex(columns=saved_feats)
    else:
        X = X_all

    X = X.astype("float64")

    # Some XGB wrappers behave best with numpy arrays in exact order
    try:
        yhat = estimator.predict(X)
    except Exception:
        yhat = estimator.predict(X.to_numpy(copy=False))

    return PredictResponse(
        geography=geo,
        model=model_name,
        years=df["year"].astype(int).tolist(),
        forecast=[float(v) for v in yhat],
        features_used=list(X.columns) if saved_feats is None else saved_feats,
    )



# #DEBUG REMOVE LATER
# @app.get("/debug/model/{name}", dependencies=[Depends(_require_auth)])
# def debug_model(name: str):
#     m = MODEL_REGISTRY.get(name)
#     if m is None:
#         return {"loaded": False}
#     est = m.get("model") if isinstance(m, dict) else m
#     return {
#         "loaded": True,
#         "wrapped": isinstance(m, dict),
#         "saved_features": (m.get("features") if isinstance(m, dict) else None),
#         "feature_names_in_": list(getattr(est, "feature_names_in_", [])) if hasattr(est, "feature_names_in_") else None,
#         "n_features_in_": getattr(est, "n_features_in_", None),
#         "estimator_type": type(est).__name__,
#     }


def run():
    import uvicorn
    uvicorn.run("ppp_api.main:app", host="0.0.0.0", port=8000, reload=False)