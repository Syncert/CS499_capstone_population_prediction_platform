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
    "linear": MODELS_DIR / "linear_model.pkl",
    "ridge":  MODELS_DIR / "ridge_model.pkl",
    "xgb":    MODELS_DIR / "xgb_model.pkl",
    "prophet": MODELS_DIR / "prophet_model.json"
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


def _load_models() -> None:
    MODEL_REGISTRY.clear()
    for name, path in MODEL_PATHS.items():
        if not path.exists():
            continue

        if name == "prophet":
            try:
                txt = path.read_text(encoding="utf-8")
                MODEL_REGISTRY[name] = {"type": "prophet_model", "model": model_from_json(txt)}
            except Exception as e:
                print(f"[WARN] Failed loading Prophet model {path}: {e}")
            continue

        try:
            with path.open("rb") as f:
                obj = pickle.load(f)
            # allow wrapped artifacts: {"model": estimator, "features": [...]}
            MODEL_REGISTRY[name] = obj
        except Exception as e:
            print(f"[WARN] Failed loading model {name} ({path}): {e}")

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
    # Prophet branch (no feature_matrix for X; only for regressors when needed)
    # ----------------------------
    if model_name == "prophet":
        reg = MODEL_REGISTRY.get("prophet")
        if reg is None:
            raise HTTPException(status_code=400, detail="Model 'prophet' is not loaded on server")

        model = reg["model"]  # loaded via model_from_json

        # Find which regressors Prophet expects (if any)
        try:
            required_regs = list(getattr(model, "extra_regressors", {}).keys())
        except Exception:
            required_regs = []

        # Build the future frame
        future = pd.DataFrame({"ds": pd.to_datetime([f"{y}-01-01" for y in years])})
        future["year"] = future["ds"].dt.year  # to join with feature_matrix rows

        if required_regs:
            # Pull required regressors from your feature matrix (by year)
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

            # Light imputations for stability
            future[required_regs] = future[required_regs].ffill().bfill()
            for c in required_regs:
                if future[c].isna().any():
                    future[c].fillna(float(df_feat[c].mean()), inplace=True)

            future = future.drop(columns=["year"])
        else:
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
    # sklearn/xgb branch (uses feature_matrix for X)
    # ----------------------------
    df = _fetch_features_window(geo, req.start_year, req.end_year)
    if df.empty:
        raise HTTPException(status_code=404, detail="No feature rows found for selection")

    IDENT_COLS = {"geo_code", "year"}
    TARGETish = {"population", "y", "target", "label"}

    # choose numeric features, exclude ids/targets
    features = [c for c in df.columns if c not in (IDENT_COLS | TARGETish)]
    features = [c for c in features if pd.api.types.is_numeric_dtype(df[c])]

    if not features:
        raise HTTPException(status_code=400, detail="No numeric feature columns available")

    # coerce to numeric to avoid object/Decimal from SQL
    X_all = df[features].apply(pd.to_numeric, errors="coerce").fillna(0.0)

    m = MODEL_REGISTRY.get(model_name)
    if m is None:
        raise HTTPException(status_code=400, detail=f"Model '{model_name}' is not loaded on server")

    # unwrap dict-wrapped artifacts like {"model": estimator, "features": [...], "rename_map": {...}}
    estimator = m.get("model") if isinstance(m, dict) else m
    if estimator is None or not hasattr(estimator, "predict"):
        raise HTTPException(status_code=400, detail=f"model '{model_name}' is not a valid predictor")

    saved_feats = (m.get("features") if isinstance(m, dict) else None) \
                  or (list(getattr(estimator, "feature_names_in_", [])) or None)

    # optional rename map to align live names to training names
    rename_map = m.get("rename_map") if isinstance(m, dict) else None
    if rename_map:
        X_all = X_all.rename(columns=rename_map)

    if saved_feats:
        missing = [c for c in saved_feats if c not in X_all.columns]
        extra   = [c for c in X_all.columns if c not in saved_feats]
        if missing:
            raise HTTPException(
                status_code=400,
                detail=f"model '{model_name}' missing required features: {missing}; extra provided: {extra}"
            )
        X = X_all.reindex(columns=saved_feats)
    else:
        X = X_all

    X = X.astype("float64")

    try:
        yhat = estimator.predict(X)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"model '{model_name}' predict failed: {e}")

    return PredictResponse(
        geography=geo,
        model=model_name,
        years=df["year"].astype(int).tolist(),
        forecast=[float(v) for v in yhat],
        features_used=list(X.columns),
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