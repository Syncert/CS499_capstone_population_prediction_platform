from __future__ import annotations
import os
import time
import pickle
from pathlib import Path
from typing import Optional, List, Dict, Any, Literal, Tuple, cast, overload
import pandas as pd
import jwt
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy import text, bindparam
from sqlalchemy.engine import Engine
from ppp_common.orm import engine as shared_engine
from prophet.serialize import model_from_json
import io, zipfile, datetime as dt


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

ModelName = Literal["linear", "ridge", "xgb", "prophet"]


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

class GeoRow(BaseModel):
    geo_code: str
    geo_name: str
    geo_type: Literal["nation","state","county"]

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

# ---------- Overloads for _roll_forward_predict; depending on conditions it can return a dataframe if part of download bundle call ----------
@overload
def _roll_forward_predict(
    estimator,
    hist_df: pd.DataFrame,
    start_next_year: int,
    end_year: int,
    feature_cols: List[str],
    *,
    use_delta: bool = ...,
    return_rows: Literal[True],
) -> Tuple[List[int], List[float], pd.DataFrame]: ...
# (years, preds, future_df)

@overload
def _roll_forward_predict(
    estimator,
    hist_df: pd.DataFrame,
    start_next_year: int,
    end_year: int,
    feature_cols: List[str],
    *,
    use_delta: bool = ...,
    return_rows: Literal[False] = ...,
) -> Tuple[List[int], List[float]]: ...
# (years, preds)

def _roll_forward_predict(
    estimator,
    hist_df: pd.DataFrame,
    start_next_year: int,
    end_year: int,
    feature_cols: List[str],
    *,
    use_delta: bool = False,
    return_rows: bool = False,
):
    rows = hist_df.copy().sort_values("year").reset_index(drop=True)
    collected: list[dict] = []

    def last_num(col: str, default: float = 0.0) -> float:
        """Return the last numeric value for column `col` in `rows`, else `default`."""
        if col in rows.columns:
            s = pd.to_numeric(rows[col], errors="coerce").dropna()
            if not s.empty:
                return float(s.iloc[-1])
        return float(default)

    def get_pop(yr: int) -> float | None:
        """Return population for a given year from `rows`, if present."""
        if "population" not in rows.columns:
            return None
        s = pd.to_numeric(
            rows.loc[rows["year"].astype(int) == int(yr), "population"],
            errors="coerce",
        ).dropna()
        return float(s.iloc[0]) if not s.empty else None

    years_out: List[int] = []
    preds_out: List[float] = []

    for y in range(int(start_next_year), int(end_year) + 1):
        # exogenous (held flat at last observed)
        unrate = last_num("unemployment_rate", 0.0)
        rent_cpi = last_num("rent_cpi_index", 0.0)

        # lags / moving average from history (actuals + prior yhat)
        lag1 = get_pop(y - 1)
        lag5 = get_pop(y - 5)
        ma_vals = [v for v in (get_pop(y - 2), get_pop(y - 1)) if v is not None]
        ma3 = (sum(ma_vals) / len(ma_vals)) if ma_vals else last_num("population", 0.0)

        feat_row: List[float] = []
        feat_map: dict = {}

        for c in feature_cols:
            if c == "year":
                v = float(y)
            elif c == "unemployment_rate":
                v = float(unrate)
            elif c == "rent_cpi_index":
                v = float(rent_cpi)
            elif c == "pop_lag1":
                v = float(lag1) if lag1 is not None else 0.0
            elif c == "pop_lag5":
                v = float(lag5) if lag5 is not None else 0.0
            elif c == "pop_ma3":
                v = float(ma3)
            elif c == "pop_yoy_growth_pct":
                v = 0.0
            elif c == "pop_cagr_5yr_pct":
                v = 0.0
            else:
                v = last_num(c, 0.0)

            feat_row.append(v)
            feat_map[c] = v

        X_next = pd.DataFrame([feat_row], columns=feature_cols).astype("float64")
        try:
            yhat_raw = float(estimator.predict(X_next.to_numpy(copy=False))[0])
        except Exception:
            yhat_raw = float(estimator.predict(X_next)[0])

        base_next = float(lag1) if (use_delta and lag1 is not None) else 0.0
        yhat_level = yhat_raw + base_next if use_delta else yhat_raw

        years_out.append(int(y))
        preds_out.append(yhat_level)

        # append to history for subsequent steps
        rows = pd.concat(
            [
                rows,
                pd.DataFrame(
                    [
                        {
                            "year": int(y),
                            "population": yhat_level,
                            "unemployment_rate": unrate,
                            "rent_cpi_index": rent_cpi,
                            "pop_lag1": lag1 if lag1 is not None else 0.0,
                            "pop_lag5": lag5 if lag5 is not None else 0.0,
                            "pop_ma3": ma3,
                        }
                    ]
                ),
            ],
            ignore_index=True,
        ).sort_values("year").reset_index(drop=True)

        if return_rows:
            feat_map["year"] = int(y)
            feat_map["population_implied"] = yhat_level
            collected.append(feat_map)

    if return_rows:
        cols = ["year"] + [c for c in feature_cols if c != "year"] + ["population_implied"]
        future_df = pd.DataFrame(collected)[cols] if collected else pd.DataFrame(columns=cols)
        return years_out, preds_out, future_df

    return years_out, preds_out



def _predict(model_name: str, X: pd.DataFrame) -> List[float]:
    m = MODEL_REGISTRY.get(model_name)
    if m is None:
        raise HTTPException(status_code=400, detail=f"Model '{model_name}' is not loaded on server")
    y = m.predict(X)  # scikit/xgboost style
    return [float(v) for v in y]

def _indicator_slice_df(geo: str, start: int, end: int) -> pd.DataFrame:
    q = text("""
      select geo_code, year, indicator_code, value::double precision as value, source, unit
      from core.indicator_values
      where geo_code=:g and year between :y0 and :y1
      order by year, indicator_code
    """)
    return pd.read_sql_query(q, _get_engine(), params={"g": geo, "y0": start, "y1": end})

def _norm_model(model: str | None) -> ModelName:
    if model is None:
        # default behavior: use linear, or look up best via /scorecard if you prefer
        return cast(ModelName, "linear")
    m = model.lower()
    if m not in ("linear", "ridge", "xgb", "prophet"):
        raise HTTPException(status_code=400, detail=f"invalid model: {model}")
    return cast(ModelName, m)

def _predictions_df(geo: str, start: int, end: int, model: str) -> pd.DataFrame:
    lit_m = _norm_model(model)
    pr = PredictRequest(geography=geo, start_year=start, end_year=end, model=lit_m)
    resp = predict(pr)
    years = resp.years
    preds = resp.forecast

    q = text("""
      select year, population::double precision as actual
      from core.population_observations
      where geo_code=:g and year between :y0 and :y1
    """)
    act = pd.read_sql_query(q, _get_engine(), params={"g": geo, "y0": start, "y1": end})
    out = pd.DataFrame({"year": years, "predicted": preds})
    out = out.merge(act, on="year", how="left")
    return out[["year", "predicted", "actual"]]

def _future_feature_rows_for_download(estimator, df_base, feature_order, start_next, end_year, use_delta):
    from typing import Tuple, List, cast
    years_preds_future = cast(
        Tuple[List[int], List[float], pd.DataFrame],
        _roll_forward_predict(
            estimator=estimator,
            hist_df=df_base,
            start_next_year=start_next,
            end_year=end_year,
            feature_cols=feature_order,
            use_delta=use_delta,
            return_rows=True,
        ),
    )
    _, _, future_rows = years_preds_future
    return future_rows

#helper for download bundle
def _as_list(x: Any) -> Optional[List[str]]:
    """Best-effort turn x into a list[str], else None."""
    if x is None:
        return None
    # common iterables: list/tuple/pd.Index/numpy array
    try:
        return [str(v) for v in list(x)]
    except Exception:
        return None

# -----------------------
# Enable CORs Middleware
# -----------------------

app = FastAPI(title=API_TITLE, version=API_VERSION)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",   # dev browser
        "http://127.0.0.1:5173",
        "http://ui:5173"           # container-to-container, just in case
    ],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

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

    #check model features for use of delta
    use_delta = bool(m.get("use_delta")) if isinstance(m, dict) else False

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

    # In-coverage prediction (years that exist in feature_matrix)
    try:
        yhat = estimator.predict(X)
    except Exception:
        yhat = estimator.predict(X.to_numpy(copy=False))

    if use_delta:
        # add back the base for each row (aligned to X)
        base = pd.to_numeric(df.loc[X.index, "pop_lag1"], errors="coerce").fillna(0.0).to_numpy("float64")
        yhat = yhat + base

    yhat = [float(v) for v in yhat]

    years_in = df["year"].astype(int).tolist()
    forecast = yhat

    # If the user asked beyond the last covered year, roll forward
    max_cov_year = max(years_in)
    if req.end_year > max_cov_year:
        # Build a base DF that includes the columns the roll-forward needs.
        # Use the *actual* population from feature_matrix for history up to max_cov_year.
        # (If you prefer to “own” the history, you can swap in your model’s yhat for 'population'.)
        df_base = df.copy()

        # exact feature order the model expects after any renames
        feature_order = list(X.columns)

        extra_years, extra_preds = _roll_forward_predict(
            estimator=estimator,
            hist_df=df_base,                  # <-- name matches helper’s signature
            start_next_year=max_cov_year + 1,
            end_year=int(req.end_year),
            feature_cols=feature_order,
            use_delta=use_delta,
            return_rows=False,
        )

        years_in += extra_years
        forecast += extra_preds

    return PredictResponse(
        geography=geo,
        model=model_name,
        years=years_in,
        forecast=forecast,
        features_used=list(X.columns) if saved_feats is None else saved_feats,
    )


@app.get("/actuals", dependencies=[Depends(_require_auth)], tags=["models"])
def actuals(geo: str, start: int, end: int):
    q = text("""
        select year, population
        from core.population_observations
        where geo_code=:g and year between :y0 and :y1
        order by year
    """)
    df = pd.read_sql_query(q, _get_engine(), params={"g": geo, "y0": start, "y1": end})
    return {"geography": geo,
            "years": df["year"].astype(int).tolist(),
            "population": df["population"].astype(float).tolist()}


# If you persist metrics during training in ml.model_metrics(geo_code, model, mse, mae, mape, trained_at)
@app.get("/scorecard", dependencies=[Depends(_require_auth)])
def scorecard(geo: str):
    q = text("""
                WITH scored AS (
                SELECT r.geo_code, r.model, r.run_id, r.trained_at,
                        h.rmse_test, h.mae_test, h.r2_test
                FROM ml.model_runs r
                LEFT JOIN ml.model_headline h USING (run_id)
                WHERE r.geo_code = :g
                ),
                best_per_model AS (
                SELECT *,
                        ROW_NUMBER() OVER (
                        PARTITION BY geo_code, model
                        ORDER BY rmse_test ASC NULLS LAST, trained_at DESC
                        ) AS rwm
                FROM scored
                ),
                picked AS (
                SELECT * FROM best_per_model WHERE rwm = 1
                )
                SELECT *,
                    ROW_NUMBER() OVER (
                        ORDER BY rmse_test ASC NULLS LAST, trained_at DESC
                    ) AS rank_within_geo_code
                FROM picked
                ORDER BY rank_within_geo_code;
    """)
    df = pd.read_sql_query(q, _get_engine(), params={"g": geo})

    best = df.iloc[0]["model"]
    if not df.empty:
        # make numbers JSON-safe (optional but nice)
        for c in ["rmse_test", "mae_test", "r2_test"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce").astype(float)
        for c in ["rank_within_model", "rank_within_geo_code"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)
        if "trained_at" in df.columns:
            df["trained_at"] = pd.to_datetime(df["trained_at"], errors="coerce") \
                                .dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    return {
        "geography": geo,
        "best_model": best,
        "metrics": df.to_dict(orient="records")
    }

@app.get("/series", dependencies=[Depends(_require_auth)])
def series(geo: str, code: str, start: int, end: int):
    q = text("""
      select year, value::double precision as value
      from core.indicator_values
      where geo_code=:g and indicator_code=:c and year between :y0 and :y1
      order by year
    """)
    df = pd.read_sql_query(q, _get_engine(), params={"g": geo, "c": code, "y0": start, "y1": end})
    return {"geography": geo, "indicator_code": code,
            "years": df["year"].astype(int).tolist(),
            "values": [float(v) if v is not None else None for v in df["value"].tolist()]}

@app.get("/geos/nations", response_model=List[GeoRow], tags=["geography"])
def geos_nations(only_with_data: bool = True):
    q = text("""
      select g.geo_code, g.geo_name, g.geo_type
      from core.geography g
      where g.geo_type='nation'
        and ((:only)=false or exists (
              select 1 from ml.feature_matrix f where f.geo_code=g.geo_code
            ))
      order by g.geo_name
    """)
    with _get_engine().connect() as c:
        df = pd.read_sql_query(q, c, params={"only": only_with_data})
    return df.to_dict(orient="records")

@app.get("/geos/states", response_model=List[GeoRow], tags=["geography"])
def geos_states(only_with_data: bool = True):
    q = text("""
      select g.geo_code, g.geo_name, g.geo_type
      from core.geography g
      where g.geo_type='state'
        and ((:only)=false or exists (
              select 1 from ml.feature_matrix f where f.geo_code=g.geo_code
            ))
      order by g.geo_name
    """)
    with _get_engine().connect() as c:
        df = pd.read_sql_query(q, c, params={"only": only_with_data})
    return df.to_dict(orient="records")

@app.get("/geos/states/{state_fips}/counties", response_model=List[GeoRow], tags=["geography"])
def geos_counties(state_fips: str, only_with_data: bool = True):
    # state_fips must be 2 chars (e.g., '06'); counties are 5-digit FIPS
    q = text("""
      select g.geo_code, g.geo_name, g.geo_type
      from core.geography g
      where g.geo_type='county' and g.geo_code like :prefix || '%'
        and ((:only)=false or exists (
              select 1 from ml.feature_matrix f where f.geo_code=g.geo_code
            ))
      order by g.geo_name
    """)
    with _get_engine().connect() as c:
        df = pd.read_sql_query(q, c, params={"prefix": state_fips, "only": only_with_data})
    return df.to_dict(orient="records")


@app.get("/download/bundle", dependencies=[Depends(_require_auth)], tags=["download"])
def download_bundle(geo: str, start: int, end: int, model: str, include_future: int = 0):
    if start > end:
        raise HTTPException(status_code=400, detail="start must be <= end")

    # Build all file contents first (strings/bytes), then write once.
    files: list[tuple[str, bytes]] = []

    # 1) indicators slice
    indicators = _indicator_slice_df(geo, start, end)
    files.append(("indicators.csv", indicators.to_csv(index=False).encode("utf-8")))

    # 2) base feature matrix window
    matrix = _fetch_features_window(geo, start, end)
    files.append(("feature_matrix.csv", matrix.to_csv(index=False).encode("utf-8")))

    def add_predictions_and_future_csvs(model_key: str):
        # predictions
        try:
            preds = _predictions_df(geo, start, end, model_key)
            fname = "predictions.csv" if model.lower() != "all" else f"predictions_{model_key}.csv"
            files.append((fname, preds.to_csv(index=False).encode("utf-8")))
        except HTTPException as e:
            files.append((f"predictions_{model_key}_ERROR.txt", (str(e.detail) + "\n").encode("utf-8")))
            return

        # optional future rows for non-prophet models
        if include_future and model_key != "prophet":
            df = _fetch_features_window(geo, start, min(end, int(matrix["year"].max()) if not matrix.empty else end))
            if df.empty:
                return
            key = f"{model_key}_{geo}"
            m = MODEL_REGISTRY.get(key)
            if m is None:
                files.append((f"future_feature_matrix_{model_key}_ERROR.txt", b"model artifact not loaded\n"))
                return
            estimator = m.get("model") if isinstance(m, dict) else m
            if estimator is None or not hasattr(estimator, "predict"):
                files.append((f"future_feature_matrix_{model_key}_ERROR.txt", b"invalid estimator\n"))
                return

            use_delta = bool(m.get("use_delta")) if isinstance(m, dict) else False

            TARGETish = {"population", "y", "target", "label"}
            IDENT_COLS = {"geo_code", "year"}
            X_all = df[[c for c in df.columns if c not in (TARGETish | IDENT_COLS)]].copy()
            if isinstance(m, dict) and m.get("rename_map"):
                X_all = X_all.rename(columns=m["rename_map"])

            feat_from_artifact = (m.get("features") if isinstance(m, dict) else None)
            feat_from_model    = getattr(estimator, "feature_names_in_", None)

            feature_order: List[str] = (
                _as_list(feat_from_artifact)
                or _as_list(feat_from_model)
                or [str(c) for c in X_all.columns]   # final fallback
            )

            max_cov_year = int(df["year"].max())
            if end > max_cov_year:
                future_rows = _future_feature_rows_for_download(
                    estimator=estimator,
                    df_base=df.copy(),
                    feature_order=feature_order,
                    start_next=max_cov_year + 1,
                    end_year=end,
                    use_delta=use_delta,
                )
                if not future_rows.empty:
                    files.append((f"future_feature_matrix_{model_key}.csv",
                                  future_rows.to_csv(index=False).encode("utf-8")))

    if model.lower() == "all":
        for mkey in ("linear", "ridge", "xgb", "prophet"):
            add_predictions_and_future_csvs(mkey)
        model_label = "all"
    else:
        add_predictions_and_future_csvs(model.lower())
        model_label = model

    readme = f"""Population Prediction Platform export
Geo: {geo}
Model: {model_label}
Range: {start}-{end}
Files:
- indicators.csv: raw indicator slice (geo/year/indicator_code/value/source/unit)
- feature_matrix.csv: model-ready features for the requested window
- predictions*.csv: predictions by model (with actuals if available)
- future_feature_matrix_*.csv: generated future rows used during roll-forward (if include_future=1)
Generated: {dt.datetime.utcnow().isoformat()}Z
"""
    files.append(("README.txt", readme.encode("utf-8")))

    # === Write ZIP exactly once, then return ===
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for fname, data in files:
            z.writestr(fname, data)

    buf.seek(0)
    fname = f"ppp_{geo}_{model_label}_{start}-{end}.zip"
    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{fname}"'},
    )

# --- indicator compare: ACS (1 & 5), BLS_UNRATE, CPI_SHELTER ---
@app.get("/features/compare", dependencies=[Depends(_require_auth)], tags=["features"])
def features_compare(geo: str, start: int, end: int, model: str = "linear"):
    """
    Returns, for the requested geo/range:
      - ACS1_TOTAL_POP (actuals only)
      - ACS5_TOTAL_POP (actuals only)
      - BLS_UNRATE     (actuals + projected)
      - CPI_SHELTER    (actuals + projected)
    'projected' for BLS/CPI comes from the same roll-forward logic you use for predictions:
    hold exogenous flat at last observation and recompute lags/ma for future years.
    """
    if start > end:
        raise HTTPException(status_code=400, detail="start must be <= end")

    # 1) pull actuals from indicator_values for the 4 codes
    codes = ("ACS1_TOTAL_POP", "ACS5_TOTAL_POP", "BLS_UNRATE", "CPI_SHELTER")

    q = text("""
    select year, indicator_code, value::double precision as value
    from core.indicator_values
    where geo_code = :g
        and indicator_code IN :codes
        and year between :y0 and :y1
    order by year
    """).bindparams(bindparam("codes", expanding=True))

    df = pd.read_sql_query(
        q,
        _get_engine(),
        params={"g": geo, "codes": tuple(codes), "y0": int(start), "y1": int(end)}, # type: ignore
    )

    # Normalize into dicts: {code: {year: value}}
    actual_by_code: Dict[str, Dict[int, float | None]] = {c: {} for c in codes}
    for _, r in df.iterrows():
        actual_by_code[str(r["indicator_code"])][int(r["year"])] = None if pd.isna(r["value"]) else float(r["value"])

    # 2) build projected series for BLS_UNRATE & CPI_SHELTER using your roll-forward
    projections: Dict[str, Dict[int, float]] = {"BLS_UNRATE": {}, "CPI_SHELTER": {}}
    if end > start:
        # we need the estimator + feature order to generate future feature rows
        key = f"{model.lower()}_{geo}"
        m = MODEL_REGISTRY.get(key)
        if m is None and model.lower() != "prophet":
            # fall back to any available estimator for this geo, or skip projection
            for fallback in ("ridge", "xgb", "linear"):
                mk = f"{fallback}_{geo}"
                if mk in MODEL_REGISTRY: 
                    m, model = MODEL_REGISTRY[mk], fallback
                    break
        if m is not None and model.lower() != "prophet":
            estimator = m.get("model") if isinstance(m, dict) else m
            if estimator is not None and hasattr(estimator, "predict"):
                # base DF covering up through last covered year (we only need for feature context)
                df_base = _fetch_features_window(geo, start, min(end, start + 1000))  # generous cap
                if not df_base.empty:
                    # rename + feature order like training
                    X_all = df_base.copy()
                    if isinstance(m, dict) and m.get("rename_map"):
                        X_all = X_all.rename(columns=m["rename_map"])
                    # pick a feature order
                    feat_art = (m.get("features") if isinstance(m, dict) else None)
                    feat_model = getattr(estimator, "feature_names_in_", None)
                    feature_order: List[str] = (
                        _as_list(feat_art) or _as_list(feat_model) or [c for c in X_all.columns if c not in ("geo_code","year")]
                    )
                    max_cov_year = int(df_base["year"].max())
                    # only project if user asked beyond coverage
                    if end > max_cov_year:
                        future_rows = _future_feature_rows_for_download(
                            estimator=estimator,
                            df_base=df_base,
                            feature_order=feature_order,
                            start_next=max_cov_year + 1,
                            end_year=end,
                            use_delta=bool(m.get("use_delta")) if isinstance(m, dict) else False,
                        )
                        # map feature columns to indicator codes
                        col_map = {"unemployment_rate": "BLS_UNRATE", "rent_cpi_index": "CPI_SHELTER"}
                        for col, code in col_map.items():
                            if col in future_rows.columns:
                                for _, r in future_rows.iterrows():
                                    projections[code][int(r["year"])] = float(r[col])

                        #future population estimates
                        pop_proj: Dict[int, float] = {}
                        if not future_rows.empty and "population_implied" in future_rows.columns:
                            for _, r in future_rows.iterrows():
                                y = int(r["year"])
                                v = r["population_implied"]
                                if pd.notna(v):
                                    pop_proj[y] = float(v)

    # 3) shape response across the full [start, end] range
    years = list(range(int(start), int(end) + 1))
    def series_for(code: str, include_projection: bool) -> Dict[str, Any]:
        actual = [actual_by_code.get(code, {}).get(y, None) for y in years]
        projected = [projections.get(code, {}).get(y, None) if include_projection else None for y in years]
        return {"code": code, "years": years, "actual": actual, "projected": projected}

    def series_pop_projected() -> Dict[str, Any]:
        projected = [pop_proj.get(y, None) for y in years]
        return {"code": "POPULATION_IMPLIED", "years": years, "actual": [None]*len(years), "projected": projected}

    return {
        "geography": geo,
        "start": start,
        "end": end,
        "model_for_projection": model,
        "series": [
            series_for("ACS1_TOTAL_POP", include_projection=False),
            series_for("ACS5_TOTAL_POP", include_projection=False),
            series_for("BLS_UNRATE",     include_projection=True),
            series_for("CPI_SHELTER",    include_projection=True),
            series_pop_projected(),   # <-- NEW
        ],
    }

def run():
    import uvicorn
    uvicorn.run("ppp_api.main:app", host="0.0.0.0", port=8000, reload=False)