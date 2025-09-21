from __future__ import annotations

TARGET_COL = "population"
YEAR_COL   = "year"

# canonical feature set from ml.feature_matrix
BASE_FEATURES: list[str] = [
    "pop_lag1", "pop_lag5", "pop_ma3",
    "pop_yoy_growth_pct", "pop_cagr_5yr_pct",
    "unemployment_rate", "rent_cpi_index",
]