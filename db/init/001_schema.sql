-- ─────────────────────────────────────────────────────────
-- Schemas
-- ─────────────────────────────────────────────────────────
create schema if not exists raw;
create schema if not exists stg;
create schema if not exists core;
create schema if not exists ml;

-- ─────────────────────────────────────────────────────────
-- core.geography
--
-- Reference table for names/types (for clean joins later)
-- ─────────────────────────────────────────────────────────

create table if not exists core.geography (
  geo_code varchar(32) primary key,
  geo_name text not null,
  geo_type varchar(16) not null   -- e.g. 'nation','state','county','place'
);

-- ─────────────────────────────────────────────────────────
-- core.population_observations
--
-- Population ground truth
-- ─────────────────────────────────────────────────────────
create table if not exists core.population_observations (
  geo_code varchar(32) not null,
  year     int         not null,
  population bigint    not null,
  primary key (geo_code, year),
  foreign key (geo_code) references core.geography(geo_code),
  constraint chk_population_positive check (population >= 0),
  constraint chk_year_reasonable check (year between 1900 and 2100)
);

create index if not exists idx_pop_year on core.population_observations(year);
create index if not exists ix_pop_geo_year on core.population_observations(geo_code, year); -- /actuals

-- ─────────────────────────────────────────────────────────
-- core.indicator_values
--
-- Long/tidy indicator store (minimal schema churn)
-- One row per (geo, year, indicator_code)
-- ─────────────────────────────────────────────────────────
create table if not exists core.indicator_values (
  geo_code       varchar(32) not null,
  year           int         not null,
  indicator_code varchar(64) not null,   -- e.g., 'BLS_UNRATE','ACS_MED_HH_INC'
  value          numeric,
  source         varchar(32),            -- 'BLS','ACS','CENSUS_BPS','FRED'
  unit           varchar(32),            -- 'percent','index','count', etc.
  etl_batch_id   varchar(64),
  updated_at     timestamptz default now(),
  primary key (geo_code, year, indicator_code),
  foreign key (geo_code) references core.geography(geo_code)
);

create index if not exists ix_indic_year on core.indicator_values(year);
create index if not exists ix_indic_code on core.indicator_values(indicator_code);
create index if not exists ix_indic_geo on core.indicator_values(geo_code);
create index if not exists ix_indic_geo_code_year on core.indicator_values(geo_code, year, indicator_code);


-- Optional: enforce known codes via a lookup (nice but not required on day 1)
create table if not exists core.indicator_catalog (
  indicator_code varchar(64) primary key,
  description    text,
  source         varchar(32),
  unit           varchar(32)
);

-- ─────────────────────────────────────────────────────────
-- Modeling matrix (population + BLS UNRATE + CPI Shelter w/ state→region→US fallback)
-- ─────────────────────────────────────────────────────────
DROP MATERIALIZED VIEW IF EXISTS ml.feature_matrix;

CREATE MATERIALIZED VIEW ml.feature_matrix AS
WITH p AS (
  SELECT
    geo_code,
    year,
    population::numeric AS population,
    LAG(population,1) OVER (PARTITION BY geo_code ORDER BY year) AS pop_lag1,
    LAG(population,5) OVER (PARTITION BY geo_code ORDER BY year) AS pop_lag5,
    AVG(population)  OVER (PARTITION BY geo_code ORDER BY year
                           ROWS BETWEEN 2 PRECEDING AND CURRENT ROW) AS pop_ma3
  FROM core.population_observations
),
u AS (  -- BLS unemployment
  SELECT geo_code, year, value::numeric AS unemployment_rate
  FROM core.indicator_values
  WHERE indicator_code = 'BLS_UNRATE'
),
cg AS ( -- CPI Shelter at exact geo (e.g., 'US'; metros would match here if loaded)
  SELECT geo_code, year, value::numeric AS cpi_geo
  FROM core.indicator_values
  WHERE indicator_code = 'CPI_SHELTER'
),
cs AS ( -- CPI fallback for states & counties via Census regions (R1–R4)
  SELECT
    p.geo_code AS geo_code,
    p.year     AS yr,
    r.value::numeric AS cpi_state_or_region
  FROM p
  LEFT JOIN core.indicator_values r
    ON r.indicator_code = 'CPI_SHELTER'
   AND r.geo_code = CASE
         -- Northeast
         WHEN substr(p.geo_code,1,2) IN ('09','23','25','33','44','50','34','36','42') THEN 'R1'
         -- Midwest
         WHEN substr(p.geo_code,1,2) IN ('17','18','26','39','55','19','20','27','29','31','38','46') THEN 'R2'
         -- South (incl. DC=11)
         WHEN substr(p.geo_code,1,2) IN ('01','05','10','11','12','13','21','22','24','28','37','40','45','47','48','51','54') THEN 'R3'
         -- West
         WHEN substr(p.geo_code,1,2) IN ('02','04','06','08','15','16','30','32','35','41','49','53','56') THEN 'R4'
         ELSE NULL
       END
   AND r.year = p.year
  WHERE (p.geo_code ~ '^\d{2}$' OR p.geo_code ~ '^\d{5}$') -- only states/counties
),
cu AS ( -- national fallback
  SELECT year AS yr, value::numeric AS cpi_us
  FROM core.indicator_values
  WHERE indicator_code = 'CPI_SHELTER' AND geo_code = 'US'
),
joined AS (
  SELECT
    p.geo_code,
    p.year,
    p.population,
    p.pop_lag1,
    p.pop_lag5,
    p.pop_ma3,
    CASE WHEN p.pop_lag1 IS NOT NULL AND p.pop_lag1 <> 0
         THEN 100.0 * (p.population - p.pop_lag1) / p.pop_lag1 END AS pop_yoy_growth_pct,
    CASE WHEN p.pop_lag5 IS NOT NULL AND p.pop_lag5 <> 0
         THEN 100.0 * (POWER(p.population / p.pop_lag5, 1.0/5) - 1.0) END AS pop_cagr_5yr_pct,
    u.unemployment_rate,
    COALESCE(cg.cpi_geo, cs.cpi_state_or_region, cu.cpi_us) AS rent_cpi_index
  FROM p
  LEFT JOIN u  ON u.geo_code = p.geo_code AND u.year = p.year
  LEFT JOIN cg ON cg.geo_code = p.geo_code AND cg.year = p.year
  LEFT JOIN cs ON cs.geo_code = p.geo_code AND cs.yr = p.year
  LEFT JOIN cu ON cu.yr = p.year
),
with_flags AS (
  SELECT
    j.*,
    -- "Full features" = target + regressors + the features your models rely on
    (j.population IS NOT NULL
     AND j.unemployment_rate IS NOT NULL
     AND j.rent_cpi_index   IS NOT NULL
     AND j.pop_lag1         IS NOT NULL
     AND j.pop_ma3          IS NOT NULL
     AND j.pop_lag5         IS NOT NULL
     AND j.pop_yoy_growth_pct IS NOT NULL
     AND j.pop_cagr_5yr_pct   IS NOT NULL) AS has_full_features
  FROM joined j
),
cutins AS (
  SELECT
    geo_code,
    MIN(year) FILTER (WHERE has_full_features) AS first_full_year
  FROM with_flags
  GROUP BY geo_code
)
SELECT f.*
FROM with_flags f
JOIN cutins s USING (geo_code)
WHERE f.has_full_features
  AND f.year >= GREATEST(2005, s.first_full_year)  -- never before 2005, and only after full coverage begins
ORDER BY f.geo_code, f.year;

-- Helpful indexes
CREATE INDEX IF NOT EXISTS feature_matrix_geo_year_idx ON ml.feature_matrix (geo_code, year);
CREATE INDEX IF NOT EXISTS feature_matrix_year_geo_idx ON ml.feature_matrix (year, geo_code);

-- ─────────────────────────────────────────────────────────
-- object: ml.model_artifacts (TABLE)
-- purpose:
--   One row per (geo_code, model). Acts as the "pointer" to the latest and/or
--   best training run. Used by drift detection and by serving layers to decide
--   which run to read.
-- grain:
--   (geo_code, model) unique
-- key columns:
--   geo_code TEXT, model TEXT
-- data ownership:
--   owned by ML training pipeline
-- write pattern:
--   upsert after each successful training; updates pointers latest_run_id/best_run_id
-- read pattern:
--   API / scoring jobs read current pointers; governance dashboards read to audit freshness
-- retention:
--   permanent (tiny table)
-- dependencies:
--   references ml.model_runs via latest_run_id / best_run_id (not FK-enforced)
-- caveats:
--   Do not store per-run metrics here; use ml.model_runs + ml.model_metrics
-- ─────────────────────────────────────────────────────────


-- one row per (geo, model family) — pointer to runs
create table if not exists ml.model_artifacts (
  geo_code       text not null,
  model          text not null,     -- 'prophet' | 'linear' | 'ridge' | 'xgb'
  data_hash      text not null,
  trained_at     timestamptz not null default now(),
  rows           int,
  year_min       int,
  year_max       int,
  artifact_path  text,
  notes          text,

  -- new: pointer(s)
  latest_run_id  uuid,              -- last successful training
  best_run_id    uuid,              -- best by your chosen metric (e.g., RMSE on test)

  primary key (geo_code, model)
);

create extension if not exists pgcrypto; -- for gen_random_uuid()

-- ─────────────────────────────────────────────────────────
-- object: ml.model_runs (TABLE)
-- purpose:
--   Append-only registry of every training run; captures params, environment,
--   data hash, split config, and artifact path.
-- grain:
--   one row per run (run_id UUID)
-- key columns:
--   run_id UUID primary key
-- write pattern:
--   insert at run start; optionally update duration_ms/artifact_path at finish
-- read pattern:
--   audit, reproducibility, leaderboard joins
-- retention:
--   permanent (keeps lineage)
-- dependencies:
--   referenced by ml.model_metrics, ml.model_forecasts
-- indexes:
--   (geo_code, model, trained_at desc) to speed “latest runs” queries
-- ─────────────────────────────────────────────────────────


create table if not exists ml.model_runs (
  run_id        uuid primary key default gen_random_uuid(),
  geo_code      text not null,
  model         text not null,
  data_hash     text not null,
  trained_at    timestamptz not null default now(),
  rows          int,
  year_min      int,
  year_max      int,
  split_year    int,                 -- e.g., 2020 for holdout
  horizon       int,                 -- forecast horizon used
  train_rows    int,
  test_rows     int,
  duration_ms   int,
  artifact_path text,
  params        jsonb,               -- hyperparams, feature flags
  env           jsonb,               -- python version, package versions, git commit
  notes         text
);

create index if not exists ix_runs_geo_model_time
  on ml.model_runs(geo_code, model, trained_at desc);

-- ─────────────────────────────────────────────────────────
-- object: ml.model_metrics (TABLE)
-- purpose:
--   Normalized metric store (rmse/mae/mape/r2, train/val/test/cv, optional folds).
-- grain:
--   (run_id, metric, scope, fold) unique; fold null means aggregate
-- write pattern:
--   bulk insert per run after evaluation
-- read pattern:
--   dashboards & selection logic (e.g., best_run_id)
-- retention:
--   permanent (tiny)
-- dependencies:
--   references ml.model_runs(run_id)
-- ─────────────────────────────────────────────────────────


create table if not exists ml.model_metrics (
  run_id   uuid not null references ml.model_runs(run_id) on delete cascade,
  metric   text not null,             -- 'rmse','mae','mape','r2', etc.
  scope    text not null,             -- 'train'|'val'|'test'|'cv'
  fold     int  not null default -1,  -- -1 = aggregate, 0..k-1 = CV folds
  value    double precision not null,
  primary key (run_id, metric, scope, fold)
);

create index if not exists ix_metrics_run on ml.model_metrics(run_id); 

-- ─────────────────────────────────────────────────────────
-- object: ml.model_headline (MATERIALIZED VIEW)
-- purpose:
--   Convenience aggregation of “headline” metrics per run for fast ranking.
-- refresh:
--   REFRESH MATERIALIZED VIEW CONCURRENTLY ml.model_headline;
-- read pattern:
--   leaderboard, artifact pointer updates
-- caveats:
--   Must be refreshed after inserting metrics
-- ─────────────────────────────────────────────────────────


create materialized view if not exists ml.model_headline as
  select run_id,
         max(value) filter (where metric='r2' and scope='test') as r2_test,
         min(value) filter (where metric='rmse' and scope='test') as rmse_test,
         min(value) filter (where metric='mae' and scope='test')  as mae_test
  from ml.model_metrics
  group by run_id;

create unique index if not exists ux_ml_model_headline_run
  on ml.model_headline(run_id);

-- ─────────────────────────────────────────────────────────
-- object: ml.model_forecasts (TABLE)
-- purpose:
--   Snapshot of predictions per run (and optional actuals for backtests).
-- grain:
--   (run_id, ds) unique (yearly; or change ds to int if you prefer)
-- write pattern:
--   bulk insert per run after forecasting
-- read pattern:
--   API-serving (via best_run_id), backtesting, error analysis
-- retention:
--   keep; if size grows, partition by year
-- dependencies:
--   references ml.model_runs(run_id)
-- indexes:
--   (geo_code, model, ds) for explorations
-- ─────────────────────────────────────────────────────────


create table if not exists ml.model_forecasts (
  run_id     uuid not null references ml.model_runs(run_id) on delete cascade,
  geo_code   text not null,
  model      text not null,
  ds         date not null,                 -- or int year if you prefer
  yhat       double precision not null,
  yhat_lo    double precision,
  yhat_hi    double precision,
  actual     double precision,              -- nullable if future
  primary key (run_id, ds)
);

create index if not exists ix_forecasts_geo_model_ds
  on ml.model_forecasts(geo_code, model, ds);

-- ─────────────────────────────────────────────────────────
-- object: ml.model_leaderboard (VIEW)
-- purpose:
--   Rank runs within (geo_code, model) by test RMSE (then recency).
-- refresh:
--   plain view; always up-to-date (depends on ml.model_headline being refreshed)
-- read pattern:
--   picking best runs; QA
-- ─────────────────────────────────────────────────────────


create view ml.model_leaderboard as
select r.geo_code, r.model, r.run_id, r.trained_at,
       h.rmse_test, h.mae_test, h.r2_test,
       row_number() over (
         partition by r.geo_code, r.model
         order by h.rmse_test asc nulls last, r.trained_at desc
       ) as rank_within_model,
       row_number() over (
         partition by r.geo_code
         order by h.rmse_test asc nulls last, r.trained_at desc
       ) as rank_within_geo_code
from ml.model_runs r
left join ml.model_headline h using (run_id);


-- ─────────────────────────────────────────────────────────
-- Upsert function
-- Lets you idempotently load indicator rows from ETL.
-- ─────────────────────────────────────────────────────────
create or replace function core.upsert_indicator_value(
  p_geo_code varchar,
  p_year int,
  p_indicator_code varchar,
  p_value numeric,
  p_source varchar,
  p_unit varchar,
  p_batch varchar
) returns void language plpgsql as $$
begin
  insert into core.indicator_values
    (geo_code, year, indicator_code, value, source, unit, etl_batch_id)
  values
    (p_geo_code, p_year, p_indicator_code, p_value, p_source, p_unit, p_batch)
  on conflict (geo_code, year, indicator_code)
  do update set
    value = excluded.value,
    source = excluded.source,
    unit = excluded.unit,
    etl_batch_id = excluded.etl_batch_id,
    updated_at = now();
end;
$$;