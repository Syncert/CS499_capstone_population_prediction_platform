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