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
-- Modeling matrix (population features + best-available CPI + BLS UNRATE)
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
cg AS ( -- CPI Shelter at exact geo (currently US-only unless you load more)
  SELECT geo_code, year, value::numeric AS cpi_geo
  FROM core.indicator_values
  WHERE indicator_code = 'CPI_SHELTER'
),
cs AS ( -- state CPI for states + counties (county inherits its state)
  SELECT
    p.geo_code AS county_code,
    p.year     AS yr,
    s.value::numeric AS cpi_state
  FROM p
  JOIN core.indicator_values s
    ON s.indicator_code = 'CPI_SHELTER'
   AND s.geo_code = CASE
                      WHEN length(p.geo_code)=5 THEN substr(p.geo_code,1,2) -- county → state
                      WHEN length(p.geo_code)=2 THEN p.geo_code             -- state
                      ELSE 'ZZ'
                    END
   AND s.year = p.year
  WHERE length(p.geo_code) IN (2,5)
),
cu AS ( -- national fallback
  SELECT year AS yr, value::numeric AS cpi_us
  FROM core.indicator_values
  WHERE indicator_code = 'CPI_SHELTER' AND geo_code = 'US'
)
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
  COALESCE(cg.cpi_geo, cs.cpi_state, cu.cpi_us) AS rent_cpi_index
FROM p
LEFT JOIN u  ON u.geo_code = p.geo_code AND u.year = p.year
LEFT JOIN cg ON cg.geo_code = p.geo_code AND cg.year = p.year
LEFT JOIN cs ON cs.county_code = p.geo_code AND cs.yr = p.year
LEFT JOIN cu ON cu.yr = p.year;

-- Helpful indexes on the MV
CREATE INDEX IF NOT EXISTS ix_fm_year ON ml.feature_matrix(year);
CREATE INDEX IF NOT EXISTS ix_fm_geo  ON ml.feature_matrix(geo_code);

-- Required for REFRESH MATERIALIZED VIEW CONCURRENTLY
CREATE UNIQUE INDEX IF NOT EXISTS uq_fm_geo_year
  ON ml.feature_matrix(geo_code, year);


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