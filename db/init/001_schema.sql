-- ─────────────────────────────────────────────────────────
-- Schemas
-- ─────────────────────────────────────────────────────────
create schema if not exists raw;
create schema if not exists stg;
create schema if not exists core;
create schema if not exists ml;

-- ─────────────────────────────────────────────────────────
-- Population ground truth
-- ─────────────────────────────────────────────────────────
create table if not exists core.population_observations (
  geo_code varchar(32) not null,
  year     int         not null,
  population bigint,
  primary key (geo_code, year)
);

create index if not exists idx_pop_year on core.population_observations(year);

-- ─────────────────────────────────────────────────────────
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
  primary key (geo_code, year, indicator_code)
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
-- Modeling matrix (pivot the shortlist you’ll start with)
-- Use a MATERIALIZED VIEW so you can refresh after ETL.
-- Add/remove indicators here without touching storage tables.
-- ─────────────────────────────────────────────────────────
create materialized view if not exists ml.feature_matrix as
select
  p.geo_code,
  p.year,
  max(case when i.indicator_code = 'BLS_UNRATE'           then i.value end) as unemployment_rate,
  max(case when i.indicator_code = 'BLS_LFPR'             then i.value end) as labor_force_participation,
  max(case when i.indicator_code = 'CENSUS_BPS_PERMITS'   then i.value end) as building_permits_total,
  max(case when i.indicator_code = 'ACS_MED_HH_INC'       then i.value end) as median_household_income,
  max(case when i.indicator_code = 'CPI_SHELTER'          then i.value end) as rent_cpi_index,
  max(case when i.indicator_code = 'BLS_JOLTS_OPENINGS'   then i.value end) as job_openings_rate
from core.population_observations p
left join core.indicator_values i
  on i.geo_code = p.geo_code and i.year = p.year
group by 1,2;

-- Helpful indexes on the MV (Postgres 12+ supports this)
create index if not exists ix_fm_year on ml.feature_matrix(year);
create index if not exists ix_fm_geo  on ml.feature_matrix(geo_code);

-- If you prefer an always-live VIEW instead of an MV, uncomment below and
-- drop the MV above. The MV is faster for modeling; the VIEW is always fresh.
-- create or replace view ml.features as
-- select * from ml.feature_matrix;

-- ─────────────────────────────────────────────────────────
-- Convenience upsert function (optional)
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