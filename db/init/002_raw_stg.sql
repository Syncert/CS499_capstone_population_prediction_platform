-- ─────────────────────────────────────────────────────────
-- RAW: External call audit logs
--   One row per upstream HTTP call (endpoint + params + response).
--   Append-only; dedupe by request_hash so safe on retries.
-- ─────────────────────────────────────────────────────────

create table if not exists raw.bls_calls (
  etl_batch_id  varchar(64) not null,
  request_hash  varchar(64) primary key, -- sha256 of {endpoint,params,payload}
  requested_at  timestamptz not null default now(),
  endpoint      text not null,
  params_json   jsonb,
  payload_json  jsonb,
  http_status   int,
  response_json jsonb,
  notes         text
);

create table if not exists raw.fred_calls (like raw.bls_calls including all);
create table if not exists raw.acs_calls  (like raw.bls_calls including all);

create index if not exists ix_raw_bls_calls_requested_at  on raw.bls_calls(requested_at);
create index if not exists ix_raw_fred_calls_requested_at on raw.fred_calls(requested_at);
create index if not exists ix_raw_acs_calls_requested_at  on raw.acs_calls(requested_at);

-- ─────────────────────────────────────────────────────────
-- RAW (normalized rows, optional but useful for lineage/diffing)
--   These mirror upstream “row-level” data so facts can be traced.
-- ─────────────────────────────────────────────────────────

create table if not exists raw.bls_points (
  etl_batch_id varchar(64) not null,
  series_id    varchar(32) not null,
  year         int         not null,
  period       varchar(8),         -- e.g., 'M01'..'M12','M13' (annual avg)
  value        text,
  primary key (series_id, year, period, etl_batch_id)
);

create index if not exists ix_raw_bls_points_year on raw.bls_points(year);

create table if not exists raw.fred_observations (
  etl_batch_id varchar(64) not null,
  series_id    varchar(32) not null,
  obs_date     date        not null,
  raw_value    text,
  primary key (series_id, obs_date, etl_batch_id)
);

create index if not exists ix_raw_fred_obs_date on raw.fred_observations(obs_date);

create table if not exists raw.acs_rows (
  etl_batch_id varchar(64) not null,
  dataset      varchar(8)  not null,  -- 'acs1' | 'acs5'
  year         int         not null,
  geo_level    varchar(16) not null,  -- 'us' | 'state' | 'county'
  state_fips   varchar(2),            -- nullable for 'us'
  geo_code     varchar(32) not null,
  geo_name     text        not null,
  var_code     varchar(32) not null,  -- e.g., 'B01001_001E'
  raw_value    text,
  primary key (dataset, year, geo_code, var_code, etl_batch_id)
);

create index if not exists ix_raw_acs_rows_year  on raw.acs_rows(year);
create index if not exists ix_raw_acs_rows_geo   on raw.acs_rows(geo_code);

-- ─────────────────────────────────────────────────────────
-- STG: Typed, deduped staging (idempotent upserts by business key)
--   These tables are where we cast types, normalize codes, and dedupe.
--   No FKs to keep staging resilient to upstream changes.
-- ─────────────────────────────────────────────────────────

-- LAUS unemployment rate (annual averages per geo/year)
create table if not exists stg.laus_unrate (
  geo_code     varchar(32) not null,
  geo_type     varchar(16) not null,   -- 'state' | 'county'
  year         int         not null,
  unrate       double precision,
  etl_batch_id varchar(64) not null,
  primary key (geo_code, year),
  constraint chk_laus_year_reasonable check (year between 1900 and 2100),
  constraint chk_laus_unrate_bounds   check (unrate is null or (unrate >= 0 and unrate <= 100))
);

create index if not exists ix_stg_laus_year on stg.laus_unrate(year);

-- CPS/BLS national unemployment rate (annual average, US only)
create table if not exists stg.unrate_us_bls (
  year         int primary key,
  value        double precision,
  etl_batch_id varchar(64) not null,
  constraint chk_us_unrate_year check (year between 1900 and 2100),
  constraint chk_us_unrate_bounds check (value is null or (value >= 0 and value <= 100))
);

create index if not exists ix_stg_unrate_us_year on stg.unrate_us_bls(year);

-- FRED CPI (yearly aggregate per series)
create table if not exists stg.cpi_yearly (
  series_id    varchar(32) not null,    -- e.g. 'CPIAUCSL', 'CUSR0000SEHC'
  year         int         not null,
  value        double precision,
  etl_batch_id varchar(64) not null,
  constraint pk_cpi_yearly primary key (series_id, year),
  constraint chk_cpi_year_reasonable check (year between 1900 and 2100),
  constraint chk_cpi_value_nonneg    check (value is null or value >= 0)
);

-- ACS population (both ACS1/ACS5 preserved; canonicalization happens in core load)
create table if not exists stg.acs_population (
  geo_code     varchar(32) not null,
  geo_name     text        not null,
  geo_type     varchar(16) not null,  -- 'nation' | 'state' | 'county'
  year         int         not null,
  source       varchar(8)  not null,  -- 'ACS1' | 'ACS5'
  population   bigint      not null,
  etl_batch_id varchar(64) not null,
  primary key (geo_code, year, source),
  constraint chk_acs_year_reasonable check (year between 1900 and 2100),
  constraint chk_acs_population_pos  check (population >= 0)
);

create index if not exists ix_stg_acs_year on stg.acs_population(year);
create index if not exists ix_stg_acs_geo  on stg.acs_population(geo_code);

-- ─────────────────────────────────────────────────────────
-- Notes
--  • RAW tables: append-only; dedupe on primary keys to tolerate retries.
--  • STG tables: idempotent upserts via (geo_code, year[, source]) keys.
--  • Core loads should read FROM stg.* and write INTO core.* (dims/facts),
--    keeping business rules and FKs out of staging.
-- ─────────────────────────────────────────────────────────
