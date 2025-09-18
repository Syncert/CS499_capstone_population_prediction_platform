-- ===============================
-- 0) WHAT'S HERE? (catalog peek)
-- ===============================
-- Schemas & tables you care about
SELECT n.nspname AS schema, c.relname AS table_name
FROM pg_class c
JOIN pg_namespace n ON n.oid = c.relnamespace
WHERE c.relkind = 'r'
  AND n.nspname IN ('core','ml','raw','stg')
ORDER BY 1,2;

-- Basic row counts
SELECT 'core.geography'               AS table, count(*) FROM core.geography
UNION ALL
SELECT 'core.population_observations' AS table, count(*) FROM core.population_observations
UNION ALL
SELECT 'core.indicator_values'        AS table, count(*) FROM core.indicator_values
UNION ALL
SELECT 'ml.feature_matrix'            AS table, count(*) FROM ml.feature_matrix;

-- ===============================
-- 1) CORE INTEGRITY CHECKS
-- ===============================
-- Exactly one pop fact per (geo_code, year)?
SELECT geo_code, year, count(*) AS dup_count
FROM core.population_observations
GROUP BY 1,2
HAVING count(*) > 1;

-- No null or non-positive populations?
SELECT count(*) AS bad_population_rows
FROM core.population_observations
WHERE population IS NULL OR population <= 0;

-- Year coverage of canonical pop facts
SELECT min(year) AS min_year, max(year) AS max_year, count(*) AS rows
FROM core.population_observations;

-- ===============================
-- 2) ACS TESTS (ACS1 + ACS5 total population mirrors)
-- indicator_code: 'ACS1_TOTAL_POP' and 'ACS5_TOTAL_POP'
-- ===============================

-- Per-indicator totals and year span
SELECT indicator_code,
       min(year) AS min_year,
       max(year) AS max_year,
       count(*)  AS rows
FROM core.indicator_values
WHERE indicator_code IN ('ACS1_TOTAL_POP','ACS5_TOTAL_POP')
GROUP BY 1
ORDER BY 1;

-- Per year, how many rows of each?
SELECT year, indicator_code, count(*) AS rows
FROM core.indicator_values
WHERE indicator_code IN ('ACS1_TOTAL_POP','ACS5_TOTAL_POP')
GROUP BY 1,2
ORDER BY 1,2;

-- Sanity: values present and > 0
SELECT indicator_code,
       SUM(CASE WHEN value IS NULL OR value <= 0 THEN 1 ELSE 0 END) AS bad_values
FROM core.indicator_values
WHERE indicator_code IN ('ACS1_TOTAL_POP','ACS5_TOTAL_POP')
GROUP BY 1
ORDER BY 1;

-- Coverage by geo_type (join to geography), example for a recent year
-- (Change 2022 to any year you care about.)
SELECT iv.year, iv.indicator_code, g.geo_type, count(*) AS rows
FROM core.indicator_values iv
JOIN core.geography g USING (geo_code)
WHERE iv.indicator_code IN ('ACS1_TOTAL_POP','ACS5_TOTAL_POP')
  AND iv.year = 2022
GROUP BY iv.year, iv.indicator_code, g.geo_type
ORDER BY 1,2,3;

-- Every canonical pop fact is backed by at least one ACS mirror?
SELECT count(*) AS pop_without_acs_mirror
FROM core.population_observations p
LEFT JOIN core.indicator_values iv
  ON iv.geo_code = p.geo_code
 AND iv.year     = p.year
 AND iv.indicator_code IN ('ACS1_TOTAL_POP','ACS5_TOTAL_POP')
WHERE iv.geo_code IS NULL;

-- ===============================
-- 3) BLS LAUS TESTS (unemployment rate, percent)
-- indicator_code: 'BLS_UNRATE'
-- ===============================

-- Yearly row counts
SELECT year, count(*) AS rows
FROM core.indicator_values
WHERE indicator_code = 'BLS_UNRATE'
GROUP BY 1
ORDER BY 1;

-- Range checks (rates should be sane, e.g., 0–50)
SELECT
  min(value) AS min_unrate,
  max(value) AS max_unrate,
  SUM(CASE WHEN value IS NULL OR value < 0 OR value > 50 THEN 1 ELSE 0 END) AS bad_rows
FROM core.indicator_values
WHERE indicator_code = 'BLS_UNRATE';

-- Geo coverage (expect lots of counties)
SELECT g.geo_type, count(*) AS rows
FROM core.indicator_values iv
JOIN core.geography g USING (geo_code)
WHERE iv.indicator_code = 'BLS_UNRATE'
GROUP BY 1
ORDER BY 1;

-- ===============================
-- 4) FRED CPI TESTS (shelter CPI index)
-- indicator_code: 'CPI_SHELTER'
-- ===============================

-- Who/what/when for CPI shelter
SELECT min(year) AS min_year, max(year) AS max_year, count(*) AS rows
FROM core.indicator_values
WHERE indicator_code = 'CPI_SHELTER';

-- Units & source check (should be 'index' and 'FRED' if you loaded that way)
SELECT source, unit, count(*) AS rows
FROM core.indicator_values
WHERE indicator_code = 'CPI_SHELTER'
GROUP BY 1,2;

-- Expect it to be US-only (unless you later add regional CPIs)
SELECT array_agg(DISTINCT geo_code) AS distinct_geos
FROM core.indicator_values
WHERE indicator_code = 'CPI_SHELTER';

-- ===============================
-- 5) DUPLICATES & FRESHNESS (all indicators)
-- ===============================

-- Duplicates in indicator mirrors? (should be none due to PK/unique)
SELECT geo_code, year, indicator_code, count(*) AS dup_count
FROM core.indicator_values
GROUP BY 1,2,3
HAVING count(*) > 1;

-- Most recent year loaded per indicator
SELECT indicator_code, max(year) AS latest_year, count(*) AS total_rows
FROM core.indicator_values
GROUP BY 1
ORDER BY 1;

-- ===============================
-- 6) FEATURE MATRIX TESTS (ml.feature_matrix)
-- Columns expected (based on your MV): unemployment_rate, labor_force_participation,
-- building_permits_total, median_household_income, rent_cpi_index, job_openings_rate
-- ===============================

-- Row parity: ML feature matrix should have one row per pop fact (same key count)
SELECT
  (SELECT count(*) FROM ml.feature_matrix)            AS fm_rows,
  (SELECT count(*) FROM core.population_observations) AS population_rows;

-- Non-null coverage per year for a couple of key features
-- (swap/add columns as you wire more indicators)
SELECT
  year,
  count(*)                                   AS rows,
  count(unemployment_rate)                   AS unrate_nonnull,
  count(rent_cpi_index)                      AS cpi_shelter_nonnull
FROM ml.feature_matrix
GROUP BY year
ORDER BY year;

-- Unique key on MV exists for concurrent refresh (should return one row)
SELECT indexname, indexdef
FROM pg_indexes
WHERE schemaname = 'ml' AND tablename = 'feature_matrix'
  AND indexname = 'uq_fm_geo_year';

-- ===============================
-- 7) OPTIONAL: RAW LANDING (if you created raw tables)
-- ===============================
-- Does raw ACS landing exist? (returns NULL if it doesn't)
SELECT to_regclass('raw.acs1_total_population') AS raw_acs1_table;

-- If it exists, count rows:
-- SELECT count(*) FROM raw.acs1_total_population;
