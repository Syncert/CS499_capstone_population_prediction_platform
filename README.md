# syncert.github.io


# Population Prediction Platform (CS 499 Capstone Project)

## 📖 Project Overview
This project builds on my artifact from **CS 370 – Emerging Trends in Computer Science**, where I implemented a Q-learning algorithm to solve a treasure maze. The capstone transforms that prototype into a **full-fledged Population Prediction Platform**, combining **software engineering, predictive algorithms, and database design**.  

The platform forecasts U.S. population trends using multiple algorithms, provides professional-quality visualizations, and supports both **web and desktop deployments** (via React and Tauri). It integrates with the U.S. Census API and leverages modern data engineering and machine learning practices.

---

## 🎯 Goals
- **Software Engineering & Design**  
  Transform the original Jupyter notebook prototype into a containerized, scalable application available via web and desktop.

- **Algorithms & Data Structures**  
  Expand beyond Q-learning to incorporate **Linear Regression, Time Series Forecasting (Prophet)**, and **Deep Q-Learning** for predictive modeling.

- **Databases**  
  Upgrade from local data objects to a **PostgreSQL relational database** with ETL pipelines (Polars/Pandas), ensuring clean, validated, and query-efficient population data.

---

## 🗂️ Artifact Enhancements by Category

### 1. Software Engineering & Design
- **Enhancement**: Build a containerized web + desktop application with interactive charts (React + Chart.js, Tauri).  
- **Outcomes Demonstrated**:  
  - Collaborative environments (Docker deployment).  
  - Professional-quality visuals (React front-end).  
  - Secure architecture (auth, hardened containers).  

### 2. Algorithms & Data Structures
- **Enhancement**: Implement multiple predictive algorithms and evaluate performance trade-offs.  
- **Outcomes Demonstrated**:  
  - Algorithmic design and evaluation.  
  - Use of innovative libraries (Scikit-Learn, Prophet, PyTorch).  
  - Communication of trade-offs (accuracy vs. efficiency).  

### 3. Databases
- **Enhancement**: Develop a relational database schema, load Census data via ETL, and integrate with API + models.  
- **Outcomes Demonstrated**:  
  - Normalized schema with indexes for performance.  
  - SQLAlchemy ORM integration.  
  - Security practices (RBAC, query sanitization).  

---

## 🧩 Course Outcomes Alignment
This project demonstrates all five CS 499 outcomes:

1. **Collaboration** – Containerization + shared database enables team use.  
2. **Communication** – React/Chart.js visualizations present results clearly.  
3. **Computing Solutions** – Balances trade-offs across algorithms and architectures.  
4. **Innovative Tools** – Modern ML, ETL, and container frameworks.  
5. **Security Mindset** – Authentication, container hardening, role-based DB access.  

---

## 📊 System Design Visualization

This section presents the three key architecture views that support the platform design.  
Each visualization highlights a different perspective of the system.

---

### 🛠️ From CS-370 Prototype to Capstone
Planned enhancements that extend the original Q-learning artifact into a full platform.

![Capstone Design Document](artifacts/images/Capstone_Design_Document.png)

---

### 🏗️ System Architecture
High-level view of external data sources, ETL, storage, training, API, and UI layers.

![System Architecture](artifacts/images/System_Architecture.png)

---

### 🔄 Request → Predict → Visualize
Sequence of interactions from user input to prediction to chart rendering.

![Request → Predict → Visualize](artifacts/images/Request_Predict_Visualize.png)

---

### 📈 ETL Training Loop
Data ingestion, staging, feature matrix refresh, and model training flow.

![ETL Training Loop](artifacts/images/ETL_Training_Loop.png)

---

👉 [Download the full Capstone Design Document (PDF)](artifacts/images/Capstone_Design_Document.pdf)

---

## 🚀 Roadmap
This project will be completed over **6 weeks**, with deliverables including:
1. Environment setup & ETL pipeline.  
2. PostgreSQL schema + data ingestion.  
3. Predictive algorithms (linear regression, forecasting, deep Q-learning).  
4. FastAPI back-end with authentication.  
5. React/Tauri front-end with charts.  
6. Final containerized stack, documentation, and presentation.  

---

## 📂 Repository Structure
```plaintext
/PopulationPredictionPlatform
├── data/         # Raw & processed Census data
├── models/       # Trained model artifacts
├── notebooks/    # Prototyping and experimentation
├── api/          # FastAPI service
├── ui/           # React + Tauri front-end
├── db/           # Postgres schema + migrations
├── etl/          # ETL scripts (Python jobs, pipelines)
├── src/          # ETL scripts (Python jobs, pipelines)
│   └── ppp_api/    # storage of code for api
│   └── ppp_common/ # storage of code for common-use across platform
│   └── ppp_etl/    # storage of all etl scripts
├── artifacts/    # Documentation and original .ipynb file
│   └── images/   # Exported diagrams and figures
├── docker-compose.yml
└── requirements.txt
```

---

# 📦 Data Sources & Indicators

## ACS — American Community Survey (Census)

**What it is**  
Annual survey from the U.S. Census Bureau, with two releases:

- **ACS1 (1-year):** Timelier; available for geographies with population ≥ 65k (nation, states, large counties/places).
- **ACS5 (5-year):** Pooled over 5 years; covers all standard geographies (nation, states, all counties, many places).

**What we load**

- **Total population (`B01001_001E`):**
  - Indicator mirrors: `ACS1_TOTAL_POP`, `ACS5_TOTAL_POP` in `core.indicator_values` *(unit: count, source: ACS)*  
  - Canonical facts: one population per `(geo_code, year)` in `core.population_observations`

**Rule of thumb**

- Use **ACS1** for nation/states (more current).
- Use **ACS5** for counties (universal coverage).

**Geography keys**

- Nation: `US`
- States: 2-digit FIPS
- Counties: 5-digit FIPS

**Why it matters**  
ACS establishes the baseline population by geography and time, anchoring the forecasting models.

---

## BLS — Local Area Unemployment Statistics (LAUS) (States & Counties)

**What it is**  
Labor force statistics for states and counties from the Bureau of Labor Statistics.

**What we load**

- **Unemployment rate (annual average, percent):** `BLS_UNRATE` in `core.indicator_values` *(unit: percent, source: BLS)*

**Series ID patterns & examples**

- **State (seasonally adjusted, annual avg):** `LASST{SS}…003`  
  Example (CA): `LASST060000000000003`
- **County (not seasonally adjusted, annual avg):** `LAUCN{SS}{CCC}…0003`  
  Example (Los Angeles County, CA): `LAUCN060370000000003`

> Notes  
> • Annual values prefer BLS’s **M13** (annual average) observations; if absent, we compute the calendar-year mean of monthly points.  
> • Coverage begins in **2009** to align with ACS availability used elsewhere.

**Why it matters**  
Unemployment reflects economic opportunity, a key driver of migration and regional population change.

---

## BLS — CPS Headline Unemployment (Nation)

**What it is**  
The national unemployment rate from the Current Population Survey (CPS), via the BLS API.

**What we load**

- **Unemployment rate (U.S.)** (seasonally adjusted, annualized): `BLS_UNRATE` with `geo_code='US'`  
  Series: `LNS14000000` (SA monthly → calendar-year mean) *(unit: percent, source: BLS)*

**Why it matters**  
Provides national coverage to complement LAUS state/county series, ensuring all three levels (US/state/county) are populated for modeling.

---

## FRED — CPI Shelter Index (U.S. + Census Regions)

**What it is**  
CPI sub-index for **Shelter**, sourced from BLS and distributed via FRED. We load **U.S.** and the **four Census regions**, then annualize.

**What we load**

- **Shelter CPI (NSA monthly → annual mean):** stored as `CPI_SHELTER` in `core.indicator_values` *(unit: index (1982–84=100), source: FRED)*
  - **U.S.:** `CUUR0000SAH1` → `geo_code='US'`
  - **Northeast:** `CUUR0100SAH1` → `geo_code='R1'`
  - **Midwest:** `CUUR0200SAH1` → `geo_code='R2'`
  - **South:** `CUUR0300SAH1` → `geo_code='R3'`
  - **West:** `CUUR0400SAH1` → `geo_code='R4'`

> Notes  
> • We use **NSA** (`CUUR…`) series and compute calendar-year means. If you prefer seasonally adjusted inputs, swap to `CUSR…` equivalents.  
> • `core.geography` includes a `region` type for `R1–R4`.

**What Shelter CPI measures**

- **Concept:** Price of housing services consumed where people live (not a house price index).
- **Components:** Rent of primary residence, Owners’ Equivalent Rent (OER), lodging away from home (small share).
- **Not included:** Home purchase prices, mortgage rates, property taxes, transaction costs.
- **Behavior:** Lags spot market rents due to lease renewals and OER methodology.

**Why it matters**  
Housing cost pressure shapes migration and location decisions. Adding Shelter CPI introduces a cost-of-living dimension to forecasts.

---

# 🧮 Feature Matrix (`ml.feature_matrix`)

**Purpose**  
A unified view aligning population, labor market, and housing-cost signals per `(geo_code, year)` for modeling.

**Inputs**

- `core.population_observations` (canonical population)
- `core.indicator_values`:
  - `BLS_UNRATE` (US via CPS; states & counties via LAUS)
  - `CPI_SHELTER` (US + regions `R1–R4` via FRED)

**CPI fallback logic**

1. Use CPI at the **exact geo** if present.  
2. Else for **states & counties**, map the state FIPS → **region (R1–R4)** and use regional CPI.  
3. Else fall back to **US** CPI.

**Region mapping**

- **R1 Northeast:** CT(09), ME(23), MA(25), NH(33), RI(44), VT(50), NJ(34), NY(36), PA(42)  
- **R2 Midwest:** IL(17), IN(18), MI(26), OH(39), WI(55), IA(19), KS(20), MN(27), MO(29), NE(31), ND(38), SD(46)  
- **R3 South (incl. DC=11):** AL(01), AR(05), DE(10), DC(11), FL(12), GA(13), KY(21), LA(22), MD(24), MS(28), NC(37), OK(40), SC(45), TN(47), TX(48), VA(51), WV(54)  
- **R4 West:** AK(02), AZ(04), CA(06), CO(08), HI(15), ID(16), MT(30), NV(32), NM(35), OR(41), UT(49), WA(53), WY(56)

**Columns (selected)**

- `population` (level value), `pop_lag1`, `pop_lag5`, `pop_ma3` (3-yr moving avg)  
- `pop_yoy_growth_pct` (YoY % growth when `year-1` exists)  
- `pop_cagr_5yr_pct` (5-yr CAGR when `year-5` exists)  
- `unemployment_rate` (BLS)  
- `rent_cpi_index` (CPI Shelter with geo→region→US fallback)

**Geographies**

- Nation: `US`  
- Regions: `R1` (Northeast), `R2` (Midwest), `R3` (South), `R4` (West)  
- States: 2-digit FIPS  
- Counties: 5-digit FIPS

---

# 🔎 Validation Tips

**Row parity & duplicates**
```sql
select count(*) from ml.feature_matrix;

select geo_code, year, count(*) c
from ml.feature_matrix
group by 1,2
having count(*) > 1;
```