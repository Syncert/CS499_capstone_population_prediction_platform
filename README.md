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
/PopulationPredictionPlatform
│── /data # Raw & processed Census data
│── /models # Trained model artifacts
│── /notebooks # Prototyping and experimentation
│── /api # FastAPI service
│── /ui # React + Tauri front-end
│── /db # Postgres schema + migrations
│── /etl # ETL scripts (Python jobs, pipelines)
|── /artifacts #documentation and original .ipynb file
│   └── images/          # Exported diagrams and figures
│── docker-compose.yml
│── requirements.txt


