# Healthcare Operations & Revenue Risk Intelligence Platform

## 📌 Project Overview
Large hospital networks face increasing challenges related to patient flow management, operational inefficiencies, and revenue leakage caused by insurance claim rejections and delayed payments. 

This capstone project is an end-to-end, industry-aligned Machine Learning pipeline and Analytics platform designed to solve these operational and financial challenges. It features two core predictive classification systems:
1. **Visit Risk Classification (Model A):** Predicts whether a hospital visit represents a Low, Medium, or High operational/clinical risk to assist with resource planning and patient prioritization.
2. **Claim Outcome Classification (Model B):** Predicts whether an insurance claim will be Paid, Pending, or Rejected prior to submission to help reduce revenue leakage and improve cash flow predictability.

The project spans the entire data lifecycle, from SQL data engineering and exploratory data analysis (EDA) to model development, evaluation, FastAPI deployment, and drift monitoring.

---

## 🎖️ Evaluation
* **Final Grade:** 100 / 100
* **Reference Visual:** Refer to `Grade.png` for submission details.

---

## 🛠️ Tech Stack
* **Database:** SQLite / SQL
* **Data Processing & EDA:** Python (Pandas, NumPy, Scikit-learn)
* **Machine Learning:** Logistic Regression, Random Forest / Gradient Boosting (SMOTE for class imbalance)
* **Model Deployment:** FastAPI, Docker
* **Monitoring & Governance:** Custom Python monitoring scripts for data drift detection

---

## 📂 Repository Structure & Project Phases

The repository is organized chronologically based on the multi-phase development approach of a real-world ML system:

### `PHASE 1` - SQL Analytics Layer
Establishes the trusted data foundation for downstream modeling.
* `healthcare_capstone.db` - Structured relational database.
* `phase1_sql.py` / `Phase1_SQL.ipynb` - SQL queries for patient flow, financial performance, and data integrity checks.

### `PHASE 2` - Exploratory Data Analysis & Data Quality
* `phase2_eda.py` / `Phase2_EDA.ipynb` - Data exploration and missing value analysis.
* `build_features.py` - Feature engineering script (visit frequency, provider rejection rate, length of stay, etc.).
* `model_table.csv` - The finalized modeling dataset.
* `Data quality report.docx` - Summary of data reliability and outliers.

### `PHASE 3` - Model Development
* **`Model A — Visit Risk Classification/`**
  * `02_risk_model.py` / `02_risk_model.ipynb` - Training pipeline for the Visit Risk model.
  * `feature_schema.json` - Feature expectations for incoming data.
  * `risk_model_joblib.zip` - Zipped model artifact (See *Note on Model Artifacts* below).
* **`Model B — Claim Outcome Classification/`**
  * `03_claim_model.py` / `03_claim_model.ipynb` - Training pipeline for Claim Outcome model.
  * `claim_model.joblib` - Saved claim model artifact.

### `PHASE 4` - Model Evaluation & Explainability
* `phase4_evaluation.py` / `Phase4_Evaluation.ipynb` - Scripts evaluating precision, recall, and F1-scores.
* `Model card document.docx` - Consolidated documentation of model performance, limitations, and demographic fairness.
* `Model Explainability Summary.docx` - Feature importance breakdowns.
* `Risk Model Evaluation Report.docx` & `Claim Model Evaluation Report.docx`

### `PHASE 5` - Model Deployment & API Integration
* `app.py` - FastAPI service for real-time predictions and health checks.
* `DockerFile` - Containerization instructions for the API.
* `Deployment Guide.docx` - Operations runbook.
* `Sample request and response documentation.docx` - API contract schemas.
* `claim_model.joblib` & `risk_model_joblib.zip` - Required artifacts for the API.

### `PHASE 6` - Monitoring, Drift Detection, and Governance
* `Monitoring script.py` - Script for detecting data and prediction drift over time.
* `Drift Detection Report.docx` - Analysis of shifts in production data.
* `Governance and Compliance Document.docx` - AI safety, system limitations, and audit logs.

### `PHASE - FINAL` - Executive Business Presentation
* `Healthcare Insights Detailed Report.docx` & `Healthcare Insights Report.docx` - Translation of technical ML outcomes into leadership-level business ROI.

---

## ⚠️ Important Note on Model Artifacts
Due to GitHub's strict file size limit of 100MB per file, the uncompressed `risk_model.joblib` file (which exceeds 100MB) has been compressed. 
* You will find it saved as **`risk_model_joblib.zip`** inside both the `PHASE 3/Model A — Visit Risk Classification/` folder and the `PHASE 5/` folder. 
* **To run the code locally or build the Docker container:** Please extract `risk_model_joblib.zip` in its respective directory so the `.joblib` file is accessible to the scripts.

---
