# 🧠 NeuroVision — Student Mental Health Risk Prediction

> End-to-end Data Science project that analyzes student well-being data and builds a supervised classification prototype to estimate mental health risk levels — from data audit to an Airflow-orchestrated pipeline and an interactive Streamlit app.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-pipeline-orange)
![Airflow](https://img.shields.io/badge/Apache-Airflow-017CEE)
![Streamlit](https://img.shields.io/badge/Streamlit-app-FF4B4B)
![Docker](https://img.shields.io/badge/Docker-ready-2496ED)

---

## ⚠️ Disclaimer

This project is **strictly educational** (mock jury brief). The model is **NOT** a diagnostic tool, a clinical screening instrument, a disciplinary decision tool or a student selection system. Predictions must never be used to make decisions about real people.

---

## 📌 Table of Contents

1. [Context & Objectives](#-context--objectives)
2. [Dataset](#-dataset)
3. [Project Structure](#-project-structure)
4. [Pipeline Overview](#-pipeline-overview)
5. [Results](#-results)
6. [Installation](#-installation)
7. [Usage](#-usage)
8. [Airflow DAG](#-airflow-dag)
9. [Streamlit App](#-streamlit-app)
10. [Limitations & Ethics](#-limitations--ethics)
11. [Tech Stack](#-tech-stack)
12. [Author](#-author)

---

## 🎯 Context & Objectives

An educational institution wants to study the factors associated with student well-being and mental health risk, and to build a prototype classifier.

| Objective | Description |
|---|---|
| Risk detection | Identify students with a higher risk of psychological difficulties |
| Prevention | Anticipate support needs |
| Factor analysis | Understand which factors are associated with risk |
| Guidance | Orient students toward appropriate resources |
| Optimization | Prioritize student support actions |

---

## 📊 Dataset

- **Source:** [Mental Health Risk Prediction Dataset – Students (Kaggle)](https://www.kaggle.com/datasets/maqdadali/mental-health-risk-prediction-dataset-students)
- **File:** `data/raw/student_mental_health.csv`
- **Target:** mental health risk level *(update with the exact column name once confirmed)*
- **Rows / columns:** *(fill in after EDA)*

---

## 🗂️ Project Structure

```
NeuroVision/
├── app/                    # Streamlit application
│   └── streamlit_app.py
├── dag/                    # Airflow DAG(s)
│   └── student_risk_pipeline.py
├── data/
│   ├── raw/                # Original dataset (never modified)
│   └── processed/          # Train/test splits (generated)
├── models/                 # Serialized pipeline (.joblib) and metrics
├── notebooks/              # EDA, preprocessing, modeling, tuning
├── reports/
│   └── figures/            # Exported charts
├── src/                    # Reusable source code
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
├── tests/                  # Unit tests
├── Dockerfile
├── docker-compose.yml
├── docker.env
├── requirements.txt
└── README.md
```

> Adjust file names to match your final implementation.

---

## 🔄 Pipeline Overview

1. **Audit & EDA** – structure, missing values, duplicates, inconsistencies, outliers, target distribution, correlations.
2. **Split** – stratified 80 % train / 20 % test, done *before* any fitting.
3. **Preprocessing (leak-free pipeline)**
   - Imputation adapted to each variable type
   - Outlier treatment (IQR / Z-score)
   - Categorical encoding
   - Numerical scaling
   - Class balancing with SMOTE (training data only, if needed)
4. **Modeling** – at least 3 classifiers trained on the same data and compared (accuracy, precision, recall, F1-score, confusion matrix, ROC-AUC).
5. **Optimization** – `GridSearchCV` / `RandomizedSearchCV` on the best models.
6. **Final evaluation** – independent test set, error analysis, limitations.
7. **Serialization** – full pipeline (preprocessing + model) saved with `joblib`.
8. **Automation & UI** – Airflow DAG and Streamlit dashboard.

Every cleaning decision is justified in the EDA / preprocessing notebooks.

---

## 📈 Results

*(Fill in after final evaluation.)*

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | – | – | – | – | – |
| Random Forest | – | – | – | – | – |
| Gradient Boosting / XGBoost | – | – | – | – | – |
| **Final tuned model** | – | – | – | – | – |

**Chosen model:** *(name)* — *(short justification linked to the business context, e.g. priority on recall for the at-risk class)*.

Key risk factors identified: *(top features)*.

---

## ⚙️ Installation

### Option 1 — Docker (recommended)

```bash
git clone https://github.com/ABDELHAFIDaz/NeuroVision.git
cd NeuroVision
docker compose --env-file docker.env up --build
```

Services (check `docker-compose.yml` for the exact ports):

| Service | URL |
|---|---|
| Airflow UI | http://localhost:8080 |
| Streamlit app | http://localhost:8501 |

### Option 2 — Local environment

```bash
git clone https://github.com/ABDELHAFIDaz/NeuroVision.git
cd NeuroVision
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## ▶️ Usage

**Train the model and save the pipeline**

```bash
python -m src.train
```

**Evaluate on the test set**

```bash
python -m src.evaluate
```

**Run the Streamlit app**

```bash
streamlit run app/streamlit_app.py
```

**Load the saved pipeline in Python**

```python
import joblib
pipeline = joblib.load("models/best_pipeline.joblib")
prediction = pipeline.predict(new_data_df)
```

---

## 🛫 Airflow DAG

The DAG `student_risk_pipeline` orchestrates:

```
prepare_data → preprocess → train_model → evaluate_model
```

- Scheduled execution (e.g. weekly)
- Retries and failure callbacks configured
- Status and errors monitored through the Airflow UI and task logs

---

## 🖥️ Streamlit App

- **Overview** – project context and disclaimer
- **Performance** – metrics, model comparison, confusion matrix, ROC curve, feature importance
- **Predict** – single student form and CSV batch prediction

---

## 🚧 Limitations & Ethics

- Educational prototype trained on a limited dataset; it may not generalize to real populations.
- Self-reported or synthetic-like data can contain bias and noise.
- False negatives (missed at-risk students) and false positives (unnecessary alerts) both have real-world consequences.
- Predictions must **never** replace professional psychological evaluation.

---

## 🧰 Tech Stack

Python · pandas · NumPy · scikit-learn · imbalanced-learn · XGBoost · matplotlib · seaborn · joblib · Apache Airflow · Streamlit · Docker
