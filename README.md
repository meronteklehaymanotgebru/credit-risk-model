# Credit Risk Scoring Model – Bati Bank & Xente

**End‑to‑end credit risk product using alternative data (eCommerce transactions)**  
*10 Academy Week 4 Challenge*

---

## Table of Contents
1. [Business Context](#business-context)  
2. [Project Structure](#project-structure)  
3. [Task 1 – Credit Scoring Business Understanding](#task-1--credit-scoring-business-understanding)  
4. [Task 2 – Exploratory Data Analysis (EDA)](#task-2--exploratory-data-analysis-eda)  
5. [Task 3 – Feature Engineering](#task-3--feature-engineering)  
6. [Task 4 – Proxy Target Variable (RFM Clustering)](#task-4--proxy-target-variable-rfm-clustering)  
7. [Task 5 – Model Training & Tracking](#task-5--model-training--tracking)  
8. [Task 6 – Deployment & CI/CD](#task-6--deployment--cicd)  
9. [Setup & Usage](#setup--usage)  
10. [Testing](#testing)  
11. [Limitations & Future Work](#limitations--future-work)  
12. [References](#references)

---

## Business Context
Bati Bank is partnering with an eCommerce platform to offer a **buy‑now‑pay‑later (BNPL)** service. To do this responsibly, the bank needs a credit scoring model that can predict a customer’s likelihood of default. Because the dataset contains **no default labels**, we engineer a **proxy target** from transactional behaviour (Recency, Frequency, Monetary – RFM) and build a machine‑learning pipeline that outputs a **risk probability score**.

The model is built to meet the interpretability and documentation requirements of the **Basel II Capital Accord**. The entire workflow is version‑controlled (Git, DVC optional), tracked with **MLflow**, containerized with **Docker**, and tested automatically via **GitHub Actions CI**.

---

## Project Structure
credit-risk-model/
├── .github/workflows/ci.yml # CI/CD pipeline (flake8 + pytest)
├── data/
│ └── raw/ # Raw transaction data
├── notebooks/
│ └── eda.ipynb # Exploratory analysis
├── src/
│ ├── init.py
│ ├── data_processing.py # Feature engineering pipeline
│ ├── proxy_target.py # RFM clustering → is_high_risk target
│ ├── train.py # Model training, tuning, MLflow tracking
│ ├── predict.py # (Optional) inference helper
│ └── api/
│ ├── init.py
│ ├── main.py # FastAPI application
│ └── pydantic_models.py # Request/response schemas
├── tests/
│ ├── init.py
│ └── test_data_processing.py # Unit tests
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .gitignore
└── README.md

---

## Task 1 – Credit Scoring Business Understanding

### Basel II Accord & Interpretability
The Basel II Accord demands that internal rating systems be **conceptually sound, predictive, and well‑documented**. Models used for credit risk must be explainable to regulators. This motivates our use of **Logistic Regression with Weight‑of‑Evidence (WoE)** as a baseline, while also testing higher‑performance models (Random Forest, XGBoost) with interpretability tools (SHAP/feature importance) to justify any uplift.

### Proxy Variable
Because the dataset has **no default label**, we engineered a proxy target using **RFM segmentation** (Task 4). The cluster with the lowest engagement (high recency, low frequency, low monetary) is labelled **high‑risk**. This is a **modelling assumption**, not an observed default. The risks include misclassification and a potential self‑fulfilling prophecy.

### Trade‑offs: Interpretable vs. High‑Performance Models
| Aspect | Interpretable (Logistic Regression + WoE) | High‑Performance (Gradient Boosting) |
|--------|-------------------------------------------|--------------------------------------|
| Explainability | Coefficients directly show feature impact | SHAP values needed; more complex |
| Regulatory acceptance | High | Requires supplementary justification |
| Performance | May be sufficient with good feature engineering | Often achieves higher accuracy |
| Stability | More stable, less prone to overfitting | Requires careful tuning and monitoring |

---

## Task 2 – Exploratory Data Analysis (EDA)
- **Dataset**: 95,662 transactions, 16 columns. No missing values.
- **Amount/Value** are extremely skewed; outliers were capped at the 99th percentile.
- **Top product categories**: `financial_services`, `airtime`.
- **Channel 3** dominates; `FraudResult` is highly imbalanced (0.2% fraud).
- **Temporal patterns**: transactions from Nov 2018 – Feb 2019 with weekly seasonality.

**Top 5 insights for feature engineering**:
1. Customer‑level aggregation (sum, mean, std, count) is essential.
2. Separate debit and credit amounts; engineer ratios.
3. Extract time features (hour, day, month, recency).
4. Cap outliers before scaling.
5. Aggregate fraud flags per customer as a weak risk signal.

*(Full notebook: `notebooks/eda.ipynb`)*

---

## Task 3 – Feature Engineering
A scikit‑learn `Pipeline` (`src/data_processing.py`) transforms raw transactions into customer‑level features:
- **TransactionAggregator**: groups by `CustomerId` and computes monetary, frequency, recency, temporal, and categorical mode features.
- **Winsorizer**: caps numeric features at the 99th percentile.
- **Preprocessor**: imputes missing values (median for numeric, most‑frequent for categorical), scales with `StandardScaler`, and one‑hot encodes categorical columns.
- The pipeline preserves `CustomerId` for merging the target later.

---

## Task 4 – Proxy Target Variable (RFM Clustering)
The target `is_high_risk` is created in `src/proxy_target.py`:
1. **RFM metrics** are calculated per customer (Recency = days since last transaction, Frequency = transaction count, Monetary = net amount sum).
2. RFM features are scaled and clustered with **K‑Means (k=3)**.
3. The cluster with the **highest recency, lowest frequency, and lowest monetary** is labelled as high‑risk (`is_high_risk = 1`).
4. The target is merged back into the processed feature set.

---

## Task 5 – Model Training & Tracking
Script: `src/train.py`

- **Data split**: 80/20, stratified, `random_state=42`.
- **Models trained**: Logistic Regression, Random Forest, XGBoost.
- **Hyperparameter tuning**: `GridSearchCV` with 5‑fold cross‑validation.
- **Metrics logged**: Accuracy, Precision, Recall, F1, ROC‑AUC.
- All experiments are tracked with **MLflow**; the best model is registered in the **MLflow Model Registry** as `CreditRiskModel`.

**Model Comparison (on test set)**:
| Model               | Accuracy | Precision | Recall  | F1      | ROC‑AUC |
|---------------------|----------|-----------|---------|---------|---------|
| LogisticRegression  | 0.9960   | 1.0000    | 0.9896  | 0.9948  | 1.0000  |
| RandomForest        | 1.0000   | 1.0000    | 1.0000  | 1.0000  | 1.0000  |
| XGBoost             | 1.0000   | 1.0000    | 1.0000  | 1.0000  | 1.0000  |

> **Note**: The very high metrics are due to the proxy target being derived from the same behavioural features used for training. This indicates that the model easily separates the RFM‑defined clusters, but it does **not** guarantee genuine default prediction. A future validation step with real repayment data is essential.

The **Random Forest** was registered as the best model (version 1).

---

## Task 6 – Deployment & CI/CD

### REST API (FastAPI)
- `src/api/main.py`: Loads the registered model from MLflow and serves a `/predict` endpoint.
- Input/Output validated with **Pydantic** models.

### Containerization
- **Dockerfile**: Python 3.12‑slim base, installs dependencies, copies the project, runs `uvicorn`.
- **docker‑compose.yml**: Exposes port 8000, mounts the `mlruns` directory.

### CI/CD Pipeline (`.github/workflows/ci.yml`)
On every push to `main` or `task-*` branches:
1. Lint code with `flake8`.
2. Run unit tests with `pytest`.
3. Build fails if linter or tests fail.

---

## Setup & Usage

### Local Installation
```bash
git clone <your-repo-url>
cd credit-risk-model
python -m venv venv
source venv/bin/activate   # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### Train the Model
```bash
python src/train.py          # runs feature pipeline, proxy target creation, training, MLflow logging
```

### Start the API
```bash
uvicorn src.api.main:app --reload
```
The API is then available at http://127.0.0.1:8000.

### Sample Request
```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"data": [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]}'
```

**Response:**
```json
{
  "risk_probability": 0.0023,
  "risk_category": "low risk",
  "model_version": "CreditRiskModel v1"
}
```

### Docker
```bash
docker-compose up --build
```

### Testing
```bash
pytest tests/ -v
```
Runs two tests for the data processing pipeline (aggregator output columns, full pipeline execution).

---

## Limitations & Future Work

* **Proxy target**: `is_high_risk` is not a true default label. The model should be validated against actual repayment data when available.
* **Data leakage**: Because the target is derived from the same transaction data as the features, the current metrics are optimistically high. In production, the target must come from an external source.
* **Feature engineering**: WoE / IV transformation was deferred; it can be added for the logistic regression model to improve interpretability.
* **Model monitoring**: Once deployed, model performance should be monitored for drift.
* **Regulatory compliance**: Additional documentation and stress‑testing would be required for an IRB‑compliant model.

---

## References

* Basel II Accord – Bank for International Settlements
* HKMA Alternative Credit Scoring Guidelines
* World Bank Credit Scoring Approaches
* Xente Challenge on Kaggle
* MLflow Documentation
* FastAPI Documentation

---
