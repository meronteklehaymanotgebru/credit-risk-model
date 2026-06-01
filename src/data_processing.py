"""Feature engineering pipeline for credit risk modelling.

Transforms raw eCommerce transaction data into customer‑level features
ready for model training.  All steps are chained in a single scikit‑learn
Pipeline for reproducibility.

**Note on WoE / IV:** Weight‑of‑Evidence transformation requires a binary target,
which is not available until the proxy target (is_high_risk) is engineered
in Task 4.  WoE encoding will be applied in the training script (src/train.py)
after the target has been created.  This avoids data leakage.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn import set_config

# Keep outputs as DataFrames for easier inspection
set_config(transform_output="pandas")


# ---------------------------------------------------------------------------
# 1. Custom aggregator – transaction → customer level
# ---------------------------------------------------------------------------
class TransactionAggregator(BaseEstimator, TransformerMixin):
    """Aggregate transaction data to one row per customer."""

    def __init__(self, amount_col="Amount", value_col="Value",
                 date_col="TransactionStartTime", id_col="CustomerId"):
        self.amount_col = amount_col
        self.value_col = value_col
        self.date_col = date_col
        self.id_col = id_col

    def fit(self, X, y=None):
        self.feature_names_in_ = X.columns.tolist()
        return self

    def transform(self, X):
        df = X.copy()

        # Ensure datetime
        if not pd.api.types.is_datetime64_any_dtype(df[self.date_col]):
            df[self.date_col] = pd.to_datetime(df[self.date_col], utc=True)

        # Separate debits and credits
        df["is_debit"] = df[self.amount_col] > 0
        df["is_credit"] = df[self.amount_col] < 0
        df["debit_amount"] = df[self.amount_col].clip(lower=0)
        df["credit_amount"] = df[self.amount_col].clip(upper=0).abs()

        # Group by customer and compute aggregates
        grouped = df.groupby(self.id_col).agg(
            # Monetary
            total_amount=("Amount", "sum"),
            mean_amount=("Amount", "mean"),
            std_amount=("Amount", "std"),
            total_value=("Value", "sum"),
            mean_value=("Value", "mean"),
            std_value=("Value", "std"),
            # Credit / Debit
            total_debit=("debit_amount", "sum"),
            total_credit=("credit_amount", "sum"),
            net_cashflow=("Amount", "sum"),
            credit_debit_ratio=("credit_amount", "sum"),   # will be divided later
            # Frequency
            transaction_count=("TransactionId", "count"),
            # Time features
            hour_mean=("TransactionStartTime", lambda x: x.dt.hour.mean()),
            day_mean=("TransactionStartTime", lambda x: x.dt.day.mean()),
            month_mean=("TransactionStartTime", lambda x: x.dt.month.mean()),
            year_mean=("TransactionStartTime", lambda x: x.dt.year.mean()),
            # Recency helper
            last_transaction_date=("TransactionStartTime", "max"),
            # Categorical mode
            most_used_channel=("ChannelId", lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan),
            most_used_product_category=("ProductCategory", lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan),
            most_used_pricing_strategy=("PricingStrategy", lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan),
            # Fraud
            fraud_count=("FraudResult", "sum"),
            fraud_ratio=("FraudResult", "mean"),
        ).reset_index()

        # Snapshot date: day after the last transaction in the dataset
        snapshot_date = df[self.date_col].max().normalize() + pd.Timedelta(days=1)
        grouped["recency_days"] = (snapshot_date - grouped["last_transaction_date"]).dt.days

        # Compute credit/debit ratio safely
        grouped["credit_debit_ratio"] = np.where(
            grouped["total_debit"] > 0,
            grouped["total_credit"] / grouped["total_debit"],
            0
        )

        # Drop intermediate columns that are not useful for modeling
        # KEEP CustomerId for merging the proxy target later
        drop_cols = ["last_transaction_date", "total_debit", "total_credit", "net_cashflow"]
        grouped = grouped.drop(columns=drop_cols, errors="ignore")

        return grouped


# ---------------------------------------------------------------------------
# 2. Outlier capper (winsorize at 99th percentile)
# ---------------------------------------------------------------------------
def winsorize_values(X, upper_percentile=99):
    """Cap numeric columns at the given percentile."""
    X = X.copy()
    for col in X.select_dtypes(include=[np.number]).columns:
        cap = np.percentile(X[col].dropna(), upper_percentile)
        X[col] = np.clip(X[col], None, cap)
    return X


Winsorizer = FunctionTransformer(winsorize_values, validate=False)


# ---------------------------------------------------------------------------
# 3. Build the full pipeline
# ---------------------------------------------------------------------------
def build_pipeline():
    """Return a scikit‑learn Pipeline that transforms raw transactions into
    customer‑level features ready for model training."""

    # Numerical columns after aggregation
    num_cols = [
        "total_amount", "mean_amount", "std_amount",
        "total_value", "mean_value", "std_value",
        "transaction_count", "hour_mean", "day_mean", "month_mean", "year_mean",
        "recency_days", "fraud_count", "fraud_ratio", "credit_debit_ratio"
    ]

    # Categorical columns after aggregation
    cat_cols = [
        "most_used_channel", "most_used_product_category", "most_used_pricing_strategy"
    ]

    # Numerical preprocessing: impute (if any missing) then scale
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    # Categorical preprocessing: impute (if any missing) then one‑hot encode
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer([
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols),
    ], remainder='passthrough')

    pipeline = Pipeline([
        ("aggregator", TransactionAggregator()),
        ("winsorizer", Winsorizer),
        ("preprocessor", preprocessor),
    ])

    return pipeline


# ---------------------------------------------------------------------------
# 4. Helper to load and process data
# ---------------------------------------------------------------------------
def load_and_process_data(raw_path="data/raw/data.csv"):
    """Load raw data and run the full pipeline, returning a DataFrame."""
    df = pd.read_csv(raw_path, parse_dates=["TransactionStartTime"])
    pipeline = build_pipeline()
    processed = pipeline.fit_transform(df)

    # Rename remainder column back to 'CustomerId' for clean merging
    if 'remainder__CustomerId' in processed.columns:
        processed.rename(columns={'remainder__CustomerId': 'CustomerId'}, inplace=True)

    return processed


if __name__ == "__main__":
    processed_df = load_and_process_data()
    print(f"Processed dataset shape: {processed_df.shape}")
    print(processed_df.head())