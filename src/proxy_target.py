"""Proxy target variable engineering using RFM segmentation.

Creates the binary target `is_high_risk` by clustering customers based on
Recency, Frequency, and Monetary value.  The cluster with the highest recency,
lowest frequency, and lowest monetary value is labeled as high‑risk (likely to
default).
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


def compute_rfm(raw_df, customer_id_col="CustomerId", date_col="TransactionStartTime",
                amount_col="Amount"):
    """Compute Recency, Frequency, Monetary metrics per customer.

    Parameters
    ----------
    raw_df : DataFrame
        Raw transaction data.
    customer_id_col : str
        Column identifying the customer.
    date_col : str
        Datetime column.
    amount_col : str
        Transaction amount column (positive = debit, negative = credit).

    Returns
    -------
    rfm : DataFrame
        Customer‑level RFM DataFrame with columns:
        CustomerId, recency, frequency, monetary.
    """
    df = raw_df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        df[date_col] = pd.to_datetime(df[date_col], utc=True)

    # Snapshot date: day after the most recent transaction in the dataset
    snapshot = df[date_col].max().normalize() + pd.Timedelta(days=1)

    rfm = df.groupby(customer_id_col).agg(
        recency=(date_col, lambda x: (snapshot - x.max()).days),
        frequency=(date_col, "count"),
        monetary=(amount_col, "sum")   # net sum (debits – credits)
    ).reset_index()

    rfm.columns = ["CustomerId", "recency", "frequency", "monetary"]
    return rfm


def assign_risk_label(rfm_df, random_state=42):
    """Scale RFM features, cluster with K‑Means (k=3), and label high‑risk.

    The high‑risk cluster is the one with the largest median recency, smallest
    median frequency, and smallest median monetary value.

    Returns
    -------
    rfm_df : DataFrame
        Input DataFrame with two added columns: 'cluster' and 'is_high_risk'.
    """
    features = rfm_df[["recency", "frequency", "monetary"]].copy()

    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)

    kmeans = KMeans(n_clusters=3, random_state=random_state, n_init=10)
    rfm_df["cluster"] = kmeans.fit_predict(scaled)

    # Identify the high‑risk cluster by median RFM values
    cluster_profile = rfm_df.groupby("cluster").agg(
        med_recency=("recency", "median"),
        med_frequency=("frequency", "median"),
        med_monetary=("monetary", "median")
    )

    # Composite score: high recency (rank ascending), low frequency (descending), low monetary (descending)
    cluster_profile["rank_rec"] = cluster_profile["med_recency"].rank(ascending=True)
    cluster_profile["rank_freq"] = cluster_profile["med_frequency"].rank(ascending=False)
    cluster_profile["rank_mon"] = cluster_profile["med_monetary"].rank(ascending=False)
    cluster_profile["risk_score"] = (
        cluster_profile["rank_rec"] + cluster_profile["rank_freq"] + cluster_profile["rank_mon"]
    )
    high_risk_cluster = cluster_profile["risk_score"].idxmax()

    rfm_df["is_high_risk"] = (rfm_df["cluster"] == high_risk_cluster).astype(int)
    return rfm_df


def create_processed_data_with_target(raw_path="data/raw/data.csv"):
    """Combine feature engineering (Task 3) and proxy target (Task 4).

    Returns
    -------
    X : DataFrame
        Feature set with CustomerId preserved.
    y : Series
        Binary target is_high_risk.
    """
    # ---- Feature pipeline (Task 3) ----
    from src.data_processing import load_and_process_data
    X = load_and_process_data(raw_path)   # now contains CustomerId column

    # ---- RFM and target ----
    raw_df = pd.read_csv(raw_path, parse_dates=["TransactionStartTime"])
    rfm = compute_rfm(raw_df)
    rfm = assign_risk_label(rfm)

    # Merge target onto features via CustomerId
    X = X.merge(rfm[["CustomerId", "is_high_risk"]], on="CustomerId", how="inner")

    # Separate target and drop CustomerId from features
    y = X.pop("is_high_risk")
    X = X.drop(columns=["CustomerId"], errors="ignore")

    return X, y


if __name__ == "__main__":
    X, y = create_processed_data_with_target()
    print("Features shape:", X.shape)
    print("Target distribution:\n", y.value_counts())