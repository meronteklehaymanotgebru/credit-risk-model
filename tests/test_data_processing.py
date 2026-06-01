import pandas as pd
import os
import pytest
from src.data_processing import TransactionAggregator, load_and_process_data

DATA_PATH = "data/raw/data.csv"
SAMPLE_PATH = "data/raw/sample_data.csv"

def get_data_path():
    """Return the real data path if it exists, otherwise the sample."""
    if os.path.exists(DATA_PATH):
        return DATA_PATH
    return SAMPLE_PATH

def test_transaction_aggregator_returns_expected_columns():
    """Test that TransactionAggregator outputs all expected feature columns."""
    path = get_data_path()
    df = pd.read_csv(path, parse_dates=["TransactionStartTime"])
    agg = TransactionAggregator()
    result = agg.fit_transform(df)

    expected_columns = [
        "CustomerId", "total_amount", "mean_amount", "std_amount",
        "total_value", "mean_value", "std_value",
        "transaction_count", "hour_mean", "day_mean", "month_mean", "year_mean",
        "recency_days", "fraud_count", "fraud_ratio", "credit_debit_ratio",
        "most_used_channel", "most_used_product_category", "most_used_pricing_strategy"
    ]

    for col in expected_columns:
        assert col in result.columns, f"Missing column: {col}"

    assert len(result) == df["CustomerId"].nunique(), "Row count should match unique customers"

def test_load_and_process_data_no_error():
    """Test that the full pipeline runs without raising an error."""
    path = get_data_path()
    processed = load_and_process_data(path)
    assert processed is not None
    assert processed.shape[0] > 0
    # CustomerId should be present after the pipeline (remainder renamed)
    assert any("CustomerId" in col for col in processed.columns), "CustomerId column missing"