import os
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder

DEMO_DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "demo_data", "transactions.csv")

_cat_le = LabelEncoder()
_country_le = LabelEncoder()


def _encode_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    try:
        df["cat_enc"] = _cat_le.fit_transform(df["merchant_category"].astype(str))
    except Exception:
        df["cat_enc"] = 0
    try:
        df["country_enc"] = _country_le.fit_transform(df["country"].astype(str))
    except Exception:
        df["country_enc"] = 0
    return df


def detect_anomalies(transactions: list[dict] | None = None, contamination: float = 0.05) -> dict:
    if transactions:
        df = pd.DataFrame(transactions)
        required = {"amount", "hour_of_day", "day_of_week"}
        if not required.issubset(df.columns):
            raise ValueError(f"transactions must include: {required}")
        if "merchant_category" not in df.columns:
            df["merchant_category"] = "unknown"
        if "country" not in df.columns:
            df["country"] = "US"
    else:
        df = pd.read_csv(DEMO_DATA_PATH)

    df = _encode_df(df)
    features = ["amount", "hour_of_day", "day_of_week", "cat_enc", "country_enc"]
    X = df[features].fillna(0)

    clf = IsolationForest(contamination=contamination, random_state=42)
    scores = clf.fit_predict(X)
    anomaly_scores = clf.decision_function(X)

    df["is_anomaly"] = scores == -1
    df["anomaly_score"] = anomaly_scores

    anomalies = df[df["is_anomaly"]].copy()
    normal = df[~df["is_anomaly"]].copy()

    anomaly_records = []
    for _, row in anomalies.iterrows():
        rec = {
            "transaction_id": int(row.get("transaction_id", 0)),
            "amount": round(float(row["amount"]), 2),
            "hour_of_day": int(row["hour_of_day"]),
            "day_of_week": int(row["day_of_week"]),
            "anomaly_score": round(float(row["anomaly_score"]), 4),
            "is_anomaly": True,
        }
        if "customer_id" in row:
            rec["customer_id"] = int(row["customer_id"])
        if "merchant_category" in row:
            rec["merchant_category"] = str(row["merchant_category"])
        if "country" in row:
            rec["country"] = str(row["country"])
        anomaly_records.append(rec)

    anomaly_records.sort(key=lambda x: x["anomaly_score"])

    return {
        "model": "IsolationForest",
        "contamination": contamination,
        "total_transactions": len(df),
        "anomalies_found": len(anomalies),
        "normal_transactions": len(normal),
        "anomaly_rate_pct": f"{len(anomalies) / len(df) * 100:.1f}%",
        "avg_anomaly_amount": round(float(anomalies["amount"].mean()), 2) if len(anomalies) else 0,
        "avg_normal_amount": round(float(normal["amount"].mean()), 2) if len(normal) else 0,
        "anomalies": anomaly_records[:20],
    }
