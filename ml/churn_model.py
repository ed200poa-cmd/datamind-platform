import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

_model: RandomForestClassifier | None = None
_le = LabelEncoder()

DEMO_DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "demo_data", "customers.csv")


def _load_and_train() -> RandomForestClassifier:
    df = pd.read_csv(DEMO_DATA_PATH)
    df["plan_encoded"] = _le.fit_transform(df["plan_type"])
    features = ["age", "tenure_months", "monthly_usage_gb", "support_tickets", "plan_encoded"]
    X = df[features]
    y = df["churned"]
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    return clf


def _get_model() -> RandomForestClassifier:
    global _model
    if _model is None:
        _model = _load_and_train()
    return _model


def predict_churn(age: int, tenure_months: int, monthly_usage_gb: float,
                  support_tickets: int, plan_type: str) -> dict:
    model = _get_model()
    try:
        plan_enc = _le.transform([plan_type])[0]
    except ValueError:
        plan_enc = 1  # default to middle class

    features = np.array([[age, tenure_months, monthly_usage_gb, support_tickets, plan_enc]])
    prob = float(model.predict_proba(features)[0][1])

    if prob >= 0.65:
        risk_level = "High"
        color = "red"
    elif prob >= 0.35:
        risk_level = "Medium"
        color = "orange"
    else:
        risk_level = "Low"
        color = "green"

    model_feat = ["age", "tenure_months", "monthly_usage_gb", "support_tickets", "plan_type"]
    importances = model.feature_importances_
    top_factors = sorted(
        zip(model_feat, importances), key=lambda x: x[1], reverse=True
    )[:3]

    return {
        "churn_probability": round(prob, 4),
        "churn_probability_pct": f"{prob * 100:.1f}%",
        "risk_level": risk_level,
        "risk_color": color,
        "top_factors": [{"feature": f, "importance": round(i, 3)} for f, i in top_factors],
        "model": "RandomForestClassifier",
        "features_used": model_feat,
    }


def batch_predict(df: pd.DataFrame) -> pd.DataFrame:
    model = _get_model()
    df = df.copy()
    df["plan_encoded"] = df["plan_type"].apply(
        lambda x: _le.transform([x])[0] if x in _le.classes_ else 1
    )
    features = ["age", "tenure_months", "monthly_usage_gb", "support_tickets", "plan_encoded"]
    probs = model.predict_proba(df[features])[:, 1]
    df["churn_probability"] = probs
    df["risk_level"] = pd.cut(
        probs, bins=[0, 0.35, 0.65, 1.0], labels=["Low", "Medium", "High"]
    )
    return df


def get_dataset_stats() -> dict:
    df = pd.read_csv(DEMO_DATA_PATH)
    total = len(df)
    churned = int(df["churned"].sum())
    return {
        "total_customers": total,
        "churned": churned,
        "retained": total - churned,
        "churn_rate_pct": f"{churned / total * 100:.1f}%",
        "avg_tenure_months": round(df["tenure_months"].mean(), 1),
        "avg_support_tickets": round(df["support_tickets"].mean(), 1),
        "high_risk_estimate": int((df["support_tickets"] >= 6).sum()),
    }
