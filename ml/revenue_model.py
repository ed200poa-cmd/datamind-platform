import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

DEMO_DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "demo_data", "revenue.csv")


def _load_data() -> pd.DataFrame:
    return pd.read_csv(DEMO_DATA_PATH)


def forecast_revenue(months_ahead: int = 3, custom_history: list[float] | None = None) -> dict:
    if custom_history and len(custom_history) >= 6:
        history = custom_history
    else:
        df = _load_data()
        history = df["revenue"].tolist()

    n = len(history)
    X = np.arange(n).reshape(-1, 1)
    y = np.array(history)

    model = LinearRegression()
    model.fit(X, y)

    # Compute residuals for confidence interval estimate
    y_pred_train = model.predict(X)
    residuals = y - y_pred_train
    std_err = float(np.std(residuals))

    forecasts = []
    for i in range(1, months_ahead + 1):
        x_fut = np.array([[n + i - 1]])
        pred = float(model.predict(x_fut)[0])
        margin = std_err * 1.96  # 95% CI
        forecasts.append({
            "period": f"Month +{i}",
            "forecast": round(pred, 2),
            "lower_bound": round(max(0, pred - margin), 2),
            "upper_bound": round(pred + margin, 2),
            "confidence": "95%",
        })

    slope = float(model.coef_[0])
    avg = float(np.mean(history))
    monthly_growth_pct = round(slope / avg * 100, 2) if avg else 0

    return {
        "model": "LinearRegression",
        "history_periods": n,
        "monthly_growth_trend_pct": monthly_growth_pct,
        "trend_direction": "upward" if slope > 0 else "downward",
        "average_historical": round(avg, 2),
        "forecasts": forecasts,
        "r_squared": round(float(model.score(X, y)), 4),
    }


def get_dataset_stats() -> dict:
    df = _load_data()
    total_rev = df["revenue"].sum()
    return {
        "total_periods": len(df),
        "total_revenue": round(total_rev, 2),
        "avg_monthly_revenue": round(df["revenue"].mean(), 2),
        "min_revenue": round(df["revenue"].min(), 2),
        "max_revenue": round(df["revenue"].max(), 2),
        "latest_revenue": round(df["revenue"].iloc[-1], 2),
        "first_revenue": round(df["revenue"].iloc[0], 2),
        "growth_total_pct": round(
            (df["revenue"].iloc[-1] - df["revenue"].iloc[0]) / df["revenue"].iloc[0] * 100, 1
        ),
    }
