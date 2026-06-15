import io
import base64
import uuid
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

_chart_store: dict[str, str] = {}

sns.set_theme(style="whitegrid", palette="muted")


def _fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def _store(b64: str) -> str:
    chart_id = str(uuid.uuid4())
    _chart_store[chart_id] = b64
    return chart_id


def get_chart(chart_id: str) -> str | None:
    return _chart_store.get(chart_id)


def distribution_chart(df: pd.DataFrame, column: str) -> str:
    fig, ax = plt.subplots(figsize=(8, 4))
    data = df[column].dropna()
    if df[column].dtype in [object, "category"]:
        vc = data.value_counts()
        ax.bar(vc.index, vc.values, color="#6366f1")
        ax.set_xlabel(column)
        ax.set_ylabel("Count")
    else:
        ax.hist(data, bins=20, color="#6366f1", edgecolor="white")
        ax.set_xlabel(column)
        ax.set_ylabel("Frequency")
    ax.set_title(f"Distribution — {column}")
    fig.tight_layout()
    return _store(_fig_to_b64(fig))


def correlation_heatmap(df: pd.DataFrame) -> str:
    numeric = df.select_dtypes(include=[np.number])
    if len(numeric.columns) < 2:
        return ""
    corr = numeric.corr()
    fig, ax = plt.subplots(figsize=(max(6, len(corr.columns)), max(5, len(corr.columns) - 1)))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=ax,
                square=True, linewidths=0.5)
    ax.set_title("Correlation Heatmap")
    fig.tight_layout()
    return _store(_fig_to_b64(fig))


def churn_risk_chart(predictions_df: pd.DataFrame) -> str:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Risk level pie
    risk_counts = predictions_df["risk_level"].value_counts()
    colors = {"High": "#ef4444", "Medium": "#f97316", "Low": "#22c55e"}
    c = [colors.get(r, "#888") for r in risk_counts.index]
    ax1.pie(risk_counts.values, labels=risk_counts.index, colors=c, autopct="%1.1f%%",
            startangle=90, wedgeprops={"edgecolor": "white", "linewidth": 2})
    ax1.set_title("Customer Risk Distribution")

    # Churn probability histogram
    ax2.hist(predictions_df["churn_probability"], bins=20, color="#6366f1", edgecolor="white")
    ax2.axvline(0.35, color="#f97316", linestyle="--", label="Medium threshold (0.35)")
    ax2.axvline(0.65, color="#ef4444", linestyle="--", label="High threshold (0.65)")
    ax2.set_xlabel("Churn Probability")
    ax2.set_ylabel("Number of Customers")
    ax2.set_title("Churn Probability Distribution")
    ax2.legend(fontsize=8)

    fig.suptitle("Churn Prediction Analysis", fontsize=14, fontweight="bold")
    fig.tight_layout()
    return _store(_fig_to_b64(fig))


def revenue_forecast_chart(history: list[float], forecasts: list[dict]) -> str:
    fig, ax = plt.subplots(figsize=(12, 5))
    hist_x = list(range(len(history)))
    ax.plot(hist_x, history, color="#6366f1", linewidth=2, label="Historical", marker="o",
            markersize=4)

    for_x = [len(history) + i for i in range(len(forecasts))]
    for_y = [f["forecast"] for f in forecasts]
    lower = [f["lower_bound"] for f in forecasts]
    upper = [f["upper_bound"] for f in forecasts]

    ax.plot(for_x, for_y, color="#22c55e", linewidth=2, linestyle="--", label="Forecast",
            marker="s", markersize=6)
    ax.fill_between(for_x, lower, upper, alpha=0.2, color="#22c55e", label="95% CI")

    # Connect history to forecast
    if history:
        ax.plot([hist_x[-1], for_x[0]], [history[-1], for_y[0]],
                color="#94a3b8", linestyle=":", linewidth=1)

    ax.set_xlabel("Period")
    ax.set_ylabel("Revenue ($)")
    ax.set_title("Revenue Forecast")
    ax.legend()
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    fig.tight_layout()
    return _store(_fig_to_b64(fig))


def segmentation_chart(segments: dict) -> str:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    names = list(segments.keys())
    counts = [segments[s]["count"] for s in names]
    colors = [segments[s]["color"] for s in names]

    ax1.bar(names, counts, color=colors, edgecolor="white", linewidth=1.5)
    ax1.set_title("Customer Segments — Size")
    ax1.set_ylabel("Number of Customers")
    for i, (n, c) in enumerate(zip(names, counts)):
        ax1.text(i, c + 0.5, str(c), ha="center", fontsize=10, fontweight="bold")

    avg_mon = [segments[s]["avg_monetary"] for s in names]
    ax2.bar(names, avg_mon, color=colors, edgecolor="white", linewidth=1.5)
    ax2.set_title("Customer Segments — Avg Spend")
    ax2.set_ylabel("Average Monetary Value ($)")
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))

    fig.suptitle("Customer Segmentation (RFM Analysis)", fontsize=14, fontweight="bold")
    fig.tight_layout()
    return _store(_fig_to_b64(fig))


def anomaly_chart(df_normal: pd.DataFrame, df_anomaly: pd.DataFrame) -> str:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.scatter(df_normal["hour_of_day"], df_normal["amount"], alpha=0.4,
                color="#6366f1", label="Normal", s=20)
    ax1.scatter(df_anomaly["hour_of_day"], df_anomaly["amount"], alpha=0.8,
                color="#ef4444", label="Anomaly", s=50, marker="x", linewidths=2)
    ax1.set_xlabel("Hour of Day")
    ax1.set_ylabel("Transaction Amount ($)")
    ax1.set_title("Anomaly Detection — Time vs Amount")
    ax1.legend()
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))

    all_amounts = pd.concat([df_normal["amount"], df_anomaly["amount"]])
    ax2.hist(df_normal["amount"], bins=30, alpha=0.6, color="#6366f1", label="Normal")
    ax2.hist(df_anomaly["amount"], bins=10, alpha=0.8, color="#ef4444", label="Anomaly")
    ax2.set_xlabel("Transaction Amount ($)")
    ax2.set_ylabel("Count")
    ax2.set_title("Amount Distribution")
    ax2.legend()

    fig.suptitle("Transaction Anomaly Detection", fontsize=14, fontweight="bold")
    fig.tight_layout()
    return _store(_fig_to_b64(fig))
