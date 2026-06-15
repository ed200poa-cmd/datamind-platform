import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

_SEGMENT_NAMES = {
    0: "Champion",
    1: "Loyal",
    2: "At Risk",
    3: "Lost",
}

_SEGMENT_COLORS = {
    "Champion": "#22c55e",
    "Loyal": "#3b82f6",
    "At Risk": "#f97316",
    "Lost": "#ef4444",
}

_SEGMENT_DESCRIPTIONS = {
    "Champion": "High recency, high frequency, high spend. Best customers — reward and retain.",
    "Loyal": "Consistent buyers. Good frequency and monetary. Nurture with loyalty programs.",
    "At Risk": "Was active but declining. Re-engage with targeted offers before they churn.",
    "Lost": "Low on all dimensions. Win-back campaign needed or accept as churned.",
}


def segment_customers(rfm_data: list[dict] | None = None, n_clusters: int = 4) -> dict:
    if rfm_data:
        df = pd.DataFrame(rfm_data)
        if not all(c in df.columns for c in ["recency", "frequency", "monetary"]):
            raise ValueError("rfm_data must have recency, frequency, monetary columns")
    else:
        df = _generate_rfm_demo()

    features = df[["recency", "frequency", "monetary"]].copy()

    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(scaled)

    # Sort clusters by monetary mean (descending) to assign stable names
    cluster_means = []
    for c in range(n_clusters):
        mask = labels == c
        cluster_means.append((c, float(features["monetary"][mask].mean())))
    sorted_clusters = sorted(cluster_means, key=lambda x: x[1], reverse=True)
    cluster_name_map = {orig: _SEGMENT_NAMES.get(rank, f"Segment {rank}")
                        for rank, (orig, _) in enumerate(sorted_clusters)}

    df["cluster"] = labels
    df["segment"] = df["cluster"].map(cluster_name_map)

    segment_summary = {}
    for seg in df["segment"].unique():
        mask = df["segment"] == seg
        segment_summary[seg] = {
            "count": int(mask.sum()),
            "pct": f"{mask.sum() / len(df) * 100:.1f}%",
            "avg_recency_days": round(float(df[mask]["recency"].mean()), 1),
            "avg_frequency": round(float(df[mask]["frequency"].mean()), 1),
            "avg_monetary": round(float(df[mask]["monetary"].mean()), 2),
            "color": _SEGMENT_COLORS.get(seg, "#888"),
            "description": _SEGMENT_DESCRIPTIONS.get(seg, ""),
        }

    customer_segments = []
    for _, row in df.iterrows():
        customer_segments.append({
            "customer_id": row.get("customer_id", _),
            "recency": int(row["recency"]),
            "frequency": int(row["frequency"]),
            "monetary": round(float(row["monetary"]), 2),
            "segment": row["segment"],
        })

    return {
        "model": "KMeans",
        "n_clusters": n_clusters,
        "total_customers": len(df),
        "segments": segment_summary,
        "customers": customer_segments[:50],  # cap at 50 for response size
        "inertia": round(float(kmeans.inertia_), 2),
    }


def _generate_rfm_demo() -> pd.DataFrame:
    np.random.seed(42)
    n = 100
    records = []
    for i in range(1, n + 1):
        bucket = np.random.choice(["best", "good", "risk", "lost"], p=[0.2, 0.3, 0.3, 0.2])
        if bucket == "best":
            rec = np.random.randint(1, 30)
            freq = np.random.randint(15, 40)
            mon = round(np.random.uniform(500, 2000), 2)
        elif bucket == "good":
            rec = np.random.randint(15, 60)
            freq = np.random.randint(6, 15)
            mon = round(np.random.uniform(150, 600), 2)
        elif bucket == "risk":
            rec = np.random.randint(60, 120)
            freq = np.random.randint(2, 8)
            mon = round(np.random.uniform(50, 200), 2)
        else:
            rec = np.random.randint(120, 365)
            freq = np.random.randint(1, 3)
            mon = round(np.random.uniform(10, 60), 2)
        records.append({"customer_id": i, "recency": rec, "frequency": freq, "monetary": mon})
    return pd.DataFrame(records)
