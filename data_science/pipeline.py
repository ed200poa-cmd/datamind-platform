import io
import uuid
import pandas as pd
from datetime import datetime

from data_science.stats import compute_stats
from data_science.charts import distribution_chart, correlation_heatmap

_analysis_store: dict[str, dict] = {}


def get_analysis(analysis_id: str) -> dict | None:
    return _analysis_store.get(analysis_id)


async def run_full_analysis(csv_bytes: bytes, filename: str) -> dict:
    analysis_id = str(uuid.uuid4())
    started_at = datetime.utcnow().isoformat()

    try:
        df = pd.read_csv(io.BytesIO(csv_bytes))
    except Exception as e:
        return {"error": f"Could not parse CSV: {e}", "analysis_id": analysis_id}

    if df.empty:
        return {"error": "CSV is empty", "analysis_id": analysis_id}

    # Statistical summary
    stats = compute_stats(df)

    # Generate charts for first 3 numeric columns + correlation heatmap
    chart_ids = []
    numeric_cols = stats["numeric_columns"][:3]
    for col in numeric_cols:
        cid = distribution_chart(df, col)
        if cid:
            chart_ids.append({"column": col, "chart_id": cid, "type": "distribution"})

    heatmap_id = correlation_heatmap(df)
    if heatmap_id:
        chart_ids.append({"column": "all_numeric", "chart_id": heatmap_id, "type": "correlation_heatmap"})

    # Simple EDA insights
    eda = _run_eda(df, stats)

    result = {
        "analysis_id": analysis_id,
        "filename": filename,
        "started_at": started_at,
        "completed_at": datetime.utcnow().isoformat(),
        "stats": stats,
        "charts": chart_ids,
        "eda": eda,
        "status": "complete",
    }
    _analysis_store[analysis_id] = result
    return result


def _run_eda(df: pd.DataFrame, stats: dict) -> dict:
    total = len(df)
    missing_pct = {col: round(cnt / total * 100, 1) for col, cnt in stats["missing_values"].items() if cnt > 0}

    high_missing = [col for col, pct in missing_pct.items() if pct > 20]
    skewed_cols = []
    for col, cstats in stats["column_stats"].items():
        if cstats["type"] == "numeric" and cstats["std"] > 0:
            mean, median = cstats["mean"], cstats["median"]
            skew = abs(mean - median) / cstats["std"]
            if skew > 0.5:
                skewed_cols.append({"column": col, "skew_ratio": round(skew, 2)})

    strong_correlations = []
    for pair in stats.get("correlations", {}).get("top_pairs", []):
        if abs(pair["correlation"]) > 0.5:
            strong_correlations.append(pair)

    recommendations = []
    if high_missing:
        recommendations.append(f"Columns with >20% missing data: {', '.join(high_missing)}. Consider imputation or removal.")
    if skewed_cols:
        top_skewed = [s["column"] for s in skewed_cols[:2]]
        recommendations.append(f"Skewed distributions detected in: {', '.join(top_skewed)}. Log transform may improve model performance.")
    if strong_correlations:
        c = strong_correlations[0]
        recommendations.append(f"Strong correlation ({c['correlation']}) between {c['col_a']} and {c['col_b']}. Watch for multicollinearity.")
    if not recommendations:
        recommendations.append("Data looks clean. No major issues detected. Ready for ML modeling.")

    return {
        "missing_pct": missing_pct,
        "high_missing_columns": high_missing,
        "skewed_columns": skewed_cols,
        "strong_correlations": strong_correlations,
        "recommendations": recommendations,
        "data_quality_score": _quality_score(df, high_missing, missing_pct),
    }


def _quality_score(df: pd.DataFrame, high_missing: list, missing_pct: dict) -> str:
    total_missing = sum(missing_pct.values())
    avg_missing = total_missing / len(df.columns) if df.columns.any() else 0
    if avg_missing < 5 and not high_missing:
        return "Excellent"
    elif avg_missing < 15:
        return "Good"
    elif avg_missing < 30:
        return "Fair"
    return "Poor"
