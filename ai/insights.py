import os
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

_SYSTEM = (
    "You are a senior data scientist giving clear, actionable business insights. "
    "Be concise and direct. Use plain English. No jargon. "
    "Always structure with: 1) Key finding, 2) Business impact, 3) Recommended action."
)


def _get_llm() -> ChatAnthropic | None:
    if not os.getenv("ANTHROPIC_API_KEY"):
        return None
    return ChatAnthropic(
        model="claude-haiku-4-5-20251001",
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
        max_tokens=512,
    )


async def _call_claude(prompt: str) -> str:
    llm = _get_llm()
    if not llm:
        return "Claude API not configured. Set ANTHROPIC_API_KEY to enable AI insights."
    resp = await llm.ainvoke([SystemMessage(content=_SYSTEM), HumanMessage(content=prompt)])
    return resp.content


async def churn_insights(result: dict) -> str:
    high = result.get("high_risk_estimate", 0)
    total = result.get("total_customers", 0)
    churn_rate = result.get("churn_rate_pct", "N/A")
    factors = result.get("top_factors", [])
    factors_text = ", ".join(f["feature"] for f in factors[:3]) if factors else "unknown"

    prompt = f"""Churn Prediction Results:
- Total customers analyzed: {total}
- Overall churn rate: {churn_rate}
- High-risk customers: {high}
- Top predictive factors: {factors_text}

Provide a 3-sentence business insight: what this means, the revenue risk, and the top action to take."""

    return await _call_claude(prompt)


async def revenue_insights(result: dict) -> str:
    forecasts = result.get("forecasts", [])
    growth = result.get("monthly_growth_trend_pct", 0)
    r2 = result.get("r_squared", 0)
    trend = result.get("trend_direction", "unknown")
    avg = result.get("average_historical", 0)

    forecast_summary = ""
    if forecasts:
        next_month = forecasts[0]
        forecast_summary = f"Next month forecast: ${next_month['forecast']:,.0f} (range: ${next_month['lower_bound']:,.0f}–${next_month['upper_bound']:,.0f})"

    prompt = f"""Revenue Forecast Results:
- Trend direction: {trend}
- Monthly growth rate: {growth}%
- Model accuracy (R²): {r2}
- Average historical revenue: ${avg:,.0f}
- {forecast_summary}

Provide a 3-sentence business insight: what this forecast means, confidence level, and what factors could change it."""

    return await _call_claude(prompt)


async def segmentation_insights(result: dict) -> str:
    segments = result.get("segments", {})
    total = result.get("total_customers", 0)

    seg_lines = []
    for name, data in segments.items():
        seg_lines.append(
            f"  - {name}: {data['count']} customers ({data['pct']}), "
            f"avg spend ${data['avg_monetary']:,.2f}"
        )
    seg_text = "\n".join(seg_lines)

    prompt = f"""Customer Segmentation Results (RFM Analysis):
Total customers: {total}
Segments found:
{seg_text}

Provide a 3-sentence business insight: which segment deserves immediate attention, why, and what marketing action to take."""

    return await _call_claude(prompt)


async def anomaly_insights(result: dict) -> str:
    total = result.get("total_transactions", 0)
    anomalies = result.get("anomalies_found", 0)
    rate = result.get("anomaly_rate_pct", "N/A")
    avg_anomaly_amt = result.get("avg_anomaly_amount", 0)
    avg_normal_amt = result.get("avg_normal_amount", 0)
    top_anomalies = result.get("anomalies", [])[:3]

    top_details = ""
    if top_anomalies:
        details = [f"${a['amount']:,.2f} at hour {a['hour_of_day']}" for a in top_anomalies]
        top_details = f"Most suspicious: {', '.join(details)}"

    prompt = f"""Anomaly Detection Results:
- Total transactions analyzed: {total}
- Anomalies detected: {anomalies} ({rate})
- Average anomalous transaction: ${avg_anomaly_amt:,.2f}
- Average normal transaction: ${avg_normal_amt:,.2f}
- {top_details}

Provide a 3-sentence business insight: what type of fraud or error this looks like, the financial exposure, and what to investigate first."""

    return await _call_claude(prompt)


async def eda_insights(stats: dict, eda: dict) -> str:
    shape = stats.get("shape", {})
    quality = eda.get("data_quality_score", "Unknown")
    recommendations = eda.get("recommendations", [])
    strong_corr = eda.get("strong_correlations", [])

    recs_text = " ".join(recommendations[:2])
    corr_text = ""
    if strong_corr:
        c = strong_corr[0]
        corr_text = f"Strongest relationship: {c['col_a']} vs {c['col_b']} (r={c['correlation']})"

    prompt = f"""Exploratory Data Analysis Results:
- Dataset: {shape.get('rows', 0)} rows × {shape.get('columns', 0)} columns
- Data quality score: {quality}
- Key findings: {recs_text}
- {corr_text}

Provide a 3-sentence executive summary: overall data health, the most important pattern found, and what ML model would work best on this data."""

    return await _call_claude(prompt)
