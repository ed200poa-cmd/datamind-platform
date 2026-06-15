from datetime import datetime
from ml.churn_model import predict_churn, get_dataset_stats as churn_stats, batch_predict
from ml.revenue_model import forecast_revenue, get_dataset_stats as revenue_stats
from ml.segmentation import segment_customers
from ml.anomaly import detect_anomalies
from ai.insights import churn_insights, revenue_insights, segmentation_insights, anomaly_insights
import pandas as pd
import os


def _bubble_wrap(data: dict, message: str = "Success") -> dict:
    return {
        "status": "success",
        "message": message,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "data": data,
    }


def _bubble_error(message: str, code: int = 400) -> dict:
    return {
        "status": "error",
        "message": message,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "error_code": code,
        "data": None,
    }


async def bubble_churn_predict(
    customer_id: str = "demo_001",
    age: int = 35,
    tenure_months: int = 12,
    monthly_usage_gb: float = 15.0,
    support_tickets: int = 3,
    plan_type: str = "standard",
) -> dict:
    try:
        result = predict_churn(age, tenure_months, monthly_usage_gb, support_tickets, plan_type)
        insight = await churn_insights({**result, **churn_stats()})
        return _bubble_wrap({
            "customer_id": customer_id,
            "prediction": result,
            "ai_insight": insight,
        }, "Churn prediction complete")
    except Exception as e:
        return _bubble_error(str(e))


async def bubble_revenue_forecast(months_ahead: int = 3) -> dict:
    try:
        result = forecast_revenue(months_ahead)
        insight = await revenue_insights(result)
        return _bubble_wrap({
            "forecast": result,
            "ai_insight": insight,
        }, "Revenue forecast complete")
    except Exception as e:
        return _bubble_error(str(e))


async def bubble_segment_customers() -> dict:
    try:
        result = segment_customers()
        insight = await segmentation_insights(result)
        return _bubble_wrap({
            "segmentation": result,
            "ai_insight": insight,
        }, "Customer segmentation complete")
    except Exception as e:
        return _bubble_error(str(e))


async def bubble_detect_anomalies() -> dict:
    try:
        result = detect_anomalies()
        insight = await anomaly_insights(result)
        return _bubble_wrap({
            "anomaly_detection": result,
            "ai_insight": insight,
        }, "Anomaly detection complete")
    except Exception as e:
        return _bubble_error(str(e))


async def bubble_upload_csv(csv_bytes: bytes, filename: str) -> dict:
    from data_science.pipeline import run_full_analysis
    try:
        result = await run_full_analysis(csv_bytes, filename)
        if "error" in result:
            return _bubble_error(result["error"])
        return _bubble_wrap(result, "CSV analysis complete")
    except Exception as e:
        return _bubble_error(str(e))


async def bubble_get_insights(analysis_id: str) -> dict:
    from data_science.pipeline import get_analysis
    result = get_analysis(analysis_id)
    if not result:
        return _bubble_error(f"Analysis {analysis_id} not found", 404)
    return _bubble_wrap(result, "Insights retrieved")
