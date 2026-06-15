import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, UploadFile, File, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

from ml.churn_model import predict_churn, get_dataset_stats as churn_stats, batch_predict
from ml.revenue_model import forecast_revenue, get_dataset_stats as revenue_stats
from ml.segmentation import segment_customers
from ml.anomaly import detect_anomalies
from data_science.pipeline import run_full_analysis, get_analysis
from data_science.charts import get_chart
from ai.insights import churn_insights, revenue_insights, segmentation_insights, anomaly_insights, eda_insights
from bubble.api import (
    bubble_churn_predict, bubble_revenue_forecast,
    bubble_segment_customers, bubble_detect_anomalies,
    bubble_upload_csv, bubble_get_insights,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    claude = "set" if os.getenv("ANTHROPIC_API_KEY") else "NOT SET"
    print(f"DataMind AI — Claude API: {claude}")
    # Pre-train models on startup so first request is fast
    try:
        from ml.churn_model import _get_model
        _get_model()
        print("Churn model: ready")
    except Exception as e:
        print(f"Churn model warmup failed: {e}")
    yield


app = FastAPI(
    title="DataMind AI — ML Analytics & Prediction Platform",
    description="Churn Prediction · Revenue Forecasting · Customer Segmentation · Anomaly Detection",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/demo_data", StaticFiles(directory="demo_data"), name="demo_data")


# ── Models ───────────────────────────────────────────────────────────────────

class ChurnRequest(BaseModel):
    age: int = 35
    tenure_months: int = 12
    monthly_usage_gb: float = 15.0
    support_tickets: int = 3
    plan_type: str = "standard"
    customer_id: str = "cust_001"

class RevenueRequest(BaseModel):
    months_ahead: int = 3
    custom_history: list[float] | None = None

class SegmentRequest(BaseModel):
    rfm_data: list[dict] | None = None
    n_clusters: int = 4

class AnomalyRequest(BaseModel):
    transactions: list[dict] | None = None
    contamination: float = 0.05


# ── Core ─────────────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    return FileResponse("static/index.html")

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "claude_api": bool(os.getenv("ANTHROPIC_API_KEY")),
        "models": ["churn", "revenue", "segmentation", "anomaly"],
        "version": "1.0.0",
    }


# ── Standard ML Endpoints ────────────────────────────────────────────────────

@app.post("/predict/churn")
async def predict_churn_endpoint(req: ChurnRequest):
    result = predict_churn(
        req.age, req.tenure_months, req.monthly_usage_gb,
        req.support_tickets, req.plan_type
    )
    insight = await churn_insights({**result, **churn_stats()})
    return {
        "customer_id": req.customer_id,
        "prediction": result,
        "dataset_stats": churn_stats(),
        "ai_insight": insight,
    }


@app.post("/predict/revenue")
async def predict_revenue_endpoint(req: RevenueRequest):
    result = forecast_revenue(req.months_ahead, req.custom_history)
    insight = await revenue_insights(result)
    from data_science.charts import revenue_forecast_chart
    from ml.revenue_model import _load_data
    df = _load_data()
    chart_id = revenue_forecast_chart(df["revenue"].tolist(), result["forecasts"])
    return {
        "forecast": result,
        "dataset_stats": revenue_stats(),
        "ai_insight": insight,
        "chart_id": chart_id,
    }


@app.post("/segment")
async def segment_endpoint(req: SegmentRequest):
    result = segment_customers(req.rfm_data, req.n_clusters)
    insight = await segmentation_insights(result)
    from data_science.charts import segmentation_chart
    chart_id = segmentation_chart(result["segments"])
    return {
        "segmentation": result,
        "ai_insight": insight,
        "chart_id": chart_id,
    }


@app.post("/detect-anomalies")
async def anomaly_endpoint(req: AnomalyRequest):
    result = detect_anomalies(req.transactions, req.contamination)
    insight = await anomaly_insights(result)
    import pandas as pd
    from data_science.charts import anomaly_chart
    if result["anomalies"]:
        all_ids = {a["transaction_id"] for a in result["anomalies"]}
        from ml.anomaly import DEMO_DATA_PATH
        df = pd.read_csv(DEMO_DATA_PATH)
        df_anom = df[df["transaction_id"].isin(all_ids)]
        df_norm = df[~df["transaction_id"].isin(all_ids)]
        chart_id = anomaly_chart(df_norm, df_anom)
    else:
        chart_id = ""
    return {
        "anomaly_detection": result,
        "ai_insight": insight,
        "chart_id": chart_id,
    }


@app.post("/analyze")
async def analyze_csv(file: UploadFile = File(...)):
    if not file.filename or not file.filename.endswith(".csv"):
        raise HTTPException(400, "Please upload a .csv file")
    content = await file.read()
    result = await run_full_analysis(content, file.filename)
    if "error" in result:
        raise HTTPException(400, result["error"])
    insight = await eda_insights(result["stats"], result["eda"])
    result["ai_insight"] = insight
    return result


@app.get("/insights/{analysis_id}")
async def get_insights(analysis_id: str):
    result = get_analysis(analysis_id)
    if not result:
        raise HTTPException(404, f"Analysis {analysis_id} not found")
    return result


@app.get("/charts/{chart_id}")
async def get_chart_endpoint(chart_id: str):
    b64 = get_chart(chart_id)
    if not b64:
        raise HTTPException(404, f"Chart {chart_id} not found")
    return {"chart_id": chart_id, "format": "png", "encoding": "base64", "data": b64}


# ── Bubble.io Endpoints ──────────────────────────────────────────────────────

@app.get("/bubble/predict/churn")
async def bubble_churn(
    customer_id: str = Query("demo_001"),
    age: int = Query(35),
    tenure_months: int = Query(12),
    monthly_usage_gb: float = Query(15.0),
    support_tickets: int = Query(3),
    plan_type: str = Query("standard"),
):
    return await bubble_churn_predict(customer_id, age, tenure_months,
                                       monthly_usage_gb, support_tickets, plan_type)


@app.get("/bubble/predict/revenue")
async def bubble_revenue(months_ahead: int = Query(3)):
    return await bubble_revenue_forecast(months_ahead)


@app.get("/bubble/segment/customers")
async def bubble_segment():
    return await bubble_segment_customers()


@app.get("/bubble/detect/anomalies")
async def bubble_anomalies():
    return await bubble_detect_anomalies()


@app.post("/bubble/upload-csv")
async def bubble_csv(file: UploadFile = File(...)):
    if not file.filename or not file.filename.endswith(".csv"):
        return {"status": "error", "message": "Please upload a .csv file", "data": None}
    content = await file.read()
    return await bubble_upload_csv(content, file.filename)


@app.get("/bubble/insights/{analysis_id}")
async def bubble_insights(analysis_id: str):
    return await bubble_get_insights(analysis_id)


@app.get("/bubble/chart/{chart_id}")
async def bubble_chart(chart_id: str):
    b64 = get_chart(chart_id)
    if not b64:
        return {"status": "error", "message": f"Chart {chart_id} not found", "data": None}
    return {
        "status": "success",
        "timestamp": __import__("datetime").datetime.utcnow().isoformat() + "Z",
        "data": {"chart_id": chart_id, "format": "png", "encoding": "base64", "image": b64},
    }


# ── Demo Run All ──────────────────────────────────────────────────────────────

@app.post("/demo/run-all")
async def demo_run_all():
    churn_result = predict_churn(38, 8, 5.2, 7, "basic")
    churn_insight = await churn_insights({**churn_result, **churn_stats()})

    rev_result = forecast_revenue(3)
    rev_insight = await revenue_insights(rev_result)

    seg_result = segment_customers()
    seg_insight = await segmentation_insights(seg_result)

    anom_result = detect_anomalies()
    anom_insight = await anomaly_insights(anom_result)

    return {
        "churn": {"result": churn_result, "insight": churn_insight},
        "revenue": {"result": rev_result, "insight": rev_insight},
        "segmentation": {
            "total_customers": seg_result["total_customers"],
            "segments": list(seg_result["segments"].keys()),
            "insight": seg_insight,
        },
        "anomaly": {
            "total": anom_result["total_transactions"],
            "found": anom_result["anomalies_found"],
            "insight": anom_insight,
        },
    }
