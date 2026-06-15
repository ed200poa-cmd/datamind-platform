# DataMind AI — ML Analytics & Prediction Platform

Production-ready machine learning platform for churn prediction, revenue forecasting, customer segmentation, and anomaly detection — powered by scikit-learn and Claude API.

**Built by Edward Kim — AI & ML Developer**

---

## What is Machine Learning? (Plain English)

Machine learning is software that learns patterns from historical data instead of following rules you write by hand. You show it thousands of past examples (customers who churned, transactions that were fraudulent, revenue from previous months), and it figures out the patterns on its own. Then it applies those patterns to new data — predicting what will happen before it does.

---

## What is a Data Science Pipeline?

A data science pipeline is the step-by-step process that turns raw data into decisions. Raw CSV → clean and format → run statistics → train ML model → generate predictions → visualize results → plain-English business insight. This platform automates every step: upload a CSV and in seconds you get charts, statistics, ML predictions, and an AI-written executive summary.

---

## How Bubble.io Connects to This API

1. Open your Bubble app → Plugins → API Connector → Add another API
2. Name it "DataMind AI", Root URL: `https://your-railway-url.up.railway.app`
3. Add a new call: **GET /bubble/predict/churn** — add query params as dynamic keys
4. Click **Initialize call** — Bubble reads the response schema automatically
5. In any workflow: **Get data from an external API** → choose your call
6. Map `result.data.prediction.churn_probability` to a Bubble text/number field
7. Display `result.data.ai_insight` in a text element for the Claude explanation
8. All Bubble endpoints return the same schema: `{ status, data, message, timestamp }` — consistent and predictable

---

## ML Models

| Model | Algorithm | Input | Output |
|---|---|---|---|
| **Churn Prediction** | RandomForestClassifier | age, tenure, usage, tickets, plan | churn_probability, risk_level |
| **Revenue Forecast** | LinearRegression | historical revenue array | next N months + confidence bands |
| **Segmentation** | KMeans | RFM (recency, frequency, monetary) | Champion / Loyal / At Risk / Lost |
| **Anomaly Detection** | IsolationForest | transaction amount, time, category | anomaly_score, is_anomaly |

---

## Industry Use Cases

| Industry | Best Model | Business Value |
|---|---|---|
| SaaS / Subscription | Churn Prediction | Reduce cancellations, trigger retention offers |
| E-commerce | Customer Segmentation | Personalize campaigns by RFM segment |
| Fintech / Banking | Anomaly Detection | Catch fraud before it completes |
| Retail / CPG | Revenue Forecasting | Inventory and budget planning |
| Healthcare | CSV Analysis | Patient outcome pattern discovery |
| Real Estate | Revenue Forecasting | Market price trend analysis |

---

## Cost Comparison

| Option | Monthly Cost | Setup Time |
|---|---|---|
| **DataMind AI on Railway** | ~$5/mo | Minutes |
| Junior Data Scientist | $7,000+/mo | Weeks to onboard |
| Analytics Agency | $5,000–20,000/mo | Months for first deliverable |
| Enterprise BI Tool (Tableau, etc.) | $500–2,000/mo | Days, no ML included |

---

## Quick Start

```bash
cd datamind
pip install -r requirements.txt
cp .env.example .env
# Add ANTHROPIC_API_KEY to .env
uvicorn main:app --reload
```

Open http://localhost:8000

---

## API Endpoints

### Standard
| Method | Path | Description |
|---|---|---|
| POST | /predict/churn | Single customer churn prediction |
| POST | /predict/revenue | Revenue forecast |
| POST | /segment | Customer segmentation |
| POST | /detect-anomalies | Anomaly detection |
| POST | /analyze | Upload CSV → full EDA |
| GET | /insights/{id} | Retrieve saved analysis |
| GET | /charts/{id} | Get chart as base64 PNG |
| POST | /demo/run-all | Run all 4 models at once |

### Bubble.io (GET-friendly, query params)
| Method | Path |
|---|---|
| GET | /bubble/predict/churn |
| GET | /bubble/predict/revenue |
| GET | /bubble/segment/customers |
| GET | /bubble/detect/anomalies |
| POST | /bubble/upload-csv |
| GET | /bubble/insights/{id} |
| GET | /bubble/chart/{id} |

---

## Environment Variables

```env
ANTHROPIC_API_KEY=sk-ant-...   # Required — enables AI insights
PORT=8000                       # Set automatically by Railway
```

---

## Tech Stack

- **Python 3.12** + FastAPI + Uvicorn
- **scikit-learn** — RandomForest, LinearRegression, KMeans, IsolationForest
- **pandas + numpy** — data processing and feature engineering
- **matplotlib + seaborn** — chart generation (base64 PNG)
- **Claude claude-haiku** — plain-English insight generation
- **Railway** — cloud deployment
