# PulseX Sri Lanka - ModelX - The Final Problem

**Real-Time Business Intelligence & Situational Awareness Platform**

---

##  Executive Summary

PulseX is not just a news aggregator; it is a **Probabilistic Risk Engine** designed to solve the "Fog of War" problem for Sri Lankan business leaders.

In a volatile economic environment, traditional news reading is too slow. PulseX ingests unstructured data (News, Social Media, Weather) and converts it into a single, actionable **Risk Score** using advanced signal processing.

PulseX answers three critical questions in real-time:

1. **Is the environment stable?** (Bayesian Risk Score)
2. **Are hidden anomalies forming?** (Hybrid Anomaly Detection)
3. **What should we do right now?** (AI-Strategic Recommendations)

---

##  System Architecture

The system follows a modular **ETL (Extract, Transform, Load)** pipeline architecture designed for high availability and low latency.

```
┌─────────────────┐
│  Data Sources   │ ← News, Social Media, Weather, Economy
└────────┬────────┘
         │
    ┌────▼─────┐
    │Ingestion │ ← Async multi-source ingestion
    └────┬─────┘
         │
    ┌────▼──────┐
    │Preprocessor│ ← Text Cleaning, Feature Extraction
    └────┬──────┘
         │
    ┌────▼────────┐
    │ ML Pipeline │ ← Anomaly, Sentiment, Trends
    └────┬────────┘
         │
    ┌────▼─────────┐
    │ Dashboard UI │ ← Real-time visualization
    └──────────────┘
```

---

##  Project Structure

```
PulseX Sri Lanka
│
├── diagnostic_import_results.json
├── README.md
├── requirements.txt
├── logs/
│   └── pulsex.log
├── notebooks/
│   ├── 01_Exploration_and_Report.ipynb
│   ├── 02_Model_Training_Report.ipynb
│   └── 03_Model_Evaluation_Metrics.ipynb
├── tests/
│   └── test_text_cleaner.py
├── data/
│   ├── models/
│   │   ├── anomaly_detector.pkl
│   │   ├── risk_scorer.pkl
│   │   └── training_report.json
│   ├── processed/
│   │   ├── dashboard_data.json
│   │   ├── evaluation_metrics.json
│   │   ├── exploration_summary.json
│   │   └── golden_test_data.csv
│   └── raw/
│       ├── historical_metrics.csv
│       ├── historical_news.csv
│       └── worldbank_FP.CPI.TOTL.ZG_LK.csv
└── src/
     ├── __init__.py
     ├── main.py
     ├── train_models.py
     ├── config.py
     ├── utils.py
     ├── generate_test_data.py
     ├── run_historical_collector.py
     ├── dashboard/
     │   ├── app.py
     │   ├── components.py
     │   └── recommendations.py
     ├── data_ingestion/
     │   ├── news_scraper.py
     │   ├── social_monitor.py
     │   ├── weather_events.py
     │   └── historical_collector.py
     ├── preprocessing/
     │   ├── feature_extractor.py
     │   └── text_cleaner.py
     └── models/
          ├── anomaly_detector.py
          ├── news_classifier.py
          ├── risk_scorer.py
          ├── sentiment_engine.py
          └── trend_analyzer.py
```

---

##  Deep Dive: Technical Innovations

### **1. Bayesian Risk Engine — The “Brain”**

**Why:** Simple averages react too quickly to single-source noise.

**How it works:**
PulseX uses **Bayesian Inference** with a Beta-Binomial Conjugate Prior.

* **Memory (Prior):** Remembers historical stability.
* **Evidence:** New anomalies update the belief.
* **Posterior:** Produces a stable, smoothed risk score.

**Result:** Avoids false spikes but reacts fast to **sustained, multi-source crises**.

---

### **2. Hybrid Anomaly Detection — Detecting the “Unknown Unknowns”**

A crisis cannot be predicted—but it can be identified statistically.

PulseX uses a weighted ensemble:

| Method             | Weight | Purpose                       |
| ------------------ | ------ | ----------------------------- |
| Isolation Forest   | 35%    | High-dimensional outliers     |
| PCA Reconstruction | 20%    | Structural breaks             |
| Z-Score            | 25%    | Simple deviations from normal |

This detects subtle and complex anomalies.

---

### **3. Multi-Lingual Root Matching Engine**

Local context matters.

Our custom engine supports English, Sinhala, and Tamil:

* **Root Matching:** flood → flooded → flooding
* **Negation Handling:** "not stable", "no crisis" → inverted sentiment
* **Language Agnostic:** Sinhala/Tamil tokenizers plug-in ready

---

### **4. Non-Text-Dependent Mathematical Signals**

Text-independent features:

* **Shannon Entropy:** Lower entropy → panic patterns
* **Zipf Deviation:** Viral anomalies via unnatural repetition

---

##  Model Validation — Stress Test

### **Cold Start Problem**

We cannot label *future* Sri Lankan crises.

### **Solution: Synthetic Injection**

A specially designed dataset was created only for internal testing:

* **500 normal samples**
* **50 crisis samples** (volatility + negative sentiment + viral volume)

### **Results (on synthetic golden data):**

* **ROC-AUC:** 1.00
* **Precision:** 1.00
* **Recall:** 1.00

### **Important Note — No 100% Guarantee**

These scores reflect performance **only on the synthetic test dataset we created**. Real‑world performance may vary because actual events are more complex and unpredictable.

PulseX provides **probabilistic risk assessments**, not absolute guarantees.

---

PulseX reliably detects any statistically defined crisis.

---

##  Installation & Usage

### 1. Setup Environment

Linux / macOS (bash/zsh):

```bash
git clone https://github.com/dilmani773/-PulseX-Sri-Lanka.git
cd "pulsex-sri-lanka" || exit
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Windows (PowerShell):

```powershell
git clone https://github.com/dilmani773/-PulseX-Sri-Lanka.git
cd "pulsex-sri-lanka"
python -m venv .\venv
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process -Force  # if activation blocked
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Environment Variables & Secrets

The project supports a few environment variables for external APIs and configuration. Do not commit secrets to the repository.

- `OPENAI_API_KEY`: Optional — used by the recommendation engine to generate LLM-based suggestions.
- `TWITTER_BEARER_TOKEN`, `REDDIT_CLIENT_ID`, `REDDIT_CLIENT_SECRET`: Optional — for social ingestion (if available).

Set a key for the current PowerShell session (temporary):

```powershell
$env:OPENAI_API_KEY = 'sk-...your-key...'
```

Persist the key for future shells (writes to user environment; requires a new session to take effect):

```powershell
setx OPENAI_API_KEY "sk-...your-key..."
```

Security note: never paste or commit API keys into code or public chats. Consider storing local secrets in a `.env` file and adding it to `.gitignore` for convenience (the project will load `.env` if `python-dotenv` is installed).

### 3. Run common commands

Use the activated virtualenv Python to ensure dependencies are available.

Train models (one-time):

```powershell
.\venv\Scripts\python.exe src\train_models.py
```

Run the real-time pipeline (ingest → features → dashboard payload):

```powershell
.\venv\Scripts\python.exe src\main.py
```

Launch the Streamlit dashboard (view at the printed Local URL):

```powershell
streamlit run src/dashboard/app.py
```

Quick smoke test for the news classifier:

```powershell
.\venv\Scripts\python.exe src\smoke_news_classifier.py
```

### 4. Models, Artifacts & Git

- Trained models and artifacts are saved under `data/models/` and `data/processed/` (e.g. `training_report.json`, `dashboard_data.json`).
- Avoid committing binary model files (`*.pkl`) to Git. If you must store large models in the repository, use Git LFS or a model registry (S3, Artifactory, MLflow).
- Recommended `.gitignore` entries: `venv/`, `__pycache__/`, `*.pyc`, `data/models/*.pkl`, `.env`.

### 5. Tests & Local Validation

Run unit tests with pytest:

```powershell
.\venv\Scripts\python.exe -m pytest tests
```

Run the smoke test for the news classifier (quick check that `news_classifier.pkl` loads and predicts):

```powershell
.\venv\Scripts\python.exe src\smoke_news_classifier.py
```

### 6. Recommendation Engine (LLM)

The recommendation engine will attempt to use OpenAI if `OPENAI_API_KEY` is set. If the key is missing or the API call fails (quota/errors), the engine falls back to rule-based templates. See `src/dashboard/recommendations.py` for details and tuning.

### 7. Troubleshooting & Notes

- If activation fails on Windows, run the `Set-ExecutionPolicy` command shown above for the session.
- Streamlit may print deprecation warnings about `use_container_width`; these are non-blocking and can be updated in `src/dashboard/components.py` by replacing `use_container_width=True` with `width='stretch'`.
- If you accidentally commit an API key, rotate it immediately and remove it from the repo history.

---

##  Team NovaX — University of Peradeniya

* **SURIYAPPERUMA H.D. (E/21/453)**
* **MADHUSHAN S.K.A.K. (E/21/245)**
* **THENNAKOON T.M.I.I.C. (E/21/407)**

Developed for the **ModelX Data Science Competition**.
