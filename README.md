# PulseX Sri Lanka

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
pulsex-sri-lanka/
│
├── src/
│   ├── main.py
│   ├── train_models.py
│   ├── config.py
│   ├── utils.py
│   │
│   ├── data_ingestion/
│   │   ├── news_scraper.py
│   │   ├── social_monitor.py
│   │   ├── weather_events.py
│   │   ├── economic_api.py
│   │   └── historical_collector.py
│   │
│   ├── preprocessing/
│   │   ├── text_cleaner.py
│   │   └── feature_extractor.py
│   │
│   ├── models/
│   │   ├── anomaly_detector.py
│   │   ├── trend_analyzer.py
│   │   ├── sentiment_engine.py
│   │   └── risk_scorer.py
│   │
│   └── dashboard/
│       ├── app.py
│       ├── components.py
│       └── recommendations.py
│
├── data/
├── notebooks/
│   ├── 01_Exploration_and_Report.ipynb
│   ├── 02_Model_Training_Report.ipynb
│   └── 03_Model_Evaluation_Metrics.ipynb
│
├── requirements.txt
└── README.md
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

### **1. Setup Environment**

```
git clone https://github.com/dilmani773/-PulseX-Sri-Lanka.git
cd pulsex-sri-lanka
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### **2. Train Models (One-Time)**

```
python src/train_models.py
```

### **3. Run Real-Time Pipeline**

```
python src/main.py
```

To simulate crisis mode:

```
INJECT_CRISIS = True
```

### **4. Launch Dashboard**

```
streamlit run src/dashboard/app.py
```

---

##  Team NovaX — University of Peradeniya

* **SURIYAPPERUMA H.D. (E/21/453)**
* **MADHUSHAN S.K.A.K. (E/21/245)**
* **THENNAKOON T.M.I.I.C. (E/21/407)**

Developed for the **ModelX Data Science Competition**.
