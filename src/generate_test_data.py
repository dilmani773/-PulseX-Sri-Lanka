import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Ensure project root is on sys.path so `import src.*` works when running
# this file as a script (e.g. `python src/generate_test_data.py`).
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import PROCESSED_DATA_DIR

# ---------------------------------------------------------
# GENERATE "GOLDEN" TEST DATA (Stress Test)
# ---------------------------------------------------------
print("Generating Synthetic Crisis Data for Validation...")

# 1. Create 500 "Normal" Days (Stable, boring news)
n_normal = 500
dates_normal = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(n_normal)]
normal_data = {
    'date': dates_normal,
    'article_count': np.random.normal(50, 5, n_normal),       # Normal volume
    'avg_sentiment': np.random.normal(0.2, 0.1, n_normal),    # Slightly positive
    'sentiment_std': np.random.normal(0.1, 0.05, n_normal),   # Low volatility
    'sentiment_min': np.random.normal(-0.2, 0.1, n_normal),
    'sentiment_max': np.random.normal(0.6, 0.1, n_normal),
    'avg_engagement': np.random.normal(100, 20, n_normal),
    'engagement_std': np.random.normal(20, 5, n_normal),
    'unique_sources': np.random.randint(5, 10, n_normal),
    'unique_categories': np.random.randint(3, 8, n_normal),
    # Topic percentages (stable)
    'pct_political': np.random.uniform(0.1, 0.3, n_normal),
    'pct_economic': np.random.uniform(0.1, 0.3, n_normal),
    'pct_infrastructure': np.random.uniform(0.1, 0.2, n_normal),
    'pct_tourism': np.random.uniform(0.1, 0.2, n_normal),
    'sentiment_velocity': np.random.normal(0, 0.05, n_normal),
    'sentiment_acceleration': np.random.normal(0, 0.01, n_normal),
    'y_true': 0  # <--- LABEL IS 0 (NORMAL)
}
df_normal = pd.DataFrame(normal_data)

# 2. Inject 50 "Crisis" Days (The Anomalies!)
# Scenario: Market Crash (High Volatility, Negative Sentiment, Huge Engagement)
n_crisis = 50
dates_crisis = [datetime(2025, 1, 1) + timedelta(days=i) for i in range(n_crisis)]
crisis_data = {
    'date': dates_crisis,
    'article_count': np.random.normal(200, 50, n_crisis),      # EXPLOSIVE volume
    'avg_sentiment': np.random.normal(-0.8, 0.1, n_crisis),    # VERY Negative
    'sentiment_std': np.random.normal(0.8, 0.1, n_crisis),     # HIGH Volatility
    'sentiment_min': np.random.normal(-0.95, 0.05, n_crisis),
    'sentiment_max': np.random.normal(-0.4, 0.2, n_crisis),
    'avg_engagement': np.random.normal(1000, 200, n_crisis),   # Viral news
    'engagement_std': np.random.normal(500, 100, n_crisis),
    'unique_sources': np.random.randint(8, 15, n_crisis),
    'unique_categories': np.random.randint(8, 12, n_crisis),
    # Topic percentages (Crisis focus)
    'pct_political': np.random.uniform(0.4, 0.8, n_crisis),
    'pct_economic': np.random.uniform(0.1, 0.2, n_crisis),
    'pct_infrastructure': np.random.uniform(0.0, 0.1, n_crisis),
    'pct_tourism': np.random.uniform(0.0, 0.1, n_crisis),
    'sentiment_velocity': np.random.normal(-0.5, 0.2, n_crisis), # Crashing fast
    'sentiment_acceleration': np.random.normal(-0.1, 0.05, n_crisis),
    'y_true': 1  # <--- LABEL IS 1 (ANOMALY)
}
df_crisis = pd.DataFrame(crisis_data)

# 3. Merge and Save
df_golden = pd.concat([df_normal, df_crisis]).sample(frac=1).reset_index(drop=True)
save_path = PROCESSED_DATA_DIR / 'golden_test_data.csv'
df_golden.to_csv(save_path, index=False)

print(f"✅ Success! Created {len(df_golden)} test samples.")
print(f"   - 500 Normal Days")
print(f"   - 50 'Crisis' Days (Anomalies)")
print(f"   - Saved to: {save_path}")