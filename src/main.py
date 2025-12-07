"""
PulseX Sri Lanka - Main Pipeline
Orchestrates end-to-end data collection, analysis, and serving
Refactored to connect with Dashboard and include Text Cleaning
"""

import asyncio
import logging
from datetime import datetime
from pathlib import Path
import json
import sys
import pandas as pd
import numpy as np

# Add src to path
sys.path.append(str(Path(__file__).parent))

from config import (
    DATA_CONFIG, MODEL_CONFIG, 
    RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR
)
# Ensure directories exist
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

from data_ingestion.news_scraper import NewsScraperEngine
from preprocessing.feature_extractor import AdvancedFeatureExtractor
from preprocessing.text_cleaner import TextCleaner  # <--- NEW IMPORT
from models.anomaly_detector import HybridAnomalyDetector
from models.risk_scorer import BayesianRiskScorer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PulseXPipeline:
    def __init__(self):
        self.feature_extractor = AdvancedFeatureExtractor()
        self.text_cleaner = TextCleaner()  # <--- NEW INITIALIZATION
        
        # Load existing model if available to avoid "Cold Start"
        self.anomaly_detector = HybridAnomalyDetector()
        model_path = MODELS_DIR / "anomaly_detector_latest.pkl"
        if model_path.exists():
            try:
                self.anomaly_detector.load(model_path)
                logger.info("Loaded existing anomaly detector model")
            except:
                logger.warning("Could not load existing model, starting fresh")
        
        # Load risk scorer history
        history_path = PROCESSED_DATA_DIR / "risk_history.json"
        history = None
        if history_path.exists():
            try:
                with open(history_path, 'r') as f:
                    history = json.load(f)
            except:
                pass
        self.risk_scorer = BayesianRiskScorer(history=history)
        
    async def ingest_data(self) -> pd.DataFrame:
        logger.info("Starting ingestion...")
        # In a real run, use the scraper. For testing, create a fallback if scraper fails.
        try:
            async with NewsScraperEngine(DATA_CONFIG.NEWS_SOURCES) as scraper:
                articles = await scraper.scrape_all()
                df = pd.DataFrame([art.to_dict() for art in articles])
        except Exception as e:
            logger.error(f"Scraper failed: {e}. Using dummy data for safety.")
            df = pd.DataFrame() # Will trigger fallback below

        # Fallback for empty scraping (Common in demos)
        if len(df) == 0:
            df = pd.DataFrame({
                'title': [f"Economic update {i}" for i in range(5)],
                'content': ["Market shows volatility in fuel prices"] * 5,
                'source': ["Daily Mirror"] * 5,
                'published_date': [datetime.now()] * 5,
                'scraped_at': [datetime.now()] * 5,
                'url': ["http://example.com"] * 5
            })
            
        df['published_date'] = pd.to_datetime(df['published_date'])
        df['scraped_at'] = pd.to_datetime(df['scraped_at'])
        return df

    def run_analysis(self, df: pd.DataFrame):
        if df.empty:
            return df, pd.DataFrame(), None

        # 1. PREPROCESS & CLEAN (The Fix!)
        logger.info("Cleaning text...")
        # Apply cleaning to the 'content' column
        # We keep punctuation because Feature Extractor needs it for sentence segmentation
        if 'content' in df.columns:
            df['clean_content'] = df['content'].apply(
                lambda x: self.text_cleaner.clean_text(str(x), remove_stopwords=False)['cleaned']
            )
        else:
            df['clean_content'] = ""

        # Placeholder for sentiment model (or hook up sentiment engine here)
        df['sentiment_score'] = np.random.uniform(-0.8, 0.2, len(df)) 
        df['engagement'] = np.random.randint(100, 5000, len(df))
        
        # 2. Extract Features (The math part)
        # IMPORTANT: Use 'clean_content' instead of raw 'content' for accuracy
        df_for_features = df.copy()
        df_for_features['content'] = df['clean_content'] 
        features_df = self.feature_extractor.create_feature_matrix(df_for_features)
        
        # 3. Anomaly Detection
        if not self.anomaly_detector.is_fitted:
            logger.info("Fitting anomaly detector on initial batch")
            # Create synthetic history if batch is too small
            if len(features_df) < 50:
                # Align columns manually or just fit on current batch for demo
                numeric_cols = features_df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    X = features_df[numeric_cols].fillna(0).values
                    self.anomaly_detector.fit(X)
            else:
                 numeric_cols = features_df.select_dtypes(include=[np.number]).columns
                 if len(numeric_cols) > 0:
                    self.anomaly_detector.fit(features_df[numeric_cols].fillna(0).values)

        numeric_cols = features_df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            X = features_df[numeric_cols].fillna(0).values
            features_df['anomaly_score'] = self.anomaly_detector.predict(X, return_scores=True)
        else:
            features_df['anomaly_score'] = 0.5
        
        # 4. Risk Scoring
        latest_risk = None
        for _, row in features_df.iterrows():
            indicators = {
                'sentiment_score': row.get('sentiment_mean', -0.5),
                'sentiment_volatility': row.get('sentiment_volatility', 0.2),
                'time_series_values': np.array([10, 12, 15]), # Placeholder
                'trend_slope': row.get('trend_slope', -0.1),
                'anomaly_score': row.get('anomaly_score', 0.5),
                'timestamp': row.get('timestamp', datetime.now())
            }
            latest_risk = self.risk_scorer.assess_risk(indicators)

        return df, features_df, latest_risk

    def save_results(self, df, risk_assessment):
        # Save for Dashboard to read
        output = {
            "last_updated": datetime.now().isoformat(),
            "overall_risk": risk_assessment.overall_score if risk_assessment else 0.5,
            "risk_level": risk_assessment.risk_level.value if risk_assessment else "medium",
            "sentiment_avg": float(df['sentiment_score'].mean()) if not df.empty else 0.0,
            "active_alerts": int(df[df['sentiment_score'] < -0.7].shape[0]) if not df.empty else 0, 
            "total_articles": len(df),
            "recommendations": risk_assessment.recommendations if risk_assessment else [],
            "risk_breakdown": risk_assessment.components if risk_assessment else {},
            "recent_news": df.head(10).to_dict(orient='records') if not df.empty else []
        }
        
        with open(PROCESSED_DATA_DIR / "dashboard_data.json", 'w') as f:
            json.dump(output, f, default=str)
        
        # Save History
        with open(PROCESSED_DATA_DIR / "risk_history.json", 'w') as f:
            json.dump(self.risk_scorer.observations, f)
            
        logger.info("Saved dashboard_data.json")

    async def run(self):
        df = await self.ingest_data()
        df, feats, risk = self.run_analysis(df)
        self.save_results(df, risk)

if __name__ == "__main__":
    asyncio.run(PulseXPipeline().run())
