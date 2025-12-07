"""
PulseX Sri Lanka - Main Pipeline
Orchestrates end-to-end data collection, analysis, and serving
Refactored to connect with Dashboard and fully integrate Sentiment & Trend analysis
"""

import asyncio
import logging
from datetime import datetime, timedelta
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
from preprocessing.text_cleaner import TextCleaner
from models.anomaly_detector import HybridAnomalyDetector
from models.risk_scorer import BayesianRiskScorer
from models.sentiment_engine import SentimentAnalyzer # <--- NEW IMPORT
from models.trend_analyzer import TrendAnalyzer     # <--- NEW IMPORT

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PulseXPipeline:
    def __init__(self):
        self.feature_extractor = AdvancedFeatureExtractor()
        self.text_cleaner = TextCleaner()
        self.sentiment_analyzer = SentimentAnalyzer() # <--- INITIALIZED
        self.trend_analyzer = TrendAnalyzer()         # <--- INITIALIZED
        
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
            logger.warning("Using synthetic fallback data.")
            # Ensure dates are distributed in the past for trend analysis to work
            dates = [datetime.now() - timedelta(hours=i*2) for i in reversed(range(50))]
            df = pd.DataFrame({
                'title': [f"Update {i}" for i in range(50)],
                'content': ["Market shows volatility in fuel prices and currency exchange. Tourism sees growth."] * 50,
                'source': ["Daily Mirror"] * 50,
                'published_date': dates,
                'scraped_at': [datetime.now()] * 50,
                'url': ["http://example.com"] * 50
            })
            
        df['published_date'] = pd.to_datetime(df['published_date'])
        df['scraped_at'] = pd.to_datetime(df['scraped_at'])
        return df

    def run_analysis(self, df: pd.DataFrame):
        if df.empty:
            return df, pd.DataFrame(), None

        # 1. PREPROCESS & SENTIMENT ANALYSIS (Real Sentiment)
        logger.info("Cleaning and running Sentiment Analysis...")
        
        df['clean_content'] = df['content'].apply(
            lambda x: self.text_cleaner.clean_text(str(x), remove_stopwords=False)['cleaned']
        )
        # Calculate real sentiment score (compound score)
        df['sentiment_results'] = df['clean_content'].apply(self.sentiment_analyzer.analyze_text)
        df['sentiment_score'] = df['sentiment_results'].apply(lambda x: x['compound'])
        df['engagement'] = np.random.randint(100, 5000, len(df)) # Keep engagement placeholder
        
        # 2. Extract Features
        df_for_features = df.copy()
        df_for_features['content'] = df['clean_content'] 
        features_df = self.feature_extractor.create_feature_matrix(df_for_features)
        
        # 3. Trend Analysis on Sentiment (NEW)
        # Aggregate sentiment by the hour to create a time series
        sentiment_series = df.set_index('published_date')['sentiment_score'].resample('H').mean().fillna(0)
        
        if len(sentiment_series) > 3:
            trend_info = self.trend_analyzer.detect_trend(sentiment_series.values)
            sentiment_trend_slope = trend_info['slope']
            sentiment_volatility = sentiment_series.std()
        else:
            sentiment_trend_slope = 0.0
            sentiment_volatility = df['sentiment_score'].std() if len(df) > 1 else 0.0
        
        # 4. Anomaly Detection
        # ... (Keep existing anomaly detection logic as is) ...
        if not self.anomaly_detector.is_fitted:
            logger.info("Fitting anomaly detector on initial batch")
            numeric_cols = features_df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0 and len(features_df) > 1:
                X = features_df[numeric_cols].fillna(0).values
                self.anomaly_detector.fit(X)

        numeric_cols = features_df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0 and self.anomaly_detector.is_fitted:
            X = features_df[numeric_cols].fillna(0).values
            features_df['anomaly_score'] = self.anomaly_detector.predict(X, return_scores=True)
        else:
            features_df['anomaly_score'] = 0.5
        
        # 5. Risk Scoring (Now uses real trend data)
        latest_risk = None
        # Use the average anomaly score from the batch for the overall risk score
        batch_anomaly_score = features_df['anomaly_score'].mean()
        
        indicators = {
            'sentiment_score': df['sentiment_score'].mean(),
            'sentiment_volatility': sentiment_volatility, # <--- REAL VOLATILITY
            'time_series_values': sentiment_series.values if len(sentiment_series) > 0 else np.array([0]),
            'trend_slope': sentiment_trend_slope,         # <--- REAL TREND SLOPE
            'anomaly_score': batch_anomaly_score,
            'timestamp': datetime.now()
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
