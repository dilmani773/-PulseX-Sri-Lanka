"""
PulseX Sri Lanka - Main Pipeline
Orchestrates end-to-end data collection (News, Social, Weather, Econ),
Analysis (Sentiment, Trends, Anomalies), and Serving.
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

# --- IMPORTS FROM ALL MODULES ---
from data_ingestion.news_scraper import NewsScraperEngine
from data_ingestion.social_monitor import SocialMediaAggregator  # <--- NEW
from data_ingestion.weather_events import WeatherEventAggregator # <--- NEW
from data_ingestion.economic_api import get_inflation, get_gdp    # <--- NEW

from preprocessing.feature_extractor import AdvancedFeatureExtractor
from preprocessing.text_cleaner import TextCleaner
from models.anomaly_detector import HybridAnomalyDetector
from models.risk_scorer import BayesianRiskScorer
from models.sentiment_engine import SentimentAnalyzer
from models.trend_analyzer import TrendAnalyzer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PulseXPipeline:
    def __init__(self):
        # 1. Initialize Processing & Modeling Components
        self.feature_extractor = AdvancedFeatureExtractor()
        self.text_cleaner = TextCleaner()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.trend_analyzer = TrendAnalyzer()
        
        # 2. Initialize Data Collectors
        self.social_aggregator = SocialMediaAggregator()
        self.weather_aggregator = WeatherEventAggregator()
        
        # 3. Load Models (Stateful)
        self.anomaly_detector = HybridAnomalyDetector()
        model_path = MODELS_DIR / "anomaly_detector_latest.pkl"
        if model_path.exists():
            try:
                self.anomaly_detector.load(model_path)
                logger.info("Loaded existing anomaly detector model")
            except:
                logger.warning("Could not load existing model, starting fresh")
        
        history_path = PROCESSED_DATA_DIR / "risk_history.json"
        history = None
        if history_path.exists():
            try:
                with open(history_path, 'r') as f:
                    history = json.load(f)
            except:
                pass
        self.risk_scorer = BayesianRiskScorer(history=history)
        
    async def ingest_unified_data(self) -> pd.DataFrame:
        """
        Ingest AND Merge News + Social Media into a single DataFrame
        """
        logger.info("Starting Unified Ingestion...")
        all_items = []

        # A. Fetch News
        try:
            async with NewsScraperEngine(DATA_CONFIG.NEWS_SOURCES) as scraper:
                articles = await scraper.scrape_all()
                for art in articles:
                    all_items.append({
                        'title': art.title,
                        'content': art.content,
                        'source': art.source, # e.g., "Daily Mirror"
                        'published_date': art.published_date,
                        'type': 'news'
                    })
        except Exception as e:
            logger.error(f"News scraper failed: {e}")

        # B. Fetch Social Media (Treat as micro-news)
        try:
            social_data = await self.social_aggregator.collect_all(["economy", "fuel", "crisis", "politics"])
            
            # Flatten Twitter
            for post in social_data.get('twitter', []):
                all_items.append({
                    'title': post.content[:50] + "...", # Use start of tweet as title
                    'content': post.content,
                    'source': 'Twitter',
                    'published_date': datetime.fromisoformat(post.timestamp) if isinstance(post.timestamp, str) else post.timestamp,
                    'type': 'social'
                })
            
            # Flatten Reddit
            for post in social_data.get('reddit', []):
                all_items.append({
                    'title': post.content[:50] + "...",
                    'content': post.content,
                    'source': 'Reddit',
                    'published_date': datetime.fromisoformat(post.timestamp) if isinstance(post.timestamp, str) else post.timestamp,
                    'type': 'social'
                })
        except Exception as e:
            logger.error(f"Social scraper failed: {e}")

        # C. Create Unified DataFrame
        df = pd.DataFrame(all_items)
        
        # Fallback if empty
        if len(df) == 0:
            logger.warning("No data collected. Using fallback.")
            df = pd.DataFrame({
                'title': ["Fallback Data"] * 10,
                'content': ["System is running on simulated data."] * 10,
                'source': ["System"] * 10,
                'published_date': [datetime.now()] * 10,
                'type': ['news'] * 10
            })
            
        df['published_date'] = pd.to_datetime(df['published_date'])
        return df

    async def collect_environment_context(self):
        """
        Collect Weather and Economic Context (Non-text signals)
        """
        # 1. Weather
        weather_data = await self.weather_aggregator.collect_all()
        weather_risk = self.weather_aggregator.calculate_weather_risk_score(weather_data)
        weather_summary = self.weather_aggregator.generate_weather_summary(weather_data)
        
        # 2. Economics (Cached)
        try:
            inflation_df = get_inflation(start=2020)
            latest_inflation = inflation_df['value'].iloc[-1] if not inflation_df.empty else 5.0
        except:
            latest_inflation = 0.0
            
        return {
            "weather_risk": weather_risk,
            "weather_summary": weather_summary,
            "latest_inflation": latest_inflation
        }

    def run_analysis(self, df: pd.DataFrame, context: dict):
        if df.empty:
            return df, pd.DataFrame(), None

        # 1. CLEANING & SENTIMENT
        logger.info("Running NLP Pipeline...")
        df['clean_content'] = df['content'].apply(
            lambda x: self.text_cleaner.clean_text(str(x), remove_stopwords=False)['cleaned']
        )
        df['sentiment_results'] = df['clean_content'].apply(self.sentiment_analyzer.analyze_text)
        df['sentiment_score'] = df['sentiment_results'].apply(lambda x: x['compound'])
        
        # 2. FEATURE EXTRACTION
        df_for_features = df.copy()
        df_for_features['content'] = df['clean_content'] 
        features_df = self.feature_extractor.create_feature_matrix(df_for_features)
        
        # 3. TREND ANALYSIS
        sentiment_series = df.set_index('published_date')['sentiment_score'].resample('H').mean().fillna(0)
        if len(sentiment_series) > 3:
            trend_info = self.trend_analyzer.detect_trend(sentiment_series.values)
            sentiment_trend_slope = trend_info['slope']
        else:
            sentiment_trend_slope = 0.0

        # 4. ANOMALY DETECTION
        if not self.anomaly_detector.is_fitted:
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
        
        # 5. RISK SCORING (Fusing Text + Weather + Trends)
        batch_anomaly_score = features_df['anomaly_score'].mean()
        
        # FUSION LOGIC:
        # If Weather Risk is High, we artificially boost the 'volatility' signal sent to the Bayesian Model
        combined_volatility = df['sentiment_score'].std() + (context['weather_risk'] * 0.5)
        
        indicators = {
            'sentiment_score': df['sentiment_score'].mean(),
            'sentiment_volatility': combined_volatility, 
            'time_series_values': sentiment_series.values if len(sentiment_series) > 0 else np.array([0]),
            'trend_slope': sentiment_trend_slope,         
            'anomaly_score': batch_anomaly_score,
            'timestamp': datetime.now()
        }
        latest_risk = self.risk_scorer.assess_risk(indicators)

        return df, features_df, latest_risk

    def save_results(self, df, risk_assessment, context):
        output = {
            "last_updated": datetime.now().isoformat(),
            "overall_risk": risk_assessment.overall_score if risk_assessment else 0.5,
            "risk_level": risk_assessment.risk_level.value if risk_assessment else "medium",
            "sentiment_avg": float(df['sentiment_score'].mean()) if not df.empty else 0.0,
            "total_articles": len(df),
            "data_breakdown": {
                "news_count": int(len(df[df['type']=='news'])),
                "social_count": int(len(df[df['type']=='social']))
            },
            # Economic & Weather Context injected here for Dashboard
            "context": {
                "weather_summary": context['weather_summary'],
                "weather_risk_score": context['weather_risk'],
                "inflation_rate": context['latest_inflation']
            },
            "recommendations": risk_assessment.recommendations if risk_assessment else [],
            "risk_breakdown": risk_assessment.components if risk_assessment else {},
            "recent_news": df.head(10).to_dict(orient='records') if not df.empty else []
        }
        
        with open(PROCESSED_DATA_DIR / "dashboard_data.json", 'w') as f:
            json.dump(output, f, default=str)
            
        with open(PROCESSED_DATA_DIR / "risk_history.json", 'w') as f:
            json.dump(self.risk_scorer.observations, f)
            
        logger.info("Saved integrated dashboard_data.json")

    async def run(self):
        # 1. Collect Text Data (News + Social)
        df = await self.ingest_unified_data()
        
        # 2. Collect Context (Weather + Econ)
        context = await self.collect_environment_context()
        
        # 3. Analyze
        df, feats, risk = self.run_analysis(df, context)
        
        # 4. Save
        self.save_results(df, risk, context)

if __name__ == "__main__":
    asyncio.run(PulseXPipeline().run())
