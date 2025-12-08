"""
PulseX Sri Lanka - Main Pipeline
Integrated with Recommendation Engine and Full Data Fusion
"""

import asyncio
import logging
from datetime import datetime, timedelta
from pathlib import Path
import json
import sys
import pandas as pd
import numpy as np
from collections import Counter

# Ensure project root is on sys.path so `src.*` imports work when running
# this module as a script (python src/main.py).
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    DATA_CONFIG, MODEL_CONFIG, 
    RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR
)

# Ensure directories exist
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# --- IMPORTS ---
from src.data_ingestion.news_scraper import NewsScraperEngine
from data_ingestion.social_monitor import SocialMediaAggregator
from data_ingestion.weather_events import WeatherEventAggregator
from data_ingestion.economic_api import get_inflation

from preprocessing.feature_extractor import AdvancedFeatureExtractor
from preprocessing.text_cleaner import TextCleaner
from models.anomaly_detector import HybridAnomalyDetector
from models.risk_scorer import BayesianRiskScorer
from models.sentiment_engine import SentimentAnalyzer
from models.trend_analyzer import TrendAnalyzer

# --- NEW IMPORT: Business Logic ---
from dashboard.recommendations import RecommendationEngine

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PulseXPipeline:
    def __init__(self):
        # 1. Initialize Engines
        self.feature_extractor = AdvancedFeatureExtractor()
        self.text_cleaner = TextCleaner()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.trend_analyzer = TrendAnalyzer()
        self.recommendation_engine = RecommendationEngine()  # <--- NEW
        
        # 2. Initialize Data Collectors
        self.social_aggregator = SocialMediaAggregator()
        self.weather_aggregator = WeatherEventAggregator()
        
        # 3. Load Stateful Models
        self.anomaly_detector = HybridAnomalyDetector()
        model_path = MODELS_DIR / "anomaly_detector_latest.pkl"
        if model_path.exists():
            try:
                self.anomaly_detector.load(model_path)
            except: pass
        
        history_path = PROCESSED_DATA_DIR / "risk_history.json"
        history = None
        if history_path.exists():
            try:
                with open(history_path, 'r') as f: history = json.load(f)
            except: pass
        self.risk_scorer = BayesianRiskScorer(history=history)
        
    async def ingest_unified_data(self) -> pd.DataFrame:
        """Ingest News + Social"""
        logger.info("Starting Ingestion...")
        all_items = []

        # News
        try:
            async with NewsScraperEngine(DATA_CONFIG.NEWS_SOURCES) as scraper:
                articles = await scraper.scrape_all()
                for art in articles:
                    all_items.append({
                        'title': art.title, 'content': art.content,
                        'source': art.source, 'published_date': art.published_date,
                        'type': 'news'
                    })
        except Exception as e: logger.error(f"News failed: {e}")

        # Social
        try:
            social = await self.social_aggregator.collect_all(["economy", "fuel", "crisis"])
            for post in social.get('twitter', []) + social.get('reddit', []):
                all_items.append({
                    'title': post.content[:50] + "...", 'content': post.content,
                    'source': post.platform.title(), 
                    'published_date': datetime.fromisoformat(post.timestamp) if isinstance(post.timestamp, str) else post.timestamp,
                    'type': 'social'
                })
        except Exception as e: logger.error(f"Social failed: {e}")

        df = pd.DataFrame(all_items)
        if len(df) == 0:
            df = pd.DataFrame({'title': ["No Data"], 'content': ["Simulated content"], 'source': ["System"], 'published_date': [datetime.now()], 'type': ['news']})
        
        df['published_date'] = pd.to_datetime(df['published_date'])
        return df

    def extract_trending_topics(self, df: pd.DataFrame) -> list:
        """Helper to find top keywords for RecommendationEngine"""
        if df.empty: return []
        text = " ".join(df['clean_content'].astype(str))
        words = text.lower().split()
        # Filter basic stopwords (very simple list for speed)
        stops = {'the', 'and', 'is', 'in', 'to', 'of', 'a', 'for', 'on', 'with', 'sri', 'lanka'}
        words = [w for w in words if w not in stops and len(w) > 4]
        
        common = Counter(words).most_common(5)
        return [{'topic': word, 'volume': count} for word, count in common]

    async def run(self):
        # 1. Ingest
        df = await self.ingest_unified_data()
        
        # 2. Context
        weather = await self.weather_aggregator.collect_all()
        weather_risk = self.weather_aggregator.calculate_weather_risk_score(weather)
        try: inflation = get_inflation()['value'].iloc[-1]
        except: inflation = 0.0

        # 3. Clean & Analyze
        df['clean_content'] = df['content'].apply(lambda x: self.text_cleaner.clean_text(str(x))['cleaned'])
        df['sentiment_score'] = df['clean_content'].apply(lambda x: self.sentiment_analyzer.analyze_text(x)['compound'])
        
        # 4. Features & Trends
        df_feats = df.copy()
        df_feats['content'] = df['clean_content']
        df_feats['scraped_at'] = df_feats['published_date']  # Add scraped_at column
        feats_df = self.feature_extractor.create_feature_matrix(df_feats)
        
        sent_series = df.set_index('published_date')['sentiment_score'].resample('H').mean().fillna(0)
        trend_slope = self.trend_analyzer.detect_trend(sent_series.values)['slope'] if len(sent_series) > 3 else 0.0
        
        # 5. Anomaly
        if not self.anomaly_detector.is_fitted and len(feats_df) > 1:
            self.anomaly_detector.fit(feats_df.select_dtypes(include=[np.number]).fillna(0).values)
        
        if self.anomaly_detector.is_fitted and len(feats_df) > 0:
            feats_df['anomaly_score'] = self.anomaly_detector.predict(feats_df.select_dtypes(include=[np.number]).fillna(0).values, return_scores=True)
        else:
            feats_df['anomaly_score'] = 0.5

        # 6. Risk Scoring
        indicators = {
            'sentiment_score': df['sentiment_score'].mean(),
            'sentiment_volatility': df['sentiment_score'].std() + (weather_risk * 0.3),
            'time_series_values': sent_series.values,
            'trend_slope': trend_slope,
            'anomaly_score': feats_df['anomaly_score'].mean(),
            'timestamp': datetime.now()
        }
        risk_assess = self.risk_scorer.assess_risk(indicators)
        
        # 7. GENERATE ADVANCED RECOMMENDATIONS (The Fix)
        trending = self.extract_trending_topics(df)
        advanced_recs = self.recommendation_engine.generate_recommendations(
            risk_level=risk_assess.risk_level.value,
            sentiment_score=indicators['sentiment_score'],
            anomaly_score=indicators['anomaly_score'],
            volatility=indicators['sentiment_volatility'],
            trending_topics=trending
        )
        
        # 8. Save
        output = {
            "last_updated": datetime.now().isoformat(),
            "overall_risk": risk_assess.overall_score,
            "sentiment_avg": indicators['sentiment_score'],
            "active_alerts": int(len(df[df['sentiment_score'] < -0.5])),
            "total_articles": len(df),
            "recommendations": self.recommendation_engine.format_for_display(advanced_recs), # Use formatted version
            "risk_breakdown": risk_assess.components,
            "context": {
                "weather_summary": self.weather_aggregator.generate_weather_summary(weather),
                "weather_risk_score": weather_risk,
                "inflation_rate": inflation
            },
            "recent_news": df.head(15).to_dict(orient='records')
        }
        
        with open(PROCESSED_DATA_DIR / "dashboard_data.json", 'w') as f: json.dump(output, f, default=str)
        with open(PROCESSED_DATA_DIR / "risk_history.json", 'w') as f: json.dump(self.risk_scorer.observations, f)
        logger.info("Pipeline Completed Successfully")

if __name__ == "__main__":
    asyncio.run(PulseXPipeline().run())
