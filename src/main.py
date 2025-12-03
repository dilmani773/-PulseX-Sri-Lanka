"""
PulseX Sri Lanka - Main Pipeline
Orchestrates end-to-end data collection, analysis, and serving
"""

import asyncio
import logging
from datetime import datetime, timedelta
from pathlib import Path
import json
import sys
from typing import Dict, List
import pandas as pd
import numpy as np

# Add src to path
sys.path.append(str(Path(__file__).parent))

from config import (
    DATA_CONFIG, MODEL_CONFIG, DB_CONFIG, 
    RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR
)
from data_ingestion.new_scraper import NewsScraperEngine
from preprocessing.feature_extractor import AdvancedFeatureExtractor
from models.anomaly_detector import HybridAnomalyDetector
from models.risk_scorer import BayesianRiskScorer, RiskLevel

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PulseXPipeline:
    """
    Main orchestration pipeline
    Coordinates all components of the system
    """
    
    def __init__(self):
        self.feature_extractor = AdvancedFeatureExtractor()
        self.anomaly_detector = HybridAnomalyDetector(
            contamination=MODEL_CONFIG.ANOMALY_CONTAMINATION,
            n_estimators=MODEL_CONFIG.ANOMALY_N_ESTIMATORS
        )
        self.risk_scorer = BayesianRiskScorer(
            weights=MODEL_CONFIG.RISK_WEIGHTS
        )
        
        self.news_data = None
        self.features = None
        self.anomalies = None
        self.risks = None
        
        logger.info("PulseX Pipeline initialized")
    
    async def ingest_data(self) -> pd.DataFrame:
        """
        Step 1: Ingest data from all sources
        """
        logger.info("Starting data ingestion...")
        
        all_articles = []
        
        # News scraping
        async with NewsScraperEngine(DATA_CONFIG.NEWS_SOURCES) as scraper:
            articles = await scraper.scrape_all()
            all_articles.extend(articles)
        
        # Convert to DataFrame
        df = pd.DataFrame([art.to_dict() for art in all_articles])
        
        if len(df) > 0:
            df['published_date'] = pd.to_datetime(df['published_date'])
            df['scraped_at'] = pd.to_datetime(df['scraped_at'])
        
        logger.info(f"Ingested {len(df)} articles from {df['source'].nunique() if len(df) > 0 else 0} sources")
        
        self.news_data = df
        return df
    
    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Step 2: Clean and preprocess data
        """
        logger.info("Preprocessing data...")
        
        if len(df) == 0:
            logger.warning("No data to preprocess")
            return df
        
        # Basic cleaning
        df = df.dropna(subset=['title', 'content'])
        df = df.drop_duplicates(subset=['id'])
        
        # Add synthetic sentiment for demo (in production, use actual sentiment model)
        df['sentiment_score'] = np.random.uniform(-1, 1, len(df))
        df['sentiment_volatility'] = np.random.uniform(0, 0.5, len(df))
        
        # Add engagement metrics (in production, fetch from actual sources)
        df['engagement'] = np.random.randint(10, 10000, len(df))
        
        logger.info(f"Preprocessed {len(df)} articles")
        
        return df
    
    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Step 3: Extract advanced features
        """
        logger.info("Extracting features...")
        
        if len(df) == 0:
            logger.warning("No data for feature extraction")
            return pd.DataFrame()
        
        # Create time-windowed features
        features_list = []
        
        # Group by hour for temporal features
        for timestamp, group in df.groupby(pd.Grouper(key='scraped_at', freq='1H')):
            if len(group) < 3:
                continue
            
            hour_features = {
                'timestamp': timestamp,
                'article_count': len(group)
            }
            
            # Temporal features
            temporal_feats = self.feature_extractor.extract_temporal_features(
                group['scraped_at'].tolist()
            )
            hour_features.update(temporal_feats)
            
            # Text complexity
            text_feats = self.feature_extractor.extract_text_complexity_features(
                group['content'].tolist()
            )
            hour_features.update(text_feats)
            
            # Sentiment dynamics
            if 'sentiment_score' in group.columns:
                sentiment_feats = self.feature_extractor.extract_sentiment_dynamics(
                    group['sentiment_score'].values,
                    group['scraped_at'].tolist()
                )
                hour_features.update(sentiment_feats)
            
            # Anomaly indicators
            if 'engagement' in group.columns:
                anomaly_feats = self.feature_extractor.extract_anomaly_indicators(
                    group['engagement'].values
                )
                hour_features.update(anomaly_feats)
            
            features_list.append(hour_features)
        
        features_df = pd.DataFrame(features_list)
        
        logger.info(f"Extracted {len(features_df)} feature vectors with {len(features_df.columns)} dimensions")
        
        self.features = features_df
        return features_df
    
    def detect_anomalies(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Step 4: Detect anomalies
        """
        logger.info("Detecting anomalies...")
        
        if len(features_df) < 10:
            logger.warning("Insufficient data for anomaly detection")
            return features_df
        
        # Select numeric features
        numeric_cols = features_df.select_dtypes(include=[np.number]).columns
        X = features_df[numeric_cols].fillna(0).values
        
        # Fit and predict (in production, load pre-trained model)
        self.anomaly_detector.fit(X)
        anomaly_scores = self.anomaly_detector.predict(X, return_scores=True)
        
        features_df['anomaly_score'] = anomaly_scores
        features_df['is_anomaly'] = anomaly_scores > np.percentile(anomaly_scores, 90)
        
        n_anomalies = features_df['is_anomaly'].sum()
        logger.info(f"Detected {n_anomalies} anomalies ({n_anomalies/len(features_df)*100:.1f}%)")
        
        self.anomalies = features_df[features_df['is_anomaly']]
        return features_df
    
    def assess_risks(self, df: pd.DataFrame, features_df: pd.DataFrame) -> List[Dict]:
        """
        Step 5: Assess risks using Bayesian scorer
        """
        logger.info("Assessing risks...")
        
        risk_assessments = []
        
        # Calculate risk for each time window
        for _, row in features_df.iterrows():
            indicators = {
                'sentiment_score': row.get('sentiment_mean', 0),
                'sentiment_volatility': row.get('sentiment_volatility', 0.1),
                'time_series_values': np.random.randn(50),  # Would use actual time series
                'trend_slope': row.get('sentiment_trend', 0),
                'trend_strength': 0.5,
                'anomaly_score': row.get('anomaly_score', 0.5),
                'source_credibility': 0.8,
                'timestamp': row.get('timestamp', datetime.now())
            }
            
            assessment = self.risk_scorer.assess_risk(indicators)
            
            risk_assessments.append({
                'timestamp': indicators['timestamp'],
                'risk_score': assessment.overall_score,
                'risk_level': assessment.risk_level.value,
                'confidence': assessment.confidence,
                'recommendations': assessment.recommendations,
                'explanation': assessment.explanation
            })
        
        logger.info(f"Generated {len(risk_assessments)} risk assessments")
        
        self.risks = risk_assessments
        return risk_assessments
    
    def generate_insights(self, df: pd.DataFrame, risks: List[Dict]) -> Dict:
        """
        Step 6: Generate actionable insights
        """
        logger.info("Generating insights...")
        
        if len(df) == 0:
            return {'status': 'no_data', 'insights': []}
        
        insights = {
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total_articles': len(df),
                'sources': df['source'].nunique(),
                'time_span_hours': (df['scraped_at'].max() - df['scraped_at'].min()).total_seconds() / 3600,
                'avg_sentiment': df['sentiment_score'].mean() if 'sentiment_score' in df.columns else 0
            },
            'risk_overview': {
                'current_risk': risks[-1]['risk_score'] if risks else 0.5,
                'risk_level': risks[-1]['risk_level'] if risks else 'medium',
                'high_risk_periods': sum(1 for r in risks if r['risk_score'] > 0.7)
            },
            'anomalies': {
                'detected': len(self.anomalies) if self.anomalies is not None else 0,
                'severity': 'high' if self.anomalies is not None and len(self.anomalies) > 5 else 'low'
            },
            'trending_topics': self._extract_trending_topics(df),
            'recommendations': risks[-1]['recommendations'] if risks else []
        }
        
        logger.info("Insights generated successfully")
        return insights
    
    def _extract_trending_topics(self, df: pd.DataFrame, top_n: int = 10) -> List[Dict]:
        """Extract trending topics from articles"""
        if len(df) == 0:
            return []
        
        # Simple word frequency analysis (in production, use proper topic modeling)
        from collections import Counter
        import re
        
        all_text = ' '.join(df['title'].tolist() + df['content'].tolist())
        words = re.findall(r'\b\w{4,}\b', all_text.lower())
        
        # Filter common words
        stopwords = {'that', 'this', 'with', 'from', 'have', 'been', 'will', 'said'}
        words = [w for w in words if w not in stopwords]
        
        word_counts = Counter(words)
        
        trending = []
        for word, count in word_counts.most_common(top_n):
            # Calculate sentiment for articles mentioning this word
            mask = df['title'].str.contains(word, case=False, na=False) | \
                   df['content'].str.contains(word, case=False, na=False)
            
            if mask.sum() > 0:
                avg_sentiment = df.loc[mask, 'sentiment_score'].mean() if 'sentiment_score' in df.columns else 0
                
                trending.append({
                    'topic': word.title(),
                    'volume': int(count),
                    'articles': int(mask.sum()),
                    'sentiment': float(avg_sentiment),
                    'trend': '↑' if count > 50 else '→'
                })
        
        return trending
    
    def save_outputs(self, insights: Dict):
        """
        Step 7: Save outputs
        """
        logger.info("Saving outputs...")
        
        timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save raw news
        if self.news_data is not None and len(self.news_data) > 0:
            news_path = RAW_DATA_DIR / f"news_{timestamp_str}.csv"
            self.news_data.to_csv(news_path, index=False)
            logger.info(f"Saved news data to {news_path}")
        
        # Save features
        if self.features is not None and len(self.features) > 0:
            features_path = PROCESSED_DATA_DIR / f"features_{timestamp_str}.csv"
            self.features.to_csv(features_path, index=False)
            logger.info(f"Saved features to {features_path}")
        
        # Save insights
        insights_path = PROCESSED_DATA_DIR / f"insights_{timestamp_str}.json"
        with open(insights_path, 'w') as f:
            json.dump(insights, f, indent=2, default=str)
        logger.info(f"Saved insights to {insights_path}")
        
        # Save models
        model_path = MODELS_DIR / "anomaly_detector_latest.pkl"
        self.anomaly_detector.save(model_path)
        logger.info(f"Saved model to {model_path}")
    
    async def run_pipeline(self):
        """
        Execute complete pipeline
        """
        logger.info("="*80)
        logger.info("Starting PulseX Pipeline Execution")
        logger.info("="*80)
        
        try:
            # Step 1: Ingest
            df = await self.ingest_data()
            
            if len(df) == 0:
                logger.error("No data ingested. Pipeline cannot continue.")
                return
            
            # Step 2: Preprocess
            df = self.preprocess_data(df)
            
            # Step 3: Extract features
            features_df = self.extract_features(df)
            
            if len(features_df) == 0:
                logger.error("No features extracted. Pipeline cannot continue.")
                return
            
            # Step 4: Detect anomalies
            features_df = self.detect_anomalies(features_df)
            
            # Step 5: Assess risks
            risks = self.assess_risks(df, features_df)
            
            # Step 6: Generate insights
            insights = self.generate_insights(df, risks)
            
            # Step 7: Save outputs
            self.save_outputs(insights)
            
            logger.info("="*80)
            logger.info("Pipeline Execution Completed Successfully")
            logger.info("="*80)
            
            # Print summary
            print("\n" + "="*80)
            print("PULSEX PIPELINE SUMMARY")
            print("="*80)
            print(f"Articles Processed: {insights['summary']['total_articles']}")
            print(f"Current Risk Level: {insights['risk_overview']['risk_level'].upper()}")
            print(f"Risk Score: {insights['risk_overview']['current_risk']:.2%}")
            print(f"Anomalies Detected: {insights['anomalies']['detected']}")
            print(f"\nTop Trending Topics:")
            for i, topic in enumerate(insights['trending_topics'][:5], 1):
                print(f"  {i}. {topic['topic']} ({topic['volume']} mentions, sentiment: {topic['sentiment']:+.2f})")
            print(f"\nRecommendations:")
            for i, rec in enumerate(insights['recommendations'][:3], 1):
                print(f"  {i}. {rec}")
            print("="*80 + "\n")
            
        except Exception as e:
            logger.error(f"Pipeline error: {str(e)}", exc_info=True)
            raise


async def main():
    """Main entry point"""
    pipeline = PulseXPipeline()
    await pipeline.run_pipeline()


if __name__ == "__main__":
    asyncio.run(main())