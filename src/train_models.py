"""
Complete Model Training Pipeline
Trains all models on historical data and saves them for dashboard use
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import pandas as pd
import numpy as np
from datetime import datetime
import logging
import json

from config import RAW_DATA_DIR, MODELS_DIR, PROCESSED_DATA_DIR
from preprocessing.feature_extractor import AdvancedFeatureExtractor
from models.anomaly_detector import HybridAnomalyDetector
from models.risk_scorer import BayesianRiskScorer
from models.sentiment_engine import SentimentAnalyzer
from models.trend_analyzer import TrendAnalyzer
from models.news_classifier import NewsClassifier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelTrainingPipeline:
    """
    Complete pipeline to train all models
    """
    
    def __init__(self):
        self.feature_extractor = AdvancedFeatureExtractor()
        self.anomaly_detector = HybridAnomalyDetector()
        self.risk_scorer = BayesianRiskScorer()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.trend_analyzer = TrendAnalyzer()
        
        self.historical_data = None
        self.features = None
        
    def load_historical_data(self):
        """Load historical data"""
        logger.info("Loading historical data...")
        
        metrics_path = RAW_DATA_DIR / 'historical_metrics.csv'
        
        if not metrics_path.exists():
            logger.error("Historical data not found. Run historical data collection first:")
            logger.error("  python src/data_ingestion/historical_collector.py")
            raise FileNotFoundError(f"No data at {metrics_path}")
        
        self.historical_data = pd.read_csv(metrics_path)
        self.historical_data['date'] = pd.to_datetime(self.historical_data['date'])
        
        logger.info(f"Loaded {len(self.historical_data)} days of historical data")
        return self.historical_data
    
    def prepare_training_features(self):
        """Prepare features for training"""
        logger.info("Preparing training features...")
        
        # Select numeric features
        feature_cols = [
            'article_count', 'avg_sentiment', 'sentiment_std',
            'sentiment_min', 'sentiment_max', 'avg_engagement',
            'engagement_std', 'unique_sources', 'unique_categories',
            'pct_political', 'pct_economic', 'pct_infrastructure', 'pct_tourism',
            'sentiment_velocity', 'sentiment_acceleration'
        ]
        
        self.features = self.historical_data[feature_cols].fillna(0).values
        
        logger.info(f"Prepared features shape: {self.features.shape}")
        return self.features
    
    def train_anomaly_detector(self):
        """Train anomaly detection model"""
        logger.info("Training anomaly detector...")
        
        self.anomaly_detector.fit(self.features)
        
        # Evaluate
        scores = self.anomaly_detector.predict(self.features, return_scores=True)
        predictions = self.anomaly_detector.predict(self.features)
        
        logger.info(f"Anomaly detector trained. Detected {predictions.sum()} anomalies")
        
        # Save model
        model_path = MODELS_DIR / 'anomaly_detector.pkl'
        self.anomaly_detector.save(model_path)
        logger.info(f"Saved model to {model_path}")
        
        return scores
    
    def train_risk_scorer(self, anomaly_scores: np.ndarray):
        """Train risk scoring model (update Bayesian priors)"""
        logger.info("Training risk scorer...")
        
        # Update Bayesian priors with historical observations
        for i in range(len(self.historical_data)):
            indicators = {
                'sentiment_score': self.historical_data.iloc[i]['avg_sentiment'],
                'sentiment_volatility': self.historical_data.iloc[i]['sentiment_std'],
                'time_series_values': np.random.randn(50),  # Simplified
                'trend_slope': self.historical_data.iloc[i]['sentiment_velocity'],
                'trend_strength': 0.5,
                'anomaly_score': anomaly_scores[i],
                'source_credibility': 0.8,
                'timestamp': self.historical_data.iloc[i]['date']
            }
            
            # This updates the Bayesian priors
            assessment = self.risk_scorer.assess_risk(indicators)
        
        logger.info("Risk scorer trained (Bayesian priors updated)")
        
        # Save (the scorer saves its learned priors)
        import joblib
        model_path = MODELS_DIR / 'risk_scorer.pkl'
        joblib.dump(self.risk_scorer, model_path)
        logger.info(f"Saved model to {model_path}")

    def train_news_classifier(self):
        """Train a lightweight news-category classifier (TF-IDF + LogisticRegression)

        Labels are generated using the rule-based `NewsClassifier` as weak labels.
        This produces a simple, reproducible classifier for the ingestion pipeline.
        """
        logger.info("Training news classifier (TF-IDF + LogisticRegression)...")

        news_path = RAW_DATA_DIR / 'historical_news.csv'
        if not news_path.exists():
            logger.warning(f"No historical news found at {news_path}; skipping news classifier training")
            return None

        import pandas as pd
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        import joblib

        df = pd.read_csv(news_path)
        if 'title' not in df.columns:
            logger.warning("historical_news.csv missing 'title' column; skipping news classifier training")
            return None

        titles = df['title'].fillna('').astype(str).values

        # Weak labels from rule-based classifier
        weak_clf = NewsClassifier()
        labels = [weak_clf.classify(t) for t in titles]

        # Keep only most frequent classes to avoid extreme class imbalance
        import collections
        counter = collections.Counter(labels)
        common = {c for c, _ in counter.most_common(10)}
        filtered = [(t, l) for t, l in zip(titles, labels) if l in common]

        if len(filtered) < 50:
            logger.warning("Not enough labeled samples for news classifier; skipping")
            return None

        X, y = zip(*filtered)

        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=20000, ngram_range=(1,2))),
            ('lr', LogisticRegression(max_iter=1000))
        ])

        pipeline.fit(X, y)

        model_path = MODELS_DIR / 'news_classifier.pkl'
        joblib.dump(pipeline, model_path)
        logger.info(f"Saved news classifier to {model_path}")
        return model_path
    
    def analyze_trends(self):
        """Analyze historical trends"""
        logger.info("Analyzing trends...")
        
        sentiment_series = self.historical_data['avg_sentiment'].values
        
        trend_info = self.trend_analyzer.detect_trend(sentiment_series)
        logger.info(f"Overall trend: {trend_info['direction']}, strength: {trend_info['strength']:.3f}")
        
        # Save trend analysis
        analysis_path = PROCESSED_DATA_DIR / 'trend_analysis.json'
        import json
        with open(analysis_path, 'w') as f:
            json.dump({
                'trend_direction': trend_info['direction'],
                'trend_slope': float(trend_info['slope']),
                'trend_strength': float(trend_info['strength']),
                'analyzed_at': datetime.now().isoformat()
            }, f, indent=2)
        
        logger.info(f"Saved trend analysis to {analysis_path}")
    
    def generate_training_report(self):
        """Generate training summary report"""
        logger.info("Generating training report...")
        
        report = {
            'training_completed_at': datetime.now().isoformat(),
            'training_data': {
                'samples': len(self.historical_data),
                'features': self.features.shape[1],
                'date_range': {
                    'start': self.historical_data['date'].min().isoformat(),
                    'end': self.historical_data['date'].max().isoformat()
                }
            },
            'models': {
                'anomaly_detector': {
                    'type': 'Hybrid Ensemble',
                    'saved_to': str(MODELS_DIR / 'anomaly_detector.pkl')
                },
                'risk_scorer': {
                    'type': 'Bayesian',
                    'saved_to': str(MODELS_DIR / 'risk_scorer.pkl')
                },
                'sentiment_analyzer': {
                    'type': 'Rule-based',
                    'note': 'No training required'
                },
                'trend_analyzer': {
                    'type': 'Statistical',
                    'note': 'No training required'
                }
            }
        }

        # If a news classifier was trained and saved, include it in the report
        news_path = MODELS_DIR / 'news_classifier.pkl'
        if news_path.exists():
            report['models']['news_classifier'] = {
                'type': 'TF-IDF + LogisticRegression',
                'saved_to': str(news_path)
            }
        
        report_path = MODELS_DIR / 'training_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Saved training report to {report_path}")
        
        return report
    
    def run_complete_training(self):
        """Run complete training pipeline"""
        print("\n" + "="*70)
        print("PULSEX SRI LANKA - MODEL TRAINING PIPELINE")
        print("="*70 + "\n")
        
        try:
            # Step 1: Load data
            self.load_historical_data()
            
            # Step 2: Prepare features
            self.prepare_training_features()
            
            # Step 3: Train anomaly detector
            anomaly_scores = self.train_anomaly_detector()
            
            # Step 4: Train risk scorer
            self.train_risk_scorer(anomaly_scores)

            # Step 5: Train news classifier
            try:
                news_model_path = self.train_news_classifier()
                if news_model_path:
                    logger.info(f"News classifier trained and saved to {news_model_path}")
            except Exception as e:
                logger.warning(f"News classifier training failed: {e}")
            
            # Step 5: Analyze trends
            self.analyze_trends()
            
            # Step 6: Generate report
            report = self.generate_training_report()
            
            print("\n" + "="*70)
            print("✅ TRAINING COMPLETE!")
            print("="*70)
            print(f"Trained on {len(self.historical_data)} days of data")
            print(f"Models saved to: {MODELS_DIR}")
            print("\nTrained models:")
            for model_name, model_info in report['models'].items():
                print(f"  • {model_name}: {model_info['type']}")
            print("\nNext step: Run dashboard to see predictions")
            print("  streamlit run src/dashboard/app.py")
            print("="*70 + "\n")
            
        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            raise


if __name__ == "__main__":
    pipeline = ModelTrainingPipeline()
    pipeline.run_complete_training()