"""
Historical Data Collection for Training
Collects past news articles and events for model training

For real implementation, you would:
1. Use news archive APIs (NewsAPI, GDELT, etc.)
2. Scrape historical articles from news websites
3. Use economic databases (World Bank, IMF, CBSL archives)
4. Collect historical social media data (if available)

For competition: We'll generate realistic synthetic historical data
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import json
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)


class HistoricalDataCollector:
    """
    Collects historical data for model training
    
    IMPORTANT: For production, replace synthetic data with:
    - Real news archives (NewsAPI, GDELT Project)
    - Economic databases (World Bank, IMF, CBSL)
    - Social media archives
    """
    
    def __init__(self, years_back: int = 3):
        """
        Initialize collector
        
        Args:
            years_back: How many years of historical data to generate
                       (3 years is sufficient for this competition)
        """
        self.years_back = years_back
        self.start_date = datetime.now() - timedelta(days=365 * years_back)
        self.end_date = datetime.now()
        
    def generate_historical_news(self, samples_per_day: int = 50) -> pd.DataFrame:
        """
        Generate realistic historical news data
        
        In production: Replace with actual news archive scraping
        """
        logger.info(f"Generating {self.years_back} years of historical news data...")
        
        # Calculate total samples
        days = (self.end_date - self.start_date).days
        total_samples = days * samples_per_day
        
        # Date range
        dates = pd.date_range(start=self.start_date, end=self.end_date, periods=total_samples)
        
        # Categories based on real Sri Lankan news
        categories = [
            'Political', 'Economic', 'Infrastructure', 'Tourism', 
            'Agriculture', 'Health', 'Education', 'Technology',
            'Energy', 'Transportation', 'Social', 'Environment'
        ]
        
        # Sources
        sources = ['Ada Derana', 'Daily Mirror', 'Hiru News', 'News First', 
                  'The Island', 'Sunday Times', 'Colombo Gazette']
        
        # Generate realistic patterns
        data = []
        
        for i, date in enumerate(dates):
            # Create trends over time
            year_progress = (date - self.start_date).days / 365.0
            
            # Simulate economic crisis period (2022-2023)
            if 2 < year_progress < 3:
                sentiment_base = -0.6
                crisis_keywords = ['fuel', 'shortage', 'crisis', 'protest', 'inflation']
                topic_weight = 0.7
            else:
                sentiment_base = np.random.uniform(-0.3, 0.3)
                crisis_keywords = ['development', 'growth', 'tourism', 'investment']
                topic_weight = 0.3
            
            # Add noise
            sentiment = sentiment_base + np.random.normal(0, 0.2)
            sentiment = np.clip(sentiment, -1, 1)
            
            # Generate title based on period
            keyword = np.random.choice(crisis_keywords)
            category = np.random.choice(categories, p=self._category_probabilities(year_progress))
            
            title = self._generate_title(keyword, category, sentiment)
            
            article = {
                'id': f'hist_{i}',
                'title': title,
                'content': f'Article content about {keyword} in {category} sector...',
                'source': np.random.choice(sources),
                'category': category,
                'language': 'en',
                'published_date': date,
                'scraped_at': date,
                'sentiment_score': sentiment,
                'engagement': int(np.random.exponential(500) + 100),
            }
            
            data.append(article)
        
        df = pd.DataFrame(data)
        logger.info(f"Generated {len(df)} historical articles")
        
        return df
    
    def _category_probabilities(self, year_progress: float) -> np.ndarray:
        """Generate category probabilities based on time period"""
        # During crisis: more political, economic news
        if 2 < year_progress < 3:
            return np.array([0.25, 0.25, 0.1, 0.05, 0.05, 0.05, 0.05, 0.05, 0.1, 0.05, 0.0, 0.0])
        else:
            return np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.08, 0.08, 0.08, 0.08, 0.08, 0.05, 0.05])
    
    def _generate_title(self, keyword: str, category: str, sentiment: float) -> str:
        """Generate realistic title"""
        if sentiment < -0.3:
            prefixes = ['Crisis:', 'Concern over', 'Warning:', 'Decline in']
        elif sentiment > 0.3:
            prefixes = ['Growth in', 'Success:', 'Improvement in', 'Boost to']
        else:
            prefixes = ['Update on', 'Report:', 'Analysis:', 'Overview of']
        
        prefix = np.random.choice(prefixes)
        return f"{prefix} {keyword} affects {category} sector"
    
    def generate_historical_metrics(self, news_df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate aggregated historical metrics for training
        
        These are the features models will learn from
        """
        logger.info("Generating historical metrics...")
        
        # Group by day
        daily_metrics = []
        
        for date, group in news_df.groupby(news_df['published_date'].dt.date):
            if len(group) < 5:
                continue
            
            metrics = {
                'date': pd.to_datetime(date),
                'article_count': len(group),
                'avg_sentiment': group['sentiment_score'].mean(),
                'sentiment_std': group['sentiment_score'].std(),
                'sentiment_min': group['sentiment_score'].min(),
                'sentiment_max': group['sentiment_score'].max(),
                'avg_engagement': group['engagement'].mean(),
                'engagement_std': group['engagement'].std(),
                'unique_sources': group['source'].nunique(),
                'unique_categories': group['category'].nunique(),
            }
            
            # Add category distribution
            for cat in ['Political', 'Economic', 'Infrastructure', 'Tourism']:
                metrics[f'pct_{cat.lower()}'] = (group['category'] == cat).sum() / len(group)
            
            # Add derived features
            metrics['sentiment_velocity'] = 0  # Will calculate in next step
            metrics['sentiment_acceleration'] = 0
            
            daily_metrics.append(metrics)
        
        df_metrics = pd.DataFrame(daily_metrics)
        
        # Calculate temporal derivatives
        df_metrics['sentiment_velocity'] = df_metrics['avg_sentiment'].diff()
        df_metrics['sentiment_acceleration'] = df_metrics['sentiment_velocity'].diff()
        
        # Fill NaN
        df_metrics = df_metrics.fillna(0)
        
        logger.info(f"Generated {len(df_metrics)} days of metrics")
        
        return df_metrics
    
    def save_historical_data(self, output_dir: Path):
        """Save all historical data"""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate news data
        news_df = self.generate_historical_news()
        
        # Save raw news
        news_path = output_dir / 'historical_news.csv'
        news_df.to_csv(news_path, index=False)
        logger.info(f"Saved historical news to {news_path}")
        
        # Generate and save metrics
        metrics_df = self.generate_historical_metrics(news_df)
        metrics_path = output_dir / 'historical_metrics.csv'
        metrics_df.to_csv(metrics_path, index=False)
        logger.info(f"Saved historical metrics to {metrics_path}")
        
        # Save metadata
        metadata = {
            'generated_at': datetime.now().isoformat(),
            'years_back': self.years_back,
            'start_date': self.start_date.isoformat(),
            'end_date': self.end_date.isoformat(),
            'total_articles': len(news_df),
            'total_days': len(metrics_df),
            'sources': news_df['source'].unique().tolist(),
            'categories': news_df['category'].unique().tolist()
        }
        
        metadata_path = output_dir / 'metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Saved metadata to {metadata_path}")
        
        return news_df, metrics_df


# REAL DATA SOURCES (for production)
class RealDataCollector:
    """
    Template for collecting REAL historical data
    
    Use these APIs for actual data:
    """
    
    def collect_from_newsapi(self, api_key: str, years: int = 3):
        """
        Collect from NewsAPI (https://newsapi.org/)
        
        Example:
        from newsapi import NewsApiClient
        newsapi = NewsApiClient(api_key=api_key)
        articles = newsapi.get_everything(
            q='Sri Lanka',
            from_param=(datetime.now() - timedelta(days=365*years)).isoformat(),
            to=datetime.now().isoformat(),
            language='en',
            sort_by='publishedAt'
        )
        """
        pass
    
    def collect_from_gdelt(self):
        """
        Collect from GDELT Project (https://www.gdeltproject.org/)
        
        GDELT has 40+ years of global news data
        Free and comprehensive for academic/research use
        """
        pass
    
    def collect_from_wayback_machine(self, url: str, years: int = 3):
        """
        Scrape historical versions from Internet Archive
        
        Example:
        from wayback import WaybackClient
        client = WaybackClient()
        for record in client.search(url, from_date=start, to_date=end):
            # Scrape archived page
            pass
        """
        pass


if __name__ == "__main__":
    from config import RAW_DATA_DIR
    
    # Generate historical data
    collector = HistoricalDataCollector(years_back=3)
    news_df, metrics_df = collector.save_historical_data(RAW_DATA_DIR)
    
    print("\n" + "="*60)
    print("HISTORICAL DATA GENERATION COMPLETE")
    print("="*60)
    print(f"News articles: {len(news_df)}")
    print(f"Date range: {news_df['published_date'].min()} to {news_df['published_date'].max()}")
    print(f"Daily metrics: {len(metrics_df)}")
    print(f"Saved to: {RAW_DATA_DIR}")
    print("="*60)