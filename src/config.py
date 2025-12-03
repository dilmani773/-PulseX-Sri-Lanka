"""
PulseX Sri Lanka - Configuration Management
Centralized configuration for all system components
"""

import os
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    # python-dotenv is optional in minimal environments; continue without loading a .env file
    load_dotenv = lambda *a, **k: None

# Base directories
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = DATA_DIR / "models"

# Create directories if they don't exist
for dir_path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)


@dataclass
class DataSourceConfig:
    """Configuration for data sources"""
    
    # Sri Lankan News Sources
    NEWS_SOURCES: List[Dict] = None
    
    # Social Media APIs
    TWITTER_BEARER_TOKEN: str = os.getenv("TWITTER_BEARER_TOKEN", "")
    REDDIT_CLIENT_ID: str = os.getenv("REDDIT_CLIENT_ID", "")
    REDDIT_CLIENT_SECRET: str = os.getenv("REDDIT_CLIENT_SECRET", "")
    
    # Economic Data APIs
    WORLD_BANK_API: str = "https://api.worldbank.org/v2/country/LK/indicator"
    CBSL_API: str = "https://www.cbsl.gov.lk"
    
    # Update frequencies (in minutes)
    NEWS_UPDATE_FREQ: int = 15
    SOCIAL_UPDATE_FREQ: int = 5
    ECONOMIC_UPDATE_FREQ: int = 60
    
    def __post_init__(self):
        if self.NEWS_SOURCES is None:
            self.NEWS_SOURCES = [
                {
                    "name": "Ada Derana",
                    "url": "http://www.adaderana.lk",
                    "language": "en",
                    "category": "general"
                },
                {
                    "name": "Daily Mirror",
                    "url": "http://www.dailymirror.lk",
                    "language": "en",
                    "category": "general"
                },
                {
                    "name": "Hiru News",
                    "url": "https://www.hirunews.lk",
                    "language": "si",
                    "category": "general"
                },
                {
                    "name": "News First",
                    "url": "https://www.newsfirst.lk",
                    "language": "en",
                    "category": "general"
                }
            ]


@dataclass
class ModelConfig:
    """Configuration for ML models"""
    
    # Anomaly Detection
    ANOMALY_CONTAMINATION: float = 0.1
    ANOMALY_N_ESTIMATORS: int = 100
    
    # Sentiment Analysis
    SENTIMENT_MODEL: str = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
    SENTIMENT_THRESHOLD_POSITIVE: float = 0.6
    SENTIMENT_THRESHOLD_NEGATIVE: float = 0.4
    
    # Trend Analysis
    TREND_SEASONALITY_MODE: str = "multiplicative"
    TREND_CHANGEPOINT_PRIOR: float = 0.05
    
    # Risk Scoring
    RISK_WEIGHTS: Dict = None
    
    # Event Classification
    EVENT_CATEGORIES: List[str] = None
    
    def __post_init__(self):
        if self.RISK_WEIGHTS is None:
            self.RISK_WEIGHTS = {
                "sentiment": 0.25,
                "volatility": 0.20,
                "trending_score": 0.15,
                "anomaly_score": 0.20,
                "source_credibility": 0.10,
                "recency": 0.10
            }
        
        if self.EVENT_CATEGORIES is None:
            self.EVENT_CATEGORIES = [
                "Political",
                "Economic",
                "Natural Disaster",
                "Infrastructure",
                "Social Unrest",
                "Health Crisis",
                "Transportation",
                "Energy & Utilities",
                "Business & Trade",
                "Technology"
            ]


@dataclass
class DashboardConfig:
    """Configuration for dashboard"""
    
    TITLE: str = "PulseX Sri Lanka - Real-Time Business Intelligence"
    PAGE_ICON: str = "🇱🇰"
    LAYOUT: str = "wide"
    
    # Refresh intervals (in seconds)
    AUTO_REFRESH_INTERVAL: int = 30
    
    # Alert thresholds
    HIGH_RISK_THRESHOLD: float = 0.7
    MEDIUM_RISK_THRESHOLD: float = 0.4
    
    # Display limits
    MAX_NEWS_ITEMS: int = 50
    MAX_TRENDS: int = 10
    
    # Color scheme
    COLORS: Dict = None
    
    def __post_init__(self):
        if self.COLORS is None:
            self.COLORS = {
                "high_risk": "#EF4444",
                "medium_risk": "#F59E0B",
                "low_risk": "#10B981",
                "neutral": "#6B7280",
                "primary": "#3B82F6",
                "background": "#F9FAFB"
            }


@dataclass
class DatabaseConfig:
    """Configuration for database"""
    
    # Redis (for caching)
    REDIS_HOST: str = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT: int = int(os.getenv("REDIS_PORT", 6379))
    REDIS_DB: int = 0
    CACHE_TTL: int = 3600  # 1 hour
    
    # PostgreSQL (for persistent storage)
    POSTGRES_HOST: str = os.getenv("POSTGRES_HOST", "localhost")
    POSTGRES_PORT: int = int(os.getenv("POSTGRES_PORT", 5432))
    POSTGRES_DB: str = os.getenv("POSTGRES_DB", "pulsex")
    POSTGRES_USER: str = os.getenv("POSTGRES_USER", "postgres")
    POSTGRES_PASSWORD: str = os.getenv("POSTGRES_PASSWORD", "")
    
    @property
    def postgres_url(self) -> str:
        return f"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"


# Initialize configurations
DATA_CONFIG = DataSourceConfig()
MODEL_CONFIG = ModelConfig()
DASHBOARD_CONFIG = DashboardConfig()
DB_CONFIG = DatabaseConfig()


# Logging configuration
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        }
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "default"
        },
        "file": {
            "class": "logging.FileHandler",
            "filename": BASE_DIR / "logs" / "pulsex.log",
            "formatter": "default"
        }
    },
    "root": {
        "level": "INFO",
        "handlers": ["console", "file"]
    }
}

# Create logs directory
(BASE_DIR / "logs").mkdir(exist_ok=True)