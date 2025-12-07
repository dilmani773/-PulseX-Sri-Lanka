"""
Advanced Feature Engineering Module
Extracts mathematical and statistical features from text and temporal data
Refactored for Robustness (NaN Handling) and Integration
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from datetime import datetime, timedelta
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import stats
from collections import Counter
import re


class AdvancedFeatureExtractor:
    """
    Extracts multi-dimensional features showcasing mathematical understanding.
    Includes safeguards against sparse data (NaN protection).
    """
    
    def __init__(self):
        self.tfidf_vectorizer = TfidfVectorizer(max_features=100, ngram_range=(1, 3))
        
    def extract_temporal_features(self, timestamps: List[datetime]) -> Dict[str, float]:
        """Extract sophisticated temporal patterns using signal processing"""
        defaults = {
            'temporal_mean': 12.0, 'temporal_std': 0.0, 'temporal_skew': 0.0,
            'temporal_kurtosis': 0.0, 'periodicity_strength': 0.0,
            'dominant_period': 0.0, 'temporal_entropy': 0.0, 'burstiness': 0.0
        }
        
        if not timestamps:
            return defaults
        
        try:
            # Convert to hourly bins
            hours = np.array([ts.hour + ts.minute/60 for ts in timestamps])
            
            # Fourier Transform to detect periodicities
            fft = np.fft.fft(np.bincount(hours.astype(int), minlength=24))
            power_spectrum = np.abs(fft[:12])
            
            # Periodicity strength
            mean_power = np.mean(power_spectrum[1:]) + 1e-6
            period_strength = np.max(power_spectrum[1:]) / mean_power
            
            return {
                'temporal_mean': float(np.mean(hours)),
                'temporal_std': float(np.std(hours)),
                'temporal_skew': float(stats.skew(hours)),
                'temporal_kurtosis': float(stats.kurtosis(hours)),
                'periodicity_strength': float(period_strength),
                'dominant_period': float(np.argmax(power_spectrum[1:]) + 1),
                'temporal_entropy': float(stats.entropy(np.bincount(hours.astype(int), minlength=24) + 1)),
                'burstiness': float(np.std(hours) / (np.mean(hours) + 1e-6)),
            }
        except Exception:
            return defaults
    
    def extract_text_complexity_features(self, texts: List[str]) -> Dict[str, float]:
        """Measure text complexity using information theory"""
        defaults = {
            'vocabulary_richness': 0.0, 'hapax_legomena_ratio': 0.0,
            'zipf_exponent': 0.0, 'lexical_entropy': 0.0,
            'ttr': 0.0, 'avg_word_length': 0.0
        }
        
        all_text = ' '.join(texts)
        words = all_text.lower().split()
        
        if not words:
            return defaults
        
        try:
            # Word frequency distribution
            word_freq = Counter(words)
            freq_values = np.array(list(word_freq.values()))
            
            # Zipf's law deviation
            ranks = np.arange(1, len(freq_values) + 1)
            sorted_freqs = np.sort(freq_values)[::-1]
            
            zipf_slope = 0
            if len(sorted_freqs) > 10:
                log_ranks = np.log(ranks[:100])
                log_freqs = np.log(sorted_freqs[:100] + 1)
                zipf_slope, _ = np.polyfit(log_ranks, log_freqs, 1)
            
            unique_words = len(word_freq)
            total_words = len(words)
            
            return {
                'vocabulary_richness': unique_words / (total_words + 1e-6),
                'hapax_legomena_ratio': sum(1 for f in freq_values if f == 1) / unique_words,
                'zipf_exponent': float(abs(zipf_slope)),
                'lexical_entropy': float(stats.entropy(freq_values)),
                'ttr': unique_words / np.sqrt(total_words + 1),
                'avg_word_length': float(np.mean([len(w) for w in words])),
            }
        except Exception:
            return defaults
    
    def extract_network_features(self, texts: List[str]) -> Dict[str, float]:
        """Extract features from co-occurrence networks"""
        defaults = {
            'network_density': 0.0, 'edge_weight_mean': 0.0,
            'edge_weight_std': 0.0, 'clustering_proxy': 0.0
        }
        
        # Build word pairs
        word_pairs = []
        for text in texts:
            words = text.lower().split()
            for i in range(len(words) - 1):
                word_pairs.append((words[i], words[i+1]))
        
        if not word_pairs:
            return defaults
            
        try:
            pair_freq = Counter(word_pairs)
            edge_weights = np.array(list(pair_freq.values()))
            
            unique_words = len(set([w for pair in word_pairs for w in pair]))
            max_edges = unique_words * (unique_words - 1)
            
            return {
                'network_density': len(pair_freq) / (max_edges + 1e-6),
                'edge_weight_mean': float(np.mean(edge_weights)),
                'edge_weight_std': float(np.std(edge_weights)),
                'clustering_proxy': float(np.sum(edge_weights > 1) / (len(edge_weights) + 1e-6))
            }
        except Exception:
            return defaults
    
    def extract_sentiment_dynamics(self, sentiments: np.ndarray) -> Dict[str, float]:
        """Analyze sentiment time series dynamics"""
        # Default values if insufficient data
        defaults = {
            'sentiment_mean': 0.0, 'sentiment_std': 0.0,
            'sentiment_velocity_mean': 0.0, 'sentiment_acceleration_mean': 0.0,
            'sentiment_volatility': 0.0, 'sentiment_turning_points': 0.0,
            'sentiment_hurst': 0.5, 'trend_slope': 0.0  # Renamed from sentiment_trend
        }
        
        if len(sentiments) < 3:
            # If we have 1 or 2 points, just return mean/std, rest 0
            if len(sentiments) > 0:
                defaults['sentiment_mean'] = float(np.mean(sentiments))
                defaults['sentiment_std'] = float(np.std(sentiments))
            return defaults
        
        try:
            # First and second derivatives
            velocity = np.gradient(sentiments)
            acceleration = np.gradient(velocity)
            
            # Turning points
            turning_points = np.sum(np.abs(np.diff(np.sign(velocity))) > 0)
            
            # Hurst exponent (simplified)
            hurst = 0.5
            if len(sentiments) > 10:
                lags = range(2, min(20, len(sentiments)//2))
                tau = [np.std(np.subtract(sentiments[lag:], sentiments[:-lag])) for lag in lags]
                if len(tau) > 2:
                    poly = np.polyfit(np.log(lags), np.log(tau), 1)
                    hurst = poly[0] * 2.0
            
            return {
                'sentiment_mean': float(np.mean(sentiments)),
                'sentiment_std': float(np.std(sentiments)),
                'sentiment_velocity_mean': float(np.mean(np.abs(velocity))),
                'sentiment_acceleration_mean': float(np.mean(np.abs(acceleration))),
                'sentiment_volatility': float(np.std(np.diff(sentiments))),
                'sentiment_turning_points': float(turning_points / len(sentiments)),
                'sentiment_hurst': float(hurst),
                'trend_slope': float(np.polyfit(range(len(sentiments)), sentiments, 1)[0]),
            }
        except Exception:
            return defaults

    def create_feature_matrix(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create comprehensive feature matrix from raw data (DataFrame-ready)"""
        feature_dicts = []
        
        # Ensure 'scraped_at' is datetime
        if not pd.api.types.is_datetime64_any_dtype(df['scraped_at']):
            df['scraped_at'] = pd.to_datetime(df['scraped_at'])
            
        # Extract features per hour
        for timestamp, group in df.groupby(pd.Grouper(key='scraped_at', freq='1H')):
            # Initialize with timestamp
            features = {'timestamp': timestamp, 'article_count': len(group)}
            
            # 1. Temporal
            t_feats = self.extract_temporal_features(group['scraped_at'].tolist())
            features.update(t_feats)
            
            # 2. Text
            if 'content' in group.columns:
                txt_feats = self.extract_text_complexity_features(group['content'].tolist())
                features.update(txt_feats)
                net_feats = self.extract_network_features(group['content'].tolist())
                features.update(net_feats)
            
            # 3. Sentiment
            if 'sentiment_score' in group.columns:
                sent_feats = self.extract_sentiment_dynamics(group['sentiment_score'].values)
                features.update(sent_feats)
            
            # 4. Anomaly Indicators (Pre-calculated aggregation)
            if 'engagement' in group.columns and len(group) > 0:
                features['engagement_mean'] = group['engagement'].mean()
            else:
                features['engagement_mean'] = 0.0
                
            feature_dicts.append(features)
        
        # Create DataFrame and Fill NaNs to prevent crashes
        features_df = pd.DataFrame(feature_dicts)
        features_df = features_df.fillna(0)
        
        return features_df

# Example usage
if __name__ == "__main__":
    extractor = AdvancedFeatureExtractor()
    
    # Create dummy data including gaps
    data = {
        'scraped_at': [datetime.now()] * 5,
        'content': ["price rise crisis", "inflation high", "economy bad", "fuel shortage", "food cost"],
        'sentiment_score': [-0.8, -0.6, -0.9, -0.5, -0.7]
    }
    df = pd.DataFrame(data)
    
    print("Extracting features...")
    matrix = extractor.create_feature_matrix(df)
    print(matrix.iloc[0])
