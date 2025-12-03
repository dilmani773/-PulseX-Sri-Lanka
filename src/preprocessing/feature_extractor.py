"""
Advanced Feature Engineering Module
Extracts mathematical and statistical features from text and temporal data
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
    Extracts multi-dimensional features showcasing mathematical understanding
    """
    
    def __init__(self):
        self.tfidf_vectorizer = TfidfVectorizer(max_features=100, ngram_range=(1, 3))
        self.fitted = False
        
    def extract_temporal_features(self, timestamps: List[datetime]) -> Dict[str, float]:
        """
        Extract sophisticated temporal patterns using signal processing
        """
        if not timestamps:
            return {}
        
        # Convert to hourly bins
        hours = np.array([ts.hour + ts.minute/60 for ts in timestamps])
        
        # Fourier Transform to detect periodicities
        fft = np.fft.fft(np.bincount(hours.astype(int), minlength=24))
        power_spectrum = np.abs(fft[:12])  # First 12 frequencies
        
        # Statistical moments
        features = {
            'temporal_mean': np.mean(hours),
            'temporal_std': np.std(hours),
            'temporal_skew': stats.skew(hours),
            'temporal_kurtosis': stats.kurtosis(hours),
            
            # Periodicity strength (dominant frequency)
            'periodicity_strength': np.max(power_spectrum[1:]) / (np.mean(power_spectrum[1:]) + 1e-6),
            'dominant_period': np.argmax(power_spectrum[1:]) + 1,
            
            # Concentration measures (entropy-based)
            'temporal_entropy': stats.entropy(np.bincount(hours.astype(int), minlength=24) + 1),
            
            # Burstiness (coefficient of variation)
            'burstiness': np.std(hours) / (np.mean(hours) + 1e-6),
        }
        
        return features
    
    def extract_text_complexity_features(self, texts: List[str]) -> Dict[str, float]:
        """
        Measure text complexity using information theory
        """
        all_text = ' '.join(texts)
        words = all_text.lower().split()
        
        if not words:
            return {}
        
        # Word frequency distribution
        word_freq = Counter(words)
        freq_values = np.array(list(word_freq.values()))
        
        # Zipf's law deviation (should follow power law)
        ranks = np.arange(1, len(freq_values) + 1)
        sorted_freqs = np.sort(freq_values)[::-1]
        
        # Log-log slope (Zipf exponent)
        if len(sorted_freqs) > 10:
            log_ranks = np.log(ranks[:100])
            log_freqs = np.log(sorted_freqs[:100] + 1)
            zipf_slope, _ = np.polyfit(log_ranks, log_freqs, 1)
        else:
            zipf_slope = 0
        
        # Vocabulary richness
        unique_words = len(word_freq)
        total_words = len(words)
        
        features = {
            'vocabulary_richness': unique_words / (total_words + 1e-6),
            'hapax_legomena_ratio': sum(1 for f in freq_values if f == 1) / unique_words,
            'zipf_exponent': abs(zipf_slope),
            
            # Shannon entropy (information content)
            'lexical_entropy': stats.entropy(freq_values),
            
            # Type-token ratio
            'ttr': unique_words / np.sqrt(total_words + 1),
            
            # Average word length (complexity indicator)
            'avg_word_length': np.mean([len(w) for w in words]),
        }
        
        return features
    
    def extract_network_features(self, texts: List[str]) -> Dict[str, float]:
        """
        Extract features from co-occurrence networks
        """
        # Build word co-occurrence matrix
        word_pairs = []
        for text in texts:
            words = text.lower().split()
            for i in range(len(words) - 1):
                word_pairs.append((words[i], words[i+1]))
        
        if not word_pairs:
            return {}
        
        pair_freq = Counter(word_pairs)
        edge_weights = np.array(list(pair_freq.values()))
        
        # Network density
        unique_words = len(set([w for pair in word_pairs for w in pair]))
        max_edges = unique_words * (unique_words - 1)
        
        features = {
            'network_density': len(pair_freq) / (max_edges + 1e-6),
            'edge_weight_mean': np.mean(edge_weights),
            'edge_weight_std': np.std(edge_weights),
            'clustering_proxy': np.sum(edge_weights > 1) / (len(edge_weights) + 1e-6)
        }
        
        return features
    
    def extract_sentiment_dynamics(self, sentiments: np.ndarray, timestamps: List[datetime]) -> Dict[str, float]:
        """
        Analyze sentiment time series dynamics
        """
        if len(sentiments) < 3:
            return {}
        
        # First and second derivatives (velocity and acceleration)
        velocity = np.gradient(sentiments)
        acceleration = np.gradient(velocity)
        
        # Turning points (local maxima/minima)
        turning_points = np.sum(np.abs(np.diff(np.sign(velocity))) > 0)
        
        # Volatility (standard deviation of changes)
        volatility = np.std(np.diff(sentiments))
        
        # Hurst exponent (long-term memory)
        def hurst_exponent(ts):
            lags = range(2, min(20, len(ts)//2))
            tau = [np.std(np.subtract(ts[lag:], ts[:-lag])) for lag in lags]
            poly = np.polyfit(np.log(lags), np.log(tau), 1)
            return poly[0] * 2.0
        
        hurst = hurst_exponent(sentiments) if len(sentiments) > 20 else 0.5
        
        features = {
            'sentiment_mean': np.mean(sentiments),
            'sentiment_std': np.std(sentiments),
            'sentiment_velocity_mean': np.mean(np.abs(velocity)),
            'sentiment_acceleration_mean': np.mean(np.abs(acceleration)),
            'sentiment_volatility': volatility,
            'sentiment_turning_points': turning_points / len(sentiments),
            'sentiment_hurst': hurst,
            'sentiment_trend': np.polyfit(range(len(sentiments)), sentiments, 1)[0],
        }
        
        return features
    
    def extract_anomaly_indicators(self, values: np.ndarray) -> Dict[str, float]:
        """
        Statistical anomaly indicators using robust statistics
        """
        if len(values) < 3:
            return {}
        
        # Median Absolute Deviation (robust to outliers)
        median = np.median(values)
        mad = np.median(np.abs(values - median))
        
        # Modified Z-scores
        modified_z = 0.6745 * (values - median) / (mad + 1e-6)
        
        # Grubbs test statistic
        mean = np.mean(values)
        std = np.std(values)
        grubbs_stat = np.max(np.abs(values - mean)) / (std + 1e-6)
        
        features = {
            'mad': mad,
            'iqr': np.percentile(values, 75) - np.percentile(values, 25),
            'max_modified_z': np.max(np.abs(modified_z)),
            'grubbs_statistic': grubbs_stat,
            'outlier_ratio': np.sum(np.abs(modified_z) > 3.5) / len(values),
        }
        
        return features
    
    def create_feature_matrix(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create comprehensive feature matrix from raw data
        """
        feature_dicts = []
        
        # Extract features per group (e.g., per source or time window)
        for _, group in df.groupby(pd.Grouper(key='scraped_at', freq='1H')):
            if len(group) == 0:
                continue
            
            features = {}
            
            # Temporal features
            if 'scraped_at' in group.columns:
                features.update(self.extract_temporal_features(group['scraped_at'].tolist()))
            
            # Text features
            if 'content' in group.columns:
                features.update(self.extract_text_complexity_features(group['content'].tolist()))
                features.update(self.extract_network_features(group['content'].tolist()))
            
            # Sentiment dynamics (if available)
            if 'sentiment_score' in group.columns:
                features.update(self.extract_sentiment_dynamics(
                    group['sentiment_score'].values,
                    group['scraped_at'].tolist()
                ))
            
            # Anomaly indicators
            if 'engagement' in group.columns:
                features.update(self.extract_anomaly_indicators(group['engagement'].values))
            
            # Metadata
            features['timestamp'] = group['scraped_at'].iloc[0]
            features['article_count'] = len(group)
            
            feature_dicts.append(features)
        
        return pd.DataFrame(feature_dicts)


# Example usage
if __name__ == "__main__":
    # Test with sample data
    extractor = AdvancedFeatureExtractor()
    
    sample_timestamps = [datetime.now() - timedelta(hours=i) for i in range(100)]
    sample_texts = [f"Sample article text number {i} with varying content" for i in range(100)]
    
    temporal_features = extractor.extract_temporal_features(sample_timestamps)
    text_features = extractor.extract_text_complexity_features(sample_texts)
    
    print("Temporal Features:", temporal_features)
    print("\nText Features:", text_features)