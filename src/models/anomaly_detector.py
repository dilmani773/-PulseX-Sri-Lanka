"""
Hybrid Anomaly Detection System
Combines multiple algorithms for robust detection
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from typing import Tuple, List, Dict
import joblib
from pathlib import Path


class HybridAnomalyDetector:
    """
    Multi-algorithm anomaly detection with ensemble voting
    Showcases understanding of:
    - Statistical methods (Z-score, MAD)
    - Tree-based isolation
    - Density estimation
    - Dimensionality reduction
    """
    
    def __init__(self, contamination: float = 0.1, n_estimators: int = 100):
        self.contamination = contamination
        self.n_estimators = n_estimators
        
        # Initialize models
        self.isolation_forest = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            random_state=42,
            n_jobs=-1
        )
        
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=0.95)  # Retain 95% variance
        
        self.is_fitted = False
        self.feature_importance_ = None
        
    def statistical_anomaly_score(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate anomaly scores using modified Z-score (MAD-based)
        More robust than standard Z-score to outliers
        """
        median = np.median(X, axis=0)
        mad = np.median(np.abs(X - median), axis=0)
        
        # Avoid division by zero
        mad = np.where(mad == 0, 1e-6, mad)
        
        # Modified Z-score
        modified_z = 0.6745 * (X - median) / mad
        
        # Aggregate across features (Mahalanobis-like)
        anomaly_scores = np.sqrt(np.sum(modified_z ** 2, axis=1))
        
        return anomaly_scores
    
    def isolation_anomaly_score(self, X: np.ndarray) -> np.ndarray:
        """
        Use Isolation Forest decision function
        Based on path length in random trees
        """
        # Negative scores mean anomalies
        scores = -self.isolation_forest.decision_function(X)
        
        # Normalize to [0, 1]
        scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-6)
        
        return scores
    
    def density_anomaly_score(self, X: np.ndarray, k: int = 10) -> np.ndarray:
        """
        Local Outlier Factor-inspired density estimation
        Uses k-nearest neighbors distance
        """
        from sklearn.neighbors import NearestNeighbors
        
        nbrs = NearestNeighbors(n_neighbors=min(k, len(X)-1))
        nbrs.fit(X)
        
        distances, _ = nbrs.kneighbors(X)
        
        # Average distance to k nearest neighbors
        avg_distances = np.mean(distances, axis=1)
        
        # Normalize
        scores = (avg_distances - avg_distances.min()) / (avg_distances.max() - avg_distances.min() + 1e-6)
        
        return scores
    
    def reconstruction_anomaly_score(self, X: np.ndarray) -> np.ndarray:
        """
        PCA reconstruction error
        Anomalies are poorly reconstructed
        """
        X_transformed = self.pca.transform(X)
        X_reconstructed = self.pca.inverse_transform(X_transformed)
        
        # Reconstruction error
        reconstruction_error = np.sum((X - X_reconstructed) ** 2, axis=1)
        
        # Normalize
        scores = (reconstruction_error - reconstruction_error.min()) / \
                 (reconstruction_error.max() - reconstruction_error.min() + 1e-6)
        
        return scores
    
    def fit(self, X: np.ndarray) -> 'HybridAnomalyDetector':
        """
        Fit all detectors
        """
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Fit PCA
        self.pca.fit(X_scaled)
        
        # Fit Isolation Forest
        self.isolation_forest.fit(X_scaled)
        
        self.is_fitted = True
        
        # Calculate feature importance (from Isolation Forest)
        self._compute_feature_importance(X_scaled)
        
        return self
    
    def _compute_feature_importance(self, X: np.ndarray):
        """
        Estimate feature importance using permutation importance
        """
        n_features = X.shape[1]
        importances = np.zeros(n_features)
        
        baseline_score = self.isolation_anomaly_score(X)
        
        for i in range(n_features):
            X_permuted = X.copy()
            np.random.shuffle(X_permuted[:, i])
            
            permuted_score = self.isolation_anomaly_score(X_permuted)
            
            # Importance = change in anomaly score
            importances[i] = np.mean(np.abs(permuted_score - baseline_score))
        
        self.feature_importance_ = importances / (importances.sum() + 1e-6)
    
    def predict(self, X: np.ndarray, return_scores: bool = False) -> np.ndarray:
        """
        Ensemble prediction using multiple methods
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Scale input
        X_scaled = self.scaler.transform(X)
        
        # Get scores from each method
        stat_scores = self.statistical_anomaly_score(X_scaled)
        iso_scores = self.isolation_anomaly_score(X_scaled)
        density_scores = self.density_anomaly_score(X_scaled)
        recon_scores = self.reconstruction_anomaly_score(X_scaled)
        
        # Ensemble: weighted average
        weights = np.array([0.25, 0.35, 0.20, 0.20])  # Tunable
        ensemble_scores = (
            weights[0] * stat_scores +
            weights[1] * iso_scores +
            weights[2] * density_scores +
            weights[3] * recon_scores
        )
        
        if return_scores:
            return ensemble_scores
        
        # Classify based on threshold
        threshold = np.percentile(ensemble_scores, (1 - self.contamination) * 100)
        predictions = (ensemble_scores > threshold).astype(int)
        
        return predictions
    
    def get_anomaly_explanations(self, X: np.ndarray, feature_names: List[str] = None) -> List[Dict]:
        """
        Explain why each instance is anomalous
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before explanation")
        
        X_scaled = self.scaler.transform(X)
        scores = self.predict(X_scaled, return_scores=True)
        
        explanations = []
        
        for i, score in enumerate(scores):
            # Find most deviant features
            instance = X_scaled[i]
            median = np.median(X_scaled, axis=0)
            deviations = np.abs(instance - median)
            
            top_features_idx = np.argsort(deviations)[-3:]  # Top 3
            
            explanation = {
                'anomaly_score': float(score),
                'is_anomaly': bool(score > np.percentile(scores, 90)),
                'contributing_features': []
            }
            
            for idx in top_features_idx:
                feat_name = feature_names[idx] if feature_names else f"feature_{idx}"
                explanation['contributing_features'].append({
                    'feature': feat_name,
                    'deviation': float(deviations[idx]),
                    'importance': float(self.feature_importance_[idx])
                })
            
            explanations.append(explanation)
        
        return explanations
    
    def save(self, path: Path):
        """Save model"""
        joblib.dump({
            'isolation_forest': self.isolation_forest,
            'scaler': self.scaler,
            'pca': self.pca,
            'feature_importance': self.feature_importance_,
            'is_fitted': self.is_fitted
        }, path)
    
    def load(self, path: Path):
        """Load model"""
        data = joblib.load(path)
        self.isolation_forest = data['isolation_forest']
        self.scaler = data['scaler']
        self.pca = data['pca']
        self.feature_importance_ = data['feature_importance']
        self.is_fitted = data['is_fitted']


# Example usage and testing
if __name__ == "__main__":
    # Generate synthetic data
    np.random.seed(42)
    
    # Normal data
    X_normal = np.random.randn(1000, 10)
    
    # Inject anomalies
    X_anomalies = np.random.randn(50, 10) * 3 + 5
    X = np.vstack([X_normal, X_anomalies])
    
    # Fit detector
    detector = HybridAnomalyDetector(contamination=0.05)
    detector.fit(X)
    
    # Predict
    predictions = detector.predict(X)
    scores = detector.predict(X, return_scores=True)
    
    print(f"Detected {predictions.sum()} anomalies")
    print(f"Score range: [{scores.min():.3f}, {scores.max():.3f}]")
    
    # Get explanations
    feature_names = [f"feature_{i}" for i in range(10)]
    explanations = detector.get_anomaly_explanations(X[:5], feature_names)
    
    print("\nSample explanations:")
    for i, exp in enumerate(explanations):
        print(f"\nInstance {i}:")
        print(f"  Anomaly score: {exp['anomaly_score']:.3f}")
        print(f"  Is anomaly: {exp['is_anomaly']}")
        print("  Top contributing features:")
        for feat in exp['contributing_features']:
            print(f"    - {feat['feature']}: deviation={feat['deviation']:.3f}")