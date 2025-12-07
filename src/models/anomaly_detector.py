"""
Hybrid Anomaly Detection System
Refactored for Real-Time Inference (Stateful)
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from typing import Tuple, List, Dict, Optional
import joblib
from pathlib import Path

class HybridAnomalyDetector:
    """
    Multi-algorithm anomaly detection optimized for Real-Time Streams.
    Separates 'fitting' (learning history) from 'scoring' (judging new data).
    """
    
    def __init__(self, contamination: float = 0.1, n_estimators: int = 100):
        self.contamination = contamination
        self.n_estimators = n_estimators
        
        # Models
        self.isolation_forest = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            random_state=42,
            n_jobs=-1
        )
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=0.95)
        self.knn = NearestNeighbors(n_neighbors=10) # Persist KNN model
        
        # State (Learned Statistics)
        self.is_fitted = False
        self.feature_importance_ = None
        self.train_median = None
        self.train_mad = None
        
        # Normalization factors (Min/Max scores seen during training)
        self.norm_params = {
            'stat': {'min': 0, 'max': 1},
            'iso': {'min': 0, 'max': 1},
            'dens': {'min': 0, 'max': 1},
            'recon': {'min': 0, 'max': 1}
        }
        
    def fit(self, X: np.ndarray) -> 'HybridAnomalyDetector':
        """
        Fit all detectors on HISTORICAL data.
        Must be called before using the dashboard.
        """
        # 1. Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # 2. Fit PCA
        self.pca.fit(X_scaled)
        
        # 3. Fit Isolation Forest
        self.isolation_forest.fit(X_scaled)
        
        # 4. Fit KNN (Density)
        # Use min(10, len-1) to avoid errors on small datasets
        k = min(10, max(1, len(X_scaled)-1))
        self.knn.set_params(n_neighbors=k)
        self.knn.fit(X_scaled)
        
        # 5. Calculate Statistical Baselines (Median & MAD) for future reference
        self.train_median = np.median(X_scaled, axis=0)
        self.train_mad = np.median(np.abs(X_scaled - self.train_median), axis=0)
        # Prevent division by zero
        self.train_mad = np.where(self.train_mad == 0, 1e-6, self.train_mad)
        
        # 6. Calibrate Normalization Ranges
        # We run the scores on the training set to find min/max for normalization
        stat_scores = self._raw_statistical_score(X_scaled)
        iso_scores = -self.isolation_forest.decision_function(X_scaled)
        dens_scores = self._raw_density_score(X_scaled)
        recon_scores = self._raw_reconstruction_score(X_scaled)
        
        self.norm_params['stat'] = {'min': stat_scores.min(), 'max': stat_scores.max()}
        self.norm_params['iso'] = {'min': iso_scores.min(), 'max': iso_scores.max()}
        self.norm_params['dens'] = {'min': dens_scores.min(), 'max': dens_scores.max()}
        self.norm_params['recon'] = {'min': recon_scores.min(), 'max': recon_scores.max()}
        
        self.is_fitted = True
        self._compute_feature_importance(X_scaled)
        
        return self

    # --- Raw Scoring Functions (Internal) ---

    def _raw_statistical_score(self, X_scaled: np.ndarray) -> np.ndarray:
        # Uses LEARNED median/mad, not current batch's median
        modified_z = 0.6745 * (X_scaled - self.train_median) / self.train_mad
        return np.sqrt(np.sum(modified_z ** 2, axis=1))

    def _raw_density_score(self, X_scaled: np.ndarray) -> np.ndarray:
        # Finds distance to training set neighbors
        distances, _ = self.knn.kneighbors(X_scaled)
        return np.mean(distances, axis=1)

    def _raw_reconstruction_score(self, X_scaled: np.ndarray) -> np.ndarray:
        X_transformed = self.pca.transform(X_scaled)
        X_reconstructed = self.pca.inverse_transform(X_transformed)
        return np.sum((X_scaled - X_reconstructed) ** 2, axis=1)

    # --- Public Methods ---

    def predict(self, X: np.ndarray, return_scores: bool = False) -> np.ndarray:
        """
        Predict on NEW data (single point or batch) without re-fitting.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Scale input using learned scaler
        X_scaled = self.scaler.transform(X)
        
        # Get Raw Scores
        raw_stat = self._raw_statistical_score(X_scaled)
        raw_iso = -self.isolation_forest.decision_function(X_scaled)
        raw_dens = self._raw_density_score(X_scaled)
        raw_recon = self._raw_reconstruction_score(X_scaled)
        
        # Normalize using learned ranges (Clip to 0-1)
        def norm(vals, key):
            p = self.norm_params[key]
            denom = p['max'] - p['min'] + 1e-6
            return np.clip((vals - p['min']) / denom, 0, 1)
            
        norm_stat = norm(raw_stat, 'stat')
        norm_iso = norm(raw_iso, 'iso')
        norm_dens = norm(raw_dens, 'dens')
        norm_recon = norm(raw_recon, 'recon')
        
        # Ensemble: weighted average
        weights = np.array([0.25, 0.35, 0.20, 0.20])
        ensemble_scores = (
            weights[0] * norm_stat +
            weights[1] * norm_iso +
            weights[2] * norm_dens +
            weights[3] * norm_recon
        )
        
        if return_scores:
            return ensemble_scores
        
        # Threshold (90th percentile of training data - approximated here)
        return (ensemble_scores > 0.7).astype(int) # Simplified threshold for runtime

    def _compute_feature_importance(self, X: np.ndarray):
        """Estimate feature importance (kept from original)"""
        n_features = X.shape[1]
        importances = np.zeros(n_features)
        baseline_score = -self.isolation_forest.decision_function(X)
        
        # Quick approximation using subset to save time
        subset_idx = np.random.choice(len(X), size=min(len(X), 100), replace=False)
        X_sub = X[subset_idx]
        baseline_sub = baseline_score[subset_idx]
        
        for i in range(n_features):
            X_permuted = X_sub.copy()
            np.random.shuffle(X_permuted[:, i])
            # Only use Isolation Forest for importance (fastest)
            permuted_score = -self.isolation_forest.decision_function(X_permuted)
            importances[i] = np.mean(np.abs(permuted_score - baseline_sub))
        
        self.feature_importance_ = importances / (importances.sum() + 1e-6)
    
    def get_anomaly_explanations(self, X: np.ndarray, feature_names: List[str] = None) -> List[Dict]:
        """Explain anomalies"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted")
        
        X_scaled = self.scaler.transform(X)
        scores = self.predict(X, return_scores=True)
        explanations = []
        
        for i, score in enumerate(scores):
            # Compare to TRAINED median
            instance = X_scaled[i]
            deviations = np.abs(instance - self.train_median)
            
            top_features_idx = np.argsort(deviations)[-3:]
            
            explanation = {
                'anomaly_score': float(score),
                'is_anomaly': bool(score > 0.7),
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
        """Save everything needed for inference"""
        joblib.dump({
            'isolation_forest': self.isolation_forest,
            'scaler': self.scaler,
            'pca': self.pca,
            'knn': self.knn,
            'train_median': self.train_median,
            'train_mad': self.train_mad,
            'norm_params': self.norm_params,
            'feature_importance': self.feature_importance_,
            'is_fitted': self.is_fitted
        }, path)
    
    def load(self, path: Path):
        """Load model"""
        data = joblib.load(path)
        self.isolation_forest = data['isolation_forest']
        self.scaler = data['scaler']
        self.pca = data['pca']
        self.knn = data['knn']
        self.train_median = data['train_median']
        self.train_mad = data['train_mad']
        self.norm_params = data['norm_params']
        self.feature_importance_ = data['feature_importance']
        self.is_fitted = data['is_fitted']
