"""
Bayesian Risk Assessment Engine
Combines multiple signals using probabilistic reasoning
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

class RiskLevel(Enum):
    """Risk classification"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    MINIMAL = "minimal"

@dataclass
class RiskAssessment:
    """Complete risk assessment output"""
    overall_score: float  # 0-1 scale
    risk_level: RiskLevel
    confidence: float  # 0-1 scale
    components: Dict[str, float]
    recommendations: List[str]
    explanation: str

class BayesianRiskScorer:
    """
    Sophisticated risk scoring using Bayesian inference.
    Now includes State Persistence and Soft-Bayesian Updates.
    """
    
    def __init__(self, weights: Dict[str, float] = None, history: Dict[str, List[float]] = None):
        """
        Initialize with component weights and optional history
        """
        self.weights = weights or {
            "sentiment": 0.25,
            "volatility": 0.20,
            "trending_score": 0.15,
            "anomaly_score": 0.20,
            "source_credibility": 0.10,
            "recency": 0.10
        }
        
        # Normalize weights
        total = sum(self.weights.values())
        self.weights = {k: v/total for k, v in self.weights.items()}
        
        # Prior distributions (Beta distributions for each component)
        self.priors = {
            component: stats.beta(a=2, b=2)  # Uninformative prior
            for component in self.weights.keys()
        }
        
        # FIXED: Load history if provided, otherwise start fresh
        if history:
            self.observations = history
        else:
            self.observations = {component: [] for component in self.weights.keys()}
    
    def sigmoid(self, x: float, k: float = 10, x0: float = 0.5) -> float:
        """Sigmoid transformation for smooth scaling"""
        return 1 / (1 + np.exp(-k * (x - x0)))
    
    def calculate_sentiment_risk(self, sentiment_score: float, 
                                  sentiment_volatility: float) -> Tuple[float, str]:
        """Calculate risk from sentiment analysis"""
        # Normalize sentiment to [0, 1]
        normalized_sentiment = (sentiment_score + 1) / 2
        
        # Risk increases with negative sentiment
        sentiment_risk = 1 - normalized_sentiment
        
        # Volatility amplifies risk (uncertainty penalty)
        volatility_penalty = self.sigmoid(sentiment_volatility, k=5, x0=0.3)
        
        combined_risk = sentiment_risk * (1 + 0.5 * volatility_penalty)
        combined_risk = np.clip(combined_risk, 0, 1)
        
        explanation = f"Sentiment: {sentiment_score:.2f}, Volatility: {sentiment_volatility:.2f}"
        
        return combined_risk, explanation
    
    def calculate_volatility_risk(self, values: np.ndarray) -> Tuple[float, str]:
        """Calculate risk from time series volatility"""
        if len(values) < 2:
            return 0.5, "Insufficient data"
        
        # Coefficient of variation (normalized volatility)
        mean_val = np.abs(np.mean(values))
        cv = np.std(values) / (mean_val + 1e-6)
        
        # Maximum drawdown
        cummax = np.maximum.accumulate(values)
        drawdown = (cummax - values) / (cummax + 1e-6)
        max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0
        
        # Combine metrics
        volatility_risk = self.sigmoid(cv, k=3, x0=0.5) * 0.6 + \
                          self.sigmoid(max_drawdown, k=5, x0=0.3) * 0.4
        
        explanation = f"CV: {cv:.2f}, Max Drawdown: {max_drawdown:.2f}"
        
        return volatility_risk, explanation
    
    def calculate_trending_risk(self, trend_slope: float, 
                                 trend_strength: float) -> Tuple[float, str]:
        """Risk from trending behavior"""
        # Negative slopes indicate increasing concern
        direction_risk = self.sigmoid(-trend_slope, k=5, x0=0)
        
        # Weak trends add uncertainty
        strength_factor = 1 - trend_strength  
        
        combined_risk = direction_risk * (1 + 0.3 * strength_factor)
        combined_risk = np.clip(combined_risk, 0, 1)
        
        explanation = f"Slope: {trend_slope:.3f}, Strength: {trend_strength:.2f}"
        
        return combined_risk, explanation
    
    def bayesian_update(self, component: str, observed_value: float) -> float:
        """
        Update belief about risk using Bayesian inference
        Uses conjugate Beta-Binomial model with soft updates
        """
        self.observations[component].append(observed_value)
        
        # Cap memory to prevent infinite growth (Performance Fix)
        if len(self.observations[component]) > 500:
             self.observations[component] = self.observations[component][-500:]

        # Get recent observations (sliding window)
        recent_obs = self.observations[component][-100:]
        
        if len(recent_obs) < 5:
            return observed_value
        
        # Soft update (Sum of probabilities rather than binary count)
        # This preserves the magnitude of risk (0.9 vs 0.6)
        successes = sum(recent_obs) 
        failures = len(recent_obs) - successes
        
        # Posterior mean calculation
        # Prior: Beta(α=2, β=2)
        alpha_post = 2 + successes
        beta_post = 2 + failures
        
        posterior_mean = alpha_post / (alpha_post + beta_post)
        
        # Blend observed value with posterior (shrinkage)
        blend_weight = min(len(recent_obs) / 50, 0.7)
        updated_value = blend_weight * posterior_mean + (1 - blend_weight) * observed_value
        
        return updated_value
    
    def assess_risk(self, indicators: Dict[str, any]) -> RiskAssessment:
        """Comprehensive risk assessment"""
        components = {}
        explanations = {}
        
        # 1. Sentiment risk
        if 'sentiment_score' in indicators:
            risk, exp = self.calculate_sentiment_risk(
                indicators['sentiment_score'],
                indicators.get('sentiment_volatility', 0.1)
            )
            components['sentiment'] = self.bayesian_update('sentiment', risk)
            explanations['sentiment'] = exp
        
        # 2. Volatility risk
        if 'time_series_values' in indicators:
            risk, exp = self.calculate_volatility_risk(indicators['time_series_values'])
            components['volatility'] = self.bayesian_update('volatility', risk)
            explanations['volatility'] = exp
        
        # 3. Trending risk
        if 'trend_slope' in indicators:
            risk, exp = self.calculate_trending_risk(
                indicators['trend_slope'],
                indicators.get('trend_strength', 0.5)
            )
            components['trending_score'] = self.bayesian_update('trending_score', risk)
            explanations['trending_score'] = exp
        
        # 4. Anomaly score
        if 'anomaly_score' in indicators:
            components['anomaly_score'] = self.bayesian_update(
                'anomaly_score',
                indicators['anomaly_score']
            )
            explanations['anomaly_score'] = f"Score: {indicators['anomaly_score']:.2f}"
        
        # 5. Source credibility
        if 'source_credibility' in indicators:
            components['source_credibility'] = 1 - indicators['source_credibility']
            explanations['source_credibility'] = f"Credibility: {indicators['source_credibility']:.2f}"
        
        # 6. Recency risk
        if 'timestamp' in indicators:
            try:
                # Handle both pandas Timestamp and python datetime
                ts = indicators['timestamp']
                now = pd.Timestamp.now() if hasattr(ts, 'tz') else pd.Timestamp.now().to_pydatetime()
                # Simple fallback if types mismatch, convert both to pandas
                if type(ts) != type(now):
                    ts = pd.to_datetime(ts)
                    now = pd.Timestamp.now()
                
                age_hours = (now - ts).total_seconds() / 3600
                recency_risk = self.sigmoid(age_hours, k=0.1, x0=24)
                components['recency'] = recency_risk
                explanations['recency'] = f"Age: {age_hours:.1f}h"
            except Exception as e:
                # Fallback if time calculation fails
                components['recency'] = 0.5
                explanations['recency'] = "Time error"
        
        # Calculate weighted overall score
        overall_score = sum(
            components.get(comp, 0.5) * weight
            for comp, weight in self.weights.items()
        )
        
        # Calculate confidence
        confidence = len(components) / len(self.weights)
        
        # Classify risk level
        if overall_score >= 0.8:
            risk_level = RiskLevel.CRITICAL
        elif overall_score >= 0.6:
            risk_level = RiskLevel.HIGH
        elif overall_score >= 0.4:
            risk_level = RiskLevel.MEDIUM
        elif overall_score >= 0.2:
            risk_level = RiskLevel.LOW
        else:
            risk_level = RiskLevel.MINIMAL
        
        # Generate recommendations and explanation
        recommendations = self._generate_recommendations(risk_level, components)
        explanation = self._create_explanation(components, explanations)
        
        return RiskAssessment(
            overall_score=overall_score,
            risk_level=risk_level,
            confidence=confidence,
            components=components,
            recommendations=recommendations,
            explanation=explanation
        )
    
    def _generate_recommendations(self, risk_level: RiskLevel, components: Dict[str, float]) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        if risk_level in [RiskLevel.CRITICAL, RiskLevel.HIGH]:
            recommendations.append("⚠️ Immediate attention required")
            recommendations.append("📊 Monitor situation closely (every 15 min)")
            if components.get('sentiment', 0) > 0.7:
                recommendations.append("💬 Public sentiment is negative - consider communication strategy")
            if components.get('volatility', 0) > 0.7:
                recommendations.append("📈 High volatility detected - prepare for rapid changes")
            if components.get('anomaly_score', 0) > 0.7:
                recommendations.append("🚨 Anomalous patterns detected - verify data sources")
        elif risk_level == RiskLevel.MEDIUM:
            recommendations.append("⚡ Moderate concern - stay vigilant")
            recommendations.append("📋 Review contingency plans")
        else:
            recommendations.append("✅ Situation stable")
            recommendations.append("📅 Continue regular monitoring")
        return recommendations
    
    def _create_explanation(self, components: Dict[str, float], explanations: Dict[str, str]) -> str:
        """Create human-readable explanation"""
        lines = ["Risk Assessment Breakdown:"]
        for comp, score in sorted(components.items(), key=lambda x: -x[1]):
            weight = self.weights.get(comp, 0)
            contrib = score * weight
            lines.append(f"  • {comp.replace('_', ' ').title()}: {score:.2%} (weight: {weight:.1%})")
            if comp in explanations:
                lines.append(f"    {explanations[comp]}")
        return "\n".join(lines)
