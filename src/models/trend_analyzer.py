"""
Time Series Trend Analysis
Forecasts and detects trends in temporal data
"""

import numpy as np
import pandas as pd
from scipy import signal, stats
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)


class TrendAnalyzer:
    """
    Advanced trend detection and forecasting
    Uses multiple techniques for robust analysis
    """
    
    def __init__(self):
        pass
    
    def detect_trend(self, values: np.ndarray, timestamps: np.ndarray = None) -> Dict:
        """
        Detect trend in time series data
        Returns trend direction, strength, and statistics
        """
        if len(values) < 3:
            return {
                'direction': 'stable',
                'slope': 0.0,
                'strength': 0.0,
                'r_squared': 0.0,
                'p_value': 1.0
            }
        
        # Create x-axis (time indices)
        x = np.arange(len(values))
        
        # Linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, values)
        
        r_squared = r_value ** 2
        
        # Determine direction
        if p_value < 0.05:  # Statistically significant
            if slope > 0:
                direction = 'increasing'
            elif slope < 0:
                direction = 'decreasing'
            else:
                direction = 'stable'
        else:
            direction = 'stable'
        
        # Trend strength (0-1)
        strength = min(abs(r_squared), 1.0)
        
        return {
            'direction': direction,
            'slope': float(slope),
            'strength': float(strength),
            'r_squared': float(r_squared),
            'p_value': float(p_value),
            'intercept': float(intercept)
        }
    
    def decompose_time_series(self, values: np.ndarray, period: int = 7) -> Dict:
        """
        Decompose time series into trend, seasonal, and residual components
        """
        if len(values) < period * 2:
            return {
                'trend': values,
                'seasonal': np.zeros_like(values),
                'residual': np.zeros_like(values)
            }
        
        # Simple moving average for trend
        trend = self._moving_average(values, window=period)
        
        # Detrended series
        detrended = values - trend
        
        # Extract seasonal component (simplified)
        seasonal = self._extract_seasonal(detrended, period)
        
        # Residual
        residual = values - trend - seasonal
        
        return {
            'trend': trend,
            'seasonal': seasonal,
            'residual': residual,
            'trend_strength': self._calculate_component_strength(trend, values),
            'seasonal_strength': self._calculate_component_strength(seasonal, values)
        }
    
    def _moving_average(self, values: np.ndarray, window: int) -> np.ndarray:
        """Calculate moving average"""
        if len(values) < window:
            return values
        
        # Use convolution for moving average
        kernel = np.ones(window) / window
        ma = np.convolve(values, kernel, mode='same')
        
        return ma
    
    def _extract_seasonal(self, values: np.ndarray, period: int) -> np.ndarray:
        """Extract seasonal component"""
        if len(values) < period:
            return np.zeros_like(values)
        
        seasonal = np.zeros_like(values)
        
        for i in range(period):
            indices = np.arange(i, len(values), period)
            if len(indices) > 0:
                seasonal[indices] = np.mean(values[indices])
        
        return seasonal
    
    def _calculate_component_strength(self, component: np.ndarray, original: np.ndarray) -> float:
        """Calculate strength of a component"""
        var_component = np.var(component)
        var_original = np.var(original)
        
        if var_original == 0:
            return 0.0
        
        strength = var_component / var_original
        return min(strength, 1.0)
    
    def detect_changepoints(self, values: np.ndarray, threshold: float = 2.0) -> List[int]:
        """
        Detect significant changepoints in time series
        Uses CUSUM (Cumulative Sum) method
        """
        if len(values) < 5:
            return []
        
        # Calculate differences
        diffs = np.diff(values)
        
        # Standardize
        mean_diff = np.mean(diffs)
        std_diff = np.std(diffs)
        
        if std_diff == 0:
            return []
        
        z_scores = (diffs - mean_diff) / std_diff
        
        # CUSUM
        cusum_pos = np.zeros(len(z_scores))
        cusum_neg = np.zeros(len(z_scores))
        
        for i in range(1, len(z_scores)):
            cusum_pos[i] = max(0, cusum_pos[i-1] + z_scores[i])
            cusum_neg[i] = min(0, cusum_neg[i-1] + z_scores[i])
        
        # Find changepoints
        changepoints = []
        for i in range(len(cusum_pos)):
            if abs(cusum_pos[i]) > threshold or abs(cusum_neg[i]) > threshold:
                changepoints.append(i)
        
        return changepoints
    
    def forecast_simple(self, values: np.ndarray, periods: int = 7) -> Dict:
        """
        Simple forecast using linear extrapolation and moving average
        """
        if len(values) < 3:
            # Not enough data, use last value
            forecast = np.full(periods, values[-1] if len(values) > 0 else 0)
            return {
                'forecast': forecast,
                'lower_bound': forecast * 0.9,
                'upper_bound': forecast * 1.1,
                'method': 'last_value'
            }
        
        # Get trend
        trend_info = self.detect_trend(values)
        slope = trend_info['slope']
        intercept = trend_info['intercept']
        
        # Forecast using linear model
        x_future = np.arange(len(values), len(values) + periods)
        forecast = slope * x_future + intercept
        
        # Calculate prediction interval (simplified)
        residuals = values - (slope * np.arange(len(values)) + intercept)
        std_residual = np.std(residuals)
        
        # 95% prediction interval
        lower_bound = forecast - 1.96 * std_residual
        upper_bound = forecast + 1.96 * std_residual
        
        return {
            'forecast': forecast,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'method': 'linear',
            'trend_slope': slope
        }
    
    def calculate_momentum(self, values: np.ndarray, window: int = 5) -> np.ndarray:
        """
        Calculate momentum (rate of change)
        """
        if len(values) < window + 1:
            return np.zeros_like(values)
        
        momentum = np.zeros_like(values, dtype=float)
        
        for i in range(window, len(values)):
            momentum[i] = values[i] - values[i - window]
        
        return momentum
    
    def detect_cycles(self, values: np.ndarray, min_period: int = 3, max_period: int = 30) -> List[Dict]:
        """
        Detect periodic cycles using FFT
        """
        if len(values) < max_period:
            return []
        
        # Detrend first
        detrended = signal.detrend(values)
        
        # FFT
        fft_vals = np.fft.fft(detrended)
        power = np.abs(fft_vals) ** 2
        freqs = np.fft.fftfreq(len(values))
        
        # Find peaks in power spectrum
        positive_freqs = freqs[:len(freqs)//2]
        positive_power = power[:len(power)//2]
        
        # Find significant peaks
        threshold = np.mean(positive_power) + 2 * np.std(positive_power)
        peak_indices = np.where(positive_power > threshold)[0]
        
        cycles = []
        for idx in peak_indices:
            if idx > 0:  # Skip DC component
                period = 1 / positive_freqs[idx] if positive_freqs[idx] != 0 else 0
                if min_period <= period <= max_period:
                    cycles.append({
                        'period': float(period),
                        'strength': float(positive_power[idx] / np.max(positive_power))
                    })
        
        # Sort by strength
        cycles.sort(key=lambda x: x['strength'], reverse=True)
        
        return cycles[:3]  # Return top 3 cycles


# Testing
if __name__ == "__main__":
    analyzer = TrendAnalyzer()
    
    # Generate test data with trend and noise
    np.random.seed(42)
    t = np.arange(100)
    trend_component = 0.5 * t
    seasonal_component = 10 * np.sin(2 * np.pi * t / 7)  # Weekly cycle
    noise = np.random.randn(100) * 5
    values = trend_component + seasonal_component + noise + 50
    
    print("\nTrend Analysis Results:")
    print("="*60)
    
    # Detect trend
    trend_info = analyzer.detect_trend(values)
    print(f"\nTrend Detection:")
    print(f"  Direction: {trend_info['direction']}")
    print(f"  Slope: {trend_info['slope']:.3f}")
    print(f"  Strength: {trend_info['strength']:.3f}")
    print(f"  R-squared: {trend_info['r_squared']:.3f}")
    
    # Decompose
    decomposition = analyzer.decompose_time_series(values, period=7)
    print(f"\nTime Series Decomposition:")
    print(f"  Trend strength: {decomposition['trend_strength']:.3f}")
    print(f"  Seasonal strength: {decomposition['seasonal_strength']:.3f}")
    
    # Detect changepoints
    changepoints = analyzer.detect_changepoints(values)
    print(f"\nChangepoints detected: {len(changepoints)}")
    if changepoints:
        print(f"  At indices: {changepoints[:5]}")
    
    # Forecast
    forecast_result = analyzer.forecast_simple(values, periods=7)
    print(f"\n7-Day Forecast:")
    print(f"  Method: {forecast_result['method']}")
    print(f"  Values: {forecast_result['forecast'][:3]}...")
    
    # Detect cycles
    cycles = analyzer.detect_cycles(values)
    print(f"\nCycles detected: {len(cycles)}")
    for cycle in cycles:
        print(f"  Period: {cycle['period']:.1f}, Strength: {cycle['strength']:.3f}")
    
    print("="*60)