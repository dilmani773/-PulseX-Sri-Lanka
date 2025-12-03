"""
AI-Powered Recommendation Engine
Generates actionable business recommendations based on risk analysis
"""

from typing import List, Dict
from datetime import datetime
import pandas as pd


class RecommendationEngine:
    """
    Generates context-aware business recommendations
    """
    
    def __init__(self):
        self.recommendation_templates = self._load_templates()
        
    def _load_templates(self) -> Dict:
        """Load recommendation templates"""
        return {
            'high_risk': {
                'sentiment_negative': [
                    {
                        'action': 'Monitor public sentiment closely',
                        'reason': 'Negative sentiment spike detected',
                        'impact': 'Early intervention can prevent reputation damage',
                        'urgency': 'HIGH'
                    },
                    {
                        'action': 'Activate crisis communication protocol',
                        'reason': 'Sentiment declining rapidly',
                        'impact': 'Proactive communication builds trust',
                        'urgency': 'HIGH'
                    }
                ],
                'anomaly_detected': [
                    {
                        'action': 'Investigate anomalous patterns immediately',
                        'reason': 'Unusual activity detected in data streams',
                        'impact': 'Early detection prevents major disruptions',
                        'urgency': 'HIGH'
                    },
                    {
                        'action': 'Brief key stakeholders on unusual trends',
                        'reason': 'Anomaly score exceeds threshold',
                        'impact': 'Preparedness reduces response time',
                        'urgency': 'HIGH'
                    }
                ],
                'volatility': [
                    {
                        'action': 'Review supply chain contingencies',
                        'reason': 'High volatility in market indicators',
                        'impact': 'Protects operations from sudden changes',
                        'urgency': 'HIGH'
                    },
                    {
                        'action': 'Hedge against price fluctuations',
                        'reason': 'Volatility index elevated',
                        'impact': 'Stabilizes costs and protects margins',
                        'urgency': 'HIGH'
                    }
                ]
            },
            'medium_risk': {
                'trending': [
                    {
                        'action': 'Monitor trending topics for opportunities',
                        'reason': 'Emerging trends detected in public discourse',
                        'impact': 'Early positioning in trends = competitive advantage',
                        'urgency': 'MEDIUM'
                    },
                    {
                        'action': 'Adjust marketing strategy to align with trends',
                        'reason': 'Shift in public attention detected',
                        'impact': 'Increased engagement and relevance',
                        'urgency': 'MEDIUM'
                    }
                ],
                'economic': [
                    {
                        'action': 'Review pricing strategy',
                        'reason': 'Economic indicators showing moderate changes',
                        'impact': 'Maintains competitiveness and profitability',
                        'urgency': 'MEDIUM'
                    },
                    {
                        'action': 'Assess inventory levels',
                        'reason': 'Supply chain signals indicate potential shifts',
                        'impact': 'Prevents stockouts or excess inventory',
                        'urgency': 'MEDIUM'
                    }
                ]
            },
            'low_risk': {
                'opportunity': [
                    {
                        'action': 'Capitalize on positive sentiment',
                        'reason': 'Public sentiment trending positive',
                        'impact': 'Amplify messaging during favorable periods',
                        'urgency': 'LOW'
                    },
                    {
                        'action': 'Expand marketing efforts',
                        'reason': 'Stable environment with positive indicators',
                        'impact': 'Growth during favorable conditions',
                        'urgency': 'LOW'
                    }
                ],
                'maintenance': [
                    {
                        'action': 'Continue regular monitoring',
                        'reason': 'Situation stable',
                        'impact': 'Maintains awareness and preparedness',
                        'urgency': 'LOW'
                    }
                ]
            }
        }
    
    def generate_recommendations(self, 
                                risk_level: str,
                                sentiment_score: float,
                                anomaly_score: float,
                                volatility: float,
                                trending_topics: List[Dict]) -> List[Dict]:
        """
        Generate recommendations based on current situation
        
        Args:
            risk_level: 'critical', 'high', 'medium', 'low', 'minimal'
            sentiment_score: -1 to 1
            anomaly_score: 0 to 1
            volatility: 0 to 1
            trending_topics: List of trending topics with metadata
            
        Returns:
            List of recommendation dictionaries
        """
        recommendations = []
        
        # Risk-based recommendations
        if risk_level in ['critical', 'high']:
            # Sentiment-driven
            if sentiment_score < -0.3:
                recommendations.extend(
                    self.recommendation_templates['high_risk']['sentiment_negative'][:1]
                )
            
            # Anomaly-driven
            if anomaly_score > 0.7:
                recommendations.extend(
                    self.recommendation_templates['high_risk']['anomaly_detected'][:1]
                )
            
            # Volatility-driven
            if volatility > 0.6:
                recommendations.extend(
                    self.recommendation_templates['high_risk']['volatility'][:1]
                )
        
        elif risk_level == 'medium':
            # Trending-driven
            if trending_topics and len(trending_topics) > 5:
                recommendations.extend(
                    self.recommendation_templates['medium_risk']['trending'][:1]
                )
            
            # Economic-driven
            if volatility > 0.3:
                recommendations.extend(
                    self.recommendation_templates['medium_risk']['economic'][:1]
                )
        
        else:  # low or minimal risk
            # Opportunity-driven
            if sentiment_score > 0.3:
                recommendations.extend(
                    self.recommendation_templates['low_risk']['opportunity'][:1]
                )
            else:
                recommendations.extend(
                    self.recommendation_templates['low_risk']['maintenance'][:1]
                )
        
        # Topic-specific recommendations
        topic_recs = self._generate_topic_recommendations(trending_topics)
        recommendations.extend(topic_recs)
        
        # Ensure we have at least 3 recommendations
        while len(recommendations) < 3:
            recommendations.append({
                'action': 'Continue standard monitoring protocols',
                'reason': 'No immediate concerns detected',
                'impact': 'Maintains situational awareness',
                'urgency': 'LOW'
            })
        
        # Limit to top 5 recommendations
        return recommendations[:5]
    
    def _generate_topic_recommendations(self, trending_topics: List[Dict]) -> List[Dict]:
        """Generate recommendations based on trending topics"""
        recommendations = []
        
        if not trending_topics:
            return recommendations
        
        # Topic-specific rules
        topic_rules = {
            'fuel': {
                'action': 'Review fuel procurement and pricing strategy',
                'reason': 'Fuel-related discussions trending',
                'impact': 'Protects from fuel cost volatility',
                'urgency': 'MEDIUM'
            },
            'tourism': {
                'action': 'Enhance tourism-related services and marketing',
                'reason': 'Tourism interest increasing',
                'impact': 'Captures growing market opportunity',
                'urgency': 'LOW'
            },
            'economy': {
                'action': 'Reassess economic exposure and strategy',
                'reason': 'Economic discussions gaining traction',
                'impact': 'Aligns business with economic trends',
                'urgency': 'MEDIUM'
            },
            'infrastructure': {
                'action': 'Monitor infrastructure developments for opportunities',
                'reason': 'Infrastructure projects being discussed',
                'impact': 'Position for infrastructure-related growth',
                'urgency': 'LOW'
            },
            'weather': {
                'action': 'Implement weather-related contingency plans',
                'reason': 'Weather concerns in public discourse',
                'impact': 'Protects operations from weather disruptions',
                'urgency': 'MEDIUM'
            }
        }
        
        # Check top trending topics
        for topic in trending_topics[:3]:
            topic_name = topic.get('topic', '').lower()
            
            for keyword, rec in topic_rules.items():
                if keyword in topic_name:
                    recommendations.append(rec)
                    break
        
        return recommendations
    
    def prioritize_recommendations(self, recommendations: List[Dict]) -> List[Dict]:
        """Sort recommendations by urgency"""
        priority_order = {'HIGH': 0, 'MEDIUM': 1, 'LOW': 2}
        
        return sorted(
            recommendations,
            key=lambda x: priority_order.get(x.get('urgency', 'LOW'), 2)
        )
    
    def format_for_display(self, recommendations: List[Dict]) -> List[Dict]:
        """Format recommendations for dashboard display"""
        formatted = []
        
        for rec in recommendations:
            formatted.append({
                'priority': rec.get('urgency', 'LOW'),
                'action': rec.get('action', ''),
                'reason': rec.get('reason', ''),
                'impact': rec.get('impact', '')
            })
        
        return formatted


class ActionPlanGenerator:
    """
    Generates detailed action plans based on recommendations
    """
    
    def generate_action_plan(self, recommendations: List[Dict], 
                            time_horizon: str = '24h') -> Dict:
        """
        Generate a structured action plan
        
        Args:
            recommendations: List of recommendations
            time_horizon: '1h', '24h', '7d', '30d'
            
        Returns:
            Structured action plan
        """
        plan = {
            'generated_at': datetime.now().isoformat(),
            'time_horizon': time_horizon,
            'immediate_actions': [],
            'short_term_actions': [],
            'ongoing_monitoring': []
        }
        
        for rec in recommendations:
            urgency = rec.get('urgency', 'LOW')
            
            if urgency == 'HIGH':
                plan['immediate_actions'].append({
                    'action': rec['action'],
                    'timeline': 'Next 1-4 hours',
                    'owner': 'Crisis Management Team',
                    'success_metric': 'Issue contained and communication deployed'
                })
            elif urgency == 'MEDIUM':
                plan['short_term_actions'].append({
                    'action': rec['action'],
                    'timeline': 'Next 24-48 hours',
                    'owner': 'Operations Team',
                    'success_metric': 'Strategy adjusted and implemented'
                })
            else:
                plan['ongoing_monitoring'].append({
                    'action': rec['action'],
                    'timeline': 'Continuous',
                    'owner': 'Analytics Team',
                    'success_metric': 'Trends tracked and reported'
                })
        
        return plan


# Testing
if __name__ == "__main__":
    engine = RecommendationEngine()
    
    # Test recommendations
    recs = engine.generate_recommendations(
        risk_level='high',
        sentiment_score=-0.6,
        anomaly_score=0.8,
        volatility=0.7,
        trending_topics=[
            {'topic': 'Fuel Prices', 'volume': 1000},
            {'topic': 'Tourism', 'volume': 500}
        ]
    )
    
    print("\nGenerated Recommendations:")
    print("="*60)
    for i, rec in enumerate(recs, 1):
        print(f"\n{i}. [{rec['urgency']}] {rec['action']}")
        print(f"   Reason: {rec['reason']}")
        print(f"   Impact: {rec['impact']}")
    
    # Test action plan
    planner = ActionPlanGenerator()
    plan = planner.generate_action_plan(recs)
    
    print("\n\nAction Plan:")
    print("="*60)
    print(f"Generated: {plan['generated_at']}")
    print(f"Time Horizon: {plan['time_horizon']}")
    print(f"\nImmediate Actions: {len(plan['immediate_actions'])}")
    print(f"Short-term Actions: {len(plan['short_term_actions'])}")
    print(f"Ongoing Monitoring: {len(plan['ongoing_monitoring'])}")
    print("="*60)