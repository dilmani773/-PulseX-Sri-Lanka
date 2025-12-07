"""
AI-Powered Recommendation Engine (Generative + Template Fallback)
Generates actionable business recommendations using LLMs or robust templates.
"""

from typing import List, Dict
import random
import os
import json
import logging

# Try to import OpenAI (Simulated for demo if missing)
try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

logger = logging.getLogger(__name__)

class RecommendationEngine:
    def __init__(self):
        self.recommendation_templates = self._load_templates()
        # Initialize OpenAI client if key is present
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=self.api_key) if (HAS_OPENAI and self.api_key) else None

    def generate_recommendations(self, 
                                risk_level: str,
                                sentiment_score: float,
                                anomaly_score: float,
                                volatility: float,
                                trending_topics: List[Dict]) -> List[Dict]:
        """
        Main entry point: Tries Generative AI first, falls back to Templates.
        """
        # 1. Try Generative AI (If configured)
        if self.client:
            try:
                logger.info("Attempting to generate AI recommendations...")
                return self._generate_with_llm(risk_level, sentiment_score, trending_topics)
            except Exception as e:
                logger.error(f"AI Generation failed: {e}. Falling back to templates.")
        
        # 2. Fallback to Rule-Based Templates (Robust & Free)
        return self._generate_from_templates(risk_level, sentiment_score, anomaly_score, volatility, trending_topics)

    def _generate_with_llm(self, risk_level, sentiment, trends) -> List[Dict]:
        """Call OpenAI to generate unique advice"""
        
        trend_text = ", ".join([t['topic'] for t in trends[:3]]) if trends else "General Market Conditions"
        
        prompt = f"""
        You are a Crisis Risk Consultant for a Sri Lankan business conglomerate.
        Current Situation:
        - Risk Level: {risk_level.upper()}
        - Public Sentiment Score: {sentiment:.2f} (-1.0 is bad, +1.0 is good)
        - Trending Topics: {trend_text}

        Generate 3 specific, actionable business recommendations in JSON format.
        Each recommendation must have:
        - "priority": "HIGH", "MEDIUM", or "LOW"
        - "action": Short title (max 5 words)
        - "reason": Why this is needed (max 10 words)
        - "impact": Business outcome (max 10 words)
        
        Return ONLY valid JSON array.
        """

        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo", # Or gpt-4
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        
        content = response.choices[0].message.content
        # Basic cleanup to ensure it's pure JSON
        content = content.replace("```json", "").replace("```", "").strip()
        return json.loads(content)

    def _generate_from_templates(self, risk_level, sentiment_score, anomaly_score, volatility, trending_topics):
        """Standard Rule-Based Logic (The reliable fallback)"""
        recommendations = []
        
        # 1. Topic-Specific (The "Smart" Template part)
        topic_recs = self._generate_topic_recommendations(trending_topics)
        recommendations.extend(topic_recs)
        
        # 2. Risk-Based
        if risk_level in ['critical', 'high']:
            source = self.recommendation_templates['high_risk']
            if sentiment_score < -0.3: recommendations.append(random.choice(source['sentiment_negative']))
            if anomaly_score > 0.7: recommendations.append(random.choice(source['anomaly_detected']))
            if volatility > 0.6: recommendations.append(random.choice(source['volatility']))
        elif risk_level == 'medium':
            recommendations.extend(random.sample(self.recommendation_templates['medium_risk']['generic'], 2))
        else:
            recommendations.extend(random.sample(self.recommendation_templates['low_risk']['generic'], 1))
            
        # 3. Fillers
        if len(recommendations) < 3:
            recommendations.append({
                'action': 'Maintain Situational Awareness',
                'reason': 'Stable indicators observed',
                'impact': 'Business continuity',
                'urgency': 'LOW'
            })
            
        # Deduplicate
        unique = {rec['action']: rec for rec in recommendations}.values()
        return list(unique)[:4]

    def _load_templates(self) -> Dict:
        """Robust backup templates"""
        return {
            'high_risk': {
                'sentiment_negative': [
                    {'action': 'Activate Crisis Protocol', 'reason': 'Sentiment collapsing', 'impact': 'Protects brand reputation', 'urgency': 'HIGH'},
                    {'action': 'Issue Stakeholder Alert', 'reason': 'Negative noise rising', 'impact': 'Transparency builds trust', 'urgency': 'HIGH'}
                ],
                'anomaly_detected': [
                    {'action': 'Audit Data Streams', 'reason': 'Critical anomaly found', 'impact': 'Prevents operational failure', 'urgency': 'HIGH'},
                    {'action': 'Deploy Rapid Response', 'reason': 'Unusual pattern detected', 'impact': 'Mitigates immediate risk', 'urgency': 'HIGH'}
                ],
                'volatility': [
                    {'action': 'Hedge Assets', 'reason': 'Extreme market volatility', 'impact': 'Financial stability', 'urgency': 'HIGH'},
                    {'action': 'Secure Supply Chain', 'reason': 'Disruption likely', 'impact': 'Operational continuity', 'urgency': 'HIGH'}
                ]
            },
            'medium_risk': {
                'generic': [
                    {'action': 'Increase Oversight', 'reason': 'Risk trending up', 'impact': 'Early warning', 'urgency': 'MEDIUM'},
                    {'action': 'Review Inventory', 'reason': 'Market shifting', 'impact': 'Prevents shortages', 'urgency': 'MEDIUM'},
                    {'action': 'Update Forecasts', 'reason': 'New data patterns', 'impact': 'Strategic alignment', 'urgency': 'MEDIUM'}
                ]
            },
            'low_risk': {
                'generic': [
                    {'action': 'Optimize Operations', 'reason': 'Stable environment', 'impact': 'Efficiency gains', 'urgency': 'LOW'},
                    {'action': 'Explore Expansion', 'reason': 'Positive sentiment', 'impact': 'Revenue growth', 'urgency': 'LOW'}
                ]
            }
        }

    def _generate_topic_recommendations(self, trending_topics: List[Dict]) -> List[Dict]:
        recs = []
        if not trending_topics: return recs
        
        # Dynamic Topic Rules
        topic_map = {
            'fuel': {'action': 'Secure Fuel Reserves', 'reason': 'Fuel detected in trends', 'impact': 'Operational continuity', 'urgency': 'HIGH'},
            'dollar': {'action': 'Hedge Forex Exposure', 'reason': 'Currency fluctuation', 'impact': 'Protects financial position', 'urgency': 'HIGH'},
            'flood': {'action': 'Activate Flood Protocols', 'reason': 'Flood warnings trending', 'impact': 'Protects physical assets', 'urgency': 'HIGH'},
            'tourism': {'action': 'Launch Promo Campaign', 'reason': 'Tourism interest rising', 'impact': 'Revenue growth', 'urgency': 'LOW'}
        }
        
        for topic in trending_topics:
            name = topic['topic'].lower()
            for key, rec in topic_map.items():
                if key in name:
                    recs.append(rec)
        return recs

    def format_for_display(self, recommendations: List[Dict]) -> List[Dict]:
        formatted = []
        for rec in recommendations:
            formatted.append({
                # Map 'urgency' to 'priority' AND ensure it defaults to 'priority' if generated by AI
                'priority': rec.get('priority', rec.get('urgency', 'LOW')).upper(), 
                'action': rec.get('action', ''),
                'reason': rec.get('reason', ''),
                'impact': rec.get('impact', '')
            })
        return formatted

class ActionPlanGenerator:
    def generate_action_plan(self, recommendations: List[Dict], time_horizon: str = '24h') -> Dict:
        return {}