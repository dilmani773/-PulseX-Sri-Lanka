"""
Multi-Lingual Sentiment Analysis Engine
Optimized for MACRO-ECONOMIC Risk (Ignores domestic/personal crime)
"""

import numpy as np
from typing import List, Dict
import logging
import re

logger = logging.getLogger(__name__)

class SentimentAnalyzer:
    def __init__(self):
        # POSITIVE ROOTS (Business Growth & Stability)
        self.positive_roots = [
            'grow', 'rise', 'boost', 'profit', 'gain', 'recover', 'stable', 'strong',
            'develop', 'invest', 'deal', 'agree', 'partner', 'support', 'aid', 'relief',
            'good', 'great', 'success', 'win', 'safe', 'restor', 'happy', 'peace', 'benefit',
            'launch', 'open', 'start', 'commence', 'approve', 'bonus', 'award', 'help',
            'resume', 'normal', 'stabil', 'medic', 'vaccin', 'touris'
        ]
        
        # NEGATIVE ROOTS (Focused on BUSINESS & MACRO Risks only)
        # REMOVED: arrest, kill, death, murder, injury, accident (Too individual)
        # KEPT: protest, strike, fuel, shortage, inflation, disaster (Systemic risks)
        self.negative_roots = [
            # Economic Shocks
            'crisis', 'crash', 'collaps', 'drop', 'fall', 'decline', 'loss', 'los', 
            'debt', 'inflat', 'bankrupt', 'recess', 'short', 'scarc', 'lack', 'fail', 
            'risk', 'threat', 'danger', 'warn', 'concern', 'worry', 'fear', 'panic',
            'fraud', 'corrupt', 'bribe', 'scam', 'default',
            
            # Operational Disruptions (Strikes, Weather, Unrest)
            'protest', 'strike', 'riot', 'violen', 'attack', 'conflict', 'fight', 'war', 
            'damag', 'destroy', 'flood', 'landslid', 'disaster', 
            'ban', 'suspend', 'clos', 'delay', 'outage', 'blackout',
            
            # Health/Systemic
            'disease', 'outbreak', 'epidemic', 'virus', 'infect'
        ]
        
        self.negations = {'not', 'no', 'never', 'neither', 'nor', 'barely', 'hardly', 'stop'}

    def analyze_text(self, text: str) -> Dict[str, float]:
        if not text:
            return {'positive': 0.0, 'neutral': 1.0, 'negative': 0.0, 'compound': 0.0}
        
        text_lower = text.lower()
        words = re.findall(r'\w+', text_lower)
        
        pos_score = 0
        neg_score = 0
        
        for i, word in enumerate(words):
            is_negated = False
            if i > 0 and words[i-1] in self.negations:
                is_negated = True
            
            # Positive Check
            if any(root in word for root in self.positive_roots):
                if is_negated: neg_score += 1.0
                else: pos_score += 1.0
            
            # Negative Check
            elif any(root in word for root in self.negative_roots):
                if is_negated: pos_score += 0.5
                else: neg_score += 1.0
            
        # Calculate Logic
        total_meaningful = pos_score + neg_score
        
        if total_meaningful == 0:
            return {'positive': 0.0, 'neutral': 1.0, 'negative': 0.0, 'compound': 0.0}
        
        raw_score = pos_score - neg_score
        # Using 3.0 to keep the score gentle (less jumping to -0.9 instantly)
        compound = np.tanh(raw_score / 3.0)
        
        return {
            'positive': 0.0, 
            'neutral': 0.0,
            'negative': 0.0,
            'compound': float(compound)
        }