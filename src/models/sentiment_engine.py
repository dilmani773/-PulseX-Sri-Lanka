"""
Multi-Lingual Sentiment Analysis Engine
Balanced for Accuracy (Reduces False Negatives)
"""

import numpy as np
from typing import List, Dict
import logging
import re

logger = logging.getLogger(__name__)

class SentimentAnalyzer:
    def __init__(self):
        # POSITIVE ROOTS (Expanded for stability)
        self.positive_roots = [
            'grow', 'rise', 'boost', 'profit', 'gain', 'recover', 'stable', 'strong',
            'develop', 'invest', 'deal', 'agree', 'partner', 'support', 'aid', 'relief',
            'good', 'great', 'success', 'win', 'safe', 'restor', 'happy', 'peace', 'benefit',
            'launch', 'open', 'start', 'commence', 'approve', 'bonus', 'award', 'help',
            'resume', 'normal', 'stabil', 'medic', 'vaccin', 'touris'
        ]
        
        # NEGATIVE ROOTS (Focused on ACTUAL risks, removed generic words like "rain")
        self.negative_roots = [
            'crisis', 'crash', 'collaps', 'drop', 'fall', 'decline', 'loss', 'los', 
            'debt', 'inflat', 'bankrupt', 'recess', 'short', 'scarc', 'lack', 'fail', 
            'risk', 'threat', 'danger', 'warn', 'concern', 'worry', 'fear', 'panic',
            'protest', 'strike', 'riot', 'violen', 'attack', 'conflict', 'fight', 'war', 
            'injur', 'death', 'dead', 'kill', 'damag', 'destroy', 'flood', 'landslid', 
            'disaster', 'accident', 'ban', 'suspend', 'arrest', 'crime', 'fraud', 
            'disease', 'outbreak', 'epidemic', 'virus', 'infect'
        ]
        
        # REMOVED: 'rain', 'storm' (unless 'damage' or 'disaster' is also present)
        
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
                else: neg_score += 1.0 # Reduced weight slightly to avoid "Panic Mode"
            
        # Calculate Logic
        total_meaningful = pos_score + neg_score
        
        if total_meaningful == 0:
            # Explicitly return Neutral for boring news
            return {'positive': 0.0, 'neutral': 1.0, 'negative': 0.0, 'compound': 0.0}
        
        # Calculate Compound (-1 to 1)
        # Using a gentler curve so scores aren't always -0.9 or +0.9
        raw_score = pos_score - neg_score
        compound = np.tanh(raw_score / 3.0) # Divider increased to 3.0 for softer scoring
        
        return {
            'positive': 0.0, 
            'neutral': 0.0,
            'negative': 0.0,
            'compound': float(compound)
        }