"""
Multi-Lingual Sentiment Analysis Engine
Uses rule-based approach with Negation Handling (Optimized for Demo)
"""

import numpy as np
from typing import List, Dict, Tuple
import logging

logger = logging.getLogger(__name__)

class SentimentAnalyzer:
    """
    Multi-lingual sentiment analysis.
    Now includes Negation Handling (e.g., 'not bad' -> Positive).
    """
    
    def __init__(self):
        self.model_loaded = False
        
        # Sentiment lexicons
        self.positive_words_en = {
            'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 'positive',
            'growth', 'success', 'improvement', 'increase', 'boost', 'progress', 'win',
            'happy', 'satisfied', 'pleased', 'benefit', 'opportunity', 'achievement',
            'stable', 'recovery', 'strong', 'up', 'bullish'
        }
        
        self.negative_words_en = {
            'bad', 'terrible', 'awful', 'poor', 'negative', 'crisis', 'problem', 'issue',
            'decline', 'decrease', 'loss', 'fail', 'failure', 'concern', 'worry', 'risk',
            'danger', 'threat', 'difficult', 'shortage', 'protest', 'strike', 'conflict',
            'crash', 'down', 'bearish', 'weak', 'inflation'
        }
        
        # Words that flip the meaning of the next word
        self.negations = {'not', 'no', 'never', 'neither', 'nor', 'barely', 'hardly'}

    def analyze_text(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment with context awareness
        """
        if not text:
            return {'positive': 0.0, 'neutral': 1.0, 'negative': 0.0, 'compound': 0.0}
        
        text_lower = text.lower()
        words = text_lower.split()
        
        pos_score = 0
        neg_score = 0
        
        i = 0
        while i < len(words):
            word = words[i]
            is_negated = False
            
            # Check previous word for negation (if not at start)
            if i > 0 and words[i-1] in self.negations:
                is_negated = True
            
            # Calculate impact
            if word in self.positive_words_en:
                if is_negated:
                    neg_score += 1  # "Not good" -> Negative
                else:
                    pos_score += 1  # "Good" -> Positive
                    
            elif word in self.negative_words_en:
                if is_negated:
                    pos_score += 0.5 # "Not bad" -> Mildly Positive
                else:
                    neg_score += 1   # "Bad" -> Negative
            
            i += 1
        
        total_meaningful_words = pos_score + neg_score
        total_words = len(words)
        
        if total_words == 0:
            return {'positive': 0.0, 'neutral': 1.0, 'negative': 0.0, 'compound': 0.0}
        
        # Calculate Logic
        final_pos = pos_score / total_words
        final_neg = neg_score / total_words
        final_neu = 1.0 - (final_pos + final_neg)
        
        # Compound Score (-1 to 1)
        # We use a normalization constant (alpha) to smooth results like VADER does
        norm_score = (pos_score - neg_score) / np.sqrt((pos_score + neg_score)**2 + 15)
        
        return {
            'positive': float(final_pos),
            'neutral': float(max(0, final_neu)),
            'negative': float(final_neg),
            'compound': float(norm_score)
        }
    
    def get_sentiment_label(self, scores: Dict[str, float]) -> str:
        compound = scores['compound']
        if compound >= 0.05: return 'positive'
        elif compound <= -0.05: return 'negative'
        else: return 'neutral'

    # (Keep analyze_batch and AspectBasedSentiment classes exactly as they were in your previous file)
    # ... [Paste the rest of the file here if you need it, or just keep the classes from before]
    
    def analyze_batch(self, texts: List[str]) -> List[Dict]:
        results = []
        for text in texts:
            scores = self.analyze_text(text)
            label = self.get_sentiment_label(scores)
            results.append({
                'text': text[:100] + '...' if len(text) > 100 else text,
                'scores': scores,
                'label': label,
                'confidence': max(scores['positive'], scores['neutral'], scores['negative'])
            })
        return results
    
    def calculate_aggregate_sentiment(self, texts: List[str]) -> Dict:
        if not texts:
            return {'average_compound': 0.0, 'overall_label': 'neutral'}
        results = self.analyze_batch(texts)
        compounds = [r['scores']['compound'] for r in results]
        avg_compound = np.mean(compounds)
        if avg_compound >= 0.05: overall = 'positive'
        elif avg_compound <= -0.05: overall = 'negative'
        else: overall = 'neutral'
        return {
            'average_compound': float(avg_compound),
            'overall_label': overall,
            'compound_std': float(np.std(compounds))
        }

class AspectBasedSentiment:
    def __init__(self):
        self.aspects = {
            'economy': ['economy', 'economic', 'gdp', 'growth', 'inflation', 'trade', 'market', 'price', 'rupee'],
            'politics': ['government', 'political', 'election', 'policy', 'minister', 'president', 'parliament'],
            'infrastructure': ['road', 'bridge', 'transport', 'infrastructure', 'construction', 'power', 'energy'],
            'healthcare': ['health', 'hospital', 'medical', 'doctor', 'medicine', 'drug'],
            'tourism': ['tourism', 'tourist', 'hotel', 'travel', 'visitor', 'arrival']
        }
        self.sentiment_analyzer = SentimentAnalyzer()
    
    def extract_aspect_sentences(self, text: str, aspect: str) -> List[str]:
        if aspect not in self.aspects: return []
        keywords = self.aspects[aspect]
        sentences = text.replace('!', '.').replace('?', '.').split('.')
        relevant_sentences = []
        for sentence in sentences:
            sentence_lower = sentence.lower()
            if any(keyword in sentence_lower for keyword in keywords):
                relevant_sentences.append(sentence.strip())
        return relevant_sentences
    
    def analyze_aspect_sentiment(self, text: str, aspect: str) -> Dict:
        sentences = self.extract_aspect_sentences(text, aspect)
        if not sentences:
            return {'aspect': aspect, 'sentiment': 'neutral', 'score': 0.0, 'mentions': 0}
        aggregate = self.sentiment_analyzer.calculate_aggregate_sentiment(sentences)
        return {
            'aspect': aspect,
            'sentiment': aggregate['overall_label'],
            'score': aggregate['average_compound'],
            'mentions': len(sentences)
        }
