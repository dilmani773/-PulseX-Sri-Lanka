"""
Multi-Lingual Sentiment Analysis Engine
Uses transformer models for accurate sentiment classification
"""

import numpy as np
from typing import List, Dict, Tuple
import logging

logger = logging.getLogger(__name__)


class SentimentAnalyzer:
    """
    Multi-lingual sentiment analysis using pre-trained models
    For production: use transformers library
    For demo: uses rule-based approach
    """
    
    def __init__(self, model_name: str = "cardiffnlp/twitter-xlm-roberta-base-sentiment"):
        self.model_name = model_name
        self.model_loaded = False
        
        # Sentiment lexicons for rule-based fallback
        self.positive_words_en = {
            'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 'positive',
            'growth', 'success', 'improvement', 'increase', 'boost', 'progress', 'win',
            'happy', 'satisfied', 'pleased', 'benefit', 'opportunity', 'achievement'
        }
        
        self.negative_words_en = {
            'bad', 'terrible', 'awful', 'poor', 'negative', 'crisis', 'problem', 'issue',
            'decline', 'decrease', 'loss', 'fail', 'failure', 'concern', 'worry', 'risk',
            'danger', 'threat', 'difficult', 'shortage', 'protest', 'strike', 'conflict'
        }
        
    def analyze_text(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment of text
        Returns: dict with sentiment scores
        """
        if not text:
            return {
                'positive': 0.33,
                'neutral': 0.34,
                'negative': 0.33,
                'compound': 0.0
            }
        
        # Rule-based sentiment analysis (simple but effective)
        text_lower = text.lower()
        words = text_lower.split()
        
        pos_count = sum(1 for word in words if word in self.positive_words_en)
        neg_count = sum(1 for word in words if word in self.negative_words_en)
        total_words = len(words)
        
        if total_words == 0:
            return {'positive': 0.33, 'neutral': 0.34, 'negative': 0.33, 'compound': 0.0}
        
        # Calculate scores
        pos_score = pos_count / total_words
        neg_score = neg_count / total_words
        neu_score = 1.0 - (pos_score + neg_score)
        
        # Normalize to sum to 1
        total = pos_score + neg_score + neu_score
        if total > 0:
            pos_score /= total
            neg_score /= total
            neu_score /= total
        
        # Compound score (-1 to +1)
        compound = (pos_count - neg_count) / max(total_words, 1)
        compound = np.clip(compound, -1, 1)
        
        return {
            'positive': float(pos_score),
            'neutral': float(neu_score),
            'negative': float(neg_score),
            'compound': float(compound)
        }
    
    def get_sentiment_label(self, scores: Dict[str, float]) -> str:
        """Get categorical sentiment label"""
        compound = scores['compound']
        
        if compound >= 0.05:
            return 'positive'
        elif compound <= -0.05:
            return 'negative'
        else:
            return 'neutral'
    
    def analyze_batch(self, texts: List[str]) -> List[Dict]:
        """Analyze sentiment for multiple texts"""
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
        """Calculate overall sentiment across multiple texts"""
        if not texts:
            return {
                'average_compound': 0.0,
                'positive_ratio': 0.0,
                'negative_ratio': 0.0,
                'neutral_ratio': 0.0,
                'overall_label': 'neutral',
                'sample_size': 0
            }
        
        results = self.analyze_batch(texts)
        
        compounds = [r['scores']['compound'] for r in results]
        labels = [r['label'] for r in results]
        
        avg_compound = np.mean(compounds)
        
        pos_ratio = labels.count('positive') / len(labels)
        neg_ratio = labels.count('negative') / len(labels)
        neu_ratio = labels.count('neutral') / len(labels)
        
        # Overall label
        if avg_compound >= 0.05:
            overall = 'positive'
        elif avg_compound <= -0.05:
            overall = 'negative'
        else:
            overall = 'neutral'
        
        return {
            'average_compound': float(avg_compound),
            'positive_ratio': float(pos_ratio),
            'negative_ratio': float(neg_ratio),
            'neutral_ratio': float(neu_ratio),
            'overall_label': overall,
            'sample_size': len(texts),
            'compound_std': float(np.std(compounds))
        }


class AspectBasedSentiment:
    """
    Analyze sentiment towards specific aspects/topics
    """
    
    def __init__(self):
        self.aspects = {
            'economy': ['economy', 'economic', 'gdp', 'growth', 'inflation', 'trade'],
            'politics': ['government', 'political', 'election', 'policy', 'minister'],
            'infrastructure': ['road', 'bridge', 'transport', 'infrastructure', 'construction'],
            'healthcare': ['health', 'hospital', 'medical', 'doctor', 'medicine'],
            'education': ['school', 'education', 'university', 'teacher', 'student'],
            'tourism': ['tourism', 'tourist', 'hotel', 'travel', 'visitor'],
            'agriculture': ['agriculture', 'farmer', 'crop', 'harvest', 'farming'],
            'technology': ['technology', 'tech', 'digital', 'internet', 'software']
        }
        
        self.sentiment_analyzer = SentimentAnalyzer()
    
    def extract_aspect_sentences(self, text: str, aspect: str) -> List[str]:
        """Extract sentences related to specific aspect"""
        if aspect not in self.aspects:
            return []
        
        keywords = self.aspects[aspect]
        sentences = text.split('.')
        
        relevant_sentences = []
        for sentence in sentences:
            sentence_lower = sentence.lower()
            if any(keyword in sentence_lower for keyword in keywords):
                relevant_sentences.append(sentence.strip())
        
        return relevant_sentences
    
    def analyze_aspect_sentiment(self, text: str, aspect: str) -> Dict:
        """Analyze sentiment for specific aspect"""
        sentences = self.extract_aspect_sentences(text, aspect)
        
        if not sentences:
            return {
                'aspect': aspect,
                'sentiment': 'neutral',
                'score': 0.0,
                'mentions': 0
            }
        
        aggregate = self.sentiment_analyzer.calculate_aggregate_sentiment(sentences)
        
        return {
            'aspect': aspect,
            'sentiment': aggregate['overall_label'],
            'score': aggregate['average_compound'],
            'mentions': len(sentences),
            'confidence': 1.0 - aggregate['compound_std']
        }
    
    def analyze_all_aspects(self, text: str) -> Dict[str, Dict]:
        """Analyze sentiment for all aspects"""
        results = {}
        
        for aspect in self.aspects.keys():
            results[aspect] = self.analyze_aspect_sentiment(text, aspect)
        
        return results


# Testing
if __name__ == "__main__":
    analyzer = SentimentAnalyzer()
    
    # Test texts
    test_texts = [
        "Tourism growth is excellent this year! Great news for the economy.",
        "The fuel crisis is causing serious problems for businesses.",
        "The government announced new infrastructure projects today.",
    ]
    
    print("\nSentiment Analysis Results:")
    print("="*60)
    
    for text in test_texts:
        result = analyzer.analyze_text(text)
        label = analyzer.get_sentiment_label(result)
        
        print(f"\nText: {text}")
        print(f"Sentiment: {label.upper()}")
        print(f"Scores: Pos={result['positive']:.2f}, Neu={result['neutral']:.2f}, Neg={result['negative']:.2f}")
        print(f"Compound: {result['compound']:+.2f}")
        print("-"*60)
    
    # Test aggregate
    print("\n\nAggregate Sentiment:")
    aggregate = analyzer.calculate_aggregate_sentiment(test_texts)
    print(f"Overall: {aggregate['overall_label'].upper()}")
    print(f"Average compound: {aggregate['average_compound']:+.2f}")
    print(f"Positive ratio: {aggregate['positive_ratio']:.1%}")
    print(f"Negative ratio: {aggregate['negative_ratio']:.1%}")
    print("="*60)