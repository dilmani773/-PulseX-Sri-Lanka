"""
Multi-Lingual Text Preprocessing
Handles Sinhala, Tamil, and English text
"""

import re
import unicodedata
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)


class TextCleaner:
    """
    Advanced text cleaning for multi-lingual content
    """
    
    def __init__(self):
        # Define stopwords for each language
        self.stopwords_en = {
            'the', 'is', 'at', 'which', 'on', 'a', 'an', 'as', 'are', 'was', 'were',
            'been', 'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
            'could', 'should', 'may', 'might', 'must', 'can', 'of', 'in', 'to', 'for',
            'with', 'by', 'from', 'up', 'about', 'into', 'through', 'during', 'before',
            'after', 'above', 'below', 'between', 'under', 'again', 'further', 'then',
            'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'both',
            'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor',
            'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very'
        }
        
        # Sinhala stopwords (common function words)
        self.stopwords_si = {
            'හා', 'සහ', 'ද', 'යි', 'ය', 'ක', 'ට', 'තුළ', 'මත', 'සඳහා', 'වන',
            'විසින්', 'මගින්', 'ගත', 'කර', 'කළ', 'කරන', 'එම', 'මෙම', 'අප', 'ඔබ'
        }
        
        # Tamil stopwords
        self.stopwords_ta = {
            'மற்றும்', 'அது', 'இது', 'என்று', 'ஆகும்', 'உள்ள', 'இருந்து', 'கொண்டு',
            'செய்', 'என', 'தான்', 'ஒரு', 'அந்த', 'இந்த'
        }
        
    def detect_language(self, text: str) -> str:
        """
        Detect primary language of text
        Returns: 'en', 'si', 'ta', or 'mixed'
        """
        # Count characters from each script
        sinhala_count = len(re.findall(r'[\u0D80-\u0DFF]', text))
        tamil_count = len(re.findall(r'[\u0B80-\u0BFF]', text))
        latin_count = len(re.findall(r'[a-zA-Z]', text))
        
        total = sinhala_count + tamil_count + latin_count
        
        if total == 0:
            return 'unknown'
        
        # Calculate percentages
        si_pct = sinhala_count / total
        ta_pct = tamil_count / total
        en_pct = latin_count / total
        
        if si_pct > 0.6:
            return 'si'
        elif ta_pct > 0.6:
            return 'ta'
        elif en_pct > 0.6:
            return 'en'
        else:
            return 'mixed'
    
    def normalize_unicode(self, text: str) -> str:
        """Normalize Unicode characters"""
        return unicodedata.normalize('NFC', text)
    
    def remove_urls(self, text: str) -> str:
        """Remove URLs"""
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        return re.sub(url_pattern, '', text)
    
    def remove_emails(self, text: str) -> str:
        """Remove email addresses"""
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        return re.sub(email_pattern, '', text)
    
    def remove_phone_numbers(self, text: str) -> str:
        """Remove phone numbers"""
        # Sri Lankan phone patterns
        patterns = [
            r'\+94\d{9}',  # International format
            r'0\d{9}',     # Local format
            r'\d{3}-\d{7}', # With dash
        ]
        for pattern in patterns:
            text = re.sub(pattern, '', text)
        return text
    
    def remove_extra_whitespace(self, text: str) -> str:
        """Remove extra whitespace"""
        # Replace multiple spaces with single space
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def remove_special_characters(self, text: str, keep_punctuation: bool = True) -> str:
        """Remove special characters while preserving text"""
        if keep_punctuation:
            # Keep letters, numbers, punctuation, and spaces from all scripts
            pattern = r'[^\w\s\u0D80-\u0DFF\u0B80-\u0BFF.!?,;:\'\"-]'
        else:
            # Keep only letters, numbers, spaces
            pattern = r'[^\w\s\u0D80-\u0DFF\u0B80-\u0BFF]'
        
        return re.sub(pattern, '', text)
    
    def remove_stopwords(self, text: str, language: str = 'en') -> str:
        """Remove stopwords for given language"""
        stopwords = self.stopwords_en
        
        if language == 'si':
            stopwords = self.stopwords_si
        elif language == 'ta':
            stopwords = self.stopwords_ta
        
        words = text.split()
        filtered_words = [word for word in words if word.lower() not in stopwords]
        
        return ' '.join(filtered_words)
    
    def clean_text(self, text: str, remove_stopwords: bool = False) -> Dict[str, str]:
        """
        Complete text cleaning pipeline
        Returns: Dict with cleaned text and metadata
        """
        if not text or not isinstance(text, str):
            return {
                'original': '',
                'cleaned': '',
                'language': 'unknown',
                'word_count': 0
            }
        
        original = text
        
        # Detect language first
        language = self.detect_language(text)
        
        # Normalize Unicode
        text = self.normalize_unicode(text)
        
        # Remove URLs, emails, phone numbers
        text = self.remove_urls(text)
        text = self.remove_emails(text)
        text = self.remove_phone_numbers(text)
        
        # Remove special characters (keep punctuation)
        text = self.remove_special_characters(text, keep_punctuation=True)
        
        # Remove extra whitespace
        text = self.remove_extra_whitespace(text)
        
        # Optionally remove stopwords
        if remove_stopwords:
            text = self.remove_stopwords(text, language)
        
        # Calculate word count
        word_count = len(text.split())
        
        return {
            'original': original,
            'cleaned': text,
            'language': language,
            'word_count': word_count
        }
    
    def clean_batch(self, texts: List[str], remove_stopwords: bool = False) -> List[Dict]:
        """Clean multiple texts"""
        return [self.clean_text(text, remove_stopwords) for text in texts]


class SentenceSegmenter:
    """
    Segment text into sentences (multi-lingual)
    """
    
    def segment(self, text: str, language: str = 'en') -> List[str]:
        """Segment text into sentences"""
        if language == 'en':
            # English sentence boundaries
            pattern = r'[.!?]+\s+'
        else:
            # For Sinhala and Tamil, use Sinhala/Tamil full stop
            pattern = r'[.!?።]+\s+'
        
        sentences = re.split(pattern, text)
        
        # Clean up empty sentences
        sentences = [s.strip() for s in sentences if s.strip()]
        
        return sentences


# Testing
if __name__ == "__main__":
    cleaner = TextCleaner()
    
    # Test with different languages
    test_texts = [
        "This is an English text with a URL http://example.com and email test@example.com",
        "මෙය සිංහල පෙළක් වේ",  # This is Sinhala text
        "இது தமிழ் உரை",  # This is Tamil text
        "Mixed සිංහල and English தமிழ் text",
    ]
    
    print("\nText Cleaning Results:")
    print("="*60)
    
    for text in test_texts:
        result = cleaner.clean_text(text)
        print(f"\nOriginal: {result['original']}")
        print(f"Cleaned: {result['cleaned']}")
        print(f"Language: {result['language']}")
        print(f"Word count: {result['word_count']}")
        print("-"*60)