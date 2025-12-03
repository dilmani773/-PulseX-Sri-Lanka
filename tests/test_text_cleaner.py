import sys
from pathlib import Path
import pytest

# Ensure src is importable when running tests from repository root
sys.path.append(str(Path(__file__).resolve().parents[1] / 'src'))

from preprocessing.text_cleaner import TextCleaner


def test_clean_text_removes_urls_emails_and_numbers():
    cleaner = TextCleaner()
    text = "This is a test http://example.com contact me at test@example.com +94123456789"
    out = cleaner.clean_text(text)
    cleaned = out['cleaned']

    assert 'http' not in cleaned
    assert '@' not in cleaned
    assert '+' not in cleaned
    assert out['language'] == 'en'


def test_remove_stopwords_option():
    cleaner = TextCleaner()
    text = "This is a sentence with the and is a"
    out_no = cleaner.clean_text(text, remove_stopwords=False)
    out_yes = cleaner.clean_text(text, remove_stopwords=True)

    assert len(out_yes['cleaned'].split()) <= len(out_no['cleaned'].split())
    # 'the' should be removed when remove_stopwords=True
    assert 'the' not in out_yes['cleaned'].lower()


def test_detect_language_multi():
    cleaner = TextCleaner()
    si = "මෙය සිංහල පෙළක් වේ"
    ta = "இது தமிழ் உரை"

    assert cleaner.detect_language(si) == 'si'
    assert cleaner.detect_language(ta) == 'ta'
