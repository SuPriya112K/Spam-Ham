"""
preprocessing.py

Handles text cleaning for the spam classifier.
This module is used both during training and at prediction time,
so the SAME cleaning logic is guaranteed to apply in both places.
"""

import re
import string
from nltk.corpus import stopwords

# Load stopwords once when this module is imported (not every function call)
STOPWORDS = set(stopwords.words('english'))


def clean_text(text: str) -> str:
    """
    Cleans a single piece of text for spam classification.

    Steps:
    1. Lowercase everything
    2. Remove URLs
    3. Remove email addresses
    4. Remove punctuation and numbers
    5. Remove extra whitespace
    6. Remove stopwords

    Args:
        text: raw input text

    Returns:
        cleaned text as a single string
    """
    if not isinstance(text, str):
        return ""

    # 1. Lowercase
    text = text.lower()

    # 2. Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)

    # 3. Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)

    # 4. Remove punctuation and numbers
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)

    # 5. Collapse extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    # 6. Remove stopwords
    words = text.split()
    words = [w for w in words if w not in STOPWORDS]

    return ' '.join(words)


def clean_text_series(texts):
    """
    Applies clean_text to a pandas Series (or any iterable of strings).
    Convenience wrapper for use in training scripts.
    """
    return [clean_text(t) for t in texts]