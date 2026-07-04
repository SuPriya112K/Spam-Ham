"""
test_preprocessing.py

Unit tests for the text cleaning logic in src/preprocessing.py.
Run with: pytest tests/test_preprocessing.py -v
"""

import sys
import os

# Add src/ to the path so we can import preprocessing.py
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from preprocessing import clean_text


def test_lowercases_text():
    result = clean_text("HELLO WORLD")
    assert result == "hello world"


def test_removes_urls():
    result = clean_text("Click here http://scam-link.com now")
    assert "http" not in result
    assert "scam-link.com" not in result


def test_removes_email_addresses():
    result = clean_text("Contact us at winner@fakemail.com today")
    assert "winner@fakemail.com" not in result
    assert "@" not in result


def test_removes_punctuation():
    result = clean_text("Wow!!! Amazing... right???")
    for char in "!.,?":
        assert char not in result


def test_removes_numbers():
    result = clean_text("Call 12345 now for 50% off")
    assert "12345" not in result
    assert "50" not in result


def test_removes_stopwords():
    result = clean_text("this is a test of the system")
    # "is", "a", "of", "the" are common stopwords and should be removed
    assert "is" not in result.split()
    assert "the" not in result.split()


def test_handles_empty_string():
    result = clean_text("")
    assert result == ""


def test_handles_none_input():
    # Non-string input should return empty string, not crash
    result = clean_text(None)
    assert result == ""


def test_handles_numbers_only_string():
    result = clean_text("12345 67890")
    assert result == ""


def test_preserves_meaningful_words():
    result = clean_text("free money winner prize")
    # These are exactly the kind of spam-signal words we want KEPT
    assert "free" in result
    assert "money" in result
    assert "winner" in result
    assert "prize" in result