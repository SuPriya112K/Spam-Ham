"""
test_model.py

Unit tests for the trained spam classifier pipeline in models/spam_classifier_pipeline.pkl.
These tests verify the model loads correctly and behaves sensibly,
NOT that it hits a specific accuracy (that's what train.py's evaluation is for).

Run with: pytest tests/test_model.py -v
"""

import os
import sys
import joblib
import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from preprocessing import clean_text

MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'spam_classifier_pipeline.pkl')


@pytest.fixture(scope="module")
def pipeline():
    """
    Loads the trained pipeline once and reuses it across all tests in this file
    (scope="module" avoids reloading the model for every single test, which would be slow).
    """
    assert os.path.exists(MODEL_PATH), (
        f"Model file not found at {MODEL_PATH}. Run 'python src/train.py' first."
    )
    return joblib.load(MODEL_PATH)


def test_model_loads(pipeline):
    assert pipeline is not None


def test_model_has_predict_method(pipeline):
    assert hasattr(pipeline, "predict")
    assert hasattr(pipeline, "predict_proba")


def test_predicts_obvious_spam(pipeline):
    text = clean_text("WIN FREE MONEY NOW! Click here to claim your prize! Limited time offer!!!")
    prediction = pipeline.predict([text])[0]
    assert prediction == "Spam"


def test_predicts_obvious_ham(pipeline):
    text = clean_text("Hi Sarah, can we move our meeting to 3pm tomorrow? Let me know if that works.")
    prediction = pipeline.predict([text])[0]
    assert prediction == "Ham"


def test_predict_proba_returns_valid_probabilities(pipeline):
    text = clean_text("Congratulations you have won a free prize")
    probabilities = pipeline.predict_proba([text])[0]

    # Probabilities must sum to ~1.0
    assert abs(sum(probabilities) - 1.0) < 0.001

    # Each probability must be between 0 and 1
    assert all(0 <= p <= 1 for p in probabilities)


def test_handles_empty_string_input(pipeline):
    # Should not crash on empty input, even if the prediction is meaningless
    prediction = pipeline.predict([""])
    assert prediction is not None


def test_batch_prediction(pipeline):
    texts = [
        clean_text("free money click now"),
        clean_text("meeting scheduled for tomorrow at noon"),
    ]
    predictions = pipeline.predict(texts)
    assert len(predictions) == 2