"""Tests for confidence/review logic and category/priority sanitisation.

These do NOT hit the network: they use a fake backend that implements the
backend protocol, so they run fast and offline.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import settings
from ml import category_classifier, priority_classifier
from ml.confidence import assess
from models.zero_shot import CandidateResult


class FakeBackend:
    def __init__(self, cat_label="Billing", cat_conf=0.9, cat_margin=0.6,
                 pri_label="High", pri_conf=0.8, pri_margin=0.5,
                 cat_failed=False, pri_failed=False):
        self.cat = CandidateResult(label=cat_label, confidence=cat_conf,
                                   margin=cat_margin, failed=cat_failed)
        self.pri = CandidateResult(label=pri_label, confidence=pri_conf,
                                   margin=pri_margin, failed=pri_failed)

    def predict_category(self, text):
        return self.cat

    def predict_priority(self, text):
        return self.pri


def test_confident_ticket_not_routed_to_review():
    be = FakeBackend()
    a = assess(be.predict_category("x"), be.predict_priority("x"), text="hello")
    assert a.needs_review is False
    assert a.category == "Billing"
    assert a.priority == "High"


def test_low_category_confidence_routes_to_review():
    be = FakeBackend(cat_conf=0.2)
    a = assess(be.predict_category("x"), be.predict_priority("x"), text="hello")
    assert a.needs_review is True
    assert any("category confidence" in r for r in a.review_reasons)


def test_small_margin_routes_to_review():
    be = FakeBackend(cat_margin=0.05)
    a = assess(be.predict_category("x"), be.predict_priority("x"), text="hello")
    assert a.needs_review is True
    assert any("margin" in r for r in a.review_reasons)


def test_critical_risk_keyword_forces_review():
    be = FakeBackend()
    a = assess(be.predict_category("x"), be.predict_priority("x"),
               text="possible security breach please help")
    assert a.needs_review is True
    assert any("critical-risk" in r for r in a.review_reasons)


def test_model_failure_routes_to_review():
    be = FakeBackend(cat_failed=True, pri_failed=True)
    a = assess(be.predict_category("x"), be.predict_priority("x"), text="hello")
    assert a.needs_review is True
    assert a.category == ""  # failure -> no prediction


def test_category_classifier_sanitises_unknown_label():
    class Weird:
        def predict_category(self, text):
            return CandidateResult(label="weird", confidence=0.9, margin=0.5)
    res = category_classifier.classify("some text", Weird())
    assert res.label in settings.CATEGORIES


def test_priority_classifier_sanitises_unknown_label():
    class Weird:
        def predict_priority(self, text):
            return CandidateResult(label="weird", confidence=0.9, margin=0.5)
    res = priority_classifier.classify("some text", Weird())
    assert res.label in settings.PRIORITIES


def test_empty_text_returns_failed():
    be = FakeBackend()
    res = category_classifier.classify("", be)
    assert res.failed
