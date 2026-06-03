"""Unit tests for the sentence-level diff helper."""

from testing.text_utils import diff_sentences


def test_diff_detects_removed_sentence():
    d = diff_sentences("Do the task. Extra injected sentence here.", "Do the task.")
    assert d["removed"] == ["Extra injected sentence here."]
    assert d["added"] == []


def test_diff_detects_added_sentence():
    d = diff_sentences("Hello world.", "Hello world. New one.")
    assert d["added"] == ["New one."]
    assert d["removed"] == []


def test_diff_identical_is_empty():
    assert diff_sentences("A. B.", "A. B.") == {"removed": [], "added": []}


def test_diff_always_returns_both_keys():
    assert set(diff_sentences("x.", "y.")) == {"removed", "added"}
