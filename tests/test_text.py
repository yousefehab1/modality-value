"""Text arm tests.

Interface conformance (fit/embed/score exist) is already covered generically
by tests/test_interface.py once TextModel is `@register`-ed. This file adds:

  1. JSON-parsing unit tests (well-formed, malformed, missing keys).
  2. The deployment-honesty fallback: a malformed/non-JSON generation must
     fall back to the *training-set class prior* in score()'s parsing path,
     and be counted into the parse-failure rate -- tested directly against
     `scores_from_generations`, with no model or PTB-XL data required.
"""
import numpy as np

from modality_value.config import SUPERCLASSES
from modality_value.modalities.text import parse_json_labels, scores_from_generations


def test_parse_well_formed_json():
    text = '{"NORM": true, "MI": false, "STTC": false, "CD": false, "HYP": true}'
    parsed = parse_json_labels(text)
    assert parsed == {"NORM": True, "MI": False, "STTC": False, "CD": False, "HYP": True}


def test_parse_json_embedded_in_extra_text():
    text = 'Sure, here is the JSON:\n{"NORM": false, "MI": true, "STTC": false, "CD": false, "HYP": false}\nHope that helps!'
    parsed = parse_json_labels(text)
    assert parsed is not None
    assert parsed["MI"] is True
    assert parsed["NORM"] is False


def test_parse_missing_key_fails():
    # Missing HYP -> must be treated as a parse failure, not silently defaulted.
    text = '{"NORM": true, "MI": false, "STTC": false, "CD": false}'
    assert parse_json_labels(text) is None


def test_parse_non_json_fails():
    assert parse_json_labels("this is definitely not json") is None
    assert parse_json_labels("") is None
    assert parse_json_labels(None) is None


def test_parse_garbage_and_truncated_json_fails():
    # A truncated generation (model ran out of tokens mid-object) is a
    # realistic failure mode and must not silently parse.
    text = '{"NORM": true, "MI": false, "STTC": '
    assert parse_json_labels(text) is None


def test_scores_from_generations_falls_back_to_prior_on_parse_failure():
    prior = np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float32)
    good = '{"NORM": true, "MI": false, "STTC": false, "CD": false, "HYP": true}'
    bad = "not json at all, model degenerated"
    scores, failure_rate = scores_from_generations([bad, good], prior)

    assert failure_rate == 0.5
    # Malformed generation -> exactly the class prior, no invented values.
    np.testing.assert_allclose(scores[0], prior)
    # Well-formed generation -> the boolean->probability mapping, not the prior.
    expected_good = np.array([0.95, 0.05, 0.05, 0.05, 0.95], dtype=np.float32)
    np.testing.assert_allclose(scores[1], expected_good)


def test_scores_from_generations_all_failures_is_100pct_and_all_prior():
    prior = np.array([0.2, 0.2, 0.2, 0.2, 0.2], dtype=np.float32)
    scores, failure_rate = scores_from_generations(["garbage", "", "{broken"], prior)
    assert failure_rate == 1.0
    for row in scores:
        np.testing.assert_allclose(row, prior)


def test_scores_from_generations_shape_matches_superclasses():
    prior = np.zeros(len(SUPERCLASSES), dtype=np.float32)
    scores, _ = scores_from_generations(["garbage"], prior)
    assert scores.shape == (1, len(SUPERCLASSES))
