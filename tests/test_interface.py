"""Interface conformance: every registered modality must honour ModalityModel.

This is what enforces the design decision in modalities/base.py -- nothing
downstream should ever need to know which concrete class it's holding.
Per-modality shape checks (score returns (n, k), embed returns (n, d)) live
alongside each modality's own tests, where the tiny fixture input is known.
"""
import pytest

from modality_value.modalities import registered_modalities
from modality_value.modalities.base import ModalityModel


@pytest.mark.parametrize("name,cls", list(registered_modalities().items()) or [("none", None)])
def test_registered_modality_has_interface(name, cls):
    if cls is None:
        pytest.skip("no modalities registered yet")
    assert issubclass(cls, ModalityModel)
    assert hasattr(cls, "fit")
    assert hasattr(cls, "embed")
    assert hasattr(cls, "score")
