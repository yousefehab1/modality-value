"""Registry of concrete ModalityModel implementations.

Phase 1/2/4 modules register themselves here (via `register`) so that
tests/test_interface.py can parametrize interface-conformance checks over
every modality without hardcoding names.
"""
from __future__ import annotations

from modality_value.modalities.base import ModalityModel

_REGISTRY: dict[str, type[ModalityModel]] = {}


def register(cls: type[ModalityModel]) -> type[ModalityModel]:
    _REGISTRY[cls.__name__] = cls
    return cls


def registered_modalities() -> dict[str, type[ModalityModel]]:
    return dict(_REGISTRY)
