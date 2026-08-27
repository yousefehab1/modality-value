"""Ensure every modality module has run its `@register` decorator before
test collection.

tests/test_interface.py parametrizes over `registered_modalities()` at
*import time* (the parametrize decorator evaluates the registry the moment
the module is collected). pytest collects test files in alphabetical order,
so without this conftest, test_interface.py would be collected before e.g.
test_text.py has had a chance to import modality_value.modalities.text and
trigger its `@register`, leaving the registry empty and the interface test
silently skipped for every modality.

Importing each modality module here (best-effort -- a phase's module may
simply not exist yet if it hasn't been built) runs its registration before
any test file, including test_interface.py, is collected.
"""
import contextlib

_MODALITY_MODULES = [
    "modality_value.modalities.tabular",
    "modality_value.modalities.waveform",
    "modality_value.modalities.waveform_features",
    "modality_value.modalities.text",
    "modality_value.modalities.molecular",
]

for _mod in _MODALITY_MODULES:
    with contextlib.suppress(ImportError):
        __import__(_mod)
