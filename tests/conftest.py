"""Import every modality module that exists so `@register`-decorated classes
are present in the registry before tests/test_interface.py's parametrize
(evaluated at collection time) reads `registered_modalities()`.

Modules from phases not yet landed in this worktree simply don't exist yet;
that's expected (parallel phases land independently), so import failures are
swallowed rather than raised.
"""
import importlib

_CANDIDATE_MODALITY_MODULES = [
    "modality_value.modalities.tabular",
    "modality_value.modalities.waveform",
    "modality_value.modalities.waveform_features",
    "modality_value.modalities.text",
    "modality_value.modalities.molecular",
]

for _mod_name in _CANDIDATE_MODALITY_MODULES:
    try:
        importlib.import_module(_mod_name)
    except ImportError:
        pass
