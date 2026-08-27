"""Ensure every phase's modality module is imported (hence registered via
`@register`) before test_interface.py's registry-based parametrization runs.

Each phase may be built independently/concurrently; a module simply not
existing yet is fine (skip it), but if it exists it must register cleanly.
"""
import importlib

for _mod in (
    "modality_value.modalities.tabular",
    "modality_value.modalities.waveform",
    "modality_value.modalities.waveform_features",
    "modality_value.modalities.text",
    "modality_value.modalities.molecular",
):
    try:
        importlib.import_module(_mod)
    except ImportError:
        pass
