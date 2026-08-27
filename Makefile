.PHONY: data tabular waveform text fusion molecular serve test

VENV := .venv/bin

data:
	bash scripts/download_ptbxl.sh
	$(VENV)/python -m modality_value.io.ptbxl --summary
	$(VENV)/python scripts/fig_data_overview.py

tabular:
	$(VENV)/python -m modality_value.modalities.tabular

waveform:
	$(VENV)/python -m modality_value.modalities.waveform

text:
	$(VENV)/python -m modality_value.modalities.text

fusion:
	$(VENV)/python -m modality_value.fusion.value

molecular:
	$(VENV)/python -m modality_value.modalities.molecular

serve:
	$(VENV)/uvicorn service.app:app --host 0.0.0.0 --port 8080

test:
	$(VENV)/pytest -v
