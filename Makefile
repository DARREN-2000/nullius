.PHONY: help test train run site api dashboard separation all clean lint

help:
	@echo "make test       - run the unit and evaluation test suite"
	@echo "make train      - generate the DICOM cohort and export models/lesion-mlp.onnx"
	@echo "make run        - end-to-end pipeline, writes out/report.json + traces + metrics"
	@echo "make site       - build docs/index.html, the published inference control plane"
	@echo "make dashboard  - render out/dashboard.html from the last report"
	@echo "make api        - serve the HTTP API on 127.0.0.1:8080"
	@echo "make separation - recompute per-feature Cohen's d / AUROC on the generator"
	@echo "make all        - train, run, dashboard and site in order"
	@echo "make clean      - remove generated artefacts"

test:
	python3 -W ignore::ResourceWarning -m unittest discover -s tests -v

run:
	python3 -W ignore::ResourceWarning scripts/run_pipeline.py

train:
	python3 -W ignore::ResourceWarning scripts/train_lesion_model.py

dashboard: run
	python3 scripts/build_dashboard.py

site:
	python3 scripts/build_site.py

separation:
	python3 -W ignore::ResourceWarning scripts/feature_separation.py

all: train run dashboard site

api:
	python3 -m nullius.api --host 127.0.0.1 --port 8080

clean:
	rm -rf out __pycache__ */__pycache__ */*/__pycache__
