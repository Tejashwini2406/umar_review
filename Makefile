PYTHON ?= python

.PHONY: audit test

audit:
	$(PYTHON) scripts/audit_datasets.py --data-root data/raw --output reports/eda

test:
	$(PYTHON) -m pytest

