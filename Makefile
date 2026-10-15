PY := .venv/bin/python

.PHONY: all setup fetch db test report clean

all: db test report

setup:
	python3 -m venv .venv
	$(PY) -m pip install -r requirements.txt
	$(PY) -m pip install -e .

fetch:            ## re-download prices (moves the data; update END_DATE in config.py first)
	$(PY) -m perp_risk.fetch

db:               ## load CSVs into SQLite and build the SQL feature tables
	$(PY) -m perp_risk.db

test:
	$(PY) -m pytest -q

report:           ## run the study and write reports/report.html, figures and tables
	$(PY) -m perp_risk.report

clean:
	rm -f data/perp.sqlite
	rm -rf reports/figures reports/tables reports/report.html
