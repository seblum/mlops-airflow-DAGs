.phony make-env:
	python3 -m venv .venv
	source .venv/bin/activate
	pip install -f requirements.txt
