.PHONY: install dev test run demo

install:
	pip install -r requirements.txt

dev:
	pip install -r requirements-dev.txt

test:
	pytest -q

run:
	uvicorn llm_semantic_cache.app:app --reload --port 8000

demo:
	python scripts/demo.py
