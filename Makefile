.PHONY: setup demo eval test clean

setup:
	pip install -r requirements.txt --break-system-packages

demo:
	python examples/demo.py --trace

eval:
	python eval/harness.py

test:
	python -m pytest -q

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache
