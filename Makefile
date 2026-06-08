PYTHON ?= python3

.PHONY: eval test lint clean

eval: ## Run deterministic evals against the golden set.
	scripts/run_eval.sh

test: ## Run all unit tests.
	$(PYTHON) -m unittest discover -s tests -v

lint: ## Run ruff linter and format checker.
	ruff check .
	ruff format --check .

clean: ## Clean up generated files.
	rm -f data/latest_run_metrics.json data/latest_eval_report.json
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
