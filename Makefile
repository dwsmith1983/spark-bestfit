.PHONY: help install install-dev install-test test test-cov clean build publish-test publish pre-commit check setup docs docs-clean benchmark benchmark-ray benchmark-all benchmark-charts validate-notebooks mutate mutate-browse mutate-results mutate-html mutate-clean

.DEFAULT_GOAL := help

help: ## Show this help message
	@echo "Usage: make [target]"
	@echo ""
	@echo "Available targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install package in editable mode
	pip install -e .

install-dev: ## Install package with development dependencies
	pip install -e ".[dev,docs]"
	pre-commit install

install-test: ## Install package with test dependencies only
	pip install -e ".[test]"

test: ## Run tests with pytest
	PYTHONPATH=src pytest

test-cov: ## Run tests with coverage report
	PYTHONPATH=src pytest --cov=src/spark_bestfit --cov-report=term-missing --cov-report=html -v

clean: ## Clean build artifacts and cache files
	rm -rf build/ dist/ *.egg-info .pytest_cache .mypy_cache .ruff_cache htmlcov/ .coverage
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

build: clean ## Build distribution packages
	python -m build

publish-test: build ## Publish to TestPyPI
	twine upload --repository testpypi dist/*

publish: build ## Publish to PyPI
	twine upload dist/*

pre-commit: ## Run pre-commit hooks on all files
	pre-commit run --all-files

check: pre-commit test ## Run all checks (pre-commit, tests)

docs: ## Build documentation
	sphinx-build -b html docs docs/_build/html

docs-clean: ## Clean documentation build
	rm -rf docs/_build

setup: install-dev ## Initial setup for development
	@echo "Development environment setup complete"
	@echo "Run 'make test' to verify everything works"

benchmark: ## Run Spark performance benchmarks (not run in CI)
	PYTHONPATH=src pytest tests/benchmarks/test_benchmark_scaling.py -v --benchmark-only --benchmark-min-rounds=20 --benchmark-save=spark-latest --benchmark-save-data

benchmark-ray: ## Run Ray performance benchmarks (requires ray installed)
	PYTHONPATH=src pytest tests/benchmarks/test_benchmark_ray.py -v --benchmark-only --benchmark-min-rounds=20 --benchmark-save=ray-latest --benchmark-save-data

benchmark-all: benchmark benchmark-ray ## Run both Spark and Ray benchmarks

benchmark-charts: ## Generate scaling charts from benchmark results
	python scripts/generate_scaling_charts.py

validate-notebooks: ## Run all example notebooks to validate they execute without errors
	@echo "Validating example notebooks..."
	@for nb in examples/*.ipynb examples/**/*.ipynb; do \
		if [ -f "$$nb" ]; then \
			echo "  Running $$nb..."; \
			PYTHONPATH=src python -m jupyter nbconvert --to notebook --execute --inplace \
				--ExecutePreprocessor.timeout=600 "$$nb" || exit 1; \
		fi; \
	done
	@echo "All notebooks validated successfully!"

# Mutation Testing (requires: pip install mutmut)
# See: https://mutmut.readthedocs.io/
# Note: Uses LocalBackend tests only (mutmut 3 trampoline incompatible with PySpark workers)
# Configuration in pyproject.toml [tool.mutmut]

mutate: ## Run mutation testing (uses pyproject.toml config)
	@echo "Running mutation tests (this may take a while)..."
	@echo "Using LocalBackend tests only (Spark tests excluded)"
	@echo "See pyproject.toml [tool.mutmut] for configuration"
	rm -rf mutants/
	PYTHONPATH=src mutmut run

mutate-browse: ## Interactive browser for mutation results
	mutmut browse

mutate-results: ## Show mutation testing results summary
	mutmut results

mutate-clean: ## Clean mutation testing cache and mutants
	rm -rf .mutmut-cache html/ mutants/
	@echo "Mutation cache cleaned"
