.PHONY: help install install-dev install-test test test-cov clean build publish-test publish pre-commit check setup docs docs-clean

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
	pytest

test-cov: ## Run tests with coverage report
	pytest --cov=src/spark_bestfit --cov-report=term-missing --cov-report=html -v

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
