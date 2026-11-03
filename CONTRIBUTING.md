# Contributing to spark-bestfit

Thank you for your interest in contributing to spark-bestfit!

Please note that this project adheres to a [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code.

## Development Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/dwsmith1983/spark-bestfit.git
   cd spark-bestfit
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Linux/macOS
   # or .venv\Scripts\activate  # Windows
   ```

3. **Install development dependencies**
   ```bash
   make install-dev
   ```
   This installs the package in editable mode with all dev dependencies and sets up pre-commit hooks.

## Pre-commit Hooks

This project uses pre-commit hooks to maintain code quality. Hooks run automatically on `git commit` and include:

- **Ruff** - Fast Python linting and auto-fixing
- **Black** - Code formatting (120 char line length)
- **isort** - Import sorting (black profile)
- **mypy** - Static type checking
- **General checks** - Trailing whitespace, YAML/JSON/TOML validation, merge conflicts

To run hooks manually on all files:
```bash
make pre-commit
```

## Code Quality Expectations

### Formatting
- Line length: 120 characters
- Use Black formatting style
- Imports sorted with isort (black profile)

### Type Hints
- All public functions should have type hints
- Use `Optional[]` for nullable parameters
- Run `mypy` to verify (included in pre-commit)

### Testing
- All new features require tests
- Bug fixes should include regression tests
- **Minimum 75% code coverage** - CI will fail below this threshold
- Run tests locally before pushing:
  ```bash
  make test      # Quick test run
  make test-cov  # With coverage report
  ```

### Mutation Testing

We use [mutmut](https://mutmut.readthedocs.io/) to verify test quality beyond coverage metrics.
Mutation testing introduces small code changes (mutations) and checks if tests catch them.

```bash
make mutate          # Run mutation testing (takes a while)
make mutate-results  # Show results summary
make mutate-browse   # Interactive browser for surviving mutants
make mutate-html     # Generate HTML report
```

**Target:** 80%+ mutation kill rate (surviving mutants indicate weak assertions)

> **Note:** Due to PySpark worker incompatibility with mutmut 3's trampoline, mutation tests
> run against LocalBackend tests only. The core fitting logic is backend-agnostic, so this
> still provides comprehensive coverage.

### Documentation
- Public APIs require docstrings (Google style)
- Update README.md for user-facing changes
- Update docs/ for new features

## Commit Messages

We use [Conventional Commits](https://www.conventionalcommits.org/) with semantic-release:

| Prefix | Description | Version Bump |
|--------|-------------|--------------|
| `feat:` | New feature | Minor (0.X.0) |
| `fix:` | Bug fix | Patch (0.0.X) |
| `perf:` | Performance improvement | Patch |
| `refactor:` | Code refactor (no behavior change) | Patch |
| `chore:` | Maintenance (no release) | None |
| `docs:` | Documentation only | None |
| `ci:` | CI/CD changes | None |

Examples:
```
feat: add support for discrete distributions
fix: handle edge case in histogram computation
chore: update pre-commit hook versions
```

## Pull Request Process

1. **Create a feature branch**
   ```bash
   git checkout -b feat/your-feature
   ```

2. **Make your changes**
   - Write code and tests
   - Ensure pre-commit hooks pass
   - Verify tests pass locally

3. **Push and create PR**
   ```bash
   git push -u origin feat/your-feature
   ```
   Then open a PR against `main`.

4. **CI checks must pass**
   - Pre-commit hooks
   - Tests across Python 3.11-3.13 and Spark 3.5/4.x matrix
   - Documentation build
   - 75% minimum coverage

   A PR that changes `scripts/spark_canary/` or `.github/workflows/spark-canary.yml` also runs
   the Spark canary (below) in dry-run mode: it prints the issues it would file instead of
   filing them. Starting the canary by hand (Run workflow, i.e. `workflow_dispatch`) from any
   branch files real issues, so only do that when you want them.

5. **Review and merge**
   - Address any feedback
   - Squash merge preferred for clean history

## Spark Release Canary

Every Monday, `.github/workflows/spark-canary.yml` looks up the newest PyPI patch of each
Spark minor from 3.5 up. It installs spark-bestfit with each one:

```bash
pip install ".[spark,test-base]" "pyspark[sql]==<version>"
```

Then it runs the Spark tests and benchmarks the newest minor against the one before it. What it
finds becomes an issue labelled `spark-canary`. When an open issue already has the same title,
a later run adds a comment if anything changed instead of opening a second one:

| Issue title | Meaning |
|-------------|---------|
| `Spark canary: tests fail on pyspark {ver}` | The Spark tests fail on that release |
| `Spark canary: performance regression on pyspark {target} vs {baseline}` | At least one benchmark's median is more than 10% slower on the newest minor than on the previous one |
| `Spark canary: spark-bestfit[spark] cannot install with pyspark {ver}` | A release inside the declared `pyspark` range can't be installed under the current pins |
| `Spark canary: pyspark {ver} passes, outside the declared range` | The tests pass on a release that the `pyspark` range in `pyproject.toml` excludes, so the range can widen. Once closed, it is never reopened or filed again |
| `Spark canary: workflow failing` | The canary itself broke; check the linked run |

The canary never closes an issue. A maintainer closes it once the fix is merged. Every issue
ends with a checklist: update the `pyspark` range in `pyproject.toml` if support changed, the
Compatibility table in `README.md`, and the Compatibility Matrix in `docs/quickstart.rst`.

## Makefile Commands

| Command | Description |
|---------|-------------|
| `make help` | Show all available commands |
| `make install-dev` | Install with dev dependencies + pre-commit |
| `make test` | Run tests |
| `make test-cov` | Run tests with coverage report |
| `make pre-commit` | Run all pre-commit hooks |
| `make check` | Run pre-commit + tests |
| `make docs` | Build documentation |
| `make clean` | Remove build artifacts |
| `make mutate` | Run mutation testing |
| `make mutate-browse` | Interactive mutation results browser |

## Questions?

Open an issue for questions or discussions about contributions.
