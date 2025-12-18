# Contributing to spark-bestfit

Thank you for your interest in contributing to spark-bestfit!

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

5. **Review and merge**
   - Address any feedback
   - Squash merge preferred for clean history

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

## Questions?

Open an issue for questions or discussions about contributions.
