# Quality Checks

This project includes comprehensive quality checks that run automatically on GitHub Actions for every push and pull request to `main` and `develop` branches.

## Running Checks Locally

To run the quality checks locally, first install the development dependencies:

```bash
pip install -r requirements-dev.txt
```

## Available Checks

### 1. Linting (Flake8 & Pylint)

**Flake8** - Checks for PEP8 compliance and syntax errors:
```bash
flake8 .
```

**Pylint** - Deep code analysis with quality scoring:
```bash
pylint src/
```

### 2. Code Formatting (Black & isort)

**Black** - Code formatter:
```bash
black .  # Format code
black --check --diff .  # Check without formatting
```

**isort** - Import statement organizer:
```bash
isort .  # Sort imports
isort --check-only --diff .  # Check without sorting
```

### 3. Type Checking (mypy)

Static type checking for Python:
```bash
mypy src/ --ignore-missing-imports
```

### 4. Security Scanning

**Bandit** - Security linter for Python:
```bash
bandit -r src/
```

**Safety** - Checks for known security vulnerabilities in dependencies:
```bash
safety check
```

**pip-audit** - Audits dependencies for vulnerabilities:
```bash
pip-audit
```

### 5. Code Complexity (Radon)

**Cyclomatic Complexity**:
```bash
radon cc src/ -a -s
```

**Maintainability Index**:
```bash
radon mi src/ -s
```

### 6. Import Validation

Validate that all modules can be imported:
```bash
python -c "import src.config; import src.data; import src.model"
python -c "import src.train; import src.evaluate; import src.visualize"
python -c "import src.presets; import src.callbacks; import src.optimizers"
python -c "import src.schedulers"
```

### 7. Streamlit App Validation

Validate the Streamlit application:
```bash
streamlit run app.py --server.headless true
```

## Configuration Files

- **`.flake8`** - Flake8 configuration (line length, exclusions, ignores)
- **`pyproject.toml`** - Black, isort, mypy, and bandit configuration
- **`.pylintrc`** - Pylint configuration (disabled checks, limits)
- **`requirements-dev.txt`** - Development dependencies for quality checks

## GitHub Actions Workflow

The workflow runs the following jobs in parallel:

1. **Python Linting** - Flake8 and Pylint
2. **Code Format Check** - Black and isort
3. **Type Checking** - mypy
4. **Security Scanning** - Bandit and Safety
5. **Dependency Vulnerability Check** - pip-audit
6. **Import Validation** - Module import checks
7. **Code Complexity** - Radon complexity analysis
8. **Streamlit App Validation** - App syntax check
9. **Quality Check Summary** - Aggregates all results

## Fixing Issues

### Auto-fixable Issues

Some issues can be automatically fixed:
```bash
# Format code with Black
black .

# Sort imports with isort
isort .
```

### Manual Fixes

For linting and type errors, review the specific files and lines mentioned in the output and fix manually.

## CI/CD

The workflow triggers on:
- Push to `main` or `develop` branches
- Pull requests to `main` or `develop` branches

All security scan reports (Bandit and Safety) are uploaded as artifacts and can be downloaded from the GitHub Actions run page.
