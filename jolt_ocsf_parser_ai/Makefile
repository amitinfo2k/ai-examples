.PHONY: install dev-install test lint format check-format clean

# Python and uv configuration
PYTHON = python3
UV = uv
PACKAGE = jolt_ocsf_parser_ai

# Ensure build dependencies are installed
build-deps:
	$(UV) pip install --upgrade pip
	$(UV) pip install hatchling

# Install the package in development mode with all dependencies
install: build-deps
	$(UV) pip install -e .

# Install development dependencies
dev-install: install
	$(UV) pip install -e ".[dev]"

# Run tests
test:
	$(PYTHON) -m pytest tests/

# Lint the code
lint:
	$(UV) run ruff check $(PACKAGE)/
	$(UV) run mypy $(PACKAGE)/

# Format the code
format:
	$(UV) run black $(PACKAGE)/
	$(UV) run isort $(PACKAGE)/

# Check formatting without making changes
check-format:
	$(UV) run black --check $(PACKAGE)/
	$(UV) run isort --check-only $(PACKAGE)/

# Clean up build artifacts
clean:
	rm -rf build/ dist/ *.egg-info/ .pytest_cache/ .mypy_cache/ .ruff_cache/
	find . -type f -name '*.py[co]' -delete
	find . -type d -name '__pycache__' -exec rm -rf {} +

# Build the package
build: build-deps
	$(PYTHON) -m build

# Install the package in production mode
prod-install: build-deps
	$(UV) pip install .

# Run the application
run:
	$(PYTHON) -m $(PACKAGE)
