# Makefile for HGT Alarm Pipeline
.PHONY: help install install-dev train test lint format clean

# Default target
help:
	@echo "HGT Alarm Pipeline - Makefile Commands"
	@echo ""
	@echo "Usage:"
	@echo "  make install          Install core dependencies"
	@echo "  make install-dev      Install development dependencies"
	@echo "  make train            Run training pipeline"
	@echo "  make test             Run test suite"
	@echo "  make lint             Run linters (ruff)"
	@echo "  make format           Run formatters (black, isort)"
	@echo "  make clean            Clean up build artifacts and caches"

# Install dependencies
install:
	pip install -e .

install-dev:
	pip install -e ".[dev,visualization]"

# Run training
train:
	python main.py --config configs/config.yaml --mode train

# Run tests
test:
	pytest tests/ -v

# Linting
lint:
	ruff check src/ tests/ pyHGT/ training_data/ main.py

# Formatting
format:
	black src/ tests/ pyHGT/ training_data/ main.py
	isort src/ tests/ pyHGT/ training_data/ main.py

# Clean up
clean:
	rm -rf __pycache__
	rm -rf .pytest_cache
	rm -rf .mypy_cache
	rm -rf .ruff_cache
	rm -rf build
	rm -rf dist
	rm -rf *.egg-info
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -type d -prune -exec rm -rf {} +
