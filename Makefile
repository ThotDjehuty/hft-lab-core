# HFT Lab Core - Makefile
# Build, test, and development commands

.PHONY: all build build-release test clean install dev lint fmt docs help

# Default target
all: build

# Build in development mode
build:
	cargo build --features python

# Build release version (optimized)
build-release:
	cargo build --release --features python

# Build Python wheel
wheel:
	maturin build --release

# Build and install for development
dev:
	maturin develop --release

# Run tests
test:
	cargo test --features python

# Run all tests with verbose output
test-verbose:
	cargo test --features python -- --nocapture

# Run benchmarks
bench:
	cargo bench --features python

# Clean build artifacts
clean:
	cargo clean
	rm -rf target/wheels/*.whl

# Install in current Python environment
install: wheel
	pip install target/wheels/*.whl --force-reinstall

# Lint code
lint:
	cargo clippy --features python -- -W clippy::all

# Format code
fmt:
	cargo fmt

# Check formatting
fmt-check:
	cargo fmt -- --check

# Generate documentation
docs:
	cargo doc --features python --no-deps --open

# Publish to crates.io (requires login)
publish-crates:
	cargo publish

# Publish to PyPI (requires login)
publish-pypi: wheel
	maturin publish

# Quick development cycle: build and test
quick: fmt build test

# Full release preparation
release: fmt-check test lint build-release wheel
	@echo "Release build complete!"
	@ls -la target/wheels/

# Help
help:
	@echo "HFT Lab Core - Available Commands:"
	@echo ""
	@echo "  make build         - Build in development mode"
	@echo "  make build-release - Build optimized release"
	@echo "  make wheel         - Build Python wheel"
	@echo "  make dev           - Build and install for development"
	@echo "  make test          - Run tests"
	@echo "  make test-verbose  - Run tests with output"
	@echo "  make bench         - Run benchmarks"
	@echo "  make clean         - Clean build artifacts"
	@echo "  make install       - Install Python wheel"
	@echo "  make lint          - Run clippy linter"
	@echo "  make fmt           - Format code"
	@echo "  make fmt-check     - Check formatting"
	@echo "  make docs          - Generate documentation"
	@echo "  make quick         - Format, build, and test"
	@echo "  make release       - Full release preparation"
	@echo ""
