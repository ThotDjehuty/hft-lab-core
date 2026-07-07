# Contributing to HFT Lab Core

We love your input! We want to make contributing to HFT Lab Core as easy and transparent as possible.

## Development Process

1. Fork the repo and create your branch from `main`.
2. If you've added code that should be tested, add tests.
3. If you've changed APIs, update the documentation.
4. Ensure the test suite passes.
5. Make sure your code lints.
6. Issue that pull request!

## Quick Start

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/hfthot-lab-core.git
cd hfthot-lab-core

# Install development dependencies
pip install maturin pytest numpy

# Build and install in development mode
maturin develop

# Run tests
cargo test --features python
```

## Code Style

### Rust

- Use `cargo fmt` to format code
- Use `cargo clippy` to lint
- Follow Rust naming conventions (snake_case for functions, CamelCase for types)
- Add documentation comments (`///`) for public items

### Python

- Follow PEP 8
- Use type hints where possible
- Add docstrings to public functions

## Testing

```bash
# Run Rust tests
cargo test --features python

# Run Rust tests with output
cargo test --features python -- --nocapture

# Run benchmarks
cargo bench --features python
```

## Pull Request Process

1. Update the README.md with details of changes if relevant
2. Update the TODO.md if any items are completed
3. The PR will be merged once you have the sign-off of at least one maintainer

## Any contributions you make will be under the MIT Software License

In short, when you submit code changes, your submissions are understood to be under the same [MIT License](LICENSE) that covers the project.

## Report bugs using GitHub's [issues](https://github.com/ThotDjehuty/hfthot-lab-core/issues)

We use GitHub issues to track public bugs. Report a bug by [opening a new issue](https://github.com/ThotDjehuty/hfthot-lab-core/issues/new).

## Write bug reports with detail, background, and sample code

**Great Bug Reports** tend to have:

- A quick summary and/or background
- Steps to reproduce
  - Be specific!
  - Give sample code if you can
- What you expected would happen
- What actually happens
- Notes (possibly including why you think this might be happening)

## License

By contributing, you agree that your contributions will be licensed under its MIT License.
