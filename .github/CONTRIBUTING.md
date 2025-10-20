# Contributing to BrainflowCytonEEGWrapper

Thank you for your interest in contributing! Here's how to get started.

## Development Setup

1. Clone the repository:
```bash
git clone https://github.com/zeyus-research/BrainflowCytonEEGWrapper
cd BrainflowCytonEEGWrapper
```

2. Install dependencies:
```bash
uv sync --all-extras --dev
```

3. Run tests:
```bash
uv run pytest tests/ -v
```

## Making Changes

1. Create a new branch for your feature/fix
2. Make your changes
3. Add tests for new functionality
4. Ensure all tests pass: `uv run pytest`
5. Submit a pull request

## Running Tests

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=BrainflowCyton

# Run specific test
uv run pytest tests/test_eeg.py::TestEEGDummyBoard::test_initialization
```

## Code Style

- Follow PEP 8 guidelines
- Use type hints where appropriate
- Add docstrings to public methods
- Keep lines under 120 characters

## Release Process

Releases are automated via GitHub Actions:

1. **Development builds**: Pushed automatically to `latest-dev` tag on every main branch commit
2. **Official releases**: Create a version tag (e.g., `v0.3.0`) to trigger PyPI publication

### Creating a Release

1. Update version in `pyproject.toml` and `BrainflowCyton/__init__.py`
2. Update `CHANGELOG.md`
3. Commit changes: `git commit -am "Release v0.3.0"`
4. Create tag: `git tag v0.3.0`
5. Push: `git push && git push --tags`

The GitHub Action will automatically:
- Run tests
- Build packages
- Create GitHub release
- Publish to PyPI (requires trusted publishing setup)

## Questions?

Open an issue or discussion on GitHub!
