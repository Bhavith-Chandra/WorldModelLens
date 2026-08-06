# Contributing to World Model Lens

Thank you for your interest in contributing to World Model Lens!

## Development Setup

1. Clone the repository:
```bash
git clone https://github.com/Bhavith-Chandra/WorldModelLens
cd world_model_lens
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install in development mode:
```bash
pip install -e ".[dev]"
```

## Code Style

We use:
- **black** for code formatting
- **ruff** for linting
- **mypy** for type checking

Run the formatter:
```bash
make format
```

Run linting and type checking:
```bash
make lint
make type-check
```

## Testing

Run tests:
```bash
pytest tests/ -v
```

Run with coverage:
```bash
make coverage
```

## Adding a New Backend

1. Create a new file in `world_model_lens/backends/`
2. Implement the `BaseModelAdapter` interface
3. Register in `backends/__init__.py`
4. Add to `backends/registry.py`
5. Add tests in `tests/`

See `backends/custom_adapter_template.py` for a detailed template.

## Pull Request Process

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests and linting
5. Submit a PR

## Code of Conduct

Be respectful and constructive in all interactions.
