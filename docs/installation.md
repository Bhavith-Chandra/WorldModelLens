# Installation

**World Model Lens** requires Python 3.10 or higher. Choose your installation method below.

## Quick Install (pip)

Install from PyPI in 10 seconds:

```bash
pip install world_model_lens
```

## Installation Methods

### Option 1: pip (Recommended for Users)

The standard Python package manager approach.

```bash
pip install world_model_lens
```

**Upgrade to latest version:**

```bash
pip install --upgrade world_model_lens
```

### Option 2: uv (Fastest - Recommended)

[uv](https://astral.sh/blog/uv) is an ultra-fast Python package installer. If you have uv installed:

```bash
uv pip install world_model_lens
```

**Or create a new project with uv:**

```bash
uv venv --python 3.11
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install world_model_lens
```

### Option 3: From Source (Development)

For contributing or using the latest unreleased features:

#### Clone the repository:

```bash
git clone https://github.com/Bhavith-Chandra/WorldModelLens.git
cd WorldModelLens
```

#### Install in editable mode with pip:

```bash
pip install -e .
```

#### Or with uv (preferred):

```bash
uv pip install -e .
```

The `-e` flag installs in "editable" mode, so changes to the code are immediately reflected without reinstalling.

## Optional Dependencies

World Model Lens is modular. Install only what you need:

### Visualization (`viz`)
For enhanced plotting and visualization exports:

```bash
pip install world_model_lens[viz]
```

Includes: Plotly enhanced features, Kaleido for static image export

### API Server (`api`)
To run the FastAPI server for remote analysis:

```bash
pip install world_model_lens[api]
```

Includes: FastAPI, Uvicorn, Pydantic, authentication, monitoring

### Documentation (`docs`)
To build documentation locally:

```bash
pip install world_model_lens[docs]
```

## Getting Help

- **Issues?** Check [GitHub Issues](https://github.com/Bhavith-Chandra/WorldModelLens/issues)
- **Questions?** Open a [Discussion](https://github.com/Bhavith-Chandra/WorldModelLens/discussions)
- **Docs?** Read the [full documentation](https://worldmodellens.readthedocs.io)
