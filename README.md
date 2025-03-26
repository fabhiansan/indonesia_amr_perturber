# Data Perturber

A Python library for perturbing AMR graphs with various types of errors.

## Installation

```bash
pip install data-perturber
```

## Usage

```python
from data_perturber import out_of_article_perturber

perturbed_graph, changelog = out_of_article_perturber.insertOutOfArticleError(original_graph)
```

## Available Perturbers

- Out-of-article perturber
- Entity perturber
- Predicate perturber
- Circumstance perturber
- Discourse perturber

## Development

To install in development mode:
```bash
pip install -e .
```

## Publishing to PyPI

1. Install build tools:
```bash
pip install wheel
```

2. Build the package:
```bash
python setup.py sdist bdist_wheel
```

2. Upload to PyPI:
```bash
twine upload dist/*
