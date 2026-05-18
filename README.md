# OpenRouter Rankings

What are the things I value in a completion large language model? Do I care about intelligence, price, or throughput?
You can try to compare these things on the OpenRouter Rankings page: https://openrouter.ai/rankings, but there are just
too many dimensions to think about.

## Development

Set up `pre-commit`:

```bash
pip install pre-commit
pre-commit install
pre-commit run --all-files
```

Install dependencies:

```bash
pip install -e .[dev]
```

## Generative AI Disclosure

This project was developed with the assistance of JetBrains' Junie, an AI agent for software development. All aspects of
its implementation were read, verified, and tested to ensure they were accurate by me (there were MANY fixes, changes,
and enhancements implemented manually by me).
