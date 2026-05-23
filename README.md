# LLM Filtering Search Tool

The goal is to consolidate multidimensional LLM metrics and benchmarks into a searchable platform.

## Motivation

What are the things I value in a completion large language model? Do I care about intelligence, price, or throughput?
You can try to compare these things on the [OpenRouter Rankings page](https://openrouter.ai/rankings), or
the [Artificial Analysis models page](https://artificialanalysis.ai/models), but neither of those tools really let you
expand your search beyond the few models you can see on the page.

The goal of this project is to:

1. Gather the latest model benchmarks and metrics from external metadata sources.
2. Allow a user to filter and sort models by all possible properties.
3. Generate some nice plots based on the user's specifications.

And, the intent is, given these tools, it will be easier for you to decide which LLM is best for your specific
situation.

Here are the metrics I care about:

- Intelligence
- Price
- Throughput (tokens/second)
- Context Window
- Release Date

So that's five dimensions. No platform that I can find out there has a tool that lets you sort and filter all possible
models by these more complex metrics. In my ideal world, I would be able to set the following parameters:

| Metric         | My Filter                    |
|----------------|------------------------------|
| Intelligence   |                              |
| Price          | $/1M output tokens <= \$1.50 |
| Throughput     | \>= 150 tokens/s             |
| Context Window | \>= 750,000 tokens           |
| Release Date   | \>= Oct. 2025                |

And then I could sort from highest to lowest intelligence, probably choosing the highest-ranked one!

## Technical Overview

1. Download model data from the [Artificial Analysis API](https://artificialanalysis.ai/api-reference#models-endpoint).
2. Download model data from the [OpenRouter API](https://openrouter.ai/docs/api/api-reference/models/get-models).
3. Pair up Artificial Analysis benchmarks and OpenRouter models
4. Populate an SQLite database with the model data.
5. Expose the data via a REST API back-end.
6. Create a Next.js front-end to retrieve the data and display it in
   an [MUI Data Grid](https://mui.com/x/react-data-grid/).

## Development

Set up `pre-commit`:

```bash
pip install pre-commit
pre-commit install
pre-commit run --all-files
```

Install dependencies:

```bash
./bin/install-backend.sh
```

Activate the virtual environment:

```bash
source .venv/bin/activate
```

## Generative AI Disclosure

This project was developed with the assistance of JetBrains' coding agent Junie. When work was fully off-loaded to the
LLM, all aspects of its implementation were read, verified, and tested to ensure they were accurate by me (there were
MANY fixes, changes, and enhancements implemented manually by me).
