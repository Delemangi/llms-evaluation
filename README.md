# LLMs Evaluation (on Macedonian data)

This is an experiment to show which LLMs are capable of extracting relevant information out of documents.

This repository contains two projects:

- [`data`](./data) - creating the dataset
- [`score`](./score) - evaluating model performance on the dataset

Both projects are initialized with `uv`. Use `uv sync` to create the virtual environments and `uv run <script>` to run the scripts.

## Data

First, a dataset with a specific format is required with at least the following columns: `Question` | `Document` | `Answer`.

There are scripts for models from different providers (currently OpenRouter, Ollama and localy). Each one of them has to be run separately and in sequence.

Steps:

1. Place dataset as `data.csv` in the root folder
2. Run `./data/ollama.py` (if you need those models)
3. Run `./data/openrouter.py` (if you need those models)
4. Run `./data/local.py` (if you need those models) (**WARNING**: Requires a GPU!)

At this point, the `data.csv` file should contain new columns with the predictions for each model.

## Score

1. Run `./score/main.py` (**WARNING**: Requires a GPU!)

At this point, `scores.csv` should be created in the root folder containing several metrics
