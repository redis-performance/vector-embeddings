# Agent guidelines

Instructions for AI coding agents (Claude Code, Copilot, Cursor, etc.) working in this repo.

## Project overview

This repository provides a complete pipeline for generating DBpedia text embeddings using
OpenAI's embedding models and publishing them as Hugging Face datasets. The pipeline
downloads the DBpedia corpus (up to 1 M records) from Hugging Face, sends batches of
title+text pairs to OpenAI's `text-embedding-3-large` model across multiple parallel
processes, saves the resulting vectors as `.npy` chunk files, then reassembles and uploads
them to the Hugging Face Hub. A separate script (`generate_ground_truth.py`) computes
brute-force cosine-similarity ground truth (train/test split + k-NN) and writes HDF5 and
`.npy` files suitable for ANN-benchmark evaluation.

## Local setup

```bash
git clone git@github.com:redis-performance/vector-embeddings.git
cd vector-embeddings

# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install all dependencies
pip install -r requirements.txt
```

Set the required environment variables before running any script:

```bash
export OPENAI_API_KEY="your-openai-api-key"   # Required — used by generate_openai_embeddings.py
export HF_TOKEN="your-huggingface-token"       # Required — used by datasets library for push_to_hub
```

### Running the pipeline

```bash
# Download the DBpedia source corpus
python download_input_huggingface.py

# Generate embeddings (small smoke-test: 100 rows)
python generate_openai_embeddings.py --nrows 100 --embedding_dimension 3072

# Upload to Hugging Face Hub
python upload_dataset_huggingface.py --embedding_dimension 3072

# Generate ground-truth nearest-neighbour files
python generate_ground_truth.py \
  --dataset filipecosta90/dbpedia-openai-1M-text-embedding-3-large-3072d \
  --max_embeddings 10000 \
  --test_size 100
```

## Branch naming

Same as human contributors: `<type>/<short-description>` (e.g. `feat/add-cohere-embeddings`).

## Coding standards

- Match the style already in the file you are editing.
- Prefer clear, minimal changes over large refactors unless explicitly asked.
- Do not add comments that describe *what* the code does — only add comments when the *why* is non-obvious.
- Do not introduce new dependencies without checking with the maintainer.

## Running tests

There is no automated test suite. Before declaring a task complete, verify your change manually:

1. Ensure `pip install -r requirements.txt` succeeds with no errors.
2. Run `generate_openai_embeddings.py` with `--nrows 100` and confirm `.npy` files appear in `output/`.
3. If you changed `generate_ground_truth.py`, run it against an existing HF dataset and inspect the output HDF5 file.
4. If you changed `upload_dataset_huggingface.py`, do a dry run against a private/test HF repo.

## How to submit changes

1. Create a branch: `git checkout -b <type>/<description>`.
2. Commit with a clear message focused on *why*, not *what*.
3. Open a pull request against `main`.
4. Do **not** push directly to `main`.

## What to avoid

- Do not reformat files unrelated to your change.
- Do not remove error handling or tests.
- Do not commit secrets, credentials, or large binary files.
- Do not amend published commits.
- Do not add new dependencies to `requirements.txt` without explicit maintainer approval.
- Do not hardcode API keys, HF tokens, or any credentials — always read them from environment variables.
- Do not delete or overwrite existing `.npy` chunk files in `output/` without confirming the intent with the user; re-generating them is expensive (OpenAI API costs money).
