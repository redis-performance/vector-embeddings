# Contributing

We treat this repo as "Open Source" within Redis: anyone who clears the bar below is welcome to contribute.

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
export OPENAI_API_KEY="your-openai-api-key"   # Required for generating embeddings
export HF_TOKEN="your-huggingface-token"       # Required for downloading/uploading HF datasets
```

### Running the pipeline

```bash
# Step 1 — download the DBpedia source corpus from Hugging Face
python download_input_huggingface.py

# Step 2 — generate OpenAI embeddings (adjust --nrows / --embedding_dimension as needed)
python generate_openai_embeddings.py \
  --nrows 10000 \
  --embedding_dimension 3072 \
  --nprocesses 4

# Step 3 — upload the resulting dataset to Hugging Face Hub
python upload_dataset_huggingface.py --embedding_dimension 3072

# Optional — generate ground-truth nearest-neighbour files (HDF5 + .npy)
python generate_ground_truth.py \
  --dataset filipecosta90/dbpedia-openai-1M-text-embedding-3-large-3072d \
  --max_embeddings 10000 \
  --test_size 100
```

## Branch naming

```
<type>/<short-description>
```

Types: `feat`, `fix`, `refactor`, `test`, `docs`, `chore`

Example: `feat/add-pipeline-mode`

## Coding standards

- Keep changes focused; one logical change per PR.
- Follow the conventions already present in the codebase (formatting, naming, error handling).
- No dead code, no commented-out blocks.

## Submitting changes

1. Fork or create a branch from `main`.
2. Make your changes with clear, atomic commits.
3. Open a pull request against `main` with a descriptive title and summary.
4. Address review comments promptly; force-push to the same branch to update.

## Testing

There is no automated test suite at this time. Before opening a PR:

1. Activate your virtual environment and ensure `pip install -r requirements.txt` succeeds cleanly.
2. Run `download_input_huggingface.py` to verify dataset download works.
3. Run `generate_openai_embeddings.py` with a small `--nrows` value (e.g. 100) to confirm the embedding pipeline produces valid `.npy` output files in `output/`.
4. If your change touches `generate_ground_truth.py` or `upload_dataset_huggingface.py`, exercise those scripts as well.
5. Make sure no new `requirements.txt` entries were added without discussion.

## Review process

- At least one maintainer approval is required before merge.
- CI must be green.
- Maintainers may request changes or close PRs that don't meet the bar — this is normal and not personal.
