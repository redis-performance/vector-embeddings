# dbpedia-openai-text-embeddings-to-huggingface

This repository provides a complete pipeline for generating DBpedia text embeddings using OpenAI's embedding models and publishing them as Hugging Face datasets. The pipeline supports generating embeddings with different dimensions from the same OpenAI model and source data, allowing you to create multiple dataset variants optimized for different use cases.

## Features

- **Flexible Embedding Dimensions**: Generate embeddings with different dimensions (e.g., 512, 1024, 1536, 3072) from the same OpenAI model
- **Scalable Processing**: Multi-process embedding generation for large datasets
- **Hugging Face Integration**: Direct upload to Hugging Face Hub
- **Resume Support**: Skip already processed chunks to resume interrupted jobs

## Generated Datasets

| Dataset link                                                                                                                                                                                 | Embedding model        | Embedding Dimensions | N Vectors |
| -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------- | -------------------- | --------- |
| [dbpedia-openai-1M-text-embedding-3-large-3072d](https://huggingface.co/datasets/filipecosta90/dbpedia-openai-1M-text-embedding-3-large-3072d) | text-embedding-3-large | 3072                 | 1M   |
