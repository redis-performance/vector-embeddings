import pandas as pd
import numpy as np
from datasets import Dataset
import os
import tqdm
import argparse

parser = argparse.ArgumentParser(
    prog="text-embedding-3-large embeddings uploader",
)
parser.add_argument("--nrows", type=int, default=1000000)
parser.add_argument("--chunksize", type=int, default=100)
parser.add_argument("--csv_filename", type=str, default="input/dbpedia_1M.csv")
parser.add_argument("--embedding_dimension", type=int, default=3072)

args = parser.parse_args()
nrows = args.nrows
chunksize = args.chunksize
csv_filename = args.csv_filename
embedding_dimension = args.embedding_dimension

print("reading *.npy files...")
df = pd.read_csv(csv_filename, nrows=nrows)
flat_embeddings = []
for row_start in tqdm.tqdm(range(0, nrows, chunksize)):
    embeddings_filename = f"output/embedded_dbpedia_1M_dim_{embedding_dimension}_{row_start}_{chunksize}.npy"
    if os.path.exists(embeddings_filename):
        with open(embeddings_filename, "rb") as f:
            embeddings = np.load(f)
            for embedding in embeddings:
                assert len(embedding) == embedding_dimension
                flat_embeddings.append(embedding)
    else:
        print(f"missing file.... {embeddings_filename} exiting right away...")
        exit(-1)

_id = df["_id"]
title = df["title"]
text = df["text"]

print("preparing the subset of the dataset")
df = pd.DataFrame(
    {"_id": _id, "title": title, "text": text, "embedding": flat_embeddings}
)

print("publishing dataset")
dataset = Dataset.from_pandas(df)
print("publishing dataset")
dataset.push_to_hub(
    f"filipecosta90/dbpedia-openai-1M-text-embedding-3-large-{embedding_dimension}d"
)
