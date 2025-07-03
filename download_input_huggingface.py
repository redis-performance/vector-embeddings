from datasets import load_dataset
import pandas as pd
import os

if not os.path.exists("input"):
    os.makedirs("input")

csv_filename = "input/dbpedia_1M.csv"
dbpedia_dataset = load_dataset("BeIR/dbpedia-entity", "corpus")
print("finished downloading the dataset")

_id = dbpedia_dataset["corpus"]["_id"][:1_000_000]
title = dbpedia_dataset["corpus"]["title"][:1_000_000]
text = dbpedia_dataset["corpus"]["text"][:1_000_000]

print("preparing the subset of the dataset")
df = pd.DataFrame({"_id": _id, "title": title, "text": text})
df = df.dropna()
print(f"storing dataset in {csv_filename}")
df.to_csv(csv_filename, index=False)
