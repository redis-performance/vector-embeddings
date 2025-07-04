"""
Simplified ground truth generator for cosine similarity on DBpedia embeddings.
"""

import argparse
import numpy as np
from datasets import load_dataset
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
import h5py


def generate_ground_truth_cosine(
    embeddings, test_size=10000, k_neighbors=100, random_state=42
):
    """
    Generate ground truth for cosine similarity using brute force nearest neighbors.

    Args:
        embeddings: numpy array of embeddings (n_samples, n_features)
        test_size: number of test queries to generate
        k_neighbors: number of nearest neighbors to find for each query
        random_state: random seed for reproducibility

    Returns:
        dict with train_embeddings, test_embeddings, neighbors, distances
    """
    print(f"Generating ground truth for {len(embeddings)} embeddings...")

    # Split into train and test sets
    train_embeddings, test_embeddings = train_test_split(
        embeddings, test_size=test_size, random_state=random_state
    )

    print(f"Train set: {len(train_embeddings)} embeddings")
    print(f"Test set: {len(test_embeddings)} embeddings")

    # Use sklearn's NearestNeighbors with cosine distance
    nbrs = NearestNeighbors(n_neighbors=k_neighbors, algorithm="brute", metric="cosine")

    print("Fitting nearest neighbors model...")
    nbrs.fit(train_embeddings)

    print("Computing ground truth neighbors...")
    distances, neighbors = nbrs.kneighbors(test_embeddings)

    return {
        "train_embeddings": train_embeddings,
        "test_embeddings": test_embeddings,
        "neighbors": neighbors,
        "distances": distances,
    }


def save_ground_truth(ground_truth, output_file):
    """
    Save ground truth to numpy files.

    Args:
        ground_truth: dict returned by generate_ground_truth_cosine
        output_file: base filename (without extension)
    """
    np.save(f"{output_file}_train.npy", ground_truth["train_embeddings"])
    np.save(f"{output_file}_test.npy", ground_truth["test_embeddings"])
    np.save(f"{output_file}_neighbors.npy", ground_truth["neighbors"])
    np.save(f"{output_file}_distances.npy", ground_truth["distances"])

    print(f"Ground truth saved to {output_file}_*.npy files")


def save_h5py(ground_truth, output_file):
    with h5py.File(f"{output_file}.hdf5", "w") as f:
        f.create_dataset("train", data=ground_truth["train_embeddings"])
        f.create_dataset("test", data=ground_truth["test_embeddings"])
        f.create_dataset("neighbors", data=ground_truth["neighbors"])
        f.create_dataset("distances", data=ground_truth["distances"])

    print(f"Ground truth saved to {output_file}.hdf5 file")


def main():
    parser = argparse.ArgumentParser(
        description="Generate ground truth for cosine similarity on DBpedia embeddings"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="filipecosta90/dbpedia-openai-1M-text-embedding-3-large-3072d",
        help="Hugging Face dataset name",
    )
    parser.add_argument(
        "--max_embeddings",
        type=int,
        default=1000,
        help="Maximum number of embeddings to use",
    )
    parser.add_argument(
        "--test_size", type=int, default=10, help="Number of test queries"
    )
    parser.add_argument(
        "--k_neighbors",
        type=int,
        default=100,
        help="Number of nearest neighbors to find",
    )
    parser.add_argument(
        "--output", type=str, default="ground_truth", help="Output file prefix"
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=12345,
        help="Random seed for reproducibility",
    )

    args = parser.parse_args()

    print(f"Loading dataset: {args.dataset}")
    dataset = load_dataset(args.dataset, split="train")

    # Extract embeddings
    embeddings = np.array(dataset["embedding"][: args.max_embeddings])
    print(f"Loaded {len(embeddings)} embeddings with dimension {embeddings.shape[1]}")

    # Generate ground truth
    ground_truth = generate_ground_truth_cosine(
        embeddings=embeddings,
        test_size=args.test_size,
        k_neighbors=args.k_neighbors,
        random_state=args.random_state,
    )

    # Save results
    save_ground_truth(ground_truth, args.output)
    save_h5py(ground_truth, args.output)

    print("Ground truth generation completed!")


if __name__ == "__main__":
    main()
