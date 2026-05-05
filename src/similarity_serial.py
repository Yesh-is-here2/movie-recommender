# similarity_serial.py
# Computes the item-item cosine similarity matrix using a single CPU core.
# This is the SERIAL BASELINE used for performance comparison.
# All movie pairs are processed sequentially — no parallelism.
# Results are cached to disk after the first run.

import numpy as np
import pickle
import time
import os
from sklearn.metrics.pairwise import cosine_similarity

# Cache path for the serial similarity matrix
SERIAL_CACHE = "data/similarity_serial.pkl"


def compute_serial(matrix):
    """
    Compute cosine similarity between all movie pairs sequentially.

    Cosine similarity measures the angle between two rating vectors.
    Movies rated similarly by users will have high cosine similarity (close to 1).
    Movies rated very differently will have low similarity (close to 0 or negative).

    This serial version uses sklearn's cosine_similarity which processes
    the entire matrix at once on a single thread. It serves as our baseline
    to compare against the parallel implementation.

    The result is an N x N matrix where result[i][j] is the similarity
    between movie i and movie j.

    Results are cached as a .pkl file so we don't recompute on every startup.
    """
    # Load from cache if available to save time on startup
    if os.path.exists(SERIAL_CACHE):
        print("✅ Loading cached serial similarity...")
        with open(SERIAL_CACHE, "rb") as f:
            return pickle.load(f)

    print("⏳ Computing similarity (serial)...")
    start = time.perf_counter()  # High-precision timer for benchmarking

    # Extract the raw numpy matrix from the pandas DataFrame
    values = matrix.values

    # Compute all pairwise cosine similarities in one operation
    # This is O(n²) in time and memory — gets expensive with large datasets
    sim_matrix = cosine_similarity(values)

    elapsed = time.perf_counter() - start
    print(f"✅ Serial done in {elapsed:.2f}s")

    # Package result with metadata for evaluation comparison
    result = {
        "matrix": sim_matrix,
        "index": list(matrix.index),  # Movie IDs in matrix row order
        "time": elapsed                # Execution time for speedup calculations
    }

    # Save to cache
    with open(SERIAL_CACHE, "wb") as f:
        pickle.dump(result, f)

    # Also save timing to a text file for easy reference
    os.makedirs("results", exist_ok=True)
    with open("results/serial_time.txt", "w") as f:
        f.write(str(elapsed))

    return result