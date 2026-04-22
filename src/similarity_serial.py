import numpy as np
import pickle
import time
import os
from sklearn.metrics.pairwise import cosine_similarity

SERIAL_CACHE = "data/similarity_serial.pkl"

def compute_serial(matrix):
    if os.path.exists(SERIAL_CACHE):
        print("✅ Loading cached serial similarity...")
        with open(SERIAL_CACHE, "rb") as f:
            return pickle.load(f)

    print("⏳ Computing similarity (serial)...")
    start = time.perf_counter()

    values = matrix.values
    sim_matrix = cosine_similarity(values)

    elapsed = time.perf_counter() - start
    print(f"✅ Serial done in {elapsed:.2f}s")

    result = {
        "matrix": sim_matrix,
        "index": list(matrix.index),
        "time": elapsed
    }

    with open(SERIAL_CACHE, "wb") as f:
        pickle.dump(result, f)

    # Save timing to results
    os.makedirs("results", exist_ok=True)
    with open("results/serial_time.txt", "w") as f:
        f.write(str(elapsed))

    return result