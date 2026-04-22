import numpy as np
import pickle
import time
import os
from multiprocessing import Pool, cpu_count
from sklearn.metrics.pairwise import cosine_similarity

PARALLEL_CACHE = "data/similarity_parallel.pkl"

def compute_chunk(args):
    chunk_indices, all_values = args
    chunk = all_values[chunk_indices]
    return chunk_indices, cosine_similarity(chunk, all_values)

def compute_parallel(matrix, n_workers=None):
    if n_workers is None:
        n_workers = min(cpu_count(), 4)

    cache_path = f"data/similarity_parallel_{n_workers}w.pkl"

    if os.path.exists(cache_path):
        print(f"✅ Loading cached parallel similarity ({n_workers} workers)...")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    print(f"⏳ Computing similarity (parallel, {n_workers} workers)...")
    start = time.perf_counter()

    values = matrix.values
    n = len(values)

    # Split indices into chunks
    indices = list(range(n))
    chunk_size = max(1, n // n_workers)
    chunks = [indices[i:i+chunk_size] for i in range(0, n, chunk_size)]
    args = [(chunk, values) for chunk in chunks]

    # Parallel execution
    with Pool(processes=n_workers) as pool:
        results = pool.map(compute_chunk, args)

    # Merge results
    sim_matrix = np.zeros((n, n))
    for chunk_indices, chunk_sim in results:
        sim_matrix[chunk_indices] = chunk_sim

    elapsed = time.perf_counter() - start
    print(f"✅ Parallel done in {elapsed:.2f}s with {n_workers} workers")

    result = {
        "matrix": sim_matrix,
        "index": list(matrix.index),
        "time": elapsed,
        "workers": n_workers
    }

    with open(cache_path, "wb") as f:
        pickle.dump(result, f)

    # Save timing
    os.makedirs("results", exist_ok=True)
    with open(f"results/parallel_time_{n_workers}w.txt", "w") as f:
        f.write(str(elapsed))

    return result

def run_evaluation(matrix):
    """Run serial vs parallel comparison for academic evaluation."""
    from src.similarity_serial import compute_serial

    print("\n📊 Running full evaluation...\n")
    serial_result = compute_serial(matrix)
    serial_time = serial_result["time"]

    worker_counts = [2, 4, 8]
    summary = []

    for w in worker_counts:
        # Clear cache to force recompute
        cache_path = f"data/similarity_parallel_{w}w.pkl"
        if os.path.exists(cache_path):
            os.remove(cache_path)

        result = compute_parallel(matrix, n_workers=w)
        speedup = serial_time / result["time"]
        summary.append({
            "workers": w,
            "time": result["time"],
            "speedup": speedup
        })
        print(f"Workers: {w} | Time: {result['time']:.2f}s | Speedup: {speedup:.2f}x")

    # Save summary CSV
    import csv
    with open("results/evaluation_summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["workers", "time", "speedup"])
        writer.writeheader()
        writer.writerows(summary)

    print("\n✅ Evaluation complete. Results saved to results/evaluation_summary.csv")
    return summary