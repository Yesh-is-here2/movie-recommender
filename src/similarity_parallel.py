# similarity_parallel.py
# Computes the item-item cosine similarity matrix using Python multiprocessing.
# This is the PARALLEL implementation — the core of our concurrent programming project.
#
# Approach:
# - The full list of movie indices is split into equal-sized chunks
# - Each chunk is assigned to a separate worker process
# - Workers compute cosine similarity for their chunk independently (no shared state)
# - Results from all workers are merged into the final similarity matrix
#
# This works well because similarity computations are embarrassingly parallel —
# computing similarity between movie A and all others is completely independent
# of computing similarity between movie B and all others.

import numpy as np
import pickle
import time
import os
from multiprocessing import Pool, cpu_count
from sklearn.metrics.pairwise import cosine_similarity

PARALLEL_CACHE = "data/similarity_parallel.pkl"


def compute_chunk(args):
    """
    Worker function executed by each parallel process.
    Computes cosine similarity for a subset (chunk) of movies
    against the full movie matrix.

    Each worker receives:
    - chunk_indices: the row indices this worker is responsible for
    - all_values: the full matrix (needed to compute similarity against all movies)

    Returns the chunk indices and the computed similarity rows so the
    main process can place them in the correct position in the final matrix.

    This function must be defined at the module level (not inside another function)
    so that Python's multiprocessing can pickle and send it to worker processes.
    """
    chunk_indices, all_values = args
    chunk = all_values[chunk_indices]  # Extract just this worker's rows
    return chunk_indices, cosine_similarity(chunk, all_values)


def compute_parallel(matrix, n_workers=None):
    """
    Compute the full item-item similarity matrix using multiple CPU cores.

    The matrix is divided into chunks — one per worker process.
    Workers run simultaneously, each computing similarity for their chunk.
    After all workers finish, results are assembled into the full N x N matrix.

    Parameters:
        matrix: the user-item rating DataFrame from preprocess.py
        n_workers: number of parallel processes (defaults to min(cpu_count, 4))

    The result is cached so subsequent app startups load instantly.
    """
    # Default to available CPU cores but cap at 4 to avoid overhead issues
    if n_workers is None:
        n_workers = min(cpu_count(), 4)

    cache_path = f"data/similarity_parallel_{n_workers}w.pkl"

    # Load from cache if this worker count was computed before
    if os.path.exists(cache_path):
        print(f"✅ Loading cached parallel similarity ({n_workers} workers)...")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    print(f"⏳ Computing similarity (parallel, {n_workers} workers)...")
    start = time.perf_counter()

    values = matrix.values
    n = len(values)

    # Split movie indices into equal chunks — one chunk per worker
    indices = list(range(n))
    chunk_size = max(1, n // n_workers)
    chunks = [indices[i:i+chunk_size] for i in range(0, n, chunk_size)]

    # Pair each chunk with the full matrix so workers have everything they need
    args = [(chunk, values) for chunk in chunks]

    # Launch worker pool and distribute chunks
    # pool.map blocks until all workers are done
    with Pool(processes=n_workers) as pool:
        results = pool.map(compute_chunk, args)

    # Merge results from all workers into one complete similarity matrix
    sim_matrix = np.zeros((n, n))
    for chunk_indices, chunk_sim in results:
        sim_matrix[chunk_indices] = chunk_sim  # Place each worker's rows in correct position

    elapsed = time.perf_counter() - start
    print(f"✅ Parallel done in {elapsed:.2f}s with {n_workers} workers")

    # Package result with metadata
    result = {
        "matrix": sim_matrix,
        "index": list(matrix.index),  # Movie IDs in row order
        "time": elapsed,
        "workers": n_workers
    }

    # Save to cache for future startups
    with open(cache_path, "wb") as f:
        pickle.dump(result, f)

    # Save timing for evaluation
    os.makedirs("results", exist_ok=True)
    with open(f"results/parallel_time_{n_workers}w.txt", "w") as f:
        f.write(str(elapsed))

    return result


def run_evaluation(matrix):
    """
    Run a full performance comparison between serial and parallel implementations.
    Tests with 2, 4, and 8 worker processes and computes speedup for each.
    Results are saved to results/evaluation_summary.csv.
    Speedup = Serial Time / Parallel Time (higher is better).
    """
    from src.similarity_serial import compute_serial

    print("\n📊 Running full evaluation...\n")

    # Get serial baseline time
    serial_result = compute_serial(matrix)
    serial_time = serial_result["time"]

    worker_counts = [2, 4, 8]
    summary = []

    for w in worker_counts:
        # Remove cached result to force fresh recompute for accurate timing
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

    # Save all results to CSV for report and graphs
    import csv
    with open("results/evaluation_summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["workers", "time", "speedup"])
        writer.writeheader()
        writer.writerows(summary)

    print("\n✅ Evaluation complete. Results saved to results/evaluation_summary.csv")
    return summary