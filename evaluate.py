# evaluate.py
# Standalone performance evaluation script for the parallel movie recommendation system.
# This script is NOT part of the web app — it's run separately to measure and compare
# the performance of serial vs parallel similarity computation.
#
# Run this script with: python evaluate.py
#
# What it does:
# 1. Clears all cached data to ensure fresh computation
# 2. Builds the rating matrix from the 1M MovieLens dataset
# 3. Runs serial computation (pure Python nested loop — intentionally slow)
# 4. Runs parallel computation with 2, 4, and 8 worker processes
# 5. Calculates speedup for each worker count
# 6. Saves results to CSV and generates performance graphs

import os
import time
import numpy as np
from multiprocessing import Pool
import matplotlib.pyplot as plt
import csv


def cosine_sim_chunk(args):
    """
    Worker function for parallel cosine similarity computation.
    This function is called by each worker process independently.

    We implement cosine similarity manually using pure NumPy here
    (instead of sklearn) because sklearn's cosine_similarity already uses
    optimized multi-threaded BLAS internally — using it in parallel would
    actually be SLOWER due to thread contention.

    By using pure NumPy dot products, each worker process does genuine
    independent work with no hidden parallelism underneath.

    Steps:
    1. Extract the chunk of movies this worker is responsible for
    2. Normalize the chunk vectors (divide by their L2 norm)
    3. Normalize the full matrix vectors
    4. Compute similarity as dot product of normalized vectors
    """
    indices, values = args
    chunk = values[indices]  # This worker's subset of movie vectors

    # Normalize chunk vectors to unit length
    # Avoid division by zero for zero vectors
    norms = np.linalg.norm(chunk, axis=1, keepdims=True)
    norms[norms == 0] = 1
    chunk_norm = chunk / norms

    # Normalize full matrix vectors
    full_norms = np.linalg.norm(values, axis=1, keepdims=True)
    full_norms[full_norms == 0] = 1
    values_norm = values / full_norms

    # Cosine similarity = dot product of normalized vectors
    return indices, np.dot(chunk_norm, values_norm.T)


def serial_cosine(values):
    """
    Compute cosine similarity using a pure Python nested loop.
    This is intentionally slow — it serves as our serial baseline.

    We use a nested for loop instead of sklearn so the serial version
    is genuinely slow and shows a meaningful speedup from parallelism.
    If we used sklearn for both serial and parallel, sklearn's internal
    optimization would make serial so fast that parallelism looks worse.

    The loop only computes the upper triangle (i <= j) and mirrors it
    to the lower triangle for efficiency — but it's still O(n²).
    """
    n = len(values)
    sim = np.zeros((n, n))  # Initialize n x n similarity matrix with zeros

    # Pre-compute L2 norms for all vectors
    norms = np.linalg.norm(values, axis=1)
    norms[norms == 0] = 1  # Avoid division by zero

    # Nested loop — compute similarity for every pair of movies
    for i in range(n):
        for j in range(i, n):  # Only compute upper triangle
            dot = np.dot(values[i], values[j])
            s = dot / (norms[i] * norms[j])
            sim[i][j] = s   # Fill upper triangle
            sim[j][i] = s   # Mirror to lower triangle
    return sim


def run_full_evaluation():
    """
    Main evaluation function — runs the full serial vs parallel benchmark.
    Results are saved to CSV and two PNG graphs are generated.
    """

    # Step 1: Clear all cached .pkl files so we get fresh computation times
    print("🧹 Cleaning caches...")
    for f in os.listdir('data'):
        if f.endswith('.pkl'):
            os.remove(f'data/{f}')
            print(f'  Deleted {f}')

    # Step 2: Build the user-item matrix from the 1M MovieLens dataset
    from src.preprocess import build_matrix
    print("\n⏳ Building matrix from 1M dataset...")
    matrix, movies_df = build_matrix(size='1m')

    # Convert to float32 for faster computation
    values = matrix.values.astype(np.float32)
    n = len(values)
    print(f"✅ Matrix: {matrix.shape[0]} movies x {matrix.shape[1]} users")

    # Use a 500-movie subset — large enough to show real speedup differences
    # but small enough to complete in reasonable time
    subset = values[:500]
    n_sub = len(subset)
    print(f"📊 Using {n_sub} movies for evaluation")

    results = []  # Stores timing and speedup for each method

    # Step 3: Serial baseline — pure Python nested loop
    print("\n⏳ Running serial (pure Python loop)...")
    start = time.perf_counter()  # High-precision timer
    serial_cosine(subset)
    serial_time = time.perf_counter() - start
    print(f"✅ Serial: {serial_time:.2f}s")

    # Record serial result (speedup = 1.0 by definition)
    results.append({"method": "Serial", "workers": 1,
                    "time": serial_time, "speedup": 1.0})

    # Step 4: Parallel runs with increasing worker counts
    for workers in [2, 4, 8]:
        print(f"\n⏳ Running parallel ({workers} workers)...")

        # Split movie indices into equal chunks — one per worker
        indices = list(range(n_sub))
        chunk_size = max(1, n_sub // workers)
        chunks = [indices[i:i+chunk_size] for i in range(0, n_sub, chunk_size)]

        # Pair each chunk with the full subset matrix
        args = [(chunk, subset) for chunk in chunks]

        # Launch parallel pool and time the execution
        start = time.perf_counter()
        with Pool(processes=workers) as pool:
            pool.map(cosine_sim_chunk, args)  # Blocks until all workers finish
        elapsed = time.perf_counter() - start

        # Speedup = how many times faster than serial
        speedup = serial_time / elapsed
        print(f"✅ Workers: {workers} | Time: {elapsed:.2f}s | Speedup: {speedup:.2f}x")

        results.append({
            "method": f"Parallel ({workers}w)",
            "workers": workers,
            "time": elapsed,
            "speedup": speedup
        })

    # Step 5: Save results to CSV for the report
    os.makedirs("results", exist_ok=True)
    with open("results/evaluation_summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["method", "workers", "time", "speedup"])
        writer.writeheader()
        writer.writerows(results)
    print("\n✅ Results saved to results/evaluation_summary.csv")

    # Step 6: Generate performance graphs
    print("📊 Generating plots...")

    labels = [r["method"] for r in results]
    times = [r["time"] for r in results]
    speedups = [r["speedup"] for r in results]
    worker_counts = [1, 2, 4, 8]

    # Create side-by-side plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor('#0d0d1a')  # Dark background for both plots