#!/usr/bin/env python3
import time
import random
import signal
import tracemalloc
import os
import inspect
import sys
from prettytable import PrettyTable
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgba
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse

# import shared utilities
from utils import (
    TIMEOUT, MAX_SIZE, SIZE_INCREMENT, DATA_TYPE, TEST_RANGE,
    generate_test_data, discover_sorting_algorithms, verify_sorting, benchmark_single
)

# global settings with environment variable overrides
NUM_PROCESSES = max(1, mp.cpu_count() - 1)

def worker_init():
    signal.signal(signal.SIGALRM, signal.SIG_IGN)

# wrapper to include size info; returns (size, result)
def run_benchmark_task(name, func, data, scaled_timeout, size):
    return size, benchmark_single(name, func, data, scaled_timeout)

def benchmark_for_complexity(algos, sizes=None, dt=None, tr=None, max_time=None, size_increment=None):
    dt = dt or DATA_TYPE
    tr = tr or TEST_RANGE
    max_time = max_time or TIMEOUT
    size_increment = size_increment or SIZE_INCREMENT
    if sizes is None:
        sizes = [100, 200, 500]
        current_size = 1000
        while current_size <= MAX_SIZE:
            sizes.append(current_size)
            current_size += size_increment
    results = {name: [] for name, _, _ in algos}
    error_reasons = {}
    total_algos = len(algos)
    completed_algos = 0
    print("Generating test data...")
    test_data = {size: generate_test_data(size, dt, tr) for size in sizes}
    print(f"[{completed_algos}/{total_algos}]", end="", flush=True)
    scaled_timeouts = {size: max_time for size in sizes}
    with ProcessPoolExecutor(max_workers=NUM_PROCESSES, initializer=worker_init) as executor:
        for name, func, algo_type in algos:
            print(f"\r[{completed_algos}/{total_algos}] {name}", end="", flush=True)
            tasks = [(size, executor.submit(run_benchmark_task, name, func, test_data[size], scaled_timeouts[size], size))
                     for size in sizes]
            for i, (size, future) in enumerate(tasks):
                print(f"\r[{completed_algos}/{total_algos}] {name} - size {size}", end="", flush=True)
                try:
                    res_size, res = future.result(timeout=scaled_timeouts[size]*2)
                except Exception as e:
                    error_reasons[name] = str(e)
                    for _, f in tasks[i+1:]:
                        f.cancel()
                    break
                if res["success"]:
                    results[name].append((size, res["time"]))
                else:
                    error_reasons[name] = res.get("error", "unknown error")
                    for _, f in tasks[i+1:]:
                        f.cancel()
                    break
            completed_algos += 1
            success_count = len(results[name])
            if success_count < len(sizes):
                error_msg = error_reasons.get(name, "unknown error")
                print(f"\r{name} ✗ {success_count}/{len(sizes)} sizes - {error_msg}" + " " * 10)
            else:
                print(f"\r{name} ✓" + " " * 30)
            if completed_algos < total_algos:
                print(f"[{completed_algos}/{total_algos}]", end="", flush=True)
    print(f"\nBenchmarked {completed_algos}/{total_algos} algorithms")
    generate_algorithm_summary(results, algos, sizes)
    return results, sizes

def plot_complexity(results, sizes):
    valid_results = {k: v for k, v in results.items() if v}
    if not valid_results:
        print("No valid algorithm results to plot")
        return False
    all_sizes = sizes
    all_times = [t for points in valid_results.values() for _, t in points]
    min_size = min(all_sizes)
    max_size = max(all_sizes)
    min_time = min(all_times) if all_times else 1e-6
    max_time = max(all_times) if all_times else 1.0
    plt.figure(figsize=(15, 10))
    n = np.logspace(np.log10(min_size*0.9), np.log10(max_size*1.1), 1000)
    mid_idx = len(sizes) // 2
    mid_size = sizes[mid_idx]
    mid_times = []
    for points in valid_results.values():
        for s, t in points:
            if s == mid_size:
                mid_times.append(t)
    scale = np.median(mid_times) if mid_times else min_time * 10
    O1 = np.ones_like(n) * scale * 0.5
    O_log = np.log2(n) * scale / np.log2(mid_size)
    O_n = n * scale / mid_size
    O_nlog = n * np.log2(n) * scale / (mid_size * np.log2(mid_size))
    O_n2 = (n**2) * scale / (mid_size**2)
    O_2n = np.minimum(2**(n/mid_size*10) * scale, max_time * 100)
    plt.fill_between(n, 0, O1, color='#CCFFCC', alpha=0.7)
    plt.fill_between(n, O1, O_log, color='#CCFFEE', alpha=0.7)
    plt.fill_between(n, O_log, O_n, color='#EEFFCC', alpha=0.7)
    plt.fill_between(n, O_n, O_nlog, color='#FFFFCC', alpha=0.7)
    plt.fill_between(n, O_nlog, O_n2, color='#FFEECC', alpha=0.7)
    plt.fill_between(n, O_n2, O_2n, color='#FFCCCC', alpha=0.7)
    complexity_handles = []
    complexity_labels = []
    line_styles = {'O(1)': O1, 'O(log n)': O_log, 'O(n)': O_n, 'O(n log n)': O_nlog, 'O(n²)': O_n2, 'O(2ⁿ)': O_2n}
    for label, curve in line_styles.items():
        line, = plt.plot(n, curve, 'k--', linewidth=1.5, alpha=0.7)
        complexity_handles.append(line)
        complexity_labels.append(label)
    markers = ['o', 's', '^', 'D', 'v', 'p', '*', 'X', 'P', 'h', '<', '>', '8', 'H']
    colors = plt.cm.tab20(np.linspace(0, 1, 20))
    avg_times = {}
    for name, points in valid_results.items():
        if points:
            avg_times[name] = sum(t for _, t in points) / len(points)
    sorted_algos = sorted(valid_results.items(), key=lambda x: avg_times.get(x[0], float('inf')))
    algo_handles = []
    algo_labels = []
    for i, (name, points) in enumerate(sorted_algos):
        marker = markers[i % len(markers)]
        color = colors[i % len(colors)]
        x_vals = [x for x, _ in points]
        y_vals = [y for _, y in points]
        line, = plt.plot(x_vals, y_vals, marker=marker, markersize=8, linewidth=2.5, color=color)
        algo_handles.append(line)
        algo_labels.append(name)
    plt.xlim(min_size * 0.8, max_size * 1.2)
    plt.ylim(min_time * 0.1, max(max_time * 10, O_n2[-1] * 2))
    plt.title('Sorting Algorithm Performance: Complexity Analysis', fontsize=16)
    plt.xlabel('Input Size (n)', fontsize=14)
    plt.ylabel('Time (seconds)', fontsize=14)
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
    region_labels = ['O(1)', 'O(log n)', 'O(n)', 'O(n log n)', 'O(n²)', 'O(2ⁿ)']
    region_colors = ['#CCFFCC', '#CCFFEE', '#EEFFCC', '#FFFFCC', '#FFEECC', '#FFCCCC']
    region_handles = [plt.Rectangle((0,0), 1, 1, color=color, alpha=0.7) for color in region_colors]
    region_labels = [f"{label} region" for label in region_labels]
    complexity_legend = plt.legend(
        region_handles + complexity_handles,
        region_labels + complexity_labels,
        loc='upper left', bbox_to_anchor=(1.01, 1), 
        fontsize=10, title="Complexity Classes",
        framealpha=0.9, ncol=1
    )
    plt.gca().add_artist(complexity_legend)
    ncols = 2 if len(algo_labels) > 15 else 1
    plt.legend(algo_handles, algo_labels, loc='upper left', bbox_to_anchor=(1.01, 0.6),
               title="Algorithms", fontsize=10, framealpha=0.9, ncol=ncols)
    plt.tight_layout()
    plt.savefig('complexity_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Plotted {len(valid_results)}/{len(results)} algorithms")
    return True

def generate_algorithm_summary(results, all_algos, sizes):
    rows = []
    for name, _, _ in all_algos:
        row = [name]
        size_results = [s for s, _ in results.get(name, [])]
        for size in sizes:
            row.append("✓" if size in size_results else "✗")
        rows.append(row)
    rows.sort(key=lambda x: x.count("✓"), reverse=True)
    print("\n=== ALGORITHM SUMMARY ===")
    header = f"{'Algorithm':<25} | " + " | ".join(f"{s}" for s in sizes)
    print(header)
    print("-" * 25 + "+" + "-" * (len(header) - 26))
    for row in rows:
        print(f"{row[0]:<25} | " + " | ".join(f"{mark}" for mark in row[1:]))
    print("\n=== STATS ===")
    success_by_size = {}
    for size in sizes:
        idx = sizes.index(size) + 1
        success_count = sum(1 for row in rows if row[idx] == "✓")
        success_by_size[size] = success_count
        print(f"Size {size}: {success_count}/{len(all_algos)} algorithms ({success_count/len(all_algos)*100:.1f}%)")
    perfect_count = sum(1 for row in rows if all(mark == "✓" for mark in row[1:]))
    print(f"\nAlgorithms completing all sizes: {perfect_count}/{len(all_algos)}")
    return success_by_size

def run_benchmark(module_name=None, sizes=None, output_file=None, size_increment=None):
    algorithms = discover_sorting_algorithms(module_name or "algorithms")
    if not algorithms:
        print("no algorithms found. exiting.")
        return
    all_passed, verified_algorithms = verify_sorting(algorithms)
    if not verified_algorithms:
        print("no algorithms passed verification. exiting.")
        return
    print("\n" + "=" * 50)
    print("complexity analysis")
    print("=" * 50)
    print(f"\nbenchmarking {len(verified_algorithms)} algorithms...\n")
    benchmark_results, sizes = benchmark_for_complexity(verified_algorithms, sizes, size_increment=size_increment)
    print("visualizing results")
    success = plot_complexity(benchmark_results, sizes)
    if output_file:
         plt.savefig(output_file, bbox_inches='tight', dpi=300)
         print(f"plot saved to {output_file}")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Benchmark sorting algorithms and visualize their complexity.')
    parser.add_argument('-m', '--module', help='Module containing sorting algorithms (default: algorithms)')
    parser.add_argument('-s', '--sizes', type=int, nargs='+', help='Input sizes to benchmark (default: exponential sequence)')
    parser.add_argument('-o', '--output', help='Output file name for the plot (default: complexity_analysis.png)')
    parser.add_argument('-t', '--timeout', type=int, help=f'Maximum time in seconds for each algorithm run (default: {TIMEOUT})')
    parser.add_argument('--max-size', type=int, help=f'Maximum input size to benchmark (default: {MAX_SIZE})')
    parser.add_argument('--size-increment', type=int, help=f'Increment between sizes (default: {SIZE_INCREMENT})')
    args = parser.parse_args()
    
    # Override global settings with command-line arguments if provided
    timeout = args.timeout or TIMEOUT
    max_size = args.max_size or MAX_SIZE
    size_increment = args.size_increment or SIZE_INCREMENT
    
    run_benchmark(args.module, args.sizes, args.output, size_increment) 