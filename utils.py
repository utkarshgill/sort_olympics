#!/usr/bin/env python3
import os
import random
import time
import signal
import inspect
import tracemalloc
from typing import List, Tuple, Dict, Any, Callable, Optional, Union

# shared configuration parameters with environment variable fallbacks
TIMEOUT = int(os.environ.get('TIMEOUT', 5))
MAX_SIZE = int(os.environ.get('MAX_SIZE', 20000))
SIZE_INCREMENT = int(os.environ.get('SIZE_INC', 5000))
TOURNAMENT_SIZE = int(os.environ.get('SIZE', 2000))
DATA_TYPE = os.environ.get('DATA_TYPE', 'random')
TEST_RANGE = int(os.environ.get('TEST_RANGE', 1000))
VIZ = os.environ.get('VIZ', '0') == '1'
BENCH = os.environ.get('BENCH', '0') == '1'

# algorithm categories
FUNDAMENTAL_ALGOS = ['bubble_sort', 'selection_sort', 'insertion_sort', 'merge_sort', 'quick_sort', 'heap_sort']
ADVANCED_ALGOS = ['shell_sort', 'tim_sort', 'intro_sort', 'library_sort', 'block_sort', 'smooth_sort']
SPECIALIZED_ALGOS = ['tree_sort', 'tournament_sort', 'patience_sort', 'cube_sort', 'comb_sort', 'cocktail_sort',
                   'gnome_sort', 'odd_even_sort', 'pancake_sort', 'strand_sort', 'exchange_sort', 'cycle_sort',
                   'recombinant_sort', 'inplace_merge_sort']
LINEAR_TIME_ALGOS = ['counting_sort', 'bucket_sort_uniform', 'bucket_sort_integer', 'radix_sort_lsd', 'radix_sort_msd', 'pigeonhole_sort']
NONCOMPARISON_ALGOS = ['spreadsort', 'burstsort', 'flashsort', 'postman_sort', 'msd_radix_sort_inplace']
THEORETICAL_ALGOS = ['bead_sort', 'merge_insertion_sort', 'i_cant_believe_it_can_sort', 'spaghetti_sort', 
                     'sorting_network', 'bitonic_sort', 'bogosort', 'stooge_sort', 'slowsort', 
                     'franceschini_mergesort', 'thorup_sort']

def generate_test_data(size: int, data_type: str, range_max: int = 1000) -> List[int]:
    """Generate test data for sorting algorithms
    
    Args:
        size: Number of elements in the data
        data_type: Type of data ('random', 'sorted', 'reversed', 'nearly_sorted')
        range_max: Maximum value for random elements
        
    Returns:
        List of integers to be sorted
    """
    if data_type == 'sorted':
        return list(range(size))
    if data_type == 'reversed':
        return list(range(size, 0, -1))
    if data_type == 'nearly_sorted':
        data = list(range(size))
        for _ in range(max(1, int(size * 0.05))):
            i, j = random.sample(range(size), 2)
            data[i], data[j] = data[j], data[i]
        return data
    # default: random
    return [random.randint(0, range_max) for _ in range(size)]

def discover_sorting_algorithms(module_name: str = "algorithms") -> List[Tuple[str, Callable, str]]:
    """
    Discover and categorize sorting algorithms from a module
    
    Args:
        module_name: Name of the module containing sorting algorithms
        
    Returns:
        List of tuples (display_name, function, category)
    """
    try:
        module = __import__(module_name)
    except ImportError:
        print(f"Error: could not import {module_name} module.")
        return []
        
    sorting_algos = []
    for name, func in inspect.getmembers(module, inspect.isfunction):
        display_name = ' '.join(word.capitalize() for word in name.split('_'))
        if name in FUNDAMENTAL_ALGOS:
            sorting_algos.append((display_name, func, "fundamental"))
        elif name in ADVANCED_ALGOS:
            sorting_algos.append((display_name, func, "advanced"))
        elif name in SPECIALIZED_ALGOS:
            sorting_algos.append((display_name, func, "specialized"))
        elif name in LINEAR_TIME_ALGOS:
            sorting_algos.append((display_name, func, "linear"))
        elif name in NONCOMPARISON_ALGOS:
            sorting_algos.append((display_name, func, "noncomparison"))
        elif name in THEORETICAL_ALGOS:
            sorting_algos.append((display_name, func, "theoretical"))
        else:
            sorting_algos.append((display_name, func, "unknown"))
    
    return sorting_algos

def print_algorithm_stats(algorithms: List[Tuple[str, Callable, str]]) -> None:
    """
    Print statistics about the discovered algorithms
    
    Args:
        algorithms: List of tuples (display_name, function, category)
    """
    print("\nALGORITHMS:")
    print(f"  Fundamental: {sum(1 for _, _, cat in algorithms if cat == 'fundamental')}")
    print(f"  Advanced: {sum(1 for _, _, cat in algorithms if cat == 'advanced')}")
    print(f"  Specialized: {sum(1 for _, _, cat in algorithms if cat == 'specialized')}")
    print(f"  Linear-time: {sum(1 for _, _, cat in algorithms if cat == 'linear')}")
    print(f"  Noncomparison: {sum(1 for _, _, cat in algorithms if cat == 'noncomparison')}")
    print(f"  Theoretical: {sum(1 for _, _, cat in algorithms if cat == 'theoretical')}")
    print(f"  Total: {len(algorithms)}")

def verify_sorting(algorithms: List[Tuple[str, Callable, str]]) -> Tuple[bool, List[Tuple[str, Callable, str]]]:
    """
    Verify that all algorithms sort correctly
    
    Args:
        algorithms: List of tuples (display_name, function, category)
        
    Returns:
        Tuple of (all_passed, valid_algorithms):
            all_passed: True if all algorithms passed verification, False otherwise
            valid_algorithms: List of algorithms that passed verification
    """
    print("\nVERIFYING ALGORITHMS...", end=" ")
    
    test_cases = [
        [5, 4, 3, 2, 1],
        [3, 1, 4, 1, 5, 9, 2, 6],
        [random.randint(0, 100) for _ in range(50)]
    ]
    
    all_passed = True
    total_algos = 0
    passed_algos = 0
    valid_algorithms = []
    
    for name, func, category in algorithms:
        if name == "BYE" or func is None:
            continue
            
        total_algos += 1
        algo_passed = True
        
        for i, test_case in enumerate(test_cases):
            test_description = ["reversed", "pattern with duplicates", "random"][i]
            expected = sorted(test_case)
            test_copy = test_case.copy()
            
            try:
                result = func(test_copy)
                # if result is None, the algorithm likely modified test_copy in-place
                if result is None:
                    result = test_copy
                    
                if result != expected:
                    print(f"\n❌ {name} failed on {test_description} input:")
                    print(f"   Input: {test_case[:10]}{'...' if len(test_case) > 10 else ''}")
                    print(f"   Expected: {expected[:10]}{'...' if len(expected) > 10 else ''}")
                    print(f"   Got: {result[:10]}{'...' if len(result) > 10 else ''}")
                    all_passed = False
                    algo_passed = False
            except Exception as e:
                print(f"\n❌ {name} threw an exception on {test_description} input: {str(e)}")
                all_passed = False
                algo_passed = False
        
        if algo_passed:
            passed_algos += 1
            valid_algorithms.append((name, func, category))
    
    print(f"✅ {passed_algos}/{total_algos}")
    return all_passed, valid_algorithms

def benchmark_single(func_name: str, func: Callable, data: List[int], max_time: int) -> Dict[str, Any]:
    """
    Benchmark a single algorithm on a given data set
    
    Args:
        func_name: Name of the algorithm
        func: Function implementing the algorithm
        data: Data to sort
        max_time: Maximum allowed time before timeout
        
    Returns:
        Dictionary with benchmark results
    """
    def timeout_handler(signum, frame):
        raise TimeoutError("timeout")
    
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(max_time)
    
    try:
        tracemalloc.start()
        data_copy = data.copy()  # avoid side effects
        start = time.time()
        result = func(data_copy)
        elapsed = time.time() - start
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        signal.alarm(0)
        
        expected = sorted(data)
        # if result is None, the algorithm likely modified data_copy in-place
        if result is None:
            result = data_copy
            
        is_correct = result == expected
        return {
            "name": func_name, 
            "time": elapsed, 
            "memory": peak/1024,
            "success": True,
            "error": None if is_correct else "incorrect sort result",
            "result": result
        }
    except TimeoutError:
        if tracemalloc.is_tracing():
            tracemalloc.stop()
        signal.alarm(0)
        return {"name": func_name, "time": float('inf'), "memory": 0, "success": False, "error": "Timeout", "result": None}
    except RecursionError:
        if tracemalloc.is_tracing():
            tracemalloc.stop()
        signal.alarm(0)
        return {"name": func_name, "time": float('inf'), "memory": 0, "success": False, "error": "Recursion depth exceeded", "result": None}
    except Exception as e:
        if tracemalloc.is_tracing():
            tracemalloc.stop()
        signal.alarm(0)
        return {"name": func_name, "time": float('inf'), "memory": 0, "success": False, "error": f"{type(e).__name__}: {str(e)}", "result": None} 