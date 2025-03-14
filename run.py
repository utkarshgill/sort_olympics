#!/usr/bin/env python3
# sort_olympics.py - Tournament Edition

import time
import random
import signal
import inspect
import tracemalloc
import os
import math
from prettytable import PrettyTable
import algorithms

# configuration from environment variables
MAX_BENCHMARK_TIME = int(os.environ.get('TIMEOUT', 5))
TOURNAMENT_SIZE = int(os.environ.get('SIZE', 2000))  # size for tournament rounds
TEST_DATA_TYPE = os.environ.get('DATA_TYPE', 'random')
TEST_RANGE = int(os.environ.get('TEST_RANGE', 1000))  # range of values for random data

# algorithm categories
FUNDAMENTAL_ALGOS = ['bubble_sort', 'selection_sort', 'insertion_sort', 
                     'merge_sort', 'quick_sort', 'heap_sort']

ADVANCED_ALGOS = ['shell_sort', 'tim_sort', 'intro_sort', 
                  'library_sort', 'block_sort', 'smooth_sort']

SPECIALIZED_ALGOS = ['tree_sort', 'tournament_sort', 'patience_sort',
                   'cube_sort', 'comb_sort', 'cocktail_sort',
                   'gnome_sort', 'odd_even_sort', 'pancake_sort',
                   'strand_sort', 'exchange_sort', 'cycle_sort',
                   'recombinant_sort', 'inplace_merge_sort']

LINEAR_TIME_ALGOS = ['counting_sort', 'bucket_sort_uniform', 'bucket_sort_integer',
                    'radix_sort_lsd', 'radix_sort_msd', 'pigeonhole_sort']

NONCOMPARISON_ALGOS = ['spreadsort', 'burstsort', 'flashsort',
                   'postman_sort', 'msd_radix_sort_inplace']

THEORETICAL_ALGOS = ['bead_sort', 'merge_insertion_sort', 'i_cant_believe_it_can_sort',
                     'spaghetti_sort', 'sorting_network', 'bitonic_sort',
                     'bogosort', 'stooge_sort', 'slowsort', 
                     'franceschini_mergesort', 'thorup_sort']

def timeout_handler(signum, frame):
    # simple signal handler for timeouts
    raise TimeoutError()

def discover_sorting_algorithms():
    # find all sorting algorithms in the module with categories
    sorting_algos = []
    
    # collect algorithms by category without printing them
    for name, func in inspect.getmembers(algorithms, inspect.isfunction):
        if name in FUNDAMENTAL_ALGOS:
            display_name = ' '.join(word.capitalize() for word in name.split('_'))
            sorting_algos.append((display_name, func, "fundamental"))
        elif name in ADVANCED_ALGOS:
            display_name = ' '.join(word.capitalize() for word in name.split('_'))
            sorting_algos.append((display_name, func, "advanced"))
        elif name in SPECIALIZED_ALGOS:
            display_name = ' '.join(word.capitalize() for word in name.split('_'))
            sorting_algos.append((display_name, func, "specialized"))
        elif name in LINEAR_TIME_ALGOS:
            display_name = ' '.join(word.capitalize() for word in name.split('_'))
            sorting_algos.append((display_name, func, "linear"))
        elif name in NONCOMPARISON_ALGOS:
            display_name = ' '.join(word.capitalize() for word in name.split('_'))
            sorting_algos.append((display_name, func, "noncomparison"))
        elif name in THEORETICAL_ALGOS:
            display_name = ' '.join(word.capitalize() for word in name.split('_'))
            sorting_algos.append((display_name, func, "theoretical"))
    
    # Print summary of algorithm counts by category
    print("\nALGORITHMS:")
    print(f"  Fundamental: {sum(1 for _, _, cat in sorting_algos if cat == 'fundamental')}")
    print(f"  Advanced: {sum(1 for _, _, cat in sorting_algos if cat == 'advanced')}")
    print(f"  Specialized: {sum(1 for _, _, cat in sorting_algos if cat == 'specialized')}")
    print(f"  Linear-time: {sum(1 for _, _, cat in sorting_algos if cat == 'linear')}")
    print(f"  Noncomparison: {sum(1 for _, _, cat in sorting_algos if cat == 'noncomparison')}")
    print(f"  Theoretical: {sum(1 for _, _, cat in sorting_algos if cat == 'theoretical')}")
    print(f"  Total: {len(sorting_algos)}")
            
    return sorting_algos

def generate_test_data(size, data_type):
    # generate different types of test data
    if data_type == 'random':
        return [random.randint(0, TEST_RANGE) for _ in range(size)]
    elif data_type == 'sorted':
        return list(range(size))
    elif data_type == 'reversed':
        return list(range(size, 0, -1))
    elif data_type == 'nearly_sorted':
        data = list(range(size))
        swaps = max(1, int(size * 0.05))
        for _ in range(swaps):
            i, j = random.sample(range(size), 2)
            data[i], data[j] = data[j], data[i]
        return data
    else:
        print(f"Unknown data type: {data_type}, using random")
        return [random.randint(0, TEST_RANGE) for _ in range(size)]

def benchmark_algorithm(sort_func, data, name):
    # benchmark a single sorting algorithm
    data_copy = data.copy()
    
    # set up timeout 
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(MAX_BENCHMARK_TIME)
    
    try:
        # track memory and time
        tracemalloc.start()
        start_time = time.time()
        result = sort_func(data_copy)
        elapsed_time = time.time() - start_time
        _, peak_memory = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        signal.alarm(0)
        
        return {
            "name": name,
            "time": elapsed_time,
            "memory": peak_memory / 1024,  # convert to KB
            "result": result,
            "timeout": False
        }
    except TimeoutError:
        tracemalloc.stop()
        signal.alarm(0)
        return {
            "name": name,
            "time": float('inf'),
            "memory": 0,
            "result": None,
            "timeout": True
        }

def run_tournament(algorithms):
    """Run a tournament-style competition between sorting algorithms"""
    print("\n" + "=" * 50)
    print(f"TOURNAMENT BEGINS! ({len(algorithms)} competitors)")
    print("=" * 50)
    
    # shuffle algorithms for random initial seeding
    random.shuffle(algorithms)
    
    # initialize results dictionary
    tournament_results = {}
    
    # create a bracket structure
    bracket_size = 2 ** math.ceil(math.log2(len(algorithms)))
    
    # add byes if needed (to fill bracket)
    contenders = algorithms.copy()
    while len(contenders) < bracket_size:
        contenders.append(("BYE", None, "bye"))
    
    # track all rounds
    all_rounds = []
    current_round = contenders
    
    # Keep track of the final match competitors
    final_match = None
    
    round_number = 1
    while len(current_round) > 1:
        print(f"\nROUND {round_number} ({len(current_round)} competitors)")
        next_round = []
        
        # pair algorithms for this round
        for i in range(0, len(current_round), 2):
            # get pair of algorithms
            algo1 = current_round[i]
            algo2 = current_round[i+1] if i+1 < len(current_round) else ("BYE", None, "bye")
            
            # handle byes
            if algo1[0] == "BYE" and algo2[0] == "BYE":
                next_round.append(algo1)  # shouldn't happen, but just in case
                continue
            elif algo1[0] == "BYE":
                next_round.append(algo2)
                print(f"  {algo2[0]} advances (opponent: BYE)")
                continue
            elif algo2[0] == "BYE":
                next_round.append(algo1)
                print(f"  {algo1[0]} advances (opponent: BYE)")
                continue
            
            # run the match
            winner, match_results = run_match(algo1, algo2)
            
            # store results
            tournament_results[(algo1[0], algo2[0])] = match_results
            
            # If this is the final round, save the competitors
            if len(current_round) == 2:
                final_match = (algo1, algo2)
            
            # add winner to next round
            next_round.append(winner)
            
            # print match result with speed difference
            if match_results["winner_time"] == float('inf') or match_results["loser_time"] == float('inf'):
                print(f"  {match_results['winner']} defeats {match_results['loser']} (timeout)")
            else:
                time_diff = match_results["loser_time"] / match_results["winner_time"]
                print(f"  {match_results['winner']} defeats {match_results['loser']} ({time_diff:.2f}x faster)")
        
        # store the round
        all_rounds.append(current_round)
        
        # move to next round
        current_round = next_round
        round_number += 1
    
    # determine 3rd place from semifinalists
    if len(all_rounds) >= 2 and len(all_rounds[-2]) >= 3:
        semifinalists = all_rounds[-2]
        # get the two semifinalists who didn't make the final
        losers = [algo for algo in semifinalists if algo not in current_round and algo[0] != "BYE"]
        
        if len(losers) == 2:
            print("\nTHIRD PLACE MATCH")
            third_place, third_match = run_match(losers[0], losers[1])
            tournament_results[(losers[0][0], losers[1][0])] = third_match
            
            # print third place match result with speed difference
            if third_match["winner_time"] == float('inf') or third_match["loser_time"] == float('inf'):
                print(f"  {third_match['winner']} defeats {third_match['loser']} (timeout)")
            else:
                time_diff = third_match["loser_time"] / third_match["winner_time"]
                print(f"  {third_match['winner']} defeats {third_match['loser']} ({time_diff:.2f}x faster)")
        
    # Champion is the last algorithm standing
    champion = current_round[0]
    
    return champion, tournament_results, all_rounds, final_match

def run_match(algo1, algo2):
    """Run a match between two algorithms on the same data"""
    # generate test data
    data = generate_test_data(TOURNAMENT_SIZE, TEST_DATA_TYPE)
    
    # run both algorithms
    result1 = benchmark_algorithm(algo1[1], data, algo1[0])
    result2 = benchmark_algorithm(algo2[1], data, algo2[0])
    
    # determine winner based on time
    if result1["timeout"] and result2["timeout"]:
        # if both timeout, winner is determined by algorithm complexity
        # we'll just pick algo2 for simplicity
        winner = algo2
        loser = algo1
        winner_time = float('inf')
        loser_time = float('inf')
    elif result1["timeout"]:
        winner = algo2
        loser = algo1
        winner_time = result2["time"]
        loser_time = float('inf')
    elif result2["timeout"]:
        winner = algo1
        loser = algo2
        winner_time = result1["time"]
        loser_time = float('inf')
    elif result1["time"] <= result2["time"]:
        winner = algo1
        loser = algo2
        winner_time = result1["time"]
        loser_time = result2["time"]
    else:
        winner = algo2
        loser = algo1
        winner_time = result2["time"]
        loser_time = result1["time"]
    
    # prepare results
    match_results = {
        "winner": winner[0],
        "loser": loser[0],
        "winner_time": winner_time,
        "loser_time": loser_time,
        "winner_memory": result1["memory"] if winner == algo1 else result2["memory"],
        "loser_memory": result1["memory"] if loser == algo1 else result2["memory"],
    }
    
    return winner, match_results

def print_tournament_results(champion, tournament_results, all_rounds, final_match):
    """Print the final tournament results"""
    print("\n" + "=" * 50)
    print("TOURNAMENT RESULTS")
    print("=" * 50)
    
    # Print the champion with emoji
    print(f"\nCHAMPION: 🏆 {champion[0]} ({champion[2].capitalize()})")
    
    # Determine 2nd place from the final match
    runner_up = None
    if final_match:
        finalist = final_match[0] if final_match[1][0] == champion[0] else final_match[1]
        if finalist and finalist[0] != "BYE" and finalist[0] != champion[0]:
            print(f"RUNNER-UP: 🥈 {finalist[0]} ({finalist[2].capitalize()})")
            runner_up = finalist
    
    # Determine 3rd place
    third_place = None
    if len(all_rounds) >= 2 and len(all_rounds[-2]) >= 3:
        semifinalists = all_rounds[-2]
        # get the two semifinalists who didn't make the final
        losers = [algo for algo in semifinalists if algo not in all_rounds[-1] and algo[0] != "BYE"]
        
        if len(losers) == 2:
            match_key = (losers[0][0], losers[1][0])
            reversed_key = (losers[1][0], losers[0][0])
            
            if match_key in tournament_results:
                third_place_name = tournament_results[match_key]["winner"]
                third_place_algo = [algo for algo in losers if algo[0] == third_place_name][0]
                print(f"THIRD PLACE: {third_place_name} ({third_place_algo[2].capitalize()})")
                third_place = third_place_algo
            elif reversed_key in tournament_results:
                third_place_name = tournament_results[reversed_key]["winner"]
                third_place_algo = [algo for algo in losers if algo[0] == third_place_name][0]
                print(f"THIRD PLACE: {third_place_name} ({third_place_algo[2].capitalize()})")
                third_place = third_place_algo

    # Get all algorithms from all rounds and their performance
    algo_performance = {}
    
    # Process all match results to get performance data for ranking
    for (algo1_name, algo2_name), match in tournament_results.items():
        # Process winner
        if match["winner"] not in algo_performance:
            algo_performance[match["winner"]] = {
                "wins": 0, 
                "losses": 0,
                "total_time": 0,
                "matches": 0
            }
        
        algo_performance[match["winner"]]["wins"] += 1
        if match["winner_time"] != float('inf'):
            algo_performance[match["winner"]]["total_time"] += match["winner_time"]
            algo_performance[match["winner"]]["matches"] += 1
            
        # Process loser
        if match["loser"] not in algo_performance:
            algo_performance[match["loser"]] = {
                "wins": 0, 
                "losses": 0,
                "total_time": 0,
                "matches": 0
            }
            
        algo_performance[match["loser"]]["losses"] += 1
        if match["loser_time"] != float('inf'):
            algo_performance[match["loser"]]["total_time"] += match["loser_time"]
            algo_performance[match["loser"]]["matches"] += 1
    
    # Calculate average performance
    for algo, perf in algo_performance.items():
        if perf["matches"] > 0:
            perf["avg_time"] = perf["total_time"] / perf["matches"]
        else:
            perf["avg_time"] = float('inf')
    
    # Get all algorithms from all rounds for category classification
    all_algorithms = {}
    for round_algos in all_rounds:
        for algo in round_algos:
            if algo[0] != "BYE":
                all_algorithms[algo[0]] = algo
    
    # Group algorithms by category
    categories = {"fundamental": [], "advanced": [], "specialized": [], "linear": [], "noncomparison": [], "theoretical": []}
    
    # Categorize all algorithms and add performance data
    for algo_name, algo in all_algorithms.items():
        if algo[2] in categories:
            performance = algo_performance.get(algo_name, {
                "wins": 0, 
                "losses": 0,
                "avg_time": float('inf'),
                "matches": 0
            })
            
            categories[algo[2]].append((algo, performance))
    
    # Print results for each category
    print("\n" + "=" * 50)
    print("RESULTS BY CATEGORY:")
    print("=" * 50)
    
    for category_name, algos in categories.items():
        if not algos:
            continue
            
        print(f"\n{category_name.upper()} ALGORITHMS:")
        
        # Sort algorithms by performance (wins first, then average time)
        sorted_algos = sorted(
            algos, 
            key=lambda x: (-x[1]["wins"], x[1]["avg_time"])
        )
        
        # Display top 3 (or all if fewer than 3)
        for i, (algo, perf) in enumerate(sorted_algos[:3], 1):
            position = "1st" if i == 1 else "2nd" if i == 2 else "3rd"
            
            # Display simplified performance stats
            if perf["matches"] > 0:
                print(f"  {position}: {algo[0]} - {perf['wins']} wins, {perf['avg_time']:.6f}s avg time")
            else:
                print(f"  {position}: {algo[0]} - {perf['wins']} wins, no valid time data")
    
    # Print notable matches
    print("\n" + "=" * 50)
    print("NOTABLE MATCHES:")
    print("=" * 50)
    
    # find the most lopsided victories
    sorted_matches = sorted(
        [(k, v) for k, v in tournament_results.items() if v['winner_time'] != float('inf') and v['loser_time'] != float('inf')],
        key=lambda x: x[1]['loser_time'] / x[1]['winner_time'] if x[1]['winner_time'] > 0 else float('inf'),
        reverse=True
    )
    
    for (algo1, algo2), match in sorted_matches[:3]:
        time_diff = match["loser_time"] / match["winner_time"] if match["winner_time"] > 0 else float('inf')
        print(f"  {match['winner']} vs {match['loser']} - {time_diff:.2f}x speed difference")

    # Add algorithm summary table at the end
    print("\n" + "=" * 50)
    print("SUMMARY:")
    print("=" * 50 + "\n")
    
    # Build a table of all algorithms with their stats
    table = PrettyTable()
    table.field_names = ["Algorithm", "Category", "Wins", "Losses", "Avg Time (s)"]
    
    # Sort algorithms by performance (wins first, then avg time)
    sorted_algorithms = []
    for algo_name, algo in all_algorithms.items():
        perf = algo_performance.get(algo_name, {
            "wins": 0, 
            "losses": 0,
            "avg_time": float('inf'),
            "matches": 0
        })
        sorted_algorithms.append((algo, perf))
    
    # Sort by wins (descending) and then by average time (ascending)
    sorted_algorithms.sort(key=lambda x: (-x[1]["wins"], x[1]["avg_time"]))
    
    # Add rows to the table
    for algo, perf in sorted_algorithms:
        avg_time = perf["total_time"] / perf["matches"] if perf["matches"] > 0 else float('inf')
        perf["avg_time"] = avg_time  # save for sorting
        
        if avg_time == float('inf'):
            time_str = "timeout"
        else:
            time_str = f"{avg_time:.6f}"
            
        # Highlight the champion with a trophy emoji
        if champion and algo[0] == champion[0]:
            algo_name = f"🏆 {algo[0]}"
        # Highlight the runner-up with a medal emoji
        elif runner_up and algo[0] == runner_up[0]:
            algo_name = f"🥈 {algo[0]}"
        else:
            algo_name = algo[0]
            
        table.add_row([
            algo_name,
            algo[2].capitalize(),
            perf["wins"],
            perf["losses"],
            time_str
        ])
    
    print(table)

def verify_sorting(algorithms):
    """
    Verify that all algorithms correctly sort the test cases.
    
    This function checks each algorithm against several test cases:
    1. A simple reversed array [5,4,3,2,1]
    2. A pattern with duplicates [3,1,4,1,5,9,2,6]
    3. A random array with 50 elements
    
    Returns True if all algorithms pass all test cases, False otherwise.
    """
    # verify algorithms produce correct results
    # can be skipped with SKIP_VERIFY=true
    if os.environ.get('SKIP_VERIFY', '').lower() == 'true':
        print("✅ Verification skipped (SKIP_VERIFY=true)")
        return True
    
    print("\nVERIFYING ALGORITHMS...", end=" ")
    
    test_cases = [
        [5, 4, 3, 2, 1],  # reversed
        [3, 1, 4, 1, 5, 9, 2, 6],  # random pattern with duplicates
        [random.randint(0, 100) for _ in range(50)]  # random data
    ]
    
    all_passed = True
    total_algos = 0
    passed_algos = 0
    
    for name, func, _ in algorithms:
        # skip BYE entries
        if name == "BYE" or func is None:
            continue
            
        total_algos += 1
        algo_passed = True
        
        for i, test_case in enumerate(test_cases):
            test_description = ["reversed", "pattern with duplicates", "random"][i]
            expected = sorted(test_case)
            
            # Make a copy to ensure the original test case isn't modified
            test_copy = test_case.copy()
            
            try:
                result = func(test_copy)
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
    
    # Simplified verification completion message on the same line
    print(f"✅ {passed_algos}/{total_algos}")
    
    return all_passed

def print_config():
    # print the current configuration
    print("\nCONFIGURATION:")
    print(f"  Timeout: {MAX_BENCHMARK_TIME}s")
    print(f"  Tournament size: {TOURNAMENT_SIZE} elements")
    print(f"  Data type: {TEST_DATA_TYPE}")
    if TEST_DATA_TYPE == 'random':
        print(f"  Value range: 0 to {TEST_RANGE}")
    if os.environ.get('SKIP_VERIFY', '').lower() == 'true':
        print("  Verification: Skipped")

def main():
    print("\nSORT OLYMPICS\n")
    
    # print current configuration
    print_config()
    
    # discover sorting algorithms with categories
    sorting_algorithms = discover_sorting_algorithms()
    
    # verify correctness
    if not verify_sorting(sorting_algorithms):
        print("WARNING: Some algorithms failed verification. Results may be incorrect.")
    
    # run tournament
    champion, tournament_results, all_rounds, final_match = run_tournament(sorting_algorithms)
    
    # print tournament results
    print_tournament_results(champion, tournament_results, all_rounds, final_match)
    
    print("\nTournament complete!")

if __name__ == "__main__":
    main()