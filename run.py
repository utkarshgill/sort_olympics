#!/usr/bin/env python3
import time, random, signal, inspect, tracemalloc, os, math
from prettytable import PrettyTable
import algorithms
import benchmark  # import the benchmark module

TIMEOUT = int(os.environ.get('TIMEOUT', 5))
TOURNAMENT_SIZE = int(os.environ.get('SIZE', 2000))
TEST_DATA_TYPE = os.environ.get('DATA_TYPE', 'random')
TEST_RANGE = int(os.environ.get('TEST_RANGE', 1000))
VIZ = os.environ.get('VIZ', '0') == '1'
BENCH = os.environ.get('BENCH', '0') == '1'

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

def discover_sorting_algorithms():
    sorting_algos = []
    for name, func in inspect.getmembers(algorithms, inspect.isfunction):
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
    # reuse the function from benchmark.py
    return benchmark.generate_test_data(size, data_type, TEST_RANGE)

def benchmark_algorithm(sort_func, data, name):
    # reuse the function from benchmark.py
    # set the global timeout first
    benchmark.TIMEOUT = TIMEOUT
    return benchmark.benchmark_single(name, sort_func, data, TIMEOUT)

def run_tournament(algorithms):
    print("\n" + "=" * 50)
    print(f"TOURNAMENT BEGINS! ({len(algorithms)} competitors)")
    print("=" * 50)
    
    random.shuffle(algorithms)
    tournament_results = {}
    bracket_size = 2 ** math.ceil(math.log2(len(algorithms)))
    
    contenders = algorithms.copy()
    while len(contenders) < bracket_size:
        contenders.append(("BYE", None, "bye"))
    
    all_rounds = []
    current_round = contenders
    final_match = None
    
    round_number = 1
    while len(current_round) > 1:
        print(f"\nROUND {round_number} ({len(current_round)} competitors)")
        next_round = []
        
        for i in range(0, len(current_round), 2):
            algo1 = current_round[i]
            algo2 = current_round[i+1] if i+1 < len(current_round) else ("BYE", None, "bye")
            
            if algo1[0] == "BYE" and algo2[0] == "BYE":
                next_round.append(algo1)
                continue
            elif algo1[0] == "BYE":
                next_round.append(algo2)
                print(f"  {algo2[0]} advances (opponent: BYE)")
                continue
            elif algo2[0] == "BYE":
                next_round.append(algo1)
                print(f"  {algo1[0]} advances (opponent: BYE)")
                continue
            
            winner, match_results = run_match(algo1, algo2)
            tournament_results[(algo1[0], algo2[0])] = match_results
            
            if len(current_round) == 2:
                final_match = (algo1, algo2)
            
            next_round.append(winner)
            
            if match_results["winner_time"] == float('inf') or match_results["loser_time"] == float('inf'):
                if match_results["winner_time"] == float('inf') and match_results["loser_time"] == float('inf'):
                    print(f"  {match_results['winner']} defeats {match_results['loser']} (both failed, winner chosen arbitrarily)")
                else:
                    print(f"  {match_results['winner']} defeats {match_results['loser']} (opponent error/timeout)")
            else:
                time_diff = match_results["loser_time"] / match_results["winner_time"]
                print(f"  {match_results['winner']} defeats {match_results['loser']} ({time_diff:.2f}x faster)")
        
        all_rounds.append(current_round)
        current_round = next_round
        round_number += 1
    
    if len(all_rounds) >= 2 and len(all_rounds[-2]) >= 3:
        semifinalists = all_rounds[-2]
        losers = [algo for algo in semifinalists if algo not in current_round and algo[0] != "BYE"]
        
        if len(losers) == 2:
            print("\nTHIRD PLACE MATCH")
            third_place, third_match = run_match(losers[0], losers[1])
            tournament_results[(losers[0][0], losers[1][0])] = third_match
            
            if third_match["winner_time"] == float('inf') or third_match["loser_time"] == float('inf'):
                print(f"  {third_match['winner']} defeats {third_match['loser']} (timeout)")
            else:
                time_diff = third_match["loser_time"] / third_match["winner_time"]
                print(f"  {third_match['winner']} defeats {third_match['loser']} ({time_diff:.2f}x faster)")
        
    champion = current_round[0]
    return champion, tournament_results, all_rounds, final_match

def run_match(algo1, algo2):
    data = generate_test_data(TOURNAMENT_SIZE, TEST_DATA_TYPE)
    
    # safely benchmark algorithm 1
    try:
        result1 = benchmark_algorithm(algo1[1], data, algo1[0])
    except (RecursionError, Exception) as e:
        print(f"  Warning: {algo1[0]} failed with error: {type(e).__name__}")
        result1 = {"success": False, "time": float('inf'), "memory": 0, "result": None, "error": str(e)}
    
    # safely benchmark algorithm 2
    try:
        result2 = benchmark_algorithm(algo2[1], data, algo2[0])
    except (RecursionError, Exception) as e:
        print(f"  Warning: {algo2[0]} failed with error: {type(e).__name__}")
        result2 = {"success": False, "time": float('inf'), "memory": 0, "result": None, "error": str(e)}
    
    if not result1["success"] and not result2["success"]:
        winner, loser = algo2, algo1
        winner_time = loser_time = float('inf')
    elif not result1["success"]:
        winner, loser = algo2, algo1
        winner_time, loser_time = result2["time"], float('inf')
    elif not result2["success"]:
        winner, loser = algo1, algo2
        winner_time, loser_time = result1["time"], float('inf')
    elif result1["time"] <= result2["time"]:
        winner, loser = algo1, algo2
        winner_time, loser_time = result1["time"], result2["time"]
    else:
        winner, loser = algo2, algo1
        winner_time, loser_time = result2["time"], result1["time"]
    
    return winner, {
        "winner": winner[0], "loser": loser[0],
        "winner_time": winner_time, "loser_time": loser_time,
        "winner_memory": result1["memory"] if winner == algo1 else result2["memory"],
        "loser_memory": result1["memory"] if loser == algo1 else result2["memory"],
    }

def print_tournament_bracket(all_rounds, tournament_results, champion, runner_up=None):
    """Print a tournament bracket visualization in a compact tree-like structure"""
    print("\n" + "=" * 50)
    print("TOURNAMENT RESULT")
    print("=" * 50 + "\n")

    # Show champion and runner-up information at the top
    print(f"CHAMPION: 🏆 {champion[0]} ({champion[2].capitalize()})")
    
    if runner_up and runner_up[0] != "BYE" and runner_up[0] != champion[0]:
        print(f"RUNNER-UP: 🥈 {runner_up[0]} ({runner_up[2].capitalize()})")
    
    # Show third place if available
    if len(all_rounds) >= 2 and len(all_rounds[-2]) >= 3:
        semifinalists = all_rounds[-2]
        losers = [algo for algo in semifinalists if algo not in all_rounds[-1] and algo[0] != "BYE"]
        
        if len(losers) == 2:
            match_key = (losers[0][0], losers[1][0])
            reversed_key = (losers[1][0], losers[0][0])
            
            third_place = None
            if match_key in tournament_results:
                third_place_name = tournament_results[match_key]["winner"]
                third_place_algo = [algo for algo in losers if algo[0] == third_place_name][0]
                print(f"THIRD PLACE: {third_place_name} ({third_place_algo[2].capitalize()})")
            elif reversed_key in tournament_results:
                third_place_name = tournament_results[reversed_key]["winner"]
                third_place_algo = [algo for algo in losers if algo[0] == third_place_name][0]
                print(f"THIRD PLACE: {third_place_name} ({third_place_algo[2].capitalize()})")
    
    print()  # Add a blank line before the bracket

    # Simple ASCII art tree bracket
    # Generate nested format with indentation to show the tournament hierarchy

    def format_competitor(competitor, is_winner=False, is_champion=False, is_runner_up=False, match_time=None):
        """Format a competitor's display with appropriate icons and time info"""
        suffix = ""
        if is_champion:
            suffix = " 🏆"
        elif is_runner_up:
            suffix = " 🥈"
            
        time_info = ""
        if match_time is not None and match_time != float('inf'):
            time_info = f" [{match_time:.6f}s]"
        elif match_time == float('inf'):
            time_info = " [timeout/error]"
            
        return f"{competitor[0]}{suffix}{time_info}"

    # Store the full bracket visualization
    lines = []
    
    # Function to print match with connections
    def add_match(algo1, algo2, depth, round_name=None, is_final=False):
        indent = "    " * depth
        
        # Find match information
        match_key = (algo1[0], algo2[0])
        reversed_key = (algo2[0], algo1[0])
        
        match_info = None
        if match_key in tournament_results:
            match_info = tournament_results[match_key]
        elif reversed_key in tournament_results:
            match_info = tournament_results[reversed_key]
            
        # Determine winner and loser
        winner = None
        loser = None
        if match_info:
            winner_name = match_info["winner"]
            if algo1[0] == winner_name:
                winner, loser = algo1, algo2
                winner_time, loser_time = match_info["winner_time"], match_info["loser_time"]
            else:
                winner, loser = algo2, algo1
                winner_time, loser_time = match_info["winner_time"], match_info["loser_time"]
        elif algo1[0] == "BYE":
            winner, loser = algo2, algo1
            winner_time, loser_time = None, None
        elif algo2[0] == "BYE":
            winner, loser = algo1, algo2
            winner_time, loser_time = None, None
        else:
            # Fallback - shouldn't happen in practice
            winner, loser = algo1, algo2
            winner_time, loser_time = None, None
            
        # Format with appropriate icons
        is_champion = is_final and winner[0] == champion[0]
        is_runner_up = is_final and runner_up and loser[0] == runner_up[0]
        
        # Print round name if provided
        if round_name:
            lines.append(f"{indent}{round_name}")
            
        # Print the competitors
        if is_final:
            # No tree characters for final match
            lines.append(f"{format_competitor(winner, True, is_champion, False, winner_time)}")
            lines.append(f"{format_competitor(loser, False, False, is_runner_up, loser_time)}")
        else:
            # Use tree characters for all other matches
            lines.append(f"{indent}├─ {format_competitor(winner, True, is_champion, False, winner_time)}")
            lines.append(f"{indent}└─ {format_competitor(loser, False, False, is_runner_up, loser_time)}")
        
        return winner, loser

    # Build the bracket tree recursively
    def build_bracket_tree(competitor, depth, current_round=len(all_rounds)-1):
        if current_round <= 0:
            return
            
        # Find the previous match that this competitor came from
        prev_round = all_rounds[current_round-1]
        for i in range(0, len(prev_round), 2):
            if i+1 >= len(prev_round):
                continue
                
            algo1 = prev_round[i]
            algo2 = prev_round[i+1]
            
            # Skip BYE vs BYE matches
            if algo1[0] == "BYE" and algo2[0] == "BYE":
                continue
                
            # Check if this match produced our competitor
            match_key = (algo1[0], algo2[0])
            reversed_key = (algo2[0], algo1[0])
            
            winner_name = None
            if match_key in tournament_results:
                winner_name = tournament_results[match_key]["winner"]
            elif reversed_key in tournament_results:
                winner_name = tournament_results[reversed_key]["winner"]
            elif algo1[0] == "BYE":
                winner_name = algo2[0]
            elif algo2[0] == "BYE":
                winner_name = algo1[0]
                
            # If this match produced our competitor, print it and continue recursively
            if winner_name == competitor[0]:
                round_name = f"{current_round}" if current_round < len(all_rounds) else None
                add_match(algo1, algo2, depth, round_name)
                
                # Recursively build for both competitors
                build_bracket_tree(algo1, depth+1, current_round-1)
                build_bracket_tree(algo2, depth+1, current_round-1)
                return

    # Find the final match
    final_round = all_rounds[-1]
    
    if len(final_round) < 2:
        lines.append(f"{champion[0]} 🏆 - Champion by default")
    else:
        # Print the final match without a round number
        algo1 = final_round[0]
        algo2 = final_round[1]
        add_match(algo1, algo2, 0, None, True)
        
        # Recursively build the bracket for each finalist
        build_bracket_tree(algo1, 1, len(all_rounds)-1)
        build_bracket_tree(algo2, 1, len(all_rounds)-1)
    
    # Add vertical lines connecting elements at the same level
    final_lines = []
    for i, line in enumerate(lines):
        # Don't try to add vertical lines to the champion and runner-up
        if i < 2:
            final_lines.append(line)
            continue
            
        # Look for patterns that need connecting vertical lines
        modified_line = line
        
        # Add vertical lines to connect items at the same level
        for depth in range(10):  # Max reasonable depth
            indent = "    " * depth
            if line.startswith(indent) and not line.startswith(indent + "│"):
                # Check if there are more items at this level following this one
                # by looking at the lines below
                found_sibling = False
                for j in range(i+1, len(lines)):
                    if lines[j].startswith(indent) and not lines[j].startswith(indent + "    "):
                        found_sibling = True
                        break
                    if lines[j].startswith(indent + "    "):
                        # This is a child, not a sibling
                        continue
                    if not lines[j].startswith(indent):
                        # We've moved past this indentation level
                        break
                        
                # If we have siblings below, add a vertical line
                if found_sibling:
                    # Replace the appropriate space with a vertical line
                    chars = list(modified_line)
                    line_position = len(indent) - 1
                    if line_position >= 0:
                        chars[line_position] = "│"
                        modified_line = "".join(chars)
        
        final_lines.append(modified_line)
    
    # Print the final tree
    for line in final_lines:
        print(line)

def print_tournament_results(champion, tournament_results, all_rounds, final_match):
    # Get the runner-up from the final match
    runner_up = None
    if final_match:
        finalist = final_match[0] if final_match[1][0] == champion[0] else final_match[1]
        if finalist and finalist[0] != "BYE" and finalist[0] != champion[0]:
            runner_up = finalist
                
    # Print tournament bracket visualization with results
    print_tournament_bracket(all_rounds, tournament_results, champion, runner_up)

    # Calculate algorithm performance statistics
    algo_performance = {}
    for (algo1_name, algo2_name), match in tournament_results.items():
        if match["winner"] not in algo_performance:
            algo_performance[match["winner"]] = {"wins": 0, "losses": 0, "total_time": 0, "matches": 0}
        
        algo_performance[match["winner"]]["wins"] += 1
        if match["winner_time"] != float('inf'):
            algo_performance[match["winner"]]["total_time"] += match["winner_time"]
            algo_performance[match["winner"]]["matches"] += 1
            
        if match["loser"] not in algo_performance:
            algo_performance[match["loser"]] = {"wins": 0, "losses": 0, "total_time": 0, "matches": 0}
            
        algo_performance[match["loser"]]["losses"] += 1
        if match["loser_time"] != float('inf'):
            algo_performance[match["loser"]]["total_time"] += match["loser_time"]
            algo_performance[match["loser"]]["matches"] += 1
    
    for algo, perf in algo_performance.items():
        perf["avg_time"] = perf["total_time"] / perf["matches"] if perf["matches"] > 0 else float('inf')
    
    all_algorithms = {}
    for round_algos in all_rounds:
        for algo in round_algos:
            if algo[0] != "BYE":
                all_algorithms[algo[0]] = algo
    
    categories = {"fundamental": [], "advanced": [], "specialized": [], "linear": [], "noncomparison": [], "theoretical": []}
    for algo_name, algo in all_algorithms.items():
        if algo[2] in categories:
            performance = algo_performance.get(algo_name, {"wins": 0, "losses": 0, "avg_time": float('inf'), "matches": 0})
            categories[algo[2]].append((algo, performance))
    
    print("\n" + "=" * 50)
    print("RESULTS BY CATEGORY")
    print("=" * 50)
    
    for category_name, algos in categories.items():
        if not algos:
            continue
            
        print(f"\n{category_name.upper()} ALGORITHMS:")
        sorted_algos = sorted(algos, key=lambda x: (-x[1]["wins"], x[1]["avg_time"]))
        
        for i, (algo, perf) in enumerate(sorted_algos[:3], 1):
            position = "1st" if i == 1 else "2nd" if i == 2 else "3rd"
            if perf["matches"] > 0:
                print(f"  {position}: {algo[0]} - {perf['wins']} wins, {perf['avg_time']:.6f}s avg time")
            else:
                print(f"  {position}: {algo[0]} - {perf['wins']} wins, no valid time data")
    
    print("\n" + "=" * 50)
    print("NOTABLE MATCHES")
    print("=" * 50)
    
    sorted_matches = sorted(
        [(k, v) for k, v in tournament_results.items() if v['winner_time'] != float('inf') and v['loser_time'] != float('inf')],
        key=lambda x: x[1]['loser_time'] / x[1]['winner_time'] if x[1]['winner_time'] > 0 else float('inf'),
        reverse=True
    )
    
    for (algo1, algo2), match in sorted_matches[:3]:
        time_diff = match["loser_time"] / match["winner_time"] if match["winner_time"] > 0 else float('inf')
        print(f"  {match['winner']} vs {match['loser']} - {time_diff:.2f}x speed difference")

    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50 + "\n")
    
    table = PrettyTable()
    table.field_names = ["Algorithm", "Category", "Wins", "Losses", "Avg Time (s)"]
    
    sorted_algorithms = []
    for algo_name, algo in all_algorithms.items():
        perf = algo_performance.get(algo_name, {"wins": 0, "losses": 0, "avg_time": float('inf'), "matches": 0})
        sorted_algorithms.append((algo, perf))
    
    sorted_algorithms.sort(key=lambda x: (-x[1]["wins"], x[1]["avg_time"]))
    
    for algo, perf in sorted_algorithms:
        avg_time = perf["total_time"] / perf["matches"] if perf["matches"] > 0 else float('inf')
        perf["avg_time"] = avg_time
        time_str = "timeout" if avg_time == float('inf') else f"{avg_time:.6f}"
        algo_name = f"🏆 {algo[0]}" if champion and algo[0] == champion[0] else f"🥈 {algo[0]}" if runner_up and algo[0] == runner_up[0] else algo[0]
        table.add_row([algo_name, algo[2].capitalize(), perf["wins"], perf["losses"], time_str])
    
    print(table)

def verify_sorting(algorithms):
    # reuse the function from benchmark.py but ensure we keep the categorization
    print("\nVERIFYING ALGORITHMS...", end=" ")
    
    test_cases = [
        [5, 4, 3, 2, 1],
        [3, 1, 4, 1, 5, 9, 2, 6],
        [random.randint(0, 100) for _ in range(50)]
    ]
    
    all_passed = True
    total_algos = 0
    passed_algos = 0
    
    for name, func, _ in algorithms:
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
    
    print(f"✅ {passed_algos}/{total_algos}")
    return all_passed

def print_config():
    print("\nCONFIGURATION:")
    print(f"  Timeout: {TIMEOUT}s")
    print(f"  Tournament size: {TOURNAMENT_SIZE} elements")
    print(f"  Data type: {TEST_DATA_TYPE}")
    if TEST_DATA_TYPE == 'random':
        print(f"  Value range: 0 to {TEST_RANGE}")
    print(f"  Visualization: {'Enabled' if VIZ else 'Disabled'}")
    if os.environ.get('SKIP_VERIFY', '').lower() == 'true':
        print("  Verification: Skipped")

def main():
    print("\n"+ "=" * 50)
    print("\nSORT OLYMPICS")
    print("=" * 50)
    print_config()
    
    # Get the complete list of sorting algorithms
    sorting_algorithms = discover_sorting_algorithms()
    all_algorithm_count = len(sorting_algorithms)
    
    # Verify the algorithms work correctly
    if not verify_sorting(sorting_algorithms):
        print("WARNING: Some algorithms failed verification. Results may be incorrect.")
    
    # Make a deep copy of the original algorithms to prevent any potential modifications
    all_algorithms = [(name, func, category) for name, func, category in sorting_algorithms]
    
    # Run the tournament
    champion, tournament_results, all_rounds, final_match = run_tournament(sorting_algorithms)
    print_tournament_results(champion, tournament_results, all_rounds, final_match)
    
    if BENCH:
        print("\n" + "=" * 50)
        print("BENCHMARK RESULTS")
        print("=" * 50)
        benchmark.TIMEOUT = TIMEOUT
        benchmark_results, sizes = benchmark.benchmark_for_complexity(
            all_algorithms, 
            None, 
            TEST_DATA_TYPE, 
            TEST_RANGE
        )
        valid_results = {name: points for name, points in benchmark_results.items() if len(points) >= 1}
        if valid_results:
            for algo, points in valid_results.items():
                print(f"{algo}: {points}")
        else:
            print("Not enough data points to display benchmark results.")
    elif VIZ:
        print("\n" + "=" * 50)
        print("VISUALISE COMPLEXITY")
        print("=" * 50)
        print(f"\nVisualising all {all_algorithm_count} algorithms...\n")
        import visualizer
        visualizer.main("all")
    
    print()

if __name__ == "__main__":
    main()