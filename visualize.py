import sys
import pygame
import random
import threading
import time
import os  # added for env variables
from utils import discover_sorting_algorithms, generate_test_data

# global settings - increased window size for 48 sorts grid + leaderboard panel
WIDTH = 1600
HEIGHT = 900
# get array size from env variable (default: 100)
SIZE = int(os.getenv("SIZE", 100))
# get delay in ms from env variable (default: 20)
DELAY = int(os.getenv("DELAY", 20))
BACKGROUND_COLOR = (0, 0, 0)
BAR_COLOR = (255, 255, 255)  # unsorted bar color
SORTED_COLOR = (0, 255, 0)   # sorted bar color (green)
# white fonts everywhere except labels; labels are rendered in yellow
TEXT_COLOR = (255, 255, 255)  # white (for instructions, leaderboard, etc.)
LABEL_COLOR = (255, 255, 0)   # yellow (for algorithm labels)
FONT_SIZE = 20

# extra layout for simultaneous mode: reserve a side panel for the leaderboard.
LEADERBOARD_WIDTH = 300                # width for the leaderboard panel
GRID_WIDTH = WIDTH - LEADERBOARD_WIDTH   # remaining width for the grid

# -------------------------------
# Custom Exception for Abort
# -------------------------------
class AbortSort(Exception):
    pass

# -------------------------------
# Visualization Helper Classes
# -------------------------------

# Single-mode visual list for individual runs (with built-in delay)
class VisualList(list):
    def __init__(self, data, callback, delay=DELAY):
        super().__init__(data)
        self.callback = callback
        self.delay = delay

    def __setitem__(self, index, value):
        if pygame.key.get_pressed()[pygame.K_ESCAPE]:
            raise AbortSort("Sorting Aborted by user.")
        super().__setitem__(index, value)
        if self.callback:
            self.callback(self)
        pygame.event.pump()  # process pending events
        pygame.time.delay(self.delay)

# Simultaneous-mode list; minimal delay (yields CPU)
class SimVisualList(list):
    def __init__(self, data):
        super().__init__(data)

    def __setitem__(self, index, value):
        if pygame.key.get_pressed()[pygame.K_ESCAPE]:
            raise AbortSort("Sorting Aborted by user.")
        super().__setitem__(index, value)
        time.sleep(DELAY / 1000.0)

# -------------------------------
# Drawing Helpers
# -------------------------------

screen = None
font = None

def draw_array(arr):
    # Draw full-screen array for individual mode.
    screen.fill(BACKGROUND_COLOR)
    bar_width = WIDTH // len(arr)
    sorted_arr = sorted(arr)
    for i, val in enumerate(arr):
        bar_height = int((val / SIZE) * HEIGHT)
        x = i * bar_width
        y = HEIGHT - bar_height
        color = SORTED_COLOR if val == sorted_arr[i] else BAR_COLOR
        pygame.draw.rect(screen, color, (x, y, bar_width, bar_height))
    pygame.display.flip()

def draw_text(text):
    text_surface = font.render(text, True, TEXT_COLOR)
    screen.blit(text_surface, (10, 10))
    pygame.display.flip()

def reset_array():
    return generate_test_data(SIZE, 'random', SIZE)

def draw_array_in_cell(arr, cell_rect, algo_name):
    # Draw the array in a grid cell (for simultaneous mode) with label.
    x, y, cell_width, cell_height = cell_rect
    pygame.draw.rect(screen, BACKGROUND_COLOR, cell_rect)
    bar_width = cell_width / len(arr)
    scale = cell_height / SIZE
    sorted_arr = sorted(arr)
    for i, val in enumerate(arr):
        bar_h = val * scale
        bar_rect = (x + i * bar_width, y + (cell_height - bar_h), bar_width, bar_h)
        color = SORTED_COLOR if val == sorted_arr[i] else BAR_COLOR
        pygame.draw.rect(screen, color, bar_rect)
    # Render the algorithm label in yellow.
    text_surface = font.render(algo_name, True, LABEL_COLOR)
    screen.blit(text_surface, (x + 5, y + 5))

# -------------------------------
# Mode Functions
# -------------------------------

def algo_wrapper(name, func, sim_list, leaderboard, lock):
    # Wrapper: time the sort and record duration in leaderboard.
    try:
        start_time = time.time()
        func(sim_list)
        duration = time.time() - start_time
        with lock:
            leaderboard.append((name, duration))
    except AbortSort:
        with lock:
            leaderboard.append((name + " (aborted)", float('inf')))

def run_simultaneous_all():
    """
    Run all algorithms concurrently in a grid and display a leaderboard.
    Press the spacebar at any time to return to the main menu.
    (Press Esc to quit at any time)
    """
    initial = reset_array()
    
    # get all algorithms
    all_algos = [(name, func) for name, func, _ in discover_sorting_algorithms()]
    
    sim_lists = []
    threads = []
    leaderboard = []  # list of tuples: (name, duration)
    lock = threading.Lock()

    # Create a copy for each algorithm and start its thread.
    for name, func in all_algos:
        sim_list = SimVisualList(list(initial))
        sim_lists.append((name, sim_list, func))
    for name, sim_list, func in sim_lists:
        t = threading.Thread(target=algo_wrapper, args=(name, func, sim_list, leaderboard, lock))
        t.start()
        threads.append(t)

    # calculate grid dimensions based on number of algorithms
    total_algos = len(all_algos)
    cols = min(8, total_algos)
    rows = (total_algos + cols - 1) // cols  # ceiling division
    
    cell_width = GRID_WIDTH // cols
    cell_height = HEIGHT // rows
    clock = pygame.time.Clock()
    running_sim = True
    while running_sim:
        for evt in pygame.event.get():
            if evt.type == pygame.QUIT or (evt.type == pygame.KEYDOWN and evt.key == pygame.K_ESCAPE):
                pygame.quit()
                sys.exit()
            elif evt.type == pygame.KEYDOWN:
                if evt.key == pygame.K_SPACE:
                    running_sim = False  # Exit to main menu immediately

        screen.fill(BACKGROUND_COLOR)
        # Draw the grid of algorithms (left panel).
        for idx, (name, sim_list, _) in enumerate(sim_lists):
            col = idx % cols
            row = idx // cols
            cell_rect = (col * cell_width, row * cell_height, cell_width, cell_height)
            try:
                draw_array_in_cell(sim_list, cell_rect, name)
            except AbortSort:
                # If aborted, skip updating this cell.
                pass

        # Draw the leaderboard (right panel).
        lb_x = GRID_WIDTH + 10
        lb_y = 10
        pygame.draw.rect(screen, BACKGROUND_COLOR, (GRID_WIDTH, 0, LEADERBOARD_WIDTH, HEIGHT))
        title_surf = font.render("Leaderboard (fastest first)", True, TEXT_COLOR)
        screen.blit(title_surf, (GRID_WIDTH + 10, lb_y))
        lb_y += FONT_SIZE + 5
        with lock:
            finished_entries = sorted(leaderboard, key=lambda x: x[1])
        rank = 1
        for name, duration in finished_entries:
            text_line = f"{rank}. {name}: {duration if duration != float('inf') else '--'}"
            line_surf = font.render(text_line, True, TEXT_COLOR)
            screen.blit(line_surf, (GRID_WIDTH + 10, lb_y))
            lb_y += FONT_SIZE + 2
            rank += 1

        pygame.display.flip()
        # If all threads finished, display completion message.
        if all(not t.is_alive() for t in threads):
            complete_text = "All sorts complete - press any key, SPACE or ESC to quit"
            comp_surf = font.render(complete_text, True, TEXT_COLOR)
            screen.blit(comp_surf, (GRID_WIDTH + 10, HEIGHT - FONT_SIZE - 10))
            pygame.display.flip()
            waiting = True
            while waiting:
                for evt in pygame.event.get():
                    if evt.type == pygame.QUIT or (evt.type == pygame.KEYDOWN and evt.key == pygame.K_ESCAPE):
                        pygame.quit()
                        sys.exit()
                    if evt.type == pygame.KEYDOWN:
                        waiting = False
            running_sim = False
        clock.tick(60)

def run_single_mode(algo_name, algo_func):
    # Run a single algorithm in full-screen individual mode.
    arr_data = reset_array()
    draw_array(arr_data)
    vlist = VisualList(arr_data, draw_array, DELAY)
    draw_text("Sorting using: " + algo_name + " | Press Esc to quit")
    pygame.display.flip()
    try:
        algo_func(vlist)
    except AbortSort:
        draw_text("Sort Aborted! | Press any key, SPACE or ESC to return to menu")
    else:
        draw_text("Sorted: " + algo_name + " - press any key, SPACE or ESC to quit")
    waiting = True
    while waiting:
        for evt in pygame.event.get():
            if evt.type == pygame.QUIT or (evt.type == pygame.KEYDOWN and evt.key == pygame.K_ESCAPE):
                pygame.quit()
                sys.exit()
            if evt.type == pygame.KEYDOWN:
                waiting = False

def main_menu():
    """
    Display the main menu with a header and a grid of algorithms.
    The header shows that pressing "0" will run "All" mode.
    (Press Esc to quit at any time)
    The user may click on a cell to choose an individual algorithm.
    """
    # get all algorithms
    all_algos = [(name, func) for name, func, _ in discover_sorting_algorithms()]
    
    selecting = True
    # calculate grid dimensions based on number of algorithms
    total_algos = len(all_algos)
    cols = min(4, total_algos)
    rows = (total_algos + cols - 1) // cols  # ceiling division
    
    cell_width = WIDTH // cols
    cell_height = (HEIGHT - 40) // rows  # leave header area
    header = "Sorting Visualizer - [0: All] | Click a cell to run individual algo | [q] or Esc to Quit"
    grid_y_offset = 40
    while selecting:
        screen.fill(BACKGROUND_COLOR)
        header_surf = font.render(header, True, TEXT_COLOR)
        screen.blit(header_surf, (10, 10))
        for idx, (name, _) in enumerate(all_algos):
            col = idx % cols
            row = idx // cols
            cell_rect = (col * cell_width, grid_y_offset + row * cell_height, cell_width, cell_height)
            pygame.draw.rect(screen, (50, 50, 50), cell_rect, 1)
            # Render cell text (algorithm labels) in yellow.
            text = f"{idx+1}. {name}"
            text_surf = font.render(text, True, LABEL_COLOR)
            screen.blit(text_surf, (cell_rect[0] + 5, cell_rect[1] + 5))
        pygame.display.flip()
        for evt in pygame.event.get():
            if evt.type == pygame.QUIT or (evt.type == pygame.KEYDOWN and evt.key == pygame.K_ESCAPE):
                return None
            elif evt.type == pygame.KEYDOWN:
                if evt.key == pygame.K_q:
                    return None
                elif evt.key == pygame.K_0:
                    return "ALL"
            elif evt.type == pygame.MOUSEBUTTONDOWN:
                mx, my = evt.pos
                if my >= grid_y_offset:
                    col = mx // cell_width
                    row = (my - grid_y_offset) // cell_height
                    index = row * cols + col
                    if 0 <= index < len(all_algos):
                        return all_algos[index]
        pygame.time.delay(20)

# -------------------------------
# Main Loop
# -------------------------------

def main(arg=None):
    global screen, font
    pygame.init()
    # open fullscreen mode
    screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.FULLSCREEN)
    pygame.display.set_caption("Sorting Visualizer")
    font = pygame.font.SysFont("Arial", FONT_SIZE)
    
    # if an arg was provided, use it; otherwise, fallback to sys.argv
    if arg is None:
        if len(sys.argv) > 1:
            arg = sys.argv[1].lower()
        else:
            arg = None
    else:
        arg = arg.lower()
    
    if arg == "all":
        run_simultaneous_all()
    elif arg:
        # get all algorithms
        all_algos_dict = {func.__name__.lower(): (name, func) for name, func, _ in discover_sorting_algorithms()}
        
        if arg in all_algos_dict:
            chosen = all_algos_dict[arg]
            run_single_mode(chosen[0], chosen[1])
        else:
            print("no matching algorithm found for:", arg)
    else:
        # no arg provided: show the main menu
        running = True
        while running:
            choice = main_menu()
            if choice is None:
                running = False
            elif choice == "ALL":
                run_simultaneous_all()
            else:
                name, func = choice
                run_single_mode(name, func)
    sys.exit(0)

if __name__ == "__main__":
    main()