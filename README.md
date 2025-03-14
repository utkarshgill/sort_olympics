# sorting olympics
yet another sorting toolkit. because the world needed more sorting analysis.

## algorithms

### basic
- [x] bubble_sort
- [x] selection_sort
- [x] insertion_sort
- [x] merge_sort
- [x] quick_sort
- [x] heap_sort

### advanced
- [x] shell_sort
- [x] tim_sort
- [x] intro_sort
- [x] library_sort
- [x] block_sort
- [x] smooth_sort

### specialized
- [x] tree_sort
- [x] tournament_sort
- [x] patience_sort
- [x] cube_sort
- [x] comb_sort
- [x] cocktail_sort
- [x] gnome_sort
- [x] odd_even_sort
- [x] pancake_sort
- [x] strand_sort
- [x] exchange_sort
- [x] cycle_sort
- [x] recombinant_sort
- [x] inplace_merge_sort

### linear-time
- [x] counting_sort
- [x] bucket_sort_uniform
- [x] bucket_sort_integer
- [x] radix_sort_lsd
- [x] radix_sort_msd
- [x] pigeonhole_sort

### noncomparison
- [x] spreadsort
- [x] burstsort
- [x] flashsort
- [x] postman_sort
- [x] msd_radix_sort_inplace

### theoretical
- [x] bead_sort
- [x] merge_insertion_sort
- [x] i_cant_believe_it_can_sort
- [x] spaghetti_sort
- [x] sorting_network
- [x] bitonic_sort
- [x] bogosort
- [x] stooge_sort
- [x] slowsort
- [x] franceschini_mergesort
- [x] thorup_sort

## tools

### visualize.py
watch sorting happen in real-time. riveting entertainment.
```bash
DELAY=10 SIZE=100 python visualize.py all
```
```bash
python visualize.py bubble_sort
```
![Untitled (4)](https://github.com/user-attachments/assets/1bf2fd62-11cd-4d66-b08a-1aa0bbd62d12)

### benchmark.py
measures how painfully slow your sorting algorithm really is.
```bash
TIMEOUT=5 python benchmark.py
```
<img width="1488" alt="tournament example" src="https://github.com/user-attachments/assets/1739ac4e-6a28-443f-8905-91208949e4ad" />

### run.py
pits algorithms against each other. may the least awful one win.
```bash
SIZE=100 TIMEOUT=5 python run.py
```
```bash
# with benchmark and visualisation
BENCH=1 VIZ=1 DATA_TYPE=random TEST_RANGE=1000 python run.py
```
```bash
engelbart@Utkarshs-MacBook-Pro sort_olympics % SIZE=100 TIMEOUT=5  python run.py

==================================================

SORT OLYMPICS
==================================================

CONFIGURATION:
  Timeout: 5s
  Tournament size: 100 elements
  Data type: random
  Value range: 0 to 1000
  Visualization: Disabled

ALGORITHMS:
  Fundamental: 6
  Advanced: 6
  Specialized: 14
  Linear-time: 6
  Noncomparison: 5
  Theoretical: 11
  Total: 48

VERIFYING ALGORITHMS... ✅ 48/48

==================================================
TOURNAMENT BEGINS! (48 competitors)
==================================================

ROUND 1 (64 competitors)
  Tim Sort defeats I Cant Believe It Can Sort (4.85x faster)
  Insertion Sort defeats Bitonic Sort (9.53x faster)
  Smooth Sort defeats Gnome Sort (6.80x faster)
  Recombinant Sort defeats Stooge Sort (484.47x faster)
  Cycle Sort defeats Bead Sort (131.08x faster)
  Strand Sort defeats Cube Sort (2.24x faster)
  Msd Radix Sort Inplace defeats Patience Sort (1.46x faster)
  Shell Sort defeats Odd Even Sort (3.97x faster)
  Heap Sort defeats Selection Sort (2.11x faster)
  Thorup Sort defeats Bogosort (2.15x faster)
  Library Sort defeats Inplace Merge Sort (4.83x faster)
  Bucket Sort Uniform defeats Bucket Sort Integer (5.41x faster)
  Merge Insertion Sort defeats Comb Sort (1.02x faster)
  Cocktail Sort defeats Spaghetti Sort (8.55x faster)
  Spreadsort defeats Slowsort (1090.43x faster)
  Postman Sort defeats Burstsort (3.51x faster)
  Merge Sort defeats Sorting Network (6.06x faster)
  Intro Sort defeats Radix Sort Lsd (3.17x faster)
  Bubble Sort defeats Flashsort (3.01x faster)
  Franceschini Mergesort defeats Pancake Sort (2.74x faster)
  Exchange Sort defeats Radix Sort Msd (3.13x faster)
  Quick Sort defeats Pigeonhole Sort (5.98x faster)
  Tree Sort defeats Tournament Sort (1.88x faster)
  Block Sort defeats Counting Sort (20.84x faster)

ROUND 2 (32 competitors)
  Tim Sort defeats Insertion Sort (1.03x faster)
  Recombinant Sort defeats Smooth Sort (2.34x faster)
  Strand Sort defeats Cycle Sort (3.52x faster)
  Shell Sort defeats Msd Radix Sort Inplace (5.33x faster)
  Heap Sort defeats Thorup Sort (24.18x faster)
  Library Sort defeats Bucket Sort Uniform (2.42x faster)
  Merge Insertion Sort defeats Cocktail Sort (3.24x faster)
  Spreadsort defeats Postman Sort (2.92x faster)
  Intro Sort defeats Merge Sort (3.06x faster)
  Franceschini Mergesort defeats Bubble Sort (2.14x faster)
  Quick Sort defeats Exchange Sort (2.51x faster)
  Block Sort defeats Tree Sort (3.33x faster)

ROUND 3 (16 competitors)
  Recombinant Sort defeats Tim Sort (1.90x faster)
  Shell Sort defeats Strand Sort (4.05x faster)
  Library Sort defeats Heap Sort (1.42x faster)
  Merge Insertion Sort defeats Spreadsort (2.54x faster)
  Intro Sort defeats Franceschini Mergesort (1.73x faster)
  Block Sort defeats Quick Sort (1.80x faster)

ROUND 4 (8 competitors)
  Shell Sort defeats Recombinant Sort (1.27x faster)
  Merge Insertion Sort defeats Library Sort (1.89x faster)
  Block Sort defeats Intro Sort (1.44x faster)

ROUND 5 (4 competitors)
  Shell Sort defeats Merge Insertion Sort (1.22x faster)
  Block Sort advances (opponent: BYE)

ROUND 6 (2 competitors)
  Shell Sort defeats Block Sort (1.21x faster)

THIRD PLACE MATCH
  Block Sort defeats Merge Insertion Sort (1.28x faster)

==================================================
TOURNAMENT RESULT
==================================================

CHAMPION: 🏆 Shell Sort (Advanced)
RUNNER-UP: 🥈 Block Sort (Advanced)

Shell Sort 🏆 [0.000038s]
Block Sort 🥈 [0.000046s]
   │5
   │├─ Shell Sort [0.000035s]
   │└─ Merge Insertion Sort [0.000043s]
   │   │4
   │   │├─ Shell Sort [0.000029s]
   │   │└─ Recombinant Sort [0.000037s]
   │   │   │3
   │   │   │├─ Recombinant Sort [0.000032s]
   │   │   │└─ Tim Sort [0.000061s]
   │   │   │   │2
   │   │   │   │├─ Tim Sort [0.000067s]
   │   │   │   │└─ Insertion Sort [0.000069s]
   │   │   │   │   │1
   │   │   │   │   │├─ Tim Sort [0.000064s]
   │   │   │   │   │└─ I Cant Believe It Can Sort [0.000310s]
   │   │   │   │   │1
   │   │   │   │   │├─ Insertion Sort [0.000076s]
   │   │   │   │    └─ Bitonic Sort [0.000725s]
   │   │   │   │2
   │   │   │   │├─ Recombinant Sort [0.000029s]
   │   │   │    └─ Smooth Sort [0.000068s]
   │   │   │       │1
   │   │   │       │├─ Smooth Sort [0.000060s]
   │   │   │       │└─ Gnome Sort [0.000407s]
   │   │   │       │1
   │   │   │       │├─ Recombinant Sort [0.000033s]
   │   │   │        └─ Stooge Sort [0.015824s]
   │   │   │3
   │   │   │├─ Shell Sort [0.000028s]
   │   │    └─ Strand Sort [0.000114s]
   │   │       │2
   │   │       │├─ Strand Sort [0.000106s]
   │   │       │└─ Cycle Sort [0.000374s]
   │   │       │   │1
   │   │       │   │├─ Cycle Sort [0.000382s]
   │   │       │   │└─ Bead Sort [0.050067s]
   │   │       │   │1
   │   │       │   │├─ Strand Sort [0.000126s]
   │   │       │    └─ Cube Sort [0.000282s]
   │   │       │2
   │   │       │├─ Shell Sort [0.000030s]
   │   │        └─ Msd Radix Sort Inplace [0.000160s]
   │   │           │1
   │   │           │├─ Msd Radix Sort Inplace [0.000170s]
   │   │           │└─ Patience Sort [0.000249s]
   │   │           │1
   │   │           │├─ Shell Sort [0.000054s]
   │   │            └─ Odd Even Sort [0.000215s]
   │   │4
   │   │├─ Merge Insertion Sort [0.000041s]
   │    └─ Library Sort [0.000077s]
   │       │3
   │       │├─ Library Sort [0.000055s]
   │       │└─ Heap Sort [0.000078s]
   │       │   │2
   │       │   │├─ Heap Sort [0.000078s]
   │       │   │└─ Thorup Sort [0.001880s]
   │       │   │   │1
   │       │   │   │├─ Heap Sort [0.000065s]
   │       │   │   │└─ Selection Sort [0.000137s]
   │       │   │   │1
   │       │   │   │├─ Thorup Sort [0.001902s]
   │       │   │    └─ Bogosort [0.004093s]
   │       │   │2
   │       │   │├─ Library Sort [0.000079s]
   │       │    └─ Bucket Sort Uniform [0.000192s]
   │       │       │1
   │       │       │├─ Library Sort [0.000060s]
   │       │       │└─ Inplace Merge Sort [0.000289s]
   │       │       │1
   │       │       │├─ Bucket Sort Uniform [0.000152s]
   │       │        └─ Bucket Sort Integer [0.000821s]
   │       │3
   │       │├─ Merge Insertion Sort [0.000056s]
   │        └─ Spreadsort [0.000142s]
   │           │2
   │           │├─ Merge Insertion Sort [0.000054s]
   │           │└─ Cocktail Sort [0.000175s]
   │           │   │1
   │           │   │├─ Merge Insertion Sort [0.000049s]
   │           │   │└─ Comb Sort [0.000050s]
   │           │   │1
   │           │   │├─ Cocktail Sort [0.000201s]
   │           │    └─ Spaghetti Sort [0.001719s]
   │           │2
   │           │├─ Spreadsort [0.000159s]
   │            └─ Postman Sort [0.000465s]
   │               │1
   │               │├─ Spreadsort [0.000175s]
   │               │└─ Slowsort [0.191085s]
   │               │1
   │               │├─ Postman Sort [0.000481s]
   │                └─ Burstsort [0.001690s]
   │5
   │├─ Block Sort
    └─ BYE
       │4
       │├─ Block Sort [0.000043s]
        └─ Intro Sort [0.000062s]
           │3
           │├─ Intro Sort [0.000052s]
           │└─ Franceschini Mergesort [0.000090s]
           │   │2
           │   │├─ Intro Sort [0.000039s]
           │   │└─ Merge Sort [0.000119s]
           │   │   │1
           │   │   │├─ Merge Sort [0.000110s]
           │   │   │└─ Sorting Network [0.000666s]
           │   │   │1
           │   │   │├─ Intro Sort [0.000047s]
           │   │    └─ Radix Sort Lsd [0.000150s]
           │   │2
           │   │├─ Franceschini Mergesort [0.000098s]
           │    └─ Bubble Sort [0.000210s]
           │       │1
           │       │├─ Bubble Sort [0.000212s]
           │       │└─ Flashsort [0.000638s]
           │       │1
           │       │├─ Franceschini Mergesort [0.000094s]
           │        └─ Pancake Sort [0.000257s]
           │3
           │├─ Block Sort [0.000045s]
            └─ Quick Sort [0.000081s]
               │2
               │├─ Quick Sort [0.000066s]
               │└─ Exchange Sort [0.000166s]
               │   │1
               │   │├─ Exchange Sort [0.000175s]
               │   │└─ Radix Sort Msd [0.000548s]
               │   │1
               │   │├─ Quick Sort [0.000070s]
               │    └─ Pigeonhole Sort [0.000418s]
               │2
               │├─ Block Sort [0.000049s]
                └─ Tree Sort [0.000164s]
                   │1
                   │├─ Tree Sort [0.000192s]
                   │└─ Tournament Sort [0.000361s]
                   │1
                   │├─ Block Sort [0.000063s]
                    └─ Counting Sort [0.001312s]

==================================================
RESULTS BY CATEGORY
==================================================

FUNDAMENTAL ALGORITHMS:
  1st: Quick Sort - 2 wins, 0.000072s avg time
  2nd: Heap Sort - 2 wins, 0.000074s avg time
  3rd: Insertion Sort - 1 wins, 0.000073s avg time

ADVANCED ALGORITHMS:
  1st: Shell Sort - 6 wins, 0.000036s avg time
  2nd: Block Sort - 5 wins, 0.000049s avg time
  3rd: Intro Sort - 3 wins, 0.000050s avg time

SPECIALIZED ALGORITHMS:
  1st: Recombinant Sort - 3 wins, 0.000033s avg time
  2nd: Strand Sort - 2 wins, 0.000115s avg time
  3rd: Exchange Sort - 1 wins, 0.000171s avg time

LINEAR ALGORITHMS:
  1st: Bucket Sort Uniform - 1 wins, 0.000172s avg time
  2nd: Radix Sort Lsd - 0 wins, 0.000150s avg time
  3rd: Pigeonhole Sort - 0 wins, 0.000418s avg time

NONCOMPARISON ALGORITHMS:
  1st: Spreadsort - 2 wins, 0.000159s avg time
  2nd: Msd Radix Sort Inplace - 1 wins, 0.000165s avg time
  3rd: Postman Sort - 1 wins, 0.000473s avg time

THEORETICAL ALGORITHMS:
  1st: Merge Insertion Sort - 4 wins, 0.000050s avg time
  2nd: Franceschini Mergesort - 2 wins, 0.000094s avg time
  3rd: Thorup Sort - 1 wins, 0.001891s avg time

==================================================
NOTABLE MATCHES
==================================================
  Spreadsort vs Slowsort - 1090.43x speed difference
  Recombinant Sort vs Stooge Sort - 484.47x speed difference
  Cycle Sort vs Bead Sort - 131.08x speed difference

==================================================
SUMMARY
==================================================

+----------------------------+---------------+------+--------+--------------+
|         Algorithm          |    Category   | Wins | Losses | Avg Time (s) |
+----------------------------+---------------+------+--------+--------------+
|       🏆 Shell Sort        |    Advanced   |  6   |   0    |   0.000036   |
|       🥈 Block Sort        |    Advanced   |  5   |   1    |   0.000049   |
|    Merge Insertion Sort    |  Theoretical  |  4   |   2    |   0.000050   |
|      Recombinant Sort      |  Specialized  |  3   |   1    |   0.000033   |
|         Intro Sort         |    Advanced   |  3   |   1    |   0.000050   |
|        Library Sort        |    Advanced   |  3   |   1    |   0.000068   |
|          Tim Sort          |    Advanced   |  2   |   1    |   0.000064   |
|         Quick Sort         |  Fundamental  |  2   |   1    |   0.000072   |
|         Heap Sort          |  Fundamental  |  2   |   1    |   0.000074   |
|   Franceschini Mergesort   |  Theoretical  |  2   |   1    |   0.000094   |
|        Strand Sort         |  Specialized  |  2   |   1    |   0.000115   |
|         Spreadsort         | Noncomparison |  2   |   1    |   0.000159   |
|        Smooth Sort         |    Advanced   |  1   |   1    |   0.000064   |
|       Insertion Sort       |  Fundamental  |  1   |   1    |   0.000073   |
|         Merge Sort         |  Fundamental  |  1   |   1    |   0.000114   |
|   Msd Radix Sort Inplace   | Noncomparison |  1   |   1    |   0.000165   |
|       Exchange Sort        |  Specialized  |  1   |   1    |   0.000171   |
|    Bucket Sort Uniform     |     Linear    |  1   |   1    |   0.000172   |
|         Tree Sort          |  Specialized  |  1   |   1    |   0.000178   |
|       Cocktail Sort        |  Specialized  |  1   |   1    |   0.000188   |
|        Bubble Sort         |  Fundamental  |  1   |   1    |   0.000211   |
|         Cycle Sort         |  Specialized  |  1   |   1    |   0.000378   |
|        Postman Sort        | Noncomparison |  1   |   1    |   0.000473   |
|        Thorup Sort         |  Theoretical  |  1   |   1    |   0.001891   |
|         Comb Sort          |  Specialized  |  0   |   1    |   0.000050   |
|       Selection Sort       |  Fundamental  |  0   |   1    |   0.000137   |
|       Radix Sort Lsd       |     Linear    |  0   |   1    |   0.000150   |
|       Odd Even Sort        |  Specialized  |  0   |   1    |   0.000215   |
|       Patience Sort        |  Specialized  |  0   |   1    |   0.000249   |
|        Pancake Sort        |  Specialized  |  0   |   1    |   0.000257   |
|         Cube Sort          |  Specialized  |  0   |   1    |   0.000282   |
|     Inplace Merge Sort     |  Specialized  |  0   |   1    |   0.000289   |
| I Cant Believe It Can Sort |  Theoretical  |  0   |   1    |   0.000310   |
|      Tournament Sort       |  Specialized  |  0   |   1    |   0.000361   |
|         Gnome Sort         |  Specialized  |  0   |   1    |   0.000407   |
|      Pigeonhole Sort       |     Linear    |  0   |   1    |   0.000418   |
|       Radix Sort Msd       |     Linear    |  0   |   1    |   0.000548   |
|         Flashsort          | Noncomparison |  0   |   1    |   0.000638   |
|      Sorting Network       |  Theoretical  |  0   |   1    |   0.000666   |
|        Bitonic Sort        |  Theoretical  |  0   |   1    |   0.000725   |
|    Bucket Sort Integer     |     Linear    |  0   |   1    |   0.000821   |
|       Counting Sort        |     Linear    |  0   |   1    |   0.001312   |
|         Burstsort          | Noncomparison |  0   |   1    |   0.001690   |
|       Spaghetti Sort       |  Theoretical  |  0   |   1    |   0.001719   |
|          Bogosort          |  Theoretical  |  0   |   1    |   0.004093   |
|        Stooge Sort         |  Theoretical  |  0   |   1    |   0.015824   |
|         Bead Sort          |  Theoretical  |  0   |   1    |   0.050067   |
|          Slowsort          |  Theoretical  |  0   |   1    |   0.191085   |
+----------------------------+---------------+------+--------+--------------+
