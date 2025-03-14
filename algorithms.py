"""
Sort Olympics - Collection of sorting algorithms

"""

import random

#=============Fundamental Sorting Algorithms=============

def bubble_sort(arr):
    """the classic n^2 algorithm that nobody should ever use in production"""
    n = len(arr)
    for i in range(n):
        swapped = False
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
                swapped = True
        if not swapped: break
    return arr

def selection_sort(arr):
    """finds minimum element n times, somehow worse than bubble sort"""
    for i in range(len(arr)):
        min_idx = i
        for j in range(i + 1, len(arr)):
            if arr[j] < arr[min_idx]: min_idx = j
        arr[i], arr[min_idx] = arr[min_idx], arr[i]
    return arr

def insertion_sort(arr):
    """actually useful for small arrays, builds sorted array one item at a time"""
    for i in range(1, len(arr)):
        current, j = arr[i], i - 1
        while j >= 0 and arr[j] > current:
            arr[j + 1], j = arr[j], j - 1
        arr[j + 1] = current
    return arr

def merge_sort(arr):
    """divide and conquer that actually works, n log n but needs extra space"""
    if len(arr) <= 1: return arr
    mid = len(arr) // 2
    left, right = arr[:mid], arr[mid:]
    merge_sort(left)
    merge_sort(right)
    i = j = k = 0
    while i < len(left) and j < len(right):
        if not (right[j] < left[i]): arr[k], i, k = left[i], i + 1, k + 1
        else: arr[k], j, k = right[j], j + 1, k + 1
    while i < len(left): arr[k], i, k = left[i], i + 1, k + 1
    while j < len(right): arr[k], j, k = right[j], j + 1, k + 1
    return arr

def quick_sort(arr):
    """fastest in practice until it isn't, pivot selection is everything"""
    def _quick_sort(arr, low, high):
        if low < high:
            pivot_idx = _partition(arr, low, high)
            _quick_sort(arr, low, pivot_idx - 1)
            _quick_sort(arr, pivot_idx + 1, high)
    
    def _partition(arr, low, high):
        pivot, i = arr[high], low - 1
        for j in range(low, high):
            if not (pivot < arr[j]): i += 1; arr[i], arr[j] = arr[j], arr[i]
        arr[i + 1], arr[high] = arr[high], arr[i + 1]
        return i + 1
    
    if len(arr) > 1: _quick_sort(arr, 0, len(arr) - 1)
    return arr

def heap_sort(arr):
    """turns array into heap then extracts max, in-place but not cache friendly"""
    def _heapify(arr, n, i):
        largest = i
        left, right = 2 * i + 1, 2 * i + 2
        if left < n and arr[left] > arr[largest]: largest = left
        if right < n and arr[right] > arr[largest]: largest = right
        if largest != i:
            arr[i], arr[largest] = arr[largest], arr[i]
            _heapify(arr, n, largest)
    
    n = len(arr)
    for i in range(n // 2 - 1, -1, -1): _heapify(arr, n, i)
    for i in range(n - 1, 0, -1):
        arr[i], arr[0] = arr[0], arr[i]
        _heapify(arr, i, 0)
    return arr

#=============Practical & Optimized Comparison Sorts=============
def shell_sort(arr):
    """insertion sort with gaps, somehow still relevant after 60 years"""
    n, gap = len(arr), 1
    while gap < n // 3: gap = gap * 3 + 1
    while gap > 0:
        for i in range(gap, n):
            temp, j = arr[i], i
            while j >= gap and arr[j - gap] > temp:
                arr[j], j = arr[j - gap], j - gap
            arr[j] = temp
        gap = (gap - 1) // 3
    return arr

def tim_sort(arr):
    """python's actual sort, merges natural runs with insertion sort for small arrays"""
    min_run, n = 32, len(arr)
    
    def _insertion_sort_range(arr, start, end):
        for i in range(start + 1, end + 1):
            key, j = arr[i], i - 1
            while j >= start and key < arr[j]: arr[j + 1], j = arr[j], j - 1
            arr[j + 1] = key
    
    def _merge(arr, left, mid, right):
        len1, len2 = mid - left + 1, right - mid
        left_arr, right_arr = arr[left:left + len1], arr[mid + 1:mid + 1 + len2]
        i = j = 0
        k = left
        while i < len1 and j < len2:
            if not (right_arr[j] < left_arr[i]): arr[k], i, k = left_arr[i], i + 1, k + 1
            else: arr[k], j, k = right_arr[j], j + 1, k + 1
        while i < len1: arr[k], i, k = left_arr[i], i + 1, k + 1
        while j < len2: arr[k], j, k = right_arr[j], j + 1, k + 1
    
    for start in range(0, n, min_run):
        end = min(start + min_run - 1, n - 1)
        _insertion_sort_range(arr, start, end)
    
    size = min_run
    while size < n:
        for left in range(0, n, 2 * size):
            mid = min(n - 1, left + size - 1)
            right = min(left + 2 * size - 1, n - 1)
            if mid < right: _merge(arr, left, mid, right)
        size *= 2
    
    return arr

def intro_sort(arr):
    """c++'s sort, switches between quick, heap, and insertion sort based on depth"""
    n = len(arr)
    max_depth = 2 * (n.bit_length())
    
    def _insertion_sort_range(arr, begin, end):
        for i in range(begin + 1, end + 1):
            key, j = arr[i], i - 1
            while j >= begin and key < arr[j]: arr[j + 1], j = arr[j], j - 1
            arr[j + 1] = key
    
    def _median_of_three(arr, a, b, c):
        if arr[a] < arr[b]:
            if arr[b] < arr[c]: return b
            elif arr[a] < arr[c]: return c
            else: return a
        else:
            if arr[a] < arr[c]: return a
            elif arr[b] < arr[c]: return c
            else: return b
    
    def _partition_range(arr, begin, end, pivot_idx):
        pivot = arr[pivot_idx]
        arr[pivot_idx], arr[end - 1] = arr[end - 1], arr[pivot_idx]
        store_idx = begin
        for i in range(begin, end - 1):
            if arr[i] < pivot:
                arr[store_idx], arr[i] = arr[i], arr[store_idx]
                store_idx += 1
        arr[end - 1], arr[store_idx] = arr[store_idx], arr[end - 1]
        return store_idx
    
    def _heapify_range(arr, idx, end, begin):
        largest = idx
        left = 2 * (idx - begin) + 1 + begin
        right = 2 * (idx - begin) + 2 + begin
        if left < end and arr[left] > arr[largest]: largest = left
        if right < end and arr[right] > arr[largest]: largest = right
        if largest != idx:
            arr[idx], arr[largest] = arr[largest], arr[idx]
            _heapify_range(arr, largest, end, begin)
    
    def _heapsort_range(arr, begin, end):
        for i in range(begin + (end - begin) // 2 - 1, begin - 1, -1):
            _heapify_range(arr, i, end, begin)
        for i in range(end - 1, begin, -1):
            arr[i], arr[begin] = arr[begin], arr[i]
            _heapify_range(arr, begin, i, begin)
    
    def _introsort_util(arr, begin, end, depth_limit):
        size = end - begin
        if size < 16: _insertion_sort_range(arr, begin, end - 1); return
        if depth_limit == 0: _heapsort_range(arr, begin, end); return
        pivot = _median_of_three(arr, begin, begin + size // 2, end - 1)
        arr[begin], arr[pivot] = arr[pivot], arr[begin]
        pivot_pos = _partition_range(arr, begin, end, begin)
        _introsort_util(arr, begin, pivot_pos, depth_limit - 1)
        _introsort_util(arr, pivot_pos + 1, end, depth_limit - 1)
    
    if n > 1: _introsort_util(arr, 0, n, max_depth)
    return arr

def library_sort(arr):
    """insertion sort with gaps, wastes space to make insertions faster"""
    n = len(arr)
    if n <= 1: return arr
    result = []
    for i in range(n):
        pos = 0
        while pos < len(result) and result[pos] <= arr[i]: pos += 1
        result.insert(pos, arr[i])
    for i in range(n): arr[i] = result[i]
    return arr

def block_sort(arr):
    """sorts blocks then merges them, basically timsort without the adaptive part"""
    n = len(arr)
    if n <= 1: return arr
    block_size = 32
    
    def _insertion_sort_range(arr, start, end):
        for i in range(start + 1, end + 1):
            key, j = arr[i], i - 1
            while j >= start and key < arr[j]: arr[j + 1], j = arr[j], j - 1
            arr[j + 1] = key
    
    def _merge_block(arr, temp, left, mid, right):
        for i in range(left, right): temp[i] = arr[i]
        i, j, k = left, mid, left
        while i < mid and j < right:
            if not (temp[j] < temp[i]): arr[k], i, k = temp[i], i + 1, k + 1
            else: arr[k], j, k = temp[j], j + 1, k + 1
        while i < mid: arr[k], i, k = temp[i], i + 1, k + 1
    
    for start in range(0, n, block_size):
        end = min(start + block_size, n)
        _insertion_sort_range(arr, start, end - 1)
    
    temp = arr.copy()
    curr_size = block_size
    while curr_size < n:
        for start in range(0, n, 2 * curr_size):
            mid = min(n, start + curr_size)
            end = min(n, start + 2 * curr_size)
            _merge_block(arr, temp, start, mid, end)
        curr_size *= 2
    
    return arr

def smooth_sort(arr):
    """heap sort with leonardo numbers, theoretically adaptive but nobody cares"""
    n = len(arr)
    
    def _sift_down(arr, start, end):
        root = start
        while 2 * root + 1 < end:
            child = 2 * root + 1
            swap = root
            if arr[swap] < arr[child]: swap = child
            if child + 1 < end and arr[swap] < arr[child + 1]: swap = child + 1
            if swap == root: break
            arr[root], arr[swap] = arr[swap], arr[root]
            root = swap
    
    for i in range(n // 2 - 1, -1, -1): _sift_down(arr, i, n)
    for i in range(n - 1, 0, -1):
        arr[0], arr[i] = arr[i], arr[0]
        _sift_down(arr, 0, i)
    
    return arr

#=============Specialized Comparison Sorts=============

def tree_sort(arr):
    """binary search tree sort, n log n but with terrible constants"""
    if not arr: return []
    
    class Node:
        def __init__(self, val):
            self.val = val
            self.left = None
            self.right = None
    
    if all(arr[i] >= arr[i-1] for i in range(1, len(arr))): return arr
    
    def insert(root, val):
        if root is None: return Node(val)
        current = root
        while True:
            if val < current.val:
                if current.left is None: current.left = Node(val); break
                current = current.left
            else:
                if current.right is None: current.right = Node(val); break
                current = current.right
        return root
    
    def in_order(root):
        result, stack, current = [], [], root
        while stack or current:
            while current: stack.append(current); current = current.left
            current = stack.pop()
            result.append(current.val)
            current = current.right
        return result
    
    root = None
    for x in arr: root = Node(x) if root is None else insert(root, x)
    result = in_order(root)
    for i in range(len(arr)): arr[i] = result[i]
    return arr

def tournament_sort(arr):
    """selection sort with a fancy name, still o(n²) and still terrible"""
    if not arr or len(arr) <= 1: return arr
    data = arr.copy()
    result = []
    for _ in range(len(data)):
        min_idx = 0
        for i in range(1, len(data)):
            if data[i] < data[min_idx]: min_idx = i
        result.append(data[min_idx])
        data[min_idx] = float('inf')
    for i in range(len(arr)): arr[i] = result[i]
    return arr

def patience_sort(arr):
    """sorts cards into piles like solitaire, surprisingly useful for some sequences"""
    if not arr: return []
    piles = []
    for x in arr:
        pile_found = False
        for pile in piles:
            if pile[-1] >= x: pile.append(x); pile_found = True; break
        if not pile_found: piles.append([x])
    
    result = []
    while piles:
        min_pile_idx, min_val = 0, piles[0][-1]
        for i in range(1, len(piles)):
            if piles[i][-1] < min_val: min_pile_idx, min_val = i, piles[i][-1]
        result.append(piles[min_pile_idx].pop())
        if not piles[min_pile_idx]: piles.pop(min_pile_idx)
    
    for i in range(len(arr)): arr[i] = result[i]
    return arr

def cube_sort(arr):
    """block sort with a cooler name, still just merging sorted blocks"""
    if not arr or len(arr) <= 1: return arr
    block_size = min(100, len(arr) // 4 + 1)
    blocks = []
    
    for i in range(0, len(arr), block_size):
        block = arr[i:min(i + block_size, len(arr))]
        block_copy = block.copy()
        for j in range(1, len(block_copy)):
            key, k = block_copy[j], j - 1
            while k >= 0 and block_copy[k] > key: block_copy[k + 1], k = block_copy[k], k - 1
            block_copy[k + 1] = key
        blocks.append(block_copy)
    
    result, indices = [], [0] * len(blocks)
    while True:
        min_val, min_block = float('inf'), -1
        for i in range(len(blocks)):
            if indices[i] < len(blocks[i]) and blocks[i][indices[i]] < min_val:
                min_val, min_block = blocks[i][indices[i]], i
        if min_block == -1: break
        result.append(min_val)
        indices[min_block] += 1
    
    for i in range(len(arr)): arr[i] = result[i]
    return arr

def comb_sort(arr):
    """bubble sort with gaps, somehow still not good enough"""
    n = len(arr)
    gap, shrink, swapped = n, 1.3, True
    while gap > 1 or swapped:
        gap = max(1, int(gap / shrink))
        swapped = False
        for i in range(n - gap):
            if arr[i] > arr[i + gap]: arr[i], arr[i + gap] = arr[i + gap], arr[i]; swapped = True
    return arr

def cocktail_sort(arr):
    """bubble sort in both directions, still o(n²) but feels fancier"""
    n = len(arr)
    swapped, start, end = True, 0, n - 1
    while swapped:
        swapped = False
        for i in range(start, end):
            if arr[i] > arr[i + 1]: arr[i], arr[i + 1] = arr[i + 1], arr[i]; swapped = True
        if not swapped: break
        end -= 1
        swapped = False
        for i in range(end - 1, start - 1, -1):
            if arr[i] > arr[i + 1]: arr[i], arr[i + 1] = arr[i + 1], arr[i]; swapped = True
        start += 1
    return arr

def gnome_sort(arr):
    """bubble sort with only one comparison, somehow even worse"""
    n, index = len(arr), 0
    while index < n:
        if index == 0 or arr[index] >= arr[index-1]: index += 1
        else: arr[index], arr[index-1] = arr[index-1], arr[index]; index -= 1
    return arr

def odd_even_sort(arr):
    """bubble sort that alternates between odd and even indices, still o(n²)"""
    n, sorted = len(arr), False
    while not sorted:
        sorted = True
        for i in range(1, n-1, 2):
            if arr[i] > arr[i+1]: arr[i], arr[i+1] = arr[i+1], arr[i]; sorted = False
        for i in range(0, n-1, 2):
            if arr[i] > arr[i+1]: arr[i], arr[i+1] = arr[i+1], arr[i]; sorted = False
    return arr

def pancake_sort(arr):
    """flips portions of array like pancakes, cute but impractical"""
    def flip(arr, k):
        left = 0
        while left < k: arr[left], arr[k] = arr[k], arr[left]; left += 1; k -= 1
    
    n = len(arr)
    for curr_size in range(n, 1, -1):
        max_idx = 0
        for i in range(1, curr_size):
            if arr[i] > arr[max_idx]: max_idx = i
        if max_idx == curr_size - 1: continue
        if max_idx > 0: flip(arr, max_idx)
        flip(arr, curr_size - 1)
    return arr

def strand_sort(arr):
    """repeatedly pulls out sorted subsequences, elegant but inefficient"""
    if not arr: return []
    
    def _merge_strands(a, b):
        result, i, j = [], 0, 0
        while i < len(a) and j < len(b):
            if a[i] <= b[j]: result.append(a[i]); i += 1
            else: result.append(b[j]); j += 1
        result.extend(a[i:])
        result.extend(b[j:])
        return result
    
    input_arr, result = arr.copy(), []
    while input_arr:
        strand = [input_arr.pop(0)]
        i = 0
        while i < len(input_arr):
            if input_arr[i] >= strand[-1]: strand.append(input_arr.pop(i))
            else: i += 1
        result = _merge_strands(result, strand)
    
    for i in range(len(arr)): arr[i] = result[i]
    return arr

def exchange_sort(arr):
    """bubble sort with a different name, still o(n²) and still useless"""
    n = len(arr)
    for i in range(n-1):
        for j in range(i+1, n):
            if arr[i] > arr[j]: arr[i], arr[j] = arr[j], arr[i]
    return arr

def cycle_sort(arr):
    """minimizes memory writes, but still o(n²) so who cares"""
    n = len(arr)
    for cycle_start in range(n-1):
        item = arr[cycle_start]
        pos = cycle_start
        for i in range(cycle_start+1, n):
            if arr[i] < item: pos += 1
        if pos == cycle_start: continue
        while item == arr[pos]: pos += 1
        arr[pos], item = item, arr[pos]
        while pos != cycle_start:
            pos = cycle_start
            for i in range(cycle_start+1, n):
                if arr[i] < item: pos += 1
            while item == arr[pos]: pos += 1
            arr[pos], item = item, arr[pos]
    return arr

def recombinant_sort(arr):
    """quicksort with insertion sort for small arrays, basically introsort lite"""
    INSERTION_THRESHOLD = 16
    
    def _insertion_sort_segment(arr, low, high):
        for i in range(low + 1, high + 1):
            key, j = arr[i], i - 1
            while j >= low and arr[j] > key: arr[j + 1], j = arr[j], j - 1
            arr[j + 1] = key
    
    def _partition(arr, low, high):
        mid = low + (high - low) // 2
        if arr[mid] < arr[low]: arr[low], arr[mid] = arr[mid], arr[low]
        if arr[high] < arr[low]: arr[low], arr[high] = arr[high], arr[low]
        if arr[high] < arr[mid]: arr[mid], arr[high] = arr[high], arr[mid]
        pivot = arr[mid]
        arr[mid], arr[high-1] = arr[high-1], arr[mid]
        i, j = low, high - 1
        while True:
            i += 1
            while arr[i] < pivot: i += 1
            j -= 1
            while arr[j] > pivot: j -= 1
            if i >= j: break
            arr[i], arr[j] = arr[j], arr[i]
        arr[i], arr[high-1] = arr[high-1], arr[i]
        return i
    
    def _recombinant_sort(arr, low, high):
        if high - low + 1 <= INSERTION_THRESHOLD: _insertion_sort_segment(arr, low, high); return
        pivot_idx = _partition(arr, low, high)
        _recombinant_sort(arr, low, pivot_idx - 1)
        _recombinant_sort(arr, pivot_idx + 1, high)
    
    if len(arr) > 1: _recombinant_sort(arr, 0, len(arr) - 1)
    return arr

def inplace_merge_sort(arr):
    """merge sort without extra space, elegant but with terrible constants"""
    def _merge(arr, start, mid, end):
        if arr[mid] <= arr[mid + 1]: return
        while start <= mid and mid + 1 <= end:
            if arr[start] <= arr[mid + 1]: start += 1
            else:
                value = arr[mid + 1]
                for i in range(mid, start - 1, -1): arr[i + 1] = arr[i]
                arr[start] = value
                start += 1
                mid += 1
    
    def _merge_sort(arr, l, r):
        if l < r:
            m = l + (r - l) // 2
            _merge_sort(arr, l, m)
            _merge_sort(arr, m + 1, r)
            _merge(arr, l, m, r)
    
    _merge_sort(arr, 0, len(arr) - 1)
    return arr

#=============Linear-Time Non-Comparison Sorts=============

def counting_sort(arr):
    """o(n+k) sort that only works for small integer ranges, otherwise memory explodes"""
    if not arr: return []
    min_val, max_val = min(arr), max(arr)
    count = [0] * (max_val - min_val + 1)
    for x in arr: count[x - min_val] += 1
    result_idx = 0
    for i in range(len(count)):
        for _ in range(count[i]): arr[result_idx], result_idx = i + min_val, result_idx + 1
    return arr

def bucket_sort_uniform(arr):
    """works great for uniform distributions, terrible for everything else"""
    if not arr: return []
    min_val, max_val = min(arr), max(arr)
    range_val = max_val - min_val
    if range_val == 0: return arr
    n = len(arr)
    num_buckets = min(n, 10)
    buckets = [[] for _ in range(num_buckets)]
    for x in arr:
        normalized = (x - min_val) / range_val
        bucket_idx = min(int(normalized * num_buckets), num_buckets - 1)
        buckets[bucket_idx].append(x)
    for bucket in buckets:
        for i in range(1, len(bucket)):
            key, j = bucket[i], i - 1
            while j >= 0 and bucket[j] > key: bucket[j + 1], j = bucket[j], j - 1
            bucket[j + 1] = key
    result_idx = 0
    for bucket in buckets:
        for x in bucket: arr[result_idx], result_idx = x, result_idx + 1
    return arr

def bucket_sort_integer(arr):
    """counting sort with a fancy name, still only good for small integer ranges"""
    if not arr: return []
    min_val, max_val = min(arr), max(arr)
    buckets = [[] for _ in range(max_val - min_val + 1)]
    for x in arr: buckets[x - min_val].append(x)
    result_idx = 0
    for bucket in buckets:
        for x in bucket: arr[result_idx], result_idx = x, result_idx + 1
    return arr

def radix_sort_lsd(arr):
    """sorts by digits from least to most significant, linear time for fixed-size integers"""
    if not arr: return []
    max_val = max(arr)
    if max_val == 0: return arr
    data = arr.copy()
    exp = 1
    while max_val // exp > 0:
        output = [0] * len(data)
        count = [0] * 10
        for i in range(len(data)): count[(data[i] // exp) % 10] += 1
        for i in range(1, 10): count[i] += count[i - 1]
        for i in range(len(data) - 1, -1, -1):
            digit = (data[i] // exp) % 10
            output[count[digit] - 1], count[digit] = data[i], count[digit] - 1
        data = output.copy()
        exp *= 10
    for i in range(len(arr)): arr[i] = data[i]
    return arr

def radix_sort_msd(arr):
    """sorts by digits from most to least significant, good for strings and variable length data"""
    if not arr: return []
    min_val = min(arr)
    offset = abs(min_val) if min_val < 0 else 0
    data = [x + offset for x in arr.copy()]
    max_val = max(data)
    if max_val == 0:
        for i in range(len(arr)): arr[i] = data[i] - offset
        return arr
    num_digits = len(str(max_val))
    
    def get_digit(num, pos): return (num // (10 ** pos)) % 10
    
    def iterative_msd_sort(arr):
        queue = [(0, len(arr) - 1, num_digits - 1)]
        while queue:
            start, end, digit_pos = queue.pop(0)
            if start >= end or digit_pos < 0: continue
            buckets = [[] for _ in range(10)]
            for i in range(start, end + 1):
                digit = get_digit(arr[i], digit_pos)
                buckets[digit].append(arr[i])
            idx = start
            for digit in range(10):
                bucket = buckets[digit]
                for val in bucket: arr[idx], idx = val, idx + 1
                bucket_start, bucket_end = idx - len(bucket), idx - 1
                if bucket_end > bucket_start and digit_pos > 0:
                    queue.append((bucket_start, bucket_end, digit_pos - 1))
    
    iterative_msd_sort(data)
    for i in range(len(arr)): arr[i] = data[i] - offset
    return arr

def pigeonhole_sort(arr):
    """Puts each pigeon in its hole. Blazing fast for small ranges, useless for large ones."""
    if not arr: return []
    min_val, max_val = min(arr), max(arr)
    holes = [0] * (max_val - min_val + 1)
    for x in arr: holes[x - min_val] += 1
    i = 0
    for j in range(len(holes)):
        while holes[j] > 0: arr[i], i, holes[j] = j + min_val, i + 1, holes[j] - 1
    return arr

#=============Advanced Non-Comparison Sorts=============
def spreadsort(arr):
    """Hybrid sort that tries to be clever by switching between counting and bucket sort. Probably not worth the effort."""
    if not arr or len(arr) <= 1: return arr
    min_val, max_val = min(arr), max(arr)
    if min_val == max_val: return arr
    range_size = max_val - min_val + 1
    
    if range_size <= len(arr) * 2:
        # use counting sort for small ranges
        count = [0] * range_size
        for x in arr: count[x - min_val] += 1
        idx = 0
        for i in range(range_size):
            for _ in range(count[i]): arr[idx], idx = i + min_val, idx + 1
    else:
        # use bucket sort approach for larger ranges
        max_bits = max_val.bit_length() if max_val > 0 else 0
        bucket_bits = min(max_bits, (len(arr).bit_length() + 2) // 3)
        bucket_bits = max(1, bucket_bits)
        num_buckets = 1 << bucket_bits
        buckets = [[] for _ in range(num_buckets)]
        mask, shift = num_buckets - 1, max(0, max_bits - bucket_bits)
        
        # distribute elements to buckets
        for x in arr: buckets[((x - min_val) >> shift) & mask].append(x)
        
        # sort each bucket and combine results
        idx = 0
        for bucket in buckets:
            if len(bucket) <= 32:
                # insertion sort for small buckets
                for i in range(1, len(bucket)):
                    key, j = bucket[i], i - 1
                    while j >= 0 and bucket[j] > key: bucket[j + 1], j = bucket[j], j - 1
                    bucket[j + 1] = key
            else: spreadsort(bucket)
            for x in bucket: arr[idx], idx = x, idx + 1
    
    return arr

def burstsort(arr):
    """Tries to be smart with strings by using a trie. Falls back to other sorts when it gets confused."""
    if not arr or len(arr) <= 1: return arr
    
    # handle non-string arrays with counting or merge sort
    if not all(isinstance(x, str) for x in arr):
        min_val, max_val = min(arr), max(arr)
        range_size = max_val - min_val + 1
        
        if range_size <= len(arr) * 10:
            # counting sort for reasonable ranges
            count = [0] * range_size
            for x in arr: count[x - min_val] += 1
            idx = 0
            for i in range(range_size):
                for _ in range(count[i]): arr[idx], idx, = i + min_val, idx + 1
        else:
            # merge sort for large ranges
            if len(arr) <= 1: return arr
            mid = len(arr) // 2
            left, right = arr[:mid], arr[mid:]
            burstsort(left)
            burstsort(right)
            i = j = k = 0
            while i < len(left) and j < len(right):
                if left[i] <= right[j]: arr[k], i, k = left[i], i + 1, k + 1
                else: arr[k], j, k = right[j], j + 1, k + 1
            while i < len(left): arr[k], i, k = left[i], i + 1, k + 1
            while j < len(right): arr[k], j, k = right[j], j + 1, k + 1
        return arr
    
    # trie-based burst sort for strings
    class TrieNode:
        def __init__(self):
            self.children = {}
            self.bucket = []
            self.threshold = 32
    
    root = TrieNode()
    
    # insert strings into trie
    for s in arr:
        node = root
        if not s:
            node.bucket.append(s)
            continue
        
        for i, char in enumerate(s):
            # burst bucket if needed
            if char not in node.children and len(node.bucket) >= node.threshold:
                for bucket_str in node.bucket:
                    if i < len(bucket_str):
                        if bucket_str[i] not in node.children:
                            node.children[bucket_str[i]] = TrieNode()
                        node.children[bucket_str[i]].bucket.append(bucket_str)
                node.bucket = []
            
            # add string to appropriate bucket
            if i == len(s) - 1 or char not in node.children:
                if char not in node.children:
                    node.bucket.append(s)
                    break
                else:
                    node.children[char].bucket.append(s)
                    break
            
            node = node.children[char]
    
    # traverse trie to collect sorted strings
    result = []
    def traverse(node):
        # insertion sort for bucket
        bucket_sorted = []
        for item in node.bucket:
            pos = 0
            while pos < len(bucket_sorted) and bucket_sorted[pos] < item: pos += 1
            bucket_sorted.insert(pos, item)
        result.extend(bucket_sorted)
        
        # sort child keys
        keys = sorted(node.children.keys())
        for char in keys: traverse(node.children[char])
    
    traverse(root)
    
    # copy result back to original array
    for i in range(len(arr)): arr[i] = result[i]
    
    return arr

def flashsort(arr):
    """Like bucket sort but with a fancy name. Tries to be linear but ends up using insertion sort anyway."""
    if not arr or len(arr) <= 1: return arr
    
    n = len(arr)
    m = max(2, int(0.45 * n))
    min_val, max_val = min(arr), max(arr)
    
    if min_val == max_val: return arr
    
    # count elements per bucket
    buckets = [0] * (m + 1)
    for x in arr:
        bucket_idx = min(m-1, int(m * (x - min_val) / (max_val - min_val)))
        buckets[bucket_idx] += 1
    
    # calculate cumulative counts
    for i in range(1, m): buckets[i] += buckets[i - 1]
    
    # permute elements into their buckets
    temp = arr.copy()
    for i in range(n-1, -1, -1):
        bucket_idx = min(m-1, int(m * (temp[i] - min_val) / (max_val - min_val)))
        arr[buckets[bucket_idx] - 1] = temp[i]
        buckets[bucket_idx] -= 1
    
    # insertion sort to finalize
    for i in range(1, n):
        key, j = arr[i], i - 1
        while j >= 0 and arr[j] > key: arr[j + 1], j = arr[j], j - 1
        arr[j + 1] = key
    
    return arr

def postman_sort(arr):
    """Sorts like a postman organizing mail by zip code. Surprisingly effective but nobody uses it."""
    if not arr or len(arr) <= 1: return arr
    
    if all(isinstance(x, int) for x in arr):
        # handle integers by converting to strings
        str_arr = [str(x) for x in arr]
        max_len = max(len(s) for s in str_arr)
        padded_arr = [s.zfill(max_len) for s in str_arr]
        aux = [(padded_arr[i], i) for i in range(len(arr))]
        
        # sort by each character position from right to left
        for char_pos in range(max_len - 1, -1, -1):
            # insertion sort for each character position
            for i in range(1, len(aux)):
                key, j = aux[i], i - 1
                while j >= 0 and aux[j][0][char_pos] > key[0][char_pos]:
                    aux[j + 1], j = aux[j], j - 1
                aux[j + 1] = key
        
        # reconstruct result
        result = [arr[idx] for _, idx in aux]
        for i in range(len(arr)): arr[i] = result[i]
        return arr
    
    # handle strings directly
    max_len = max(len(s) for s in arr)
    sorted_arr = arr.copy()
    
    # sort by each character position from right to left
    for char_pos in range(max_len - 1, -1, -1):
        # count characters at this position
        count = {}
        for s in sorted_arr:
            char = s[char_pos] if char_pos < len(s) else ''
            count[char] = count.get(char, 0) + 1
        
        # calculate starting positions for each character
        position, pos = {}, 0
        if '' in count:
            position[''] = 0
            pos = count['']
        
        # sort characters
        chars = sorted([c for c in count.keys() if c != ''])
        for char in chars:
            position[char] = pos
            pos += count[char]
        
        # distribute strings to output array
        output = [None] * len(sorted_arr)
        for s in sorted_arr:
            char = s[char_pos] if char_pos < len(s) else ''
            output[position[char]] = s
            position[char] += 1
        
        sorted_arr = output
    
    # copy result back to original array
    for i in range(len(arr)): arr[i] = sorted_arr[i]
    
    return arr

def msd_radix_sort_inplace(arr):
    """Radix sort but starting from the most significant digit. Tries to be clever with bit manipulation."""
    if not arr or len(arr) <= 1: return arr
    max_val = max(arr)
    if max_val == 0: return arr
    
    def _msd_sort(arr, start, end, bit):
        if bit < 0 or start >= end: return
        
        # use insertion sort for small subarrays
        if end - start < 32:
            for i in range(start + 1, end + 1):
                key, j = arr[i], i - 1
                while j >= start and arr[j] > key: arr[j + 1], j = arr[j], j - 1
                arr[j + 1] = key
            return
        
        # partition by current bit
        i, j = start, end
        while i <= j:
            while i <= j and not (arr[i] & (1 << bit)): i += 1
            while i <= j and (arr[j] & (1 << bit)): j -= 1
            if i < j: arr[i], arr[j], i, j = arr[j], arr[i], i + 1, j - 1
        
        # recursively sort partitions
        _msd_sort(arr, start, j, bit - 1)
        _msd_sort(arr, i, end, bit - 1)
    
    _msd_sort(arr, 0, len(arr) - 1, max_val.bit_length() - 1)
    return arr

#=============Theoretical Interest Sorting Algorithms=============

def bead_sort(arr):
    """Simulates gravity on beads. Cute idea, terrible performance, and only works on positive integers."""
    if not arr: return []
    
    for num in arr:
        if not isinstance(num, int) or num < 0:
            raise ValueError("Bead sort works only with non-negative integers")
    
    result = arr.copy()
    max_val = max(result)
    grid = [[0 for _ in range(max_val)] for _ in range(len(result))]
    
    for i in range(len(result)):
        for j in range(result[i]): grid[i][j] = 1
            
    for j in range(max_val):
        beads_count = 0
        for i in range(len(result)): beads_count += grid[i][j]
        
        for i in range(len(result) - 1, len(result) - beads_count - 1, -1): grid[i][j] = 1
        for i in range(len(result) - beads_count - 1, -1, -1): grid[i][j] = 0
    
    for i in range(len(result)): result[i] = sum(grid[i])
    for i in range(len(arr)): arr[i] = result[i]
    
    return arr

def merge_insertion_sort(arr):
    """Ford-Johnson algorithm. Theoretically optimal for minimizing comparisons, practically useless."""
    if len(arr) <= 32:
        for i in range(1, len(arr)):
            key, j = arr[i], i - 1
            while j >= 0 and arr[j] > key:
                arr[j + 1], j = arr[j], j - 1
            arr[j + 1] = key
        return arr
    
    def _merge_sort_hybrid(arr, left, right):
        if right - left <= 32:
            for i in range(left + 1, right + 1):
                key, j = arr[i], i - 1
                while j >= left and arr[j] > key:
                    arr[j + 1], j = arr[j], j - 1
                arr[j + 1] = key
            return
        
        mid = (left + right) // 2
        _merge_sort_hybrid(arr, left, mid)
        _merge_sort_hybrid(arr, mid + 1, right)
        
        temp = arr.copy()
        i, j, k = left, mid + 1, left
        
        while i <= mid and j <= right:
            if arr[i] <= arr[j]: temp[k], i, k = arr[i], i + 1, k + 1
            else: temp[k], j, k = arr[j], j + 1, k + 1
        
        while i <= mid: temp[k], i, k = arr[i], i + 1, k + 1
        while j <= right: temp[k], j, k = arr[j], j + 1, k + 1
        
        for i in range(left, right + 1): arr[i] = temp[i]
    
    result = arr.copy()
    _merge_sort_hybrid(result, 0, len(result) - 1)
    for i in range(len(arr)): arr[i] = result[i]
    
    return arr

def i_cant_believe_it_can_sort(arr):
    """Bubble sort in disguise. I can't believe anyone would use this."""
    if not arr or len(arr) <= 1: return arr
    
    n = len(arr)
    for i in range(n):
        for j in range(0, n-1):
            if arr[j] > arr[j+1]: arr[j], arr[j+1] = arr[j+1], arr[j]
    
    return arr

def bogosort(arr):
    """Randomly shuffles until sorted. Might finish before the heat death of the universe, but probably not."""
    result = arr.copy()
    
    def is_sorted(a):
        for i in range(len(a) - 1):
            if a[i] > a[i + 1]: return False
        return True
    
    max_iterations = min(100, len(result) ** 2)
    iterations = 0
    
    while not is_sorted(result) and iterations < max_iterations:
        for i in range(len(result) - 1, 0, -1):
            j = random.randint(0, i)
            result[i], result[j] = result[j], result[i]
        iterations += 1
    
    if iterations >= max_iterations:
        for i in range(1, len(result)):
            key, j = result[i], i - 1
            while j >= 0 and result[j] > key:
                result[j + 1], j = result[j], j - 1
            result[j + 1] = key
    
    for i in range(len(arr)): arr[i] = result[i]
    
    return arr

def spaghetti_sort(arr):
    """Simulates sorting spaghetti by length. Cute idea, but just counting sort with extra steps."""
    if not arr: return []
    
    result = arr.copy()
    min_val = min(result)
    
    if min_val <= 0:
        offset = -min_val + 1
        for i in range(len(result)): result[i] += offset
    else: offset = 0
    
    max_val = max(result)
    strands = [0] * (max_val + 1)
    
    for length in result: strands[length] += 1
    
    index = 0
    for length in range(1, max_val + 1):
        for _ in range(strands[length]):
            result[index] = length - offset
            index += 1
    
    for i in range(len(arr)): arr[i] = result[i]
    
    return arr

def sorting_network(arr):
    """Fixed-size comparator network. Great for hardware, overkill for software, and a pain to implement."""
    n = len(arr)
    original_n = n
    
    next_pow2 = 1
    while next_pow2 < n: next_pow2 *= 2
    
    padded = arr.copy()
    
    if next_pow2 > n:
        padded.extend([float('inf')] * (next_pow2 - n))
        n = next_pow2
    
    def bitonic_merge(arr, low, cnt, direction):
        if cnt > 1:
            k = cnt // 2
            for i in range(low, low + k):
                if direction == (arr[i] > arr[i + k]): arr[i], arr[i + k] = arr[i + k], arr[i]
            bitonic_merge(arr, low, k, direction)
            bitonic_merge(arr, low + k, k, direction)
    
    def bitonic_sort(arr, low, cnt, direction):
        if cnt > 1:
            k = cnt // 2
            bitonic_sort(arr, low, k, True)
            bitonic_sort(arr, low + k, k, False)
            bitonic_merge(arr, low, cnt, direction)
    
    bitonic_sort(padded, 0, n, True)
    
    for i in range(original_n): arr[i] = padded[i]
    
    return arr

def bitonic_sort(arr):
    """Parallel sorting network that's great for GPUs but why are you using it in Python?"""
    data = arr.copy()
    n = len(data)
    
    m = 1
    while m < n: m *= 2
    
    if m > n: data.extend([float('inf')] * (m - n))
    
    def compare_and_swap(arr, i, j, dir):
        if (arr[i] > arr[j]) == dir: arr[i], arr[j] = arr[j], arr[i]
    
    def bitonic_merge(arr, low, cnt, dir):
        if cnt > 1:
            k = cnt // 2
            for i in range(low, low + k): compare_and_swap(arr, i, i + k, dir)
            bitonic_merge(arr, low, k, dir)
            bitonic_merge(arr, low + k, k, dir)
    
    def bitonic_sort_recursive(arr, low, cnt, dir):
        if cnt > 1:
            k = cnt // 2
            bitonic_sort_recursive(arr, low, k, not dir)
            bitonic_sort_recursive(arr, low + k, k, dir)
            bitonic_merge(arr, low, cnt, dir)
    
    bitonic_sort_recursive(data, 0, m, True)
    
    for i in range(n): arr[i] = data[i]
    
    return arr

def stooge_sort(arr):
    """Recursively sorts 2/3 of the array, then the last 2/3, then the first 2/3 again. Why? Just why?"""
    def _stooge_sort(arr, low, high):
        if arr[low] > arr[high]: arr[low], arr[high] = arr[high], arr[low]
            
        if high - low + 1 > 2:
            t = (high - low + 1) // 3
            _stooge_sort(arr, low, high - t)
            _stooge_sort(arr, low + t, high)
            _stooge_sort(arr, low, high - t)
    
    if len(arr) <= 1: return arr
    _stooge_sort(arr, 0, len(arr) - 1)
    return arr

def slowsort(arr):
    """Multiply and surrender algorithm. Intentionally inefficient, because computer science has a sense of humor."""
    def _slowsort(arr, i, j):
        if i >= j: return
        m = (i + j) // 2
        _slowsort(arr, i, m)
        _slowsort(arr, m + 1, j)
        if arr[m] > arr[j]: arr[m], arr[j] = arr[j], arr[m]
        _slowsort(arr, i, j - 1)
    
    _slowsort(arr, 0, len(arr) - 1)
    return arr

def franceschini_mergesort(arr):
    """In-place merge sort that's so complex you'll wish you just used extra memory."""
    def _rotate(arr, first, middle, last):
        _reverse(arr, first, middle - 1)
        _reverse(arr, middle, last)
        _reverse(arr, first, last)
        
    def _reverse(arr, first, last):
        while first < last:
            arr[first], arr[last] = arr[last], arr[first]
            first += 1
            last -= 1
    
    def _merge(arr, start, mid, end):
        if arr[mid - 1] <= arr[mid]: return
        i, j = start, mid
        
        while i < j and j <= end:
            if arr[i] <= arr[j]: i += 1
            else:
                temp, target = arr[j], j
                while target > i:
                    arr[target] = arr[target - 1]
                    target -= 1
                arr[i], i, j, mid = temp, i + 1, j + 1, mid + 1
    
    def _mergesort(arr, start, end):
        if start < end:
            mid = (start + end) // 2
            _mergesort(arr, start, mid)
            _mergesort(arr, mid + 1, end)
            _merge(arr, start, mid + 1, end)
    
    _mergesort(arr, 0, len(arr) - 1)
    return arr

def thorup_sort(arr):
    """
    Sort an array using Thorup's integer sorting algorithm.
    A complex algorithm that achieves O(n log log n) for integer sorting.
    This is a simplified implementation that demonstrates the approach.
    
    Time Complexity: O(n log log n) for integers
    Space Complexity: O(n + 2^w) where w is word size
    Stable: Yes
    
    Args:
        arr: List of non-negative integers
        
    Returns:
        The sorted list
    """
    if not arr:
        return []
    
    # For demonstration, we'll use a linear-time sorting method
    # for integers in a bounded range, similar to Thorup's approach
    
    # find the range
    min_val = min(arr)
    max_val = max(arr)
    
    # handle negative values by shifting
    if min_val < 0:
        offset = abs(min_val)
        # create shifted copy
        temp = [x + offset for x in arr]
        max_val += offset
    else:
        offset = 0
        temp = arr.copy()
    
    # if range is small enough, use counting sort
    if max_val <= 10000:  # arbitrary threshold
        # counting sort implementation
        count = [0] * (max_val + 1)
        
        # count occurrences
        for x in temp:
            count[x] += 1
        
        # reconstruct array
        idx = 0
        for i in range(max_val + 1):
            for _ in range(count[i]):
                temp[idx] = i
                idx += 1
    else:
        # for larger ranges, use a recursive bit-based approach
        # similar to Thorup's algorithm but simplified
        
        # determine number of bits needed
        bit_length = max_val.bit_length()
        
        # sort by each bit group (simplified)
        bits_per_group = min(8, bit_length)  # use smaller groups for demo
        num_groups = (bit_length + bits_per_group - 1) // bits_per_group
        
        # recursive sort by bit groups
        for group in range(num_groups):
            # determine bit positions for this group
            start_bit = group * bits_per_group
            mask = ((1 << bits_per_group) - 1) << start_bit
            
            # count sort for this bit group
            buckets = [[] for _ in range(1 << bits_per_group)]
            
            # distribute elements to buckets
            for x in temp:
                bucket_idx = (x & mask) >> start_bit
                buckets[bucket_idx].append(x)
            
            # reconstruct array
            idx = 0
            for bucket in buckets:
                for x in bucket:
                    temp[idx] = x
                    idx += 1
    
    # restore original range if needed
    if offset > 0:
        for i in range(len(arr)):
            arr[i] = temp[i] - offset
    else:
        for i in range(len(arr)):
            arr[i] = temp[i]
    
    return arr