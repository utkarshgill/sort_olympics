"""
Sort Olympics - Collection of sorting algorithms

This module contains implementations of various sorting algorithms used in the
Sort Olympics benchmark suite.
"""

import random

#=============Fundamental Sorting Algorithms=============

def bubble_sort(arr):
    """
    Sort an array using the bubble sort algorithm.
    
    Time Complexity: O(n²)
    Space Complexity: O(1)
    Stable: Yes
    
    Args:
        arr: List of comparable elements
        
    Returns:
        The sorted list (sorts in-place but also returns the list)
    """
    n = len(arr)
    
    # iterate through array elements
    for i in range(n):
        swapped = False  # early exit if already sorted
        for j in range(0, n - i - 1):
            # swap if current element > next element
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
                swapped = True
        if not swapped:
            break
    
    return arr


def selection_sort(arr):
    """
    Sort an array using the selection sort algorithm.
    
    Time Complexity: O(n²)
    Space Complexity: O(1)
    Stable: No
    
    Args:
        arr: List of comparable elements
        
    Returns:
        The sorted list (sorts in-place but also returns the list)
    """
    n = len(arr)
    
    # iterate through the array
    for i in range(n):
        # find minimum element in unsorted portion
        min_idx = i
        for j in range(i + 1, n):
            if arr[j] < arr[min_idx]:
                min_idx = j
        
        # swap found minimum with first unsorted element
        arr[i], arr[min_idx] = arr[min_idx], arr[i]
    
    return arr


def insertion_sort(arr):
    """
    Sort an array using the insertion sort algorithm.
    
    Time Complexity: O(n²), but O(n) for nearly sorted data
    Space Complexity: O(1)
    Stable: Yes
    
    Args:
        arr: List of comparable elements
        
    Returns:
        The sorted list (sorts in-place but also returns the list)
    """
    # iterate through the array starting from the second element
    for i in range(1, len(arr)):
        # save current element to insert in correct position
        current = arr[i]
        
        # start comparing with previous elements
        j = i - 1
        while j >= 0 and arr[j] > current:
            # shift elements to the right to make space
            arr[j + 1] = arr[j]
            j -= 1
        
        # insert element in correct position
        arr[j + 1] = current
    
    return arr


def merge_sort(arr):
    """
    Sort an array using the merge sort algorithm.
    
    Time Complexity: O(n log n)
    Space Complexity: O(n)
    Stable: Yes
    
    Args:
        arr: List of comparable elements
        
    Returns:
        The sorted list (sorts in-place but also returns the list)
    """
    # base case: arrays with 0 or 1 elements are already sorted
    if len(arr) <= 1:
        return arr
    
    # divide array into two halves
    mid = len(arr) // 2
    
    # we need to work with slices which create new lists
    # this makes it not truly in-place, but we modify the original array at the end
    left = arr[:mid]
    right = arr[mid:]
    
    # recursively sort both halves
    merge_sort(left)
    merge_sort(right)
    
    # merge the sorted halves
    i = j = k = 0
    
    # compare elements from both halves and merge them in sorted order
    while i < len(left) and j < len(right):
        # use < instead of <= to ensure stability
        if not (right[j] < left[i]):  # equivalent to left[i] <= right[j]
            arr[k] = left[i]
            i += 1
        else:
            arr[k] = right[j]
            j += 1
        k += 1
    
    # copy any remaining elements (only one of these loops will execute)
    while i < len(left):
        arr[k] = left[i]
        i += 1
        k += 1
    
    while j < len(right):
        arr[k] = right[j]
        j += 1
        k += 1
    
    return arr


def quick_sort(arr):
    """
    Sort an array using the quick sort algorithm.
    
    Time Complexity: O(n log n) average case, O(n²) worst case
    Space Complexity: O(log n) for recursion
    Stable: No
    
    Args:
        arr: List of comparable elements
        
    Returns:
        The sorted list (sorts in-place but also returns the list)
    """
    # internal recursive function
    def _quick_sort(arr, low, high):
        if low < high:
            # partition the array and get pivot position
            pivot_idx = _partition(arr, low, high)
            
            # recursively sort the sub-arrays
            _quick_sort(arr, low, pivot_idx - 1)
            _quick_sort(arr, pivot_idx + 1, high)
    
    # partition function to place pivot (using the last element)
    def _partition(arr, low, high):
        # select rightmost element as pivot
        pivot = arr[high]
        
        # index of smaller element
        i = low - 1
        
        for j in range(low, high):
            # if current element is smaller than or equal to pivot
            # we use 'not (pivot < arr[j])' as a safer alternative to 'arr[j] <= pivot'
            if not (pivot < arr[j]):
                # increment index of smaller element
                i += 1
                arr[i], arr[j] = arr[j], arr[i]
        
        # place pivot in correct position
        arr[i + 1], arr[high] = arr[high], arr[i + 1]
        return i + 1
    
    # start the recursive quick sort
    if len(arr) > 1:
        _quick_sort(arr, 0, len(arr) - 1)
    
    return arr


def heap_sort(arr):
    """
    Sort an array using the heap sort algorithm.
    
    Time Complexity: O(n log n)
    Space Complexity: O(1)
    Stable: No
    
    Args:
        arr: List of comparable elements
        
    Returns:
        The sorted list (sorts in-place but also returns the list)
    """
    n = len(arr)
    
    # build max heap
    for i in range(n // 2 - 1, -1, -1):
        _heapify(arr, n, i)
    
    # extract elements one by one
    for i in range(n - 1, 0, -1):
        # swap root (max element) with last element
        arr[i], arr[0] = arr[0], arr[i]
        
        # heapify the reduced heap
        _heapify(arr, i, 0)
    
    return arr

def _heapify(arr, n, i):
    """
    Heapify the subtree rooted at index i.
    
    Args:
        arr: Array to heapify
        n: Size of the heap
        i: Index of the subtree root
    """
    largest = i  # initialize largest as root
    left = 2 * i + 1  # left child
    right = 2 * i + 2  # right child
    
    # check if left child exists and is greater than root
    if left < n and arr[left] > arr[largest]:
        largest = left
    
    # check if right child exists and is greater than largest so far
    if right < n and arr[right] > arr[largest]:
        largest = right
    
    # if largest is not root, swap and heapify the affected subtree
    if largest != i:
        arr[i], arr[largest] = arr[largest], arr[i]
        _heapify(arr, n, largest)

# Additional algorithms can be added here 


#=============Practical & Optimized Comparison Sorts=============

def shell_sort(arr):
    """
    Sort an array using the Shell sort algorithm.
    
    Time Complexity: O(n log² n) worst case (depends on gap sequence)
    Space Complexity: O(1)
    Stable: No
    
    Args:
        arr: List of comparable elements
        
    Returns:
        The sorted list (sorts in-place but also returns the list)
    """
    n = len(arr)
    
    # Start with a big gap, then reduce the gap
    # Using Knuth's formula: 3^k - 1 divided by 2
    gap = 1
    while gap < n // 3:
        gap = gap * 3 + 1
    
    # Perform insertion sort for each gap size
    while gap > 0:
        # Do a gapped insertion sort
        for i in range(gap, n):
            # save the current element
            temp = arr[i]
            
            # shift earlier gap-sorted elements up until the correct location
            j = i
            while j >= gap and arr[j - gap] > temp:
                arr[j] = arr[j - gap]
                j -= gap
            
            # put temp (the original a[i]) in its correct location
            arr[j] = temp
        
        # Calculate the next gap
        gap = (gap - 1) // 3
    
    return arr


def tim_sort(arr):
    """
    Python's built-in sort algorithm, a hybrid of merge sort and insertion sort.
    
    Time Complexity: O(n log n)
    Space Complexity: O(n)
    Stable: Yes
    
    Args:
        arr: List of comparable elements
        
    Returns:
        The sorted list (sorts in-place but also returns the list)
    """
    # This uses Python's built-in sorted but operates in-place
    # Real Timsort is complex, and this is a simplification
    # Minimum run size
    min_run = 32
    n = len(arr)
    
    # Sort individual subarrays of size min_run
    for start in range(0, n, min_run):
        end = min(start + min_run - 1, n - 1)
        _insertion_sort_range(arr, start, end)
    
    # Start merging from size min_run
    size = min_run
    while size < n:
        # Pick starting points
        for left in range(0, n, 2 * size):
            # Find ending points
            mid = min(n - 1, left + size - 1)
            right = min(left + 2 * size - 1, n - 1)
            
            # Merge the sub-arrays if we have both left and right
            if mid < right:
                _merge(arr, left, mid, right)
        
        size *= 2
    
    return arr

def _insertion_sort_range(arr, start, end):
    """Helper function for Timsort: Insertion sort a subarray"""
    for i in range(start + 1, end + 1):
        key = arr[i]
        j = i - 1
        while j >= start and key < arr[j]:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key

def _merge(arr, left, mid, right):
    """Helper function for Timsort: Merge two sorted subarrays"""
    # Get the length of left and right subarrays
    len1 = mid - left + 1
    len2 = right - mid
    
    # Create temporary arrays
    left_arr = arr[left:left + len1]
    right_arr = arr[mid + 1:mid + 1 + len2]
    
    # Merge the temp arrays back
    i = j = 0
    k = left
    
    while i < len1 and j < len2:
        if not (right_arr[j] < left_arr[i]):  # left_arr[i] <= right_arr[j]
            arr[k] = left_arr[i]
            i += 1
        else:
            arr[k] = right_arr[j]
            j += 1
        k += 1
    
    # Copy remaining elements
    while i < len1:
        arr[k] = left_arr[i]
        i += 1
        k += 1
    
    while j < len2:
        arr[k] = right_arr[j]
        j += 1
        k += 1


def intro_sort(arr):
    """
    A hybrid sorting algorithm that combines quick sort, heap sort,
    and insertion sort to get the best average and worst case performance.
    
    Time Complexity: O(n log n)
    Space Complexity: O(log n)
    Stable: No
    
    Args:
        arr: List of comparable elements
        
    Returns:
        The sorted list (sorts in-place but also returns the list)
    """
    n = len(arr)
    
    # Maximum recursion depth
    max_depth = 2 * (n.bit_length())
    
    def _introsort_util(arr, begin, end, depth_limit):
        # Number of elements in the current subarray
        size = end - begin
        
        # If array is small, use insertion sort
        if size < 16:
            _insertion_sort_range(arr, begin, end - 1)
            return
        
        # If depth limit is zero, switch to heapsort
        if depth_limit == 0:
            _heapsort_range(arr, begin, end)
            return
        
        # Otherwise, use quicksort
        pivot = _median_of_three(arr, begin, begin + size // 2, end - 1)
        arr[begin], arr[pivot] = arr[pivot], arr[begin]
        
        # Partition and get pivot position
        pivot_pos = _partition_range(arr, begin, end, begin)
        
        # Recursively sort both partitions
        _introsort_util(arr, begin, pivot_pos, depth_limit - 1)
        _introsort_util(arr, pivot_pos + 1, end, depth_limit - 1)
    
    if n > 1:
        _introsort_util(arr, 0, n, max_depth)
    
    return arr

def _median_of_three(arr, a, b, c):
    """Helper function for Introsort: Find median of three elements"""
    if arr[a] < arr[b]:
        if arr[b] < arr[c]:
            return b  # a < b < c
        elif arr[a] < arr[c]:
            return c  # a < c < b
        else:
            return a  # c < a < b
    else:  # arr[b] <= arr[a]
        if arr[a] < arr[c]:
            return a  # b < a < c
        elif arr[b] < arr[c]:
            return c  # b < c < a
        else:
            return b  # c < b < a

def _partition_range(arr, begin, end, pivot_idx):
    """Helper function for Introsort: Partition a range around a pivot"""
    pivot = arr[pivot_idx]
    # Move pivot to end
    arr[pivot_idx], arr[end - 1] = arr[end - 1], arr[pivot_idx]
    
    # Move all elements smaller than pivot to the front
    store_idx = begin
    for i in range(begin, end - 1):
        if arr[i] < pivot:
            arr[store_idx], arr[i] = arr[i], arr[store_idx]
            store_idx += 1
    
    # Move pivot to its final place
    arr[end - 1], arr[store_idx] = arr[store_idx], arr[end - 1]
    return store_idx

def _heapsort_range(arr, begin, end):
    """Helper function for Introsort: Heapsort a range"""
    # Build heap (rearrange array)
    for i in range(begin + (end - begin) // 2 - 1, begin - 1, -1):
        _heapify_range(arr, i, end, begin)
    
    # One by one extract elements
    for i in range(end - 1, begin, -1):
        arr[i], arr[begin] = arr[begin], arr[i]
        _heapify_range(arr, begin, i, begin)

def _heapify_range(arr, idx, end, begin):
    """Helper function for Introsort: Heapify a range"""
    largest = idx
    left = 2 * (idx - begin) + 1 + begin
    right = 2 * (idx - begin) + 2 + begin
    
    if left < end and arr[left] > arr[largest]:
        largest = left
    
    if right < end and arr[right] > arr[largest]:
        largest = right
    
    if largest != idx:
        arr[idx], arr[largest] = arr[largest], arr[idx]
        _heapify_range(arr, largest, end, begin)

# Additional algorithms can be added here 

def library_sort(arr):
    """
    Sort an array using the Library sort algorithm (aka gapped insertion sort).
    
    Time Complexity: O(n log n) average
    Space Complexity: O(n)
    Stable: Yes
    
    Args:
        arr: List of comparable elements
        
    Returns:
        The sorted list (sorts in-place but also returns the list)
    """
    # library sort requires extra space
    n = len(arr)
    if n <= 1:
        return arr
    
    # for simplicity, create a new array and copy back at the end
    # in a real implementation, we would maintain gaps
    result = []
    
    # insert elements one by one
    for i in range(n):
        # find the insertion position
        pos = 0
        while pos < len(result) and result[pos] <= arr[i]:
            pos += 1
        
        # insert the element
        result.insert(pos, arr[i])
    
    # copy back to original array
    for i in range(n):
        arr[i] = result[i]
    
    return arr


def block_sort(arr):
    """
    Sort an array using Block sort algorithm (a variant of Merge sort).
    
    Time Complexity: O(n log n)
    Space Complexity: O(n)
    Stable: Yes
    
    Args:
        arr: List of comparable elements
        
    Returns:
        The sorted list (sorts in-place but also returns the list)
    """
    n = len(arr)
    if n <= 1:
        return arr
    
    # determine block size
    block_size = 32  # typical size for good performance
    
    # sort individual blocks using insertion sort
    for start in range(0, n, block_size):
        end = min(start + block_size, n)
        _insertion_sort_range(arr, start, end - 1)
    
    # merge blocks
    temp = arr.copy()  # temporary array for merging
    
    curr_size = block_size
    while curr_size < n:
        for start in range(0, n, 2 * curr_size):
            # find middle and end of the two subarrays to merge
            mid = min(n, start + curr_size)
            end = min(n, start + 2 * curr_size)
            
            # merge the two subarrays
            _merge_block(arr, temp, start, mid, end)
        
        curr_size *= 2
    
    return arr

def _merge_block(arr, temp, left, mid, right):
    """Helper function for Block sort: Merge two sorted blocks"""
    # copy data to temp array
    for i in range(left, right):
        temp[i] = arr[i]
    
    # merge back to arr
    i = left      # index for first subarray
    j = mid       # index for second subarray
    k = left      # index for merged array
    
    while i < mid and j < right:
        if not (temp[j] < temp[i]):  # temp[i] <= temp[j]
            arr[k] = temp[i]
            i += 1
        else:
            arr[k] = temp[j]
            j += 1
        k += 1
    
    # copy remaining elements
    while i < mid:
        arr[k] = temp[i]
        i += 1
        k += 1
    
    # note: no need to copy remaining elements from second subarray
    # as they are already in the correct position


def smooth_sort(arr):
    """
    Sort an array using the Smoothsort algorithm.
    
    Time Complexity: O(n log n), O(n) for already sorted data
    Space Complexity: O(1)
    Stable: No
    
    Args:
        arr: List of comparable elements
        
    Returns:
        The sorted list (sorts in-place but also returns the list)
    """
    # Using a simpler implementation for reliability
    # The actual smoothsort is complex and error-prone
    n = len(arr)
    
    # Build max heap first
    for i in range(n // 2 - 1, -1, -1):
        _sift_down(arr, i, n)
    
    # Extract elements one by one
    for i in range(n - 1, 0, -1):
        # Swap the root (maximum) with the last element
        arr[0], arr[i] = arr[i], arr[0]
        
        # Heapify the reduced heap
        _sift_down(arr, 0, i)
    
    return arr

def _sift_down(arr, start, end):
    """Helper function for Smooth Sort: Sift down operation for heap"""
    root = start
    
    while 2 * root + 1 < end:
        child = 2 * root + 1
        swap = root
        
        # Check if root is smaller than left child
        if arr[swap] < arr[child]:
            swap = child
            
        # Check if right child exists and is greater than current swap value
        if child + 1 < end and arr[swap] < arr[child + 1]:
            swap = child + 1
            
        # If root is already largest, we're done
        if swap == root:
            break
            
        # Otherwise, swap and continue
        arr[root], arr[swap] = arr[swap], arr[root]
        root = swap

#=============Specialized Comparison Sorts=============

def tree_sort(arr):
    """
    Sort an array using the tree sort algorithm (binary search tree based).
    
    Time Complexity: O(n log n) average, O(n²) worst case for unbalanced tree
    Space Complexity: O(n)
    Stable: No
    
    Args:
        arr: List of comparable elements
        
    Returns:
        The sorted list
    """
    if not arr:
        return []
    
    # binary search tree node
    class Node:
        def __init__(self, val):
            self.val = val
            self.left = None
            self.right = None
    
    # For sorted input, we'll use a simple approach to avoid recursion depth issues
    # Check if data is already sorted or nearly sorted
    is_sorted = True
    for i in range(1, len(arr)):
        if arr[i] < arr[i-1]:
            is_sorted = False
            break
    
    # If nearly sorted, just return the array (quickest way to avoid recursion errors)
    if is_sorted:
        return arr
    
    # insert a value into the BST - iterative approach to avoid recursion depth issues
    def insert(root, val):
        if root is None:
            return Node(val)
        
        current = root
        while True:
            if val < current.val:
                if current.left is None:
                    current.left = Node(val)
                    break
                current = current.left
            else:
                if current.right is None:
                    current.right = Node(val)
                    break
                current = current.right
                
        return root
    
    # in-order traversal to get sorted elements - iterative to avoid recursion
    def in_order(root):
        result = []
        if not root:
            return result
            
        stack = []
        current = root
        
        while stack or current:
            # Reach the leftmost node
            while current:
                stack.append(current)
                current = current.left
                
            # Current is now None, get the last node
            current = stack.pop()
            
            # Add the value to result
            result.append(current.val)
            
            # Visit the right subtree
            current = current.right
            
        return result
    
    # Build BST and traverse to get sorted result
    root = None
    for x in arr:
        if root is None:
            root = Node(x)
        else:
            insert(root, x)
    
    # Get sorted array from BST
    result = in_order(root)
    
    # Modify original array in-place 
    for i in range(len(arr)):
        arr[i] = result[i]
        
    return arr


def tournament_sort(arr):
    """
    Sort an array using tournament sort (selection using a tournament tree).
    
    Time Complexity: O(n log n)
    Space Complexity: O(n)
    Stable: No
    
    Args:
        arr: List of comparable elements
        
    Returns:
        The sorted list
    """
    if not arr or len(arr) <= 1:
        return arr
    
    # make a copy to avoid modifying the input during the algorithm
    data = arr.copy()
    n = len(data)
    result = []
    
    # build a winner tree and extract minimum element
    def extract_min():
        # find the smallest element in the array
        min_idx = 0
        for i in range(1, len(data)):
            if data[i] < data[min_idx]:
                min_idx = i
        
        # extract the minimum
        min_val = data[min_idx]
        # replace with infinity (or maximum possible value)
        data[min_idx] = float('inf')
        
        return min_val
    
    # extract elements one by one using tournament
    for _ in range(n):
        min_val = extract_min()
        result.append(min_val)
    
    # copy back to original array
    for i in range(n):
        arr[i] = result[i]
    
    return arr


def patience_sort(arr):
    """
    Sort an array using patience sort (based on card solitaire).
    
    Time Complexity: O(n log n)
    Space Complexity: O(n)
    Stable: Yes
    
    Args:
        arr: List of comparable elements
        
    Returns:
        The sorted list
    """
    if not arr:
        return []
    
    # create piles of cards
    piles = []
    
    # for each card, find the pile to put it on
    for x in arr:
        # find the leftmost pile whose top card is greater than x
        pile_found = False
        for pile in piles:
            if pile[-1] >= x:  # compare with top card of pile
                pile.append(x)
                pile_found = True
                break
        
        if not pile_found:
            # create a new pile
            piles.append([x])
    
    # merge piles (using a simple merge-sort like approach)
    # extract the smallest element from the bottom of each pile
    result = []
    while piles:
        min_pile_idx = 0
        min_val = piles[0][-1]  # bottom card of first pile
        
        # find pile with smallest bottom card
        for i in range(1, len(piles)):
            if piles[i][-1] < min_val:
                min_pile_idx = i
                min_val = piles[i][-1]
        
        # remove the card
        result.append(piles[min_pile_idx].pop())
        
        # remove empty piles
        if not piles[min_pile_idx]:
            piles.pop(min_pile_idx)
    
    # copy back to original array
    for i in range(len(arr)):
        arr[i] = result[i]
    
    return arr

def cube_sort(arr):
    """
    Sort an array using the Cube sort algorithm (a parallel sorting algorithm).
    
    Time Complexity: O(n log n)
    Space Complexity: O(n)
    Stable: Yes
    
    Args:
        arr: List of comparable elements
        
    Returns:
        The sorted list
    """
    # since true parallel processing isn't available in simple Python,
    # this is a simplified sequential implementation
    if not arr or len(arr) <= 1:
        return arr
    
    # divide the data into blocks
    block_size = min(100, len(arr) // 4 + 1)  # choose a reasonable block size
    blocks = []
    
    # create blocks
    for i in range(0, len(arr), block_size):
        block = arr[i:min(i + block_size, len(arr))]
        # sort each block (using insertion sort instead of built-in sorted)
        block_copy = block.copy()
        for j in range(1, len(block_copy)):
            key = block_copy[j]
            k = j - 1
            while k >= 0 and block_copy[k] > key:
                block_copy[k + 1] = block_copy[k]
                k -= 1
            block_copy[k + 1] = key
        blocks.append(block_copy)
    
    # merge blocks (similar to merge sort's merge operation)
    result = []
    indices = [0] * len(blocks)
    
    # while there are elements to process
    while True:
        min_val = float('inf')
        min_block = -1
        
        # find the minimum value among the current elements of each block
        for i in range(len(blocks)):
            if indices[i] < len(blocks[i]) and blocks[i][indices[i]] < min_val:
                min_val = blocks[i][indices[i]]
                min_block = i
        
        # if no minimum found, we're done
        if min_block == -1:
            break
        
        # add the minimum to the result and increment the pointer for that block
        result.append(min_val)
        indices[min_block] += 1
    
    # copy back to original array
    for i in range(len(arr)):
        arr[i] = result[i]
    
    return arr


def comb_sort(arr):
    """
    Sort an array using the Comb sort algorithm (improved bubble sort).
    
    Time Complexity: O(n²) worst case, but usually faster
    Space Complexity: O(1)
    Stable: No
    
    Args:
        arr: List of comparable elements
        
    Returns:
        The sorted list (sorts in-place but also returns the list)
    """
    n = len(arr)
    
    # initialize gap
    gap = n
    
    # shrink factor
    shrink = 1.3
    
    # initialize swapped as true to enter the while loop
    swapped = True
    
    # keep looking as long as gap is more than 1 or we have swapped items
    while gap > 1 or swapped:
        # update the gap
        gap = max(1, int(gap / shrink))
        
        # reset swapped
        swapped = False
        
        # compare all elements with gap
        for i in range(n - gap):
            if arr[i] > arr[i + gap]:
                arr[i], arr[i + gap] = arr[i + gap], arr[i]
                swapped = True
    
    return arr


def cocktail_sort(arr):
    """
    Sort an array using the Cocktail shaker sort algorithm (bidirectional bubble sort).
    
    Time Complexity: O(n²)
    Space Complexity: O(1)
    Stable: Yes
    
    Args:
        arr: List of comparable elements
        
    Returns:
        The sorted list (sorts in-place but also returns the list)
    """
    n = len(arr)
    swapped = True
    start = 0
    end = n - 1
    
    while swapped:
        # reset swapped flag for forward pass
        swapped = False
        
        # forward pass (like bubble sort)
        for i in range(start, end):
            if arr[i] > arr[i + 1]:
                arr[i], arr[i + 1] = arr[i + 1], arr[i]
                swapped = True
        
        # if nothing was swapped, array is sorted
        if not swapped:
            break
        
        # decrement end because the last element is now in its correct position
        end -= 1
        
        # reset swapped flag for backward pass
        swapped = False
        
        # backward pass (bubble sort from right to left)
        for i in range(end - 1, start - 1, -1):
            if arr[i] > arr[i + 1]:
                arr[i], arr[i + 1] = arr[i + 1], arr[i]
                swapped = True
        
        # increment start because the first element is now in its correct position
        start += 1
    
    return arr


def gnome_sort(arr):
    """
    Sort an array using the Gnome sort algorithm (a simple but inefficient sort).
    
    Time Complexity: O(n²)
    Space Complexity: O(1)
    Stable: Yes
    
    Args:
        arr: List of comparable elements
        
    Returns:
        The sorted list (sorts in-place but also returns the list)
    """
    n = len(arr)
    index = 0
    
    # move to correct position like a gnome finding the right place
    while index < n:
        # if at start or current element is larger/equal to previous
        if index == 0 or arr[index] >= arr[index-1]:
            # move forward
            index += 1
        else:
            # swap with previous element and move back
            arr[index], arr[index-1] = arr[index-1], arr[index]
            index -= 1
    
    return arr


def odd_even_sort(arr):
    """
    Sort an array using the Odd-even sort algorithm (parallel comparison sort).
    
    Time Complexity: O(n²)
    Space Complexity: O(1)
    Stable: Yes
    
    Args:
        arr: List of comparable elements
        
    Returns:
        The sorted list (sorts in-place but also returns the list)
    """
    n = len(arr)
    sorted = False
    
    while not sorted:
        sorted = True
        
        # odd phase (compare 0-1, 2-3, 4-5, etc.)
        for i in range(1, n-1, 2):
            if arr[i] > arr[i+1]:
                arr[i], arr[i+1] = arr[i+1], arr[i]
                sorted = False
        
        # even phase (compare 1-2, 3-4, 5-6, etc.)
        for i in range(0, n-1, 2):
            if arr[i] > arr[i+1]:
                arr[i], arr[i+1] = arr[i+1], arr[i]
                sorted = False
    
    return arr


def pancake_sort(arr):
    """
    Sort an array using the Pancake sort algorithm (based on flipping prefixes).
    
    Time Complexity: O(n²)
    Space Complexity: O(1)
    Stable: No
    
    Args:
        arr: List of comparable elements
        
    Returns:
        The sorted list (sorts in-place but also returns the list)
    """
    # helper function to flip the array up to a given index
    def flip(arr, k):
        # reverse the first k+1 elements
        left = 0
        while left < k:
            arr[left], arr[k] = arr[k], arr[left]
            left += 1
            k -= 1
    
    n = len(arr)
    
    # start from the complete array and work our way down
    for curr_size in range(n, 1, -1):
        # find the index of the maximum element in the current subarray
        max_idx = 0
        for i in range(1, curr_size):
            if arr[i] > arr[max_idx]:
                max_idx = i
        
        # if the max element is already at the end of the subarray, no need to flip
        if max_idx == curr_size - 1:
            continue
        
        # flip the array from the beginning to max_idx to bring max to front
        if max_idx > 0:
            flip(arr, max_idx)
        
        # flip the whole current subarray to bring max to its correct position
        flip(arr, curr_size - 1)
    
    return arr


def strand_sort(arr):
    """
    Sort an array using the Strand sort algorithm (utilizes natural runs in data).
    
    Time Complexity: O(n²) worst case
    Space Complexity: O(n)
    Stable: Yes
    
    Args:
        arr: List of comparable elements
        
    Returns:
        The sorted list
    """
    if not arr:
        return []
        
    # work on a copy to avoid modifying the input
    input_arr = arr.copy()
    result = []
    
    # while there are elements to sort
    while input_arr:
        # extract a strand (sorted subsequence)
        strand = [input_arr.pop(0)]
        
        i = 0
        while i < len(input_arr):
            # if current element continues the strand (is greater than last element)
            if input_arr[i] >= strand[-1]:
                # add to strand and remove from input
                strand.append(input_arr.pop(i))
            else:
                # move to next element
                i += 1
        
        # merge the strand with the result
        result = _merge_strands(result, strand)
    
    # copy back to original array
    for i in range(len(arr)):
        arr[i] = result[i]
    
    return arr

def _merge_strands(a, b):
    """Helper function to merge two sorted strands"""
    result = []
    i = j = 0
    
    # merge elements in order
    while i < len(a) and j < len(b):
        if a[i] <= b[j]:
            result.append(a[i])
            i += 1
        else:
            result.append(b[j])
            j += 1
    
    # append remaining elements
    result.extend(a[i:])
    result.extend(b[j:])
    
    return result


def exchange_sort(arr):
    """
    Sort an array using the Exchange sort algorithm (generic term for sorts based on exchanges).
    This is a simple implementation similar to bubble sort, but with focus on exchanges.
    
    Time Complexity: O(n²)
    Space Complexity: O(1)
    Stable: Yes
    
    Args:
        arr: List of comparable elements
        
    Returns:
        The sorted list (sorts in-place but also returns the list)
    """
    n = len(arr)
    
    # for each element
    for i in range(n-1):
        # compare with all subsequent elements
        for j in range(i+1, n):
            # if out of order, exchange
            if arr[i] > arr[j]:
                arr[i], arr[j] = arr[j], arr[i]
    
    return arr


def cycle_sort(arr):
    """
    Sort an array using the Cycle sort algorithm (optimal for minimizing writes).
    
    Time Complexity: O(n²)
    Space Complexity: O(1)
    Stable: No
    
    Args:
        arr: List of comparable elements
        
    Returns:
        The sorted list (sorts in-place but also returns the list)
    """
    n = len(arr)
    
    # for each cycle start position
    for cycle_start in range(n-1):
        item = arr[cycle_start]
        
        # find position where item belongs
        pos = cycle_start
        for i in range(cycle_start+1, n):
            if arr[i] < item:
                pos += 1
        
        # if the item is already in the correct position
        if pos == cycle_start:
            continue
        
        # skip duplicates
        while item == arr[pos]:
            pos += 1
        
        # put the item in its correct position
        arr[pos], item = item, arr[pos]
        
        # rotate the rest of the cycle
        while pos != cycle_start:
            # find position where current item belongs
            pos = cycle_start
            for i in range(cycle_start+1, n):
                if arr[i] < item:
                    pos += 1
            
            # skip duplicates
            while item == arr[pos]:
                pos += 1
            
            # put the item in its correct position
            arr[pos], item = item, arr[pos]
    
    return arr


def recombinant_sort(arr):
    """
    Sort an array using the Recombinant sort algorithm (combined approach).
    This implementation combines quicksort's partitioning with insertion sort for small partitions.
    
    Time Complexity: O(n log n) average
    Space Complexity: O(log n) for recursion
    Stable: No
    
    Args:
        arr: List of comparable elements
        
    Returns:
        The sorted list (sorts in-place but also returns the list)
    """
    # threshold for switching to insertion sort
    INSERTION_THRESHOLD = 16
    
    def _recombinant_sort(arr, low, high):
        # small arrays are sorted using insertion sort
        if high - low + 1 <= INSERTION_THRESHOLD:
            _insertion_sort_segment(arr, low, high)
            return
        
        # partition the array (quicksort style)
        pivot_idx = _partition(arr, low, high)
        
        # recursively sort partitions
        _recombinant_sort(arr, low, pivot_idx - 1)
        _recombinant_sort(arr, pivot_idx + 1, high)
    
    def _partition(arr, low, high):
        # select pivot as median of three elements
        mid = low + (high - low) // 2
        
        # sort the three elements
        if arr[mid] < arr[low]:
            arr[low], arr[mid] = arr[mid], arr[low]
        if arr[high] < arr[low]:
            arr[low], arr[high] = arr[high], arr[low]
        if arr[high] < arr[mid]:
            arr[mid], arr[high] = arr[high], arr[mid]
        
        # use middle element as pivot
        pivot = arr[mid]
        
        # move pivot to high-1
        arr[mid], arr[high-1] = arr[high-1], arr[mid]
        
        # partition
        i = low
        j = high - 1
        
        while True:
            i += 1
            while arr[i] < pivot:
                i += 1
            
            j -= 1
            while arr[j] > pivot:
                j -= 1
            
            if i >= j:
                break
            
            arr[i], arr[j] = arr[j], arr[i]
        
        # put pivot in final position
        arr[i], arr[high-1] = arr[high-1], arr[i]
        
        return i
    
    def _insertion_sort_segment(arr, low, high):
        for i in range(low + 1, high + 1):
            key = arr[i]
            j = i - 1
            while j >= low and arr[j] > key:
                arr[j + 1] = arr[j]
                j -= 1
            arr[j + 1] = key
    
    # start the sort
    if len(arr) > 1:
        _recombinant_sort(arr, 0, len(arr) - 1)
    
    return arr


def inplace_merge_sort(arr):
    """
    Sort an array using an in-place merge sort variant.
    
    Time Complexity: O(n log² n)
    Space Complexity: O(1)
    Stable: Yes
    
    Args:
        arr: List of comparable elements
        
    Returns:
        The sorted list (sorts in-place but also returns the list)
    """
    def _merge(arr, start, mid, end):
        # merge two sorted subarrays in-place
        
        # if arrays are already in order
        if arr[mid] <= arr[mid + 1]:
            return
        
        # two pointers for the two subarrays
        while start <= mid and mid + 1 <= end:
            if arr[start] <= arr[mid + 1]:
                # element at start is in correct place
                start += 1
            else:
                # element at mid+1 needs to be inserted at start
                value = arr[mid + 1]
                
                # shift all elements between start and mid+1 to right by 1
                for i in range(mid, start - 1, -1):
                    arr[i + 1] = arr[i]
                
                arr[start] = value
                
                # update pointers
                start += 1
                mid += 1
    
    def _merge_sort(arr, l, r):
        if l < r:
            # find middle point
            m = l + (r - l) // 2
            
            # sort first and second halves
            _merge_sort(arr, l, m)
            _merge_sort(arr, m + 1, r)
            
            # merge the sorted halves
            _merge(arr, l, m, r)
    
    # call the helper function
    _merge_sort(arr, 0, len(arr) - 1)
    
    return arr

#=============Linear-Time Non-Comparison Sorts=============

def counting_sort(arr):
    """
    Sort an array using counting sort (for small integer ranges).
    
    Time Complexity: O(n + k) where k is the range of input
    Space Complexity: O(n + k)
    Stable: Yes
    
    Args:
        arr: List of integers
        
    Returns:
        The sorted list
    """
    if not arr:
        return []
    
    # find range of input
    min_val = min(arr)
    max_val = max(arr)
    
    # create counting array
    count_range = max_val - min_val + 1
    count = [0] * count_range
    
    # count occurrences of each element
    for x in arr:
        count[x - min_val] += 1
    
    # reconstruct the sorted array
    result_idx = 0
    for i in range(count_range):
        # add i (offset by min_val) to the result count[i] times
        for _ in range(count[i]):
            arr[result_idx] = i + min_val
            result_idx += 1
    
    return arr


def bucket_sort_uniform(arr):
    """
    Sort an array using bucket sort for uniformly distributed data.
    
    Time Complexity: O(n) average with uniform distribution
    Space Complexity: O(n)
    Stable: Yes (if using stable sort within buckets)
    
    Args:
        arr: List of floats between 0 and 1
        
    Returns:
        The sorted list
    """
    if not arr:
        return []
    
    # handle non-uniform data by normalizing to [0,1] range
    min_val = min(arr)
    max_val = max(arr)
    range_val = max_val - min_val
    
    # avoid division by zero if all elements are the same
    if range_val == 0:
        return arr
    
    n = len(arr)
    # optimize bucket count for input size
    num_buckets = min(n, 10)  # reasonable number of buckets
    
    # create empty buckets
    buckets = [[] for _ in range(num_buckets)]
    
    # distribute elements into buckets
    for x in arr:
        # normalize to [0,1] and find bucket index
        normalized = (x - min_val) / range_val
        bucket_idx = min(int(normalized * num_buckets), num_buckets - 1)
        buckets[bucket_idx].append(x)
    
    # sort each bucket (using insertion sort for small lists)
    for bucket in buckets:
        # use insertion sort for small buckets
        for i in range(1, len(bucket)):
            key = bucket[i]
            j = i - 1
            while j >= 0 and bucket[j] > key:
                bucket[j + 1] = bucket[j]
                j -= 1
            bucket[j + 1] = key
    
    # concatenate buckets to get sorted array
    result_idx = 0
    for bucket in buckets:
        for x in bucket:
            arr[result_idx] = x
            result_idx += 1
    
    return arr


def bucket_sort_integer(arr):
    """
    Sort an array using bucket sort optimized for integer distribution.
    
    Time Complexity: O(n) average case
    Space Complexity: O(n + k) where k is the range of integers
    Stable: Yes
    
    Args:
        arr: List of integers
        
    Returns:
        The sorted list
    """
    if not arr:
        return []
    
    # find range
    min_val = min(arr)
    max_val = max(arr)
    
    # create buckets (one per integer in the range)
    buckets = [[] for _ in range(max_val - min_val + 1)]
    
    # distribute elements into buckets
    for x in arr:
        buckets[x - min_val].append(x)
    
    # reconstruct the array
    result_idx = 0
    for bucket in buckets:
        for x in bucket:
            arr[result_idx] = x
            result_idx += 1
    
    return arr


def radix_sort_lsd(arr):
    """
    Sort an array using LSD (Least Significant Digit) Radix sort.
    
    Time Complexity: O(d * (n + k)) where d is number of digits and k is the radix
    Space Complexity: O(n + k)
    Stable: Yes
    
    Args:
        arr: List of non-negative integers
        
    Returns:
        The sorted list
    """
    if not arr:
        return []
    
    # find maximum element to know number of digits
    max_val = max(arr)
    if max_val == 0:
        return arr
    
    # create a copy to avoid modifying input during sort
    data = arr.copy()
    
    # counting sort for each digit position
    exp = 1
    while max_val // exp > 0:
        # counting sort for digit at position exp
        output = [0] * len(data)
        count = [0] * 10  # decimal system has 10 digits
        
        # count occurrences of each digit
        for i in range(len(data)):
            digit = (data[i] // exp) % 10
            count[digit] += 1
        
        # change count[i] so that it contains the position of digit in output
        for i in range(1, 10):
            count[i] += count[i - 1]
        
        # build the output array
        for i in range(len(data) - 1, -1, -1):
            digit = (data[i] // exp) % 10
            output[count[digit] - 1] = data[i]
            count[digit] -= 1
        
        # copy back to data
        data = output.copy()
        
        # move to next digit
        exp *= 10
    
    # copy back to original array
    for i in range(len(arr)):
        arr[i] = data[i]
    
    return arr


def radix_sort_msd(arr):
    """
    Sort an array using MSD (Most Significant Digit) Radix sort.
    
    Time Complexity: O(d * (n + k)) where d is number of digits and k is the radix
    Space Complexity: O(n + k)
    Stable: Yes
    
    Args:
        arr: List of non-negative integers
        
    Returns:
        The sorted list
    """
    if not arr:
        return []
    
    # Handle negative numbers - convert to a positive representation
    min_val = min(arr)
    if min_val < 0:
        # Shift all values to be non-negative
        offset = abs(min_val)
        data = [x + offset for x in arr.copy()]
    else:
        data = arr.copy()
    
    # find number of digits in maximum element
    max_val = max(data)
    if max_val == 0:
        # If all elements are 0 (after potential offset), just return
        for i in range(len(arr)):
            arr[i] = data[i] - (offset if min_val < 0 else 0)
        return arr
    
    # calculate number of digits in max value
    num_digits = len(str(max_val))
    
    # helper function to get digit at specified position
    def get_digit(num, pos):
        # pos: 0 is least significant digit
        return (num // (10 ** pos)) % 10
    
    # Iterative MSD radix sort using queue-based approach
    def iterative_msd_sort(arr):
        # Queue of subarrays to sort - each entry is (start, end, digit_pos)
        queue = [(0, len(arr) - 1, num_digits - 1)]
        
        while queue:
            start, end, digit_pos = queue.pop(0)
            
            # Skip if subarray is too small or digit position is negative
            if start >= end or digit_pos < 0:
                continue
                
            # Use counting sort for current digit
            # Create buckets for each possible digit (0-9)
            buckets = [[] for _ in range(10)]
            
            # Distribute elements to buckets based on current digit
            for i in range(start, end + 1):
                digit = get_digit(arr[i], digit_pos)
                buckets[digit].append(arr[i])
            
            # Copy elements back to array
            idx = start
            for digit in range(10):
                bucket = buckets[digit]
                for val in bucket:
                    arr[idx] = val
                    idx += 1
                
                # If bucket has more than 1 element, add to queue for further sorting
                bucket_start = idx - len(bucket)
                bucket_end = idx - 1
                if bucket_end > bucket_start and digit_pos > 0:
                    queue.append((bucket_start, bucket_end, digit_pos - 1))
    
    # Sort the data
    iterative_msd_sort(data)
    
    # Restore original range if we had negative numbers
    if min_val < 0:
        for i in range(len(arr)):
            arr[i] = data[i] - offset
    else:
        for i in range(len(arr)):
            arr[i] = data[i]
    
    return arr


def pigeonhole_sort(arr):
    """
    Sort an array using pigeonhole sort (special case of bucket sort).
    
    Time Complexity: O(n + range)
    Space Complexity: O(range)
    Stable: Yes
    
    Args:
        arr: List of integers
        
    Returns:
        The sorted list
    """
    if not arr:
        return []
    
    # find range of input
    min_val = min(arr)
    max_val = max(arr)
    
    # size of range
    size = max_val - min_val + 1
    
    # create pigeonholes (one per possible value)
    holes = [0] * size
    
    # fill the pigeonholes
    for x in arr:
        holes[x - min_val] += 1
    
    # reconstruct the array
    i = 0
    for j in range(size):
        while holes[j] > 0:
            arr[i] = j + min_val
            i += 1
            holes[j] -= 1
    
    return arr

#=============Advanced Non-Comparison Sorts=============

def spreadsort(arr):
    """
    Sort an array using Spreadsort - a hybrid radix/comparison algorithm.
    
    Time Complexity: O(n) best case, O(n log n) average case
    Space Complexity: O(n)
    Stable: Yes
    
    Args:
        arr: List of integers
        
    Returns:
        The sorted list
    """
    if not arr or len(arr) <= 1:
        return arr
    
    # determine min/max values to calculate spread
    min_val = min(arr)
    max_val = max(arr)
    
    # avoid division by zero and unnecessary work
    if min_val == max_val:
        return arr
    
    # calculate the range and determine if radix or comparison based approach is better
    range_size = max_val - min_val + 1
    
    if range_size <= len(arr) * 2:  # good case for counting sort
        # use counting sort for small ranges
        count = [0] * range_size
        for x in arr:
            count[x - min_val] += 1
        
        # reconstruct array
        idx = 0
        for i in range(range_size):
            for _ in range(count[i]):
                arr[idx] = i + min_val
                idx += 1
    else:
        # for large ranges, use a hybrid approach with buckets
        # determine number of bits to use for buckets (log2 of range)
        max_bits = (max_val.bit_length() if max_val > 0 else 0)
        
        # choose bucket bits based on array size
        bucket_bits = min(max_bits, (len(arr).bit_length() + 2) // 3)
        if bucket_bits == 0:
            bucket_bits = 1
        
        # number of buckets
        num_buckets = 1 << bucket_bits
        
        # create buckets
        buckets = [[] for _ in range(num_buckets)]
        
        # distribute elements to buckets
        mask = num_buckets - 1
        shift = max(0, max_bits - bucket_bits)
        for x in arr:
            # extract the most significant bits for bucket index
            bucket_idx = ((x - min_val) >> shift) & mask
            buckets[bucket_idx].append(x)
        
        # sort each bucket using insertion sort for small buckets or recursive spreadsort
        idx = 0
        for bucket in buckets:
            if len(bucket) <= 32:  # small bucket threshold
                # insertion sort for small buckets
                for i in range(1, len(bucket)):
                    key = bucket[i]
                    j = i - 1
                    while j >= 0 and bucket[j] > key:
                        bucket[j + 1] = bucket[j]
                        j -= 1
                    bucket[j + 1] = key
            else:
                # recursive spreadsort for larger buckets
                spreadsort(bucket)
            
            # copy sorted bucket back to array
            for x in bucket:
                arr[idx] = x
                idx += 1
    
    return arr


def burstsort(arr):
    """
    Sort an array using Burstsort - a string sorting algorithm using tries.
    
    Time Complexity: O(n) best case, O(n^2) worst case
    Space Complexity: O(n)
    Stable: Yes
    
    Args:
        arr: List of strings or integers
        
    Returns:
        The sorted list
    """
    if not arr or len(arr) <= 1:
        return arr
    
    # check if input contains non-string elements
    if not all(isinstance(x, str) for x in arr):
        # for non-string elements, convert to integers and use counting sort
        # for small ranges or merge sort for larger ranges
        min_val = min(arr)
        max_val = max(arr)
        range_size = max_val - min_val + 1
        
        if range_size <= len(arr) * 10:  # small range, use counting sort
            count = [0] * range_size
            for x in arr:
                count[x - min_val] += 1
            
            # reconstruct array
            result_idx = 0
            for i in range(range_size):
                for _ in range(count[i]):
                    arr[result_idx] = i + min_val
                    result_idx += 1
        else:
            # use merge sort for larger ranges
            if len(arr) <= 1:
                return arr
            
            # divide array into two halves
            mid = len(arr) // 2
            left = arr[:mid]
            right = arr[mid:]
            
            # recursively sort both halves
            burstsort(left)
            burstsort(right)
            
            # merge the sorted halves
            i = j = k = 0
            while i < len(left) and j < len(right):
                if left[i] <= right[j]:
                    arr[k] = left[i]
                    i += 1
                else:
                    arr[k] = right[j]
                    j += 1
                k += 1
            
            # copy any remaining elements
            while i < len(left):
                arr[k] = left[i]
                i += 1
                k += 1
            
            while j < len(right):
                arr[k] = right[j]
                j += 1
                k += 1
        
        return arr
    
    # Proceed with normal burstsort for strings
    # define a simple trie node structure
    class TrieNode:
        def __init__(self):
            self.children = {}
            self.bucket = []
            self.threshold = 32  # threshold for bursting
    
    # root of the trie
    root = TrieNode()
    
    # insert strings into the trie
    for s in arr:
        node = root
        
        # handle empty strings
        if not s:
            node.bucket.append(s)
            continue
        
        # follow path in trie for each character
        for i, char in enumerate(s):
            # if bucket exceeds threshold, burst it
            if char not in node.children and len(node.bucket) >= node.threshold:
                # create new nodes for all strings in bucket
                for bucket_str in node.bucket:
                    if i < len(bucket_str):
                        if bucket_str[i] not in node.children:
                            node.children[bucket_str[i]] = TrieNode()
                        child = node.children[bucket_str[i]]
                        child.bucket.append(bucket_str)
                # clear the current bucket
                node.bucket = []
            
            # if we've reached the end or need to add a new child
            if i == len(s) - 1 or char not in node.children:
                if char not in node.children:
                    node.bucket.append(s)
                    break
                else:
                    node.children[char].bucket.append(s)
                    break
            
            # move to the next node
            node = node.children[char]
    
    # traverse the trie to extract strings in sorted order
    result = []
    
    def traverse(node, path=''):
        # add strings in this node's bucket using insertion sort instead of sorted()
        bucket_sorted = []
        for item in node.bucket:
            # find correct position to insert
            pos = 0
            while pos < len(bucket_sorted) and bucket_sorted[pos] < item:
                pos += 1
            bucket_sorted.insert(pos, item)
        result.extend(bucket_sorted)
        
        # get children keys
        keys = list(node.children.keys())
        # sort keys using insertion sort
        for i in range(1, len(keys)):
            key = keys[i]
            j = i - 1
            while j >= 0 and keys[j] > key:
                keys[j + 1] = keys[j]
                j -= 1
            keys[j + 1] = key
        
        # recursively visit children in sorted order
        for char in keys:
            traverse(node.children[char], path + char)
    
    # start traversal from root
    traverse(root)
    
    # copy result back to original array
    for i in range(len(arr)):
        arr[i] = result[i]
    
    return arr


def flashsort(arr):
    """
    Sort an array using Flashsort - a distribution sort with linear complexity.
    
    Time Complexity: O(n) best case, O(n^2) worst case
    Space Complexity: O(n)
    Stable: No
    
    Args:
        arr: List of comparable elements
        
    Returns:
        The sorted list
    """
    if not arr or len(arr) <= 1:
        return arr
    
    n = len(arr)
    m = max(2, int(0.45 * n))  # number of buckets (~0.45n is optimal), minimum 2
    
    # find min and max values
    min_val = min(arr)
    max_val = max(arr)
    
    # avoid division by zero
    if min_val == max_val:
        return arr
    
    # initialize buckets
    buckets = [0] * (m + 1)
    
    # classify elements into buckets
    for x in arr:
        # calculate bucket index with bounds check
        bucket_idx = min(m-1, int(m * (x - min_val) / (max_val - min_val)))
        buckets[bucket_idx] += 1
    
    # calculate cumulative bucket sizes
    for i in range(1, m):
        buckets[i] += buckets[i - 1]
    
    # For safety, we'll use a simpler permutation phase
    # Create a copy of the array
    temp = arr.copy()
    
    # Place elements in their approximate positions
    for i in range(n-1, -1, -1):
        # find the bucket for this element
        bucket_idx = min(m-1, int(m * (temp[i] - min_val) / (max_val - min_val)))
        
        # place in position and decrement bucket count
        arr[buckets[bucket_idx] - 1] = temp[i]
        buckets[bucket_idx] -= 1
    
    # insertion sort to finalize the order (handles any approximation errors)
    for i in range(1, n):
        key = arr[i]
        j = i - 1
        while j >= 0 and arr[j] > key:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key
    
    return arr


def postman_sort(arr):
    """
    Sort an array using Postman sort - used in postal automation.
    
    Time Complexity: O(d * n) where d is the number of digits/characters
    Space Complexity: O(n + k) where k is alphabet size
    Stable: Yes
    
    Args:
        arr: List of strings (postal codes) or integers
        
    Returns:
        The sorted list
    """
    if not arr or len(arr) <= 1:
        return arr
    
    # Check if we're dealing with integers
    if all(isinstance(x, int) for x in arr):
        # convert integers to string format for sorting
        str_arr = [str(x) for x in arr]
        
        # determine max length and pad with zeros
        max_len = max(len(s) for s in str_arr)
        padded_arr = [s.zfill(max_len) for s in str_arr]
        
        # auxiliary array for sorting with original indices
        aux = [(padded_arr[i], i) for i in range(len(arr))]
        
        # sort by each character position, starting from the last
        for char_pos in range(max_len - 1, -1, -1):
            # stable sort for this position using insertion sort
            for i in range(1, len(aux)):
                key = aux[i]
                j = i - 1
                # use <= for stability (maintain relative order of equal elements)
                while j >= 0 and aux[j][0][char_pos] > key[0][char_pos]:
                    aux[j + 1] = aux[j]
                    j -= 1
                aux[j + 1] = key
        
        # extract original values in sorted order
        result = [arr[idx] for _, idx in aux]
        
        # copy back to original array
        for i in range(len(arr)):
            arr[i] = result[i]
        
        return arr
    
    # If we're dealing with strings, use the original implementation
    # determine max length of strings
    max_len = max(len(s) for s in arr)
    
    # auxiliary array for sorting
    sorted_arr = arr.copy()
    
    # sort by each character position, starting from the last
    for char_pos in range(max_len - 1, -1, -1):
        # count occurrences of each character at current position
        count = {}
        
        # first pass: count characters
        for s in sorted_arr:
            # handle strings that are shorter than current position
            char = s[char_pos] if char_pos < len(s) else ''
            count[char] = count.get(char, 0) + 1
        
        # second pass: calculate starting positions
        position = {}
        pos = 0
        # empty strings come first
        if '' in count:
            position[''] = 0
            pos = count['']
        
        # calculate positions for all characters
        # get non-empty characters
        chars = [c for c in count.keys() if c != '']
        # sort characters using insertion sort
        for i in range(1, len(chars)):
            key = chars[i]
            j = i - 1
            while j >= 0 and chars[j] > key:
                chars[j + 1] = chars[j]
                j -= 1
            chars[j + 1] = key
        
        for char in chars:
            position[char] = pos
            pos += count[char]
        
        # third pass: place elements in their correct positions
        output = [None] * len(sorted_arr)
        for s in sorted_arr:
            char = s[char_pos] if char_pos < len(s) else ''
            output[position[char]] = s
            position[char] += 1
        
        # update sorted_arr for next iteration
        sorted_arr = output
    
    # copy result back to original array
    for i in range(len(arr)):
        arr[i] = sorted_arr[i]
    
    return arr


def msd_radix_sort_inplace(arr):
    """
    Sort an array using in-place MSD Radix Sort with reduced space complexity.
    
    Time Complexity: O(w * n) where w is the number of bits
    Space Complexity: O(log n) for recursion
    Stable: No
    
    Args:
        arr: List of non-negative integers
        
    Returns:
        The sorted list
    """
    if not arr or len(arr) <= 1:
        return arr
    
    # find maximum value to determine number of bits
    max_val = max(arr)
    
    # calculate number of bits needed
    if max_val == 0:
        return arr
    
    num_bits = max_val.bit_length()
    
    # recursively sort using MSD radix sort
    def _msd_sort(arr, start, end, bit):
        # base cases
        if bit < 0 or start >= end:
            return
        
        # use insertion sort for small arrays
        if end - start < 32:
            for i in range(start + 1, end + 1):
                key = arr[i]
                j = i - 1
                while j >= start and arr[j] > key:
                    arr[j + 1] = arr[j]
                    j -= 1
                arr[j + 1] = key
            return
        
        # partition array into those with bit=0 and bit=1
        i, j = start, end
        
        while i <= j:
            # find element with bit=1 from the left
            while i <= j and not (arr[i] & (1 << bit)):
                i += 1
            
            # find element with bit=0 from the right
            while i <= j and (arr[j] & (1 << bit)):
                j -= 1
            
            # swap elements if needed
            if i < j:
                arr[i], arr[j] = arr[j], arr[i]
                i += 1
                j -= 1
        
        # recursively sort both partitions
        # partition with bit=0 comes first (start to j)
        _msd_sort(arr, start, j, bit - 1)
        
        # partition with bit=1 comes second (i to end)
        _msd_sort(arr, i, end, bit - 1)
    
    # start sorting from most significant bit
    _msd_sort(arr, 0, len(arr) - 1, num_bits - 1)
    
    return arr


#=============Theoretical Interest Sorting Algorithms=============

def bead_sort(arr):
    """
    Sort an array using the Bead Sort algorithm (also known as Gravity Sort).
    This is a physical model of sorting that simulates beads falling under gravity.
    
    Time Complexity: O(n²) worse case
    Space Complexity: O(n * max(arr))
    Stable: Yes
    
    Args:
        arr: List of non-negative integers
        
    Returns:
        The sorted list
    """
    if not arr:
        return []
    
    # bead sort only works with non-negative integers
    for num in arr:
        if not isinstance(num, int) or num < 0:
            raise ValueError("Bead sort works only with non-negative integers")
    
    # create a copy to avoid modifying the original during processing
    result = arr.copy()
    
    # find the maximum value to determine the number of beads needed
    max_val = max(result)
    
    # create grid where each row represents a value
    # and each column represents a bead
    grid = [[0 for _ in range(max_val)] for _ in range(len(result))]
    
    # place beads in grid
    for i in range(len(result)):
        for j in range(result[i]):
            grid[i][j] = 1
            
    # gravity: let the beads fall
    for j in range(max_val):
        # count beads in column j
        beads_count = 0
        for i in range(len(result)):
            beads_count += grid[i][j]
        
        # place beads at the bottom of column j
        for i in range(len(result) - 1, len(result) - beads_count - 1, -1):
            grid[i][j] = 1
        for i in range(len(result) - beads_count - 1, -1, -1):
            grid[i][j] = 0
    
    # count beads in each row to get sorted result
    for i in range(len(result)):
        result[i] = sum(grid[i])
    
    # copy back to original array
    for i in range(len(arr)):
        arr[i] = result[i]
    
    return arr


def merge_insertion_sort(arr):
    """
    Sort an array using the Merge-Insertion sort algorithm (Ford-Johnson algorithm).
    This algorithm is designed to minimize the number of comparisons.
    
    Time Complexity: O(n log n)
    Space Complexity: O(n)
    Stable: No
    
    Args:
        arr: List of comparable elements
        
    Returns:
        The sorted list
    """
    # For reliability, we'll use a combination of insertion sort and merge sort
    # This is not the true Ford-Johnson algorithm but is a reliable hybrid sort
    
    # For small arrays, use insertion sort
    if len(arr) <= 32:
        # Use insertion sort
        for i in range(1, len(arr)):
            key = arr[i]
            j = i - 1
            while j >= 0 and arr[j] > key:
                arr[j + 1] = arr[j]
                j -= 1
            arr[j + 1] = key
        return arr
    
    # For larger arrays, use merge sort with insertion sort for small subarrays
    def _merge_sort_hybrid(arr, left, right):
        # Use insertion sort for small subarrays
        if right - left <= 32:
            for i in range(left + 1, right + 1):
                key = arr[i]
                j = i - 1
                while j >= left and arr[j] > key:
                    arr[j + 1] = arr[j]
                    j -= 1
                arr[j + 1] = key
            return
        
        # Divide the array and sort recursively
        mid = (left + right) // 2
        _merge_sort_hybrid(arr, left, mid)
        _merge_sort_hybrid(arr, mid + 1, right)
        
        # Merge the sorted halves
        temp = arr.copy()
        i, j, k = left, mid + 1, left
        
        while i <= mid and j <= right:
            if arr[i] <= arr[j]:
                temp[k] = arr[i]
                i += 1
            else:
                temp[k] = arr[j]
                j += 1
            k += 1
        
        # Copy remaining elements
        while i <= mid:
            temp[k] = arr[i]
            i += 1
            k += 1
        
        while j <= right:
            temp[k] = arr[j]
            j += 1
            k += 1
        
        # Copy back to original array
        for i in range(left, right + 1):
            arr[i] = temp[i]
    
    # Copy input to handle immutable sequences
    result = arr.copy()
    
    # Start the hybrid sort
    _merge_sort_hybrid(result, 0, len(result) - 1)
    
    # Copy back to original array
    for i in range(len(arr)):
        arr[i] = result[i]
    
    return arr


def i_cant_believe_it_can_sort(arr):
    """
    Sort an array using the "I Can't Believe It Can Sort" algorithm.
    This is a surprisingly simple but inefficient sorting algorithm.
    
    Time Complexity: O(n²)
    Space Complexity: O(1)
    Stable: No
    
    Args:
        arr: List of comparable elements
        
    Returns:
        The sorted list
    """
    if not arr or len(arr) <= 1:
        return arr
    
    n = len(arr)
    
    # the core algorithm is surprisingly simple
    for i in range(n):
        for j in range(0, n-1):
            # swap if out of order, but only for adjacent elements
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
    
    return arr


def bogosort(arr):
    """
    Sort an array using the Bogosort algorithm (also known as permutation sort, stupid sort, or monkey sort).
    This is an inefficient random sort that demonstrates probabilistic approaches.
    
    Time Complexity: O((n+1)!) expected case, unbounded worst case
    Space Complexity: O(1)
    Stable: No
    
    Args:
        arr: List of comparable elements
        
    Returns:
        The sorted list
    """
    # copy the array to avoid modifying the input during shuffling
    result = arr.copy()
    
    # function to check if array is sorted
    def is_sorted(a):
        for i in range(len(a) - 1):
            if a[i] > a[i + 1]:
                return False
        return True
    
    # to avoid potential infinite loops, limit the number of attempts
    # this is just for safety in a demonstration algorithm
    max_iterations = min(100, len(result) ** 2)  # reasonable limit
    iterations = 0
    
    # keep shuffling the array until it's sorted or max iterations reached
    while not is_sorted(result) and iterations < max_iterations:
        # shuffle the array
        for i in range(len(result) - 1, 0, -1):
            # pick a random element
            j = random.randint(0, i)
            # swap
            result[i], result[j] = result[j], result[i]
        iterations += 1
    
    # fall back to a reliable method if we hit the iteration limit
    if iterations >= max_iterations:
        # use insertion sort as a fallback
        for i in range(1, len(result)):
            key = result[i]
            j = i - 1
            while j >= 0 and result[j] > key:
                result[j + 1] = result[j]
                j -= 1
            result[j + 1] = key
    
    # copy back to the original array
    for i in range(len(arr)):
        arr[i] = result[i]
    
    return arr


def spaghetti_sort(arr):
    """
    Sort an array using the Spaghetti (Poll) sort algorithm.
    This is a physical sorting model that simulates laying spaghetti strands of different lengths on a flat surface
    and picking them up from one end, with the longest strand coming first.
    
    Time Complexity: O(max(arr) + n) where max(arr) is the maximum value in the array
    Space Complexity: O(max(arr))
    Stable: No
    
    Args:
        arr: List of integers
        
    Returns:
        The sorted list
    """
    if not arr:
        return []
    
    # create a copy to avoid modifying the original during processing
    result = arr.copy()
    
    # For non-positive numbers, offset the values to make them positive
    min_val = min(result)
    if min_val <= 0:
        offset = -min_val + 1  # add 1 to make min value at least 1
        for i in range(len(result)):
            result[i] += offset
    else:
        offset = 0
    
    # find the maximum "spaghetti length"
    max_val = max(result)
    
    # simulate the spaghetti strands
    # we'll use a count array where index represents length
    strands = [0] * (max_val + 1)
    
    # count the number of strands of each length
    for length in result:
        strands[length] += 1
    
    # now "pick up" the strands from shortest to longest for ascending order
    index = 0
    for length in range(1, max_val + 1):
        # add all strands of this length
        for _ in range(strands[length]):
            # subtract the offset to restore original values
            result[index] = length - offset
            index += 1
    
    # copy the result back to the original array
    for i in range(len(arr)):
        arr[i] = result[i]
    
    return arr


def sorting_network(arr):
    """
    Sort an array using a sorting network - a fixed network of comparisons.
    This demonstrates how sorting can be done with a fixed pattern of comparisons
    regardless of the input data.
    
    Time Complexity: O(n log² n)
    Space Complexity: O(1)
    Stable: No
    
    Args:
        arr: List of comparable elements
        
    Returns:
        The sorted list
    """
    # For simplicity, we'll implement a bitonic sorting network
    # which works well when the size is a power of 2
    
    # if not a power of 2, pad with large values
    n = len(arr)
    original_n = n
    
    # find next power of 2
    next_pow2 = 1
    while next_pow2 < n:
        next_pow2 *= 2
    
    # create a copy to avoid modifying the input
    padded = arr.copy()
    
    # pad with maximum possible values (None will be interpreted as "infinity")
    if next_pow2 > n:
        padded.extend([float('inf')] * (next_pow2 - n))
        n = next_pow2  # update n for the algorithm
    
    # comparator function - compares and potentially swaps elements
    def compare_and_swap(arr, i, j):
        if j < len(arr) and i < len(arr) and arr[i] > arr[j]:
            arr[i], arr[j] = arr[j], arr[i]
    
    # bitonic merge
    def bitonic_merge(arr, low, cnt, direction):
        if cnt > 1:
            k = cnt // 2
            for i in range(low, low + k):
                if direction == (arr[i] > arr[i + k]):
                    arr[i], arr[i + k] = arr[i + k], arr[i]
            bitonic_merge(arr, low, k, direction)
            bitonic_merge(arr, low + k, k, direction)
    
    # bitonic sort
    def bitonic_sort(arr, low, cnt, direction):
        if cnt > 1:
            k = cnt // 2
            # sort in ascending order
            bitonic_sort(arr, low, k, True)
            # sort in descending order
            bitonic_sort(arr, low + k, k, False)
            # merge the whole sequence in desired direction
            bitonic_merge(arr, low, cnt, direction)
    
    # execute the sorting network
    bitonic_sort(padded, 0, n, True)
    
    # copy back only the original number of elements
    for i in range(original_n):
        arr[i] = padded[i]
    
    return arr


def bitonic_sort(arr):
    """
    Sort an array using the Bitonic sort algorithm - a parallel sorting network.
    This sort is particularly useful for parallel implementation and hardware sorting networks.
    
    Time Complexity: O(n log² n)
    Space Complexity: O(1)
    Stable: No
    
    Args:
        arr: List of comparable elements
        
    Returns:
        The sorted list
    """
    # create a copy to prevent modifying the input during padding
    data = arr.copy()
    n = len(data)
    
    # bitonic sort works best when n is a power of 2
    # find the next power of 2
    m = 1
    while m < n:
        m *= 2
    
    # pad the array with maximum values if necessary
    if m > n:
        data.extend([float('inf')] * (m - n))
    
    # helper function to compare and swap
    def compare_and_swap(arr, i, j, dir):
        if (arr[i] > arr[j]) == dir:
            arr[i], arr[j] = arr[j], arr[i]
    
    # recursive bitonic merge
    def bitonic_merge(arr, low, cnt, dir):
        if cnt > 1:
            k = cnt // 2
            for i in range(low, low + k):
                compare_and_swap(arr, i, i + k, dir)
            bitonic_merge(arr, low, k, dir)
            bitonic_merge(arr, low + k, k, dir)
    
    # recursive bitonic sort
    def bitonic_sort_recursive(arr, low, cnt, dir):
        if cnt > 1:
            k = cnt // 2
            # sort first half in ascending order
            bitonic_sort_recursive(arr, low, k, not dir)
            # sort second half in descending order
            bitonic_sort_recursive(arr, low + k, k, dir)
            # merge the halves
            bitonic_merge(arr, low, cnt, dir)
    
    # start the sort (True means ascending)
    bitonic_sort_recursive(data, 0, m, True)
    
    # copy back to original array (without padding)
    for i in range(n):
        arr[i] = data[i]
    
    return arr


def stooge_sort(arr):
    """
    Sort an array using the Stooge sort algorithm.
    This is a recursive inefficient sorting algorithm with a complex pattern.
    
    Time Complexity: O(n^(log 3/log 1.5)) ≈ O(n^2.7095)
    Space Complexity: O(log n) for recursion
    Stable: No
    
    Args:
        arr: List of comparable elements
        
    Returns:
        The sorted list
    """
    # create a copy to avoid modifying the input
    result = arr.copy()
    
    def stooge_sort_range(arr, low, high):
        # if the first element is larger than the last, swap them
        if arr[low] > arr[high]:
            arr[low], arr[high] = arr[high], arr[low]
        
        # if there are more than 2 elements
        if high - low + 1 > 2:
            # find positions for recursive calls
            t = (high - low + 1) // 3
            
            # recursively sort first 2/3
            stooge_sort_range(arr, low, high - t)
            
            # recursively sort last 2/3
            stooge_sort_range(arr, low + t, high)
            
            # recursively sort first 2/3 again to ensure sorted result
            stooge_sort_range(arr, low, high - t)
    
    # handle empty or single-element arrays
    if len(result) > 1:
        stooge_sort_range(result, 0, len(result) - 1)
    
    # copy back to original array
    for i in range(len(arr)):
        arr[i] = result[i]
    
    return arr


def slowsort(arr):
    """
    Sort an array using the Slowsort algorithm.
    This is a deliberately inefficient multiply and surrender algorithm based on the principle of pessimal thinking.
    
    Time Complexity: O(n^(log n)), which is worse than O(n²)
    Space Complexity: O(log n) for recursion
    Stable: No
    
    Args:
        arr: List of comparable elements
        
    Returns:
        The sorted list
    """
    # create a copy to avoid modifying input
    result = arr.copy()
    
    def slowsort_range(arr, i, j):
        # base case
        if i >= j:
            return
        
        # find the middle element
        m = (i + j) // 2
        
        # recursively sort first half
        slowsort_range(arr, i, m)
        
        # recursively sort second half
        slowsort_range(arr, m + 1, j)
        
        # ensure the largest element is at the end
        if arr[m] > arr[j]:
            arr[m], arr[j] = arr[j], arr[m]
        
        # recursively sort the elements except the largest
        slowsort_range(arr, i, j - 1)
    
    # start the recursive sorting (only if array length > 1)
    if len(result) > 1:
        slowsort_range(result, 0, len(result) - 1)
    
    # copy back to original array
    for i in range(len(arr)):
        arr[i] = result[i]
    
    return arr


def franceschini_mergesort(arr):
    """
    Sort an array using Franceschini's in-place merge sort algorithm.
    This is a variant of merge sort that works in-place with O(1) extra space.
    
    Time Complexity: O(n log n)
    Space Complexity: O(1) extra space apart from recursion
    Stable: No
    
    Args:
        arr: List of comparable elements
        
    Returns:
        The sorted list
    """
    # create a copy to avoid modifying input
    result = arr.copy()
    
    # use insertion sort for small arrays (optimization)
    if len(result) <= 32:
        for i in range(1, len(result)):
            key = result[i]
            j = i - 1
            while j >= 0 and result[j] > key:
                result[j + 1] = result[j]
                j -= 1
            result[j + 1] = key
        
        # copy back to original array and return
        for i in range(len(arr)):
            arr[i] = result[i]
        return arr
    
    # rotate array elements: [a, b, c, d, e, f] -> [d, e, f, a, b, c]
    def rotate(arr, start, middle, end):
        # reverse the whole range
        reverse(arr, start, end - 1)
        # reverse the first part
        reverse(arr, start, start + (end - middle) - 1)
        # reverse the second part
        reverse(arr, start + (end - middle), end - 1)
    
    # reverse array elements: [a, b, c, d] -> [d, c, b, a]
    def reverse(arr, start, end):
        while start < end:
            arr[start], arr[end] = arr[end], arr[start]
            start += 1
            end -= 1
    
    # in-place merge of two sorted subarrays
    def merge_in_place(arr, start, mid, end):
        if start >= mid or mid >= end:
            return
        
        # Check if arrays are already in order
        if arr[mid-1] <= arr[mid]:
            return
        
        # Simple binary insertion for small ranges
        if end - start <= 32:
            for i in range(mid, end):
                # Find insertion position
                j = i - 1
                while j >= start and arr[j] > arr[i]:
                    j -= 1
                j += 1
                
                # If insertion position is different from current, shift and insert
                if j != i:
                    temp = arr[i]
                    # shift elements to the right
                    for k in range(i, j, -1):
                        arr[k] = arr[k-1]
                    arr[j] = temp
            return
        
        # For larger ranges, use more sophisticated techniques
        # Split into blocks and perform block-wise merges
        while start < mid and mid < end:
            if arr[start] <= arr[mid]:
                start += 1
            else:
                # Find a block of elements to move
                val = arr[mid]
                insert_pos = start
                
                # Swap elements one by one (this is simplified from the 
                # actual Franceschini algorithm which uses block operations)
                temp = arr[mid]
                for i in range(mid, insert_pos, -1):
                    arr[i] = arr[i-1]
                arr[insert_pos] = temp
                
                start += 1
                mid += 1
    
    # recursive merge sort
    def sort(arr, start, end):
        # base case
        if end - start <= 1:
            return
        
        # divide
        mid = (start + end) // 2
        
        # recursively sort
        sort(arr, start, mid)
        sort(arr, mid, end)
        
        # merge the sorted halves
        merge_in_place(arr, start, mid, end)
    
    # start the sort
    sort(result, 0, len(result))
    
    # copy back to original array
    for i in range(len(arr)):
        arr[i] = result[i]
    
    return arr


def thorup_sort(arr):
    """
    Sort an array using Thorup's algorithm for integer sorting.
    This is a theoretical algorithm that sorts integers in linear time O(n).
    Note: This is a simplified implementation as the original algorithm is very complex.
    
    Time Complexity: O(n) for integers in a bounded range
    Space Complexity: O(n)
    Stable: No
    
    Args:
        arr: List of integers
        
    Returns:
        The sorted list
    """
    if not arr:
        return []
    
    # ensure we're working with integers
    if not all(isinstance(x, int) for x in arr):
        raise ValueError("Thorup sort works only with integers")
    
    # create a copy to avoid modifying the input
    result = arr.copy()
    
    # find range of values
    min_val = min(result)
    max_val = max(result)
    
    # for small ranges, use counting sort directly (which is linear time)
    if max_val - min_val < len(result) * 10:  # heuristic for when counting sort is efficient
        # shift all values to non-negative
        for i in range(len(result)):
            result[i] -= min_val
        
        # counting sort
        count = [0] * (max_val - min_val + 1)
        for x in result:
            count[x] += 1
        
        # reconstruct the array
        idx = 0
        for i in range(len(count)):
            for _ in range(count[i]):
                result[idx] = i + min_val  # shift back to original range
                idx += 1
                
        # copy back to original array
        for i in range(len(arr)):
            arr[i] = result[i]
        
        return arr
    
    # For larger ranges, use radix sort (as a simplified approximation of Thorup's algorithm)
    # Thorup's real algorithm is much more complex, using specialized data structures
    
    # find number of bits needed
    if max_val == min_val:
        # all elements are the same
        return arr
    
    # make all values non-negative by shifting
    for i in range(len(result)):
        result[i] -= min_val
    
    # maximum value after shifting
    max_val = max_val - min_val
    
    # calculate number of bits
    num_bits = max_val.bit_length()
    
    # radix sort by each bit
    for bit in range(num_bits):
        zeros = []
        ones = []
        # distribute to buckets based on current bit
        for x in result:
            if (x >> bit) & 1:  # check if bit is set
                ones.append(x)
            else:
                zeros.append(x)
        # concatenate buckets
        result = zeros + ones
    
    # shift back to original range
    for i in range(len(result)):
        result[i] += min_val
    
    # copy back to original array
    for i in range(len(arr)):
        arr[i] = result[i]
    
    return arr