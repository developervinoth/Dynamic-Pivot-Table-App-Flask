import multiprocessing as mp
from functools import partial
import time

def process_item_pair(item1, list2, action_func):
    """
    Process one item from list1 against all items in list2
    """
    results = []
    for item2 in list2:
        # Perform your action here
        result = action_func(item1, item2)
        results.append(result)
    return results

def your_action_function(item1, item2):
    """
    Define your custom action here
    Replace this with whatever operation you want to perform
    """
    # Example: simple computation
    return item1 * item2 + len(str(item1)) + len(str(item2))

def parallel_nested_loop(list1, list2, action_func, num_processes=None):
    """
    Process nested loops using multiprocessing
    """
    if num_processes is None:
        num_processes = mp.cpu_count()
    
    # Create a partial function with list2 and action_func pre-filled
    worker_func = partial(process_item_pair, list2=list2, action_func=action_func)
    
    # Use multiprocessing Pool
    with mp.Pool(processes=num_processes) as pool:
        results = pool.map(worker_func, list1)
    
    return results

# Example usage
if __name__ == "__main__":
    # Create sample large lists
    list1 = list(range(1000))
    list2 = list(range(500))
    
    print(f"Processing {len(list1)} x {len(list2)} = {len(list1) * len(list2)} operations")
    
    # Time the parallel execution
    start_time = time.time()
    results = parallel_nested_loop(list1, list2, your_action_function)
    end_time = time.time()
    
    print(f"Parallel execution time: {end_time - start_time:.2f} seconds")
    print(f"Results shape: {len(results)} outer results, {len(results[0])} inner results each")