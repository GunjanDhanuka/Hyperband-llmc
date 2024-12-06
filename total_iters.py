# import numpy as np

# max_iters = 500
# eta = 3

# s_max = int(np.log(max_iters) / np.log(eta))
# print(f"s_max: {s_max}")

# total_iterations = 0
# for s in reversed(range(s_max + 1)):
#     n = int(np.ceil((s_max + 1) * (eta ** s) / (s + 1)))
#     r = int(max_iters * (eta ** (-s)))
#     for i in range(s+1):
#         n_i = int(n * (eta ** (-i)))
#         r_i = int(r * (eta ** i))
#         total_iterations += n * r

# print(f"Total iterations: {total_iterations}")

import numpy as np
from math import ceil, log

def calculate_hyperband_iterations(max_iter, eta):
    # Calculate s_max
    s_max = int(log(max_iter) / log(eta))
    print(f"s_max = {s_max}")
    
    total_iters = 0
    total_configs = 0
    
    # For each bracket from s_max down to 0
    for s in range(s_max, -1, -1):
        # Calculate initial number of configurations
        n = ceil((s_max + 1) * (eta ** s) / (s + 1))
        # Calculate initial iterations per config
        r = max_iter * (eta ** (-s))
        
        total_configs += n
        
        print(f"\nBracket s={s}:")
        bracket_total = 0
        prev_ri = 0
        ri = 0
        
        # For each round in the bracket
        for i in range(s + 1):
            # Calculate ni and ri for this round
            ni = int(n * (eta ** (-i)))
            prev_ri = ri
            ri = int(r * (eta ** i))

            
            # Calculate iterations for this round
            round_iters = ni * (ri - prev_ri)
            bracket_total += round_iters
            
            print(f"Round {i}: {ni} configs × {ri} iters = {round_iters}")
        
        print(f"Bracket total: {bracket_total}")
        total_iters += bracket_total
    
    print(f"\nTotal iterations across all brackets: {total_iters}")
    print(f"Total configurations: {total_configs}")
    
    # Calculate approximate runtime and cost
    seconds_per_iter = 2.5
    total_hours = (total_iters * seconds_per_iter) / 3600
    cost_per_hour = 1.5
    total_cost = total_hours * cost_per_hour
    
    print(f"\nEstimated runtime: {total_hours:.2f} hours")
    print(f"Estimated cost at ${cost_per_hour}/hour: ${total_cost:.2f}")
    
    return total_iters

# Example usage
max_iter = 735  # 3^5
eta = 3
total = calculate_hyperband_iterations(max_iter, eta)
