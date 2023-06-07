from itertools import combinations
import numpy as np


def characteristic_function(N):
    # Define your characteristic function here
    # This is a dummy function that assigns a random payoff to each coalition
    v = {}
    for i in range(len(N) + 1):
        for coalition in combinations(N, i):
            v[coalition] = np.random.randint(0, 100)
    return v


def excess(coalition, payoff, v):
    return v[coalition] - sum(payoff[i] for i in coalition)


def nucleolus(N, v):
    # Step 1: Calculate the pre-imputation
    # Assign to each player their singleton coalition value
    payoff = {i: v[(i,)] for i in N}
    remaining_value = v[tuple(N)] - sum(payoff.values())
    for i in N:
        # Distribute the remaining value equally among the players
        payoff[i] += remaining_value / len(N)

    # Step 2: Calculate the excess function for each coalition
    excesses = {coalition: excess(coalition, payoff, v)
                for coalition in v.keys()}

    # Step 3: Order the coalitions by excess
    ordered_coalitions = sorted(excesses, key=excesses.get, reverse=True)

    # Step 4: Find the nucleolus
    max_excess = excesses[ordered_coalitions[0]]
    for coalition in ordered_coalitions:
        if excesses[coalition] < max_excess:
            max_excess = excesses[coalition]
            for i in coalition:
                payoff[i] += (max_excess - excesses[coalition]) / \
                    len(coalition)

    return payoff


N = (1, 2, 3)  # Define the set of players
v = characteristic_function(N)  # Define the characteristic function
nucleolus_payoff = nucleolus(N, v)  # Calculate the nucleolus
print(nucleolus_payoff)
