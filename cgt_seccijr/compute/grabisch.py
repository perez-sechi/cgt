from typing import Any

from decimal import *
from math import factorial
from itertools import combinations

from cgt_seccijr.util.float import float_round_to_zero


def calculate_interaction_ij(
    i: int, j: int, n_set: list, v: callable, original: Any
):
    interaction_ij = Decimal(0.0)
    n = len(n_set)
    n_minus_ij = n_set[:]
    n_minus_ij.remove(i)
    n_minus_ij.remove(j)

    for k in range(n - 2 + 1):
        zeta_k = (
            Decimal(factorial(n - k - 2)) * Decimal(factorial(k))
        ) / Decimal(factorial(n - 1))

        s_minus_ij_subsets = combinations(n_minus_ij, k)

        for subset in s_minus_ij_subsets:
            sublist = list(subset)
            mu_sublist = Decimal(v(n_set, sublist, original))
            mu_sublist_plus_i = Decimal(v(n_set, sublist + [i], original))
            mu_sublist_plus_j = Decimal(v(n_set, sublist + [j], original))
            mu_sublist_plus_ij = Decimal(v(n_set, sublist + [i, j], original))
            interaction_ij += zeta_k * (mu_sublist_plus_ij - mu_sublist_plus_i - mu_sublist_plus_j + mu_sublist)

    return i, j, float_round_to_zero(interaction_ij)