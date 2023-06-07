from typing import Any

from decimal import *
from typing import List
from math import factorial
from itertools import combinations, chain


def float_round_to_zero(x):
    precision = getcontext().prec
    lowest_num = Decimal(f"1e-{precision}")
    return float(x) if abs(x) > lowest_num else 0


def calculate_sh_i(
    i: int, n_set: list, v: callable, original: Any
):
    sh_i = Decimal(0.0)
    n = len(n_set)
    n_minus_i = n_set[:]
    n_minus_i.remove(i)
    power_set = chain.from_iterable(
        combinations(n_minus_i, subset_size)
        for subset_size in range(0, len(n_minus_i) + 1)
    )
    for subset in power_set:
        sublist = list(subset)
        s = len(sublist)
        v_sublist_plus_i = Decimal(v(n_set, sublist + [i], original))
        v_sublist = Decimal(v(n_set, sublist, original))
        proportion = (
            Decimal(factorial(s)) * Decimal(factorial(n - s - 1))
        ) / Decimal(factorial(n - 1))
        x_O_i = proportion * (v_sublist_plus_i - v_sublist)
        sh_i += x_O_i

    sh_i = sh_i / Decimal(n)

    return i, sh_i


def calculate_cost_sh_i(
    i: int, n_set: list, c: callable, original: Any
):
    sh_i = Decimal(0.0)
    n = len(n_set)
    power_set = chain.from_iterable(
        combinations(n_set, subset_size)
        for subset_size in range(1, n + 1)
    )
    for s_set in power_set:
        if i not in s_set:
            continue
        s = len(s_set)
        s_list = list(s_set)
        s_minus_i = s_list[:]
        s_minus_i.remove(i)
        c_s_list = Decimal(c(n_set, s_list, original))
        c_s_minus_i = Decimal(c(n_set, s_minus_i, original))
        proportion = (
            Decimal(factorial(n - s)) * Decimal(factorial(s - 1))
        ) / Decimal(factorial(n))
        c_O_i = proportion * (c_s_list - c_s_minus_i)
        sh_i += c_O_i

    return i, sh_i