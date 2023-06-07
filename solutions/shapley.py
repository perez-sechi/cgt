import os
import sys
from typing import Any

from decimal import *
from typing import List

try:
    file_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    file_dir = "."

sys.path.append(os.path.join(file_dir, "..", "..", "."))
from solutions.sh_i import calculate_cost_sh_i, calculate_sh_i, float_round_to_zero


def cost(n: int, c: callable, original: Any) -> List[List[float]]:
    n_set = list(range(n))
    sh_i = [0.0 for i in n_set]

    for i in n_set:
        i_proc, sh_i_proc = calculate_cost_sh_i(i, n_set, c, original)
        sh_i[i_proc] = float_round_to_zero(sh_i_proc)

    return sh_i


def cost_diff(n: int, c: callable, original: Any) -> List[List[float]]:
    n_set = list(range(n))
    sh_i = [0.0 for i in n_set]

    for i in n_set:
        i_proc, sh_i_proc = calculate_cost_sh_i(i, n_set, c, original)
        sh_i[i_proc] = sh_i_proc

    sh_ij = [[0.0 for j in n_set] for i in n_set]
    for j in n_set:
        n_minus_j = n_set[:]
        n_minus_j.remove(j)
        for i in n_set:
            if i != j:
                i_minus_j_proc, sh_i_minus_j_proc = calculate_cost_sh_i(
                    i, n_minus_j, c, original
                )
                sh_ij[i_minus_j_proc][j] = float_round_to_zero(
                    sh_i[i_minus_j_proc] - sh_i_minus_j_proc
                )

    return sh_ij


def exact(n: int, v: callable, original: Any) -> List[List[float]]:
    n_set = list(range(n))
    sh_i = [0.0 for i in n_set]

    for i in n_set:
        i_proc, sh_i_proc = calculate_sh_i(i, n_set, v, original)
        sh_i[i_proc] = float_round_to_zero(sh_i_proc)

    return sh_i


def exac_n_set(n_set: List[int], v: callable, original: Any) -> List[List[float]]:
    sh_i = {}

    for i in n_set:
        i_proc, sh_i_proc = calculate_sh_i(i, n_set, v, original)
        sh_i[i_proc] = float_round_to_zero(sh_i_proc)

    return sh_i


def exact_diff(n: int, v: callable, original: Any) -> List[List[float]]:
    n_set = list(range(n))
    sh_i = [0.0 for i in n_set]

    for i in n_set:
        i_proc, sh_i_proc = calculate_sh_i(i, n_set, v, original)
        sh_i[i_proc] = sh_i_proc

    sh_ij = [[0.0 for j in n_set] for i in n_set]

    for j in n_set:
        n_minus_j = n_set[:]
        n_minus_j.remove(j)
        for i in n_set:
            if i != j:
                i_minus_j_proc, sh_i_minus_j_proc = calculate_sh_i(
                    i, n_minus_j, v, original
                )
                sh_ij[i_minus_j_proc][j] = float_round_to_zero(
                    sh_i[i_minus_j_proc] - sh_i_minus_j_proc
                )

    return sh_ij
