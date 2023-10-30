from typing import Any
import multiprocess as mp
from decimal import *
from typing import List

from cgt_seccijr.util import float_round_to_zero
from cgt_seccijr.sh_i import calculate_cost_sh_i, calculate_sh_i


def cost(n: int, c: callable, original: Any) -> List[List[float]]:
    n_set = list(range(n))
    sh_i = [0.0 for i in n_set]

    with mp.Pool() as pool:
        results = pool.starmap(
            calculate_cost_sh_i,
            [[i, n_set, c, original] for i in n_set]
        )
        for result in results:
            i_proc, sh_i_proc = result
            sh_i[i_proc] = float_round_to_zero(sh_i_proc)

    return sh_i


def cost_diff(n: int, c: callable, original: Any) -> List[List[float]]:
    n_set = list(range(n))
    sh_i = [0.0 for i in n_set]

    with mp.Pool() as pool:
        results = pool.starmap(
            calculate_cost_sh_i,
            [[i, n_set, c, original] for i in n_set]
        )
        for result in results:
            i_proc, sh_i_proc = result
            sh_i[i_proc] = sh_i_proc

    sh_ij = [[0.0 for j in n_set] for i in n_set]
    for j in n_set:
        n_minus_j = n_set[:]
        n_minus_j.remove(j)
        with mp.Pool() as pool:
            results = pool.starmap(
                calculate_cost_sh_i,
                [[i, n_minus_j, c, original] for i in n_set if i != j]
            )
            for result in results:
                i_minus_j_proc, sh_i_minus_j_proc = result
                sh_ij[i_minus_j_proc][j] = float_round_to_zero(
                    sh_i[i_minus_j_proc] - sh_i_minus_j_proc
                )

    return sh_ij


def exact(n: int, v: callable, original: Any) -> List[List[float]]:
    n_set = list(range(n))
    sh_i = [0.0 for i in n_set]

    with mp.Pool() as pool:
        results = pool.starmap(
            calculate_sh_i,
            [[i, n_set, v, original] for i in n_set]
        )
        for result in results:
            i_proc, sh_i_proc = result
            sh_i[i_proc] = float_round_to_zero(sh_i_proc)

    return sh_i


def exac_n_set(n_set: List[int], v: callable, original: Any) -> List[List[float]]:
    sh_i = {}

    with mp.Pool() as pool:
        results = pool.starmap(
            calculate_sh_i,
            [[i, n_set, v, original] for i in n_set]
        )
        for result in results:
            i_proc, sh_i_proc = result
            sh_i[i_proc] = float_round_to_zero(sh_i_proc)

    return sh_i


def exact_diff(n: int, v: callable, original: Any) -> List[List[float]]:
    n_set = list(range(n))
    sh_i = [0.0 for i in n_set]

    with mp.Pool() as pool:
        results = pool.starmap(
            calculate_sh_i,
            [[i, n_set, v, original] for i in n_set]
        )
        for result in results:
            i_proc, sh_i_proc = result
            sh_i[i_proc] = sh_i_proc

    sh_ij = [[0.0 for j in n_set] for i in n_set]

    for j in n_set:
        n_minus_j = n_set[:]
        n_minus_j.remove(j)
        with mp.Pool() as pool:
            results = pool.starmap(
                calculate_sh_i,
                [[i, n_minus_j, v, original] for i in n_set if i != j]
            )
            for result in results:
                i_minus_j_proc, sh_i_minus_j_proc = result
                sh_ij[i_minus_j_proc][j] = float_round_to_zero(
                    sh_i[i_minus_j_proc] - sh_i_minus_j_proc
                )

    return sh_ij
