from typing import Any

from decimal import *
from typing import List
from random import shuffle
from math import ceil


def float_round_to_zero(x):
    precision = getcontext().prec
    lowest_num = Decimal(f"1e-{precision}")
    return float(x) if abs(x) > lowest_num else 0


def shuffle_x_O_i(l: int, i: int, n_set: list):
    r = n_set[:]
    r.remove(i)
    shuffle(r)
    return r[0:l]


def shuffle_x_O_ij(l: int, i: int, j: int, n_set: list):
    r = n_set[:]
    r.remove(i)
    r.remove(j)
    shuffle(r)
    return r[0:l]


def sample_x_O_i(l: int, i: int, n_set: list, v: callable, original: Any):
    g = shuffle_x_O_i(l, i, n_set)
    return Decimal(v(n_set, g + [i], original)) - Decimal(v(n_set, g, original))


def sample_x_O_ij(
    l: int, i: int, j: int, n_set: list, v: callable, original: Any
):
    g = shuffle_x_O_ij(l, i, j, n_set)
    v_predj = Decimal(v(n_set, g, original))
    v_predj_i = Decimal(v(n_set, g + [i], original))
    v_predj_j = Decimal(v(n_set, g + [j], original))
    v_predj_i_j = Decimal(v(n_set, g + [i, j], original))
    return v_predj_i_j - v_predj_i - v_predj_j + v_predj


def estimate_s_2_il(
    l: int, i: int, n_set: list, v: callable, original: Any, m_exp_il: int
):
    sum_cuad_l = Decimal(0.0)
    sh_l_i = Decimal(0.0)

    for c in range(m_exp_il):
        x_O_i = sample_x_O_i(l, i, n_set, v, original)
        sh_l_i += x_O_i
        sum_cuad_l += Context.power(x_O_i, Decimal(2))

    s_2_il = Decimal((
        sum_cuad_l - Context.power(sh_l_i, Decimal(2)) / Decimal(m_exp_il)
    ) / Decimal(m_exp_il - 1))

    return l, i, sh_l_i, s_2_il


def estimate_s_2_ijl(
    l: int, i: int, j: int, n_set: list,
    v: callable, original: Any, m_exp_ijl: int
):
    sum_cuad_l = Decimal(0.0)
    sh_l_ij = Decimal(0.0)

    for c in range(m_exp_ijl):
        x_O_ij = sample_x_O_ij(l, i, j, n_set, v, original)
        sh_l_ij += x_O_ij
        sum_cuad_l += Context.power(x_O_ij, 2)

    s_2_ijl = Decimal((
        sum_cuad_l - Context.power(sh_l_ij, 2) / m_exp_ijl
    ) / Decimal(m_exp_ijl - 1))

    return l, i, j, sh_l_ij, s_2_ijl


def estimate_sh_l_i(
    l: int, i: int, n_set: list, v: callable, original: Any,
    m_exp_il: int, m_st_il: int, sh_l_i: Decimal
):
    for c in range(m_st_il):
        x_O_i = sample_x_O_i(l, i, n_set, v, original)
        sh_l_i += x_O_i

    sh_l_i = sh_l_i / Decimal(m_exp_il + m_st_il)

    return l, i, sh_l_i


def estimate_sh_l_ij(
    l: int, i: int, j: int, n_set: list, v: callable, original: Any,
    m_exp_ijl: int, m_st_ijl: int, sh_l_ij: Decimal
):
    for c in range(m_st_ijl):
        x_O_ij = sample_x_O_ij(l, i, j, n_set, v, original)
        sh_l_ij += x_O_ij

    sh_l_ij = sh_l_ij / Decimal(m_exp_ijl + m_st_ijl)

    return l, i, j, sh_l_ij


def castro(n: int, m: int, v: callable, original: Any) -> List[float]:
    n_set = list(n_set)
    m_exp_il = ceil(m / (2 * n * n))
    sh_l_i = [[Decimal(0.0) for i in n_set] for i in n_set]
    s_2_il = [[Decimal(0.0) for i in n_set] for i in n_set]
    m_st_il = [[0 for i in n_set] for i in n_set]

    for l in n_set:
        for i in n_set:
            l_proc, i_proc, sh_l_i_proc, s_2_il_proc = estimate_s_2_il(
                l, i, n_set, v, original, m_exp_il)
            sh_l_i[l_proc][i_proc] += sh_l_i_proc
            s_2_il[l_proc][i_proc] = s_2_il_proc

    s_2_i_k_sum = 0
    for o in s_2_il:
        for p in o:
            s_2_ij_k_sum += float(p)

    for l in n_set:
        for i in n_set:
            m_il = m * (s_2_il[l][i] ** 2) / s_2_i_k_sum
            m_st_il[l][i] = ceil(m_il - m_exp_il) if m_il > m_exp_il else 1

    for l in n_set:
        for i in n_set:
            l_proc, i_proc, sh_l_i_proc = estimate_sh_l_i(
                l, i, n_set, v, original, m_exp_il, m_st_il[l][i], sh_l_i[l][i])
            sh_l_i[l_proc][i_proc] = sh_l_i_proc

    sh_st_opt_i = [0.0 for i in n_set]
    for l in n_set:
        for i in n_set:
            sh_st_opt_i[i] += sh_l_i[l][i]
    for i in n_set:
        sh_st_opt_i[i] = float(sh_st_opt_i[i] / Decimal(n))

    return sh_st_opt_i


def castro_interaction_index(n: int, m: int, v: callable, original: Any) -> List[List[float]]:
    n_set = list(n_set)
    m_exp_ijl = ceil(m / (2 * n * n * n))
    sh_l_ij = [[[Decimal(0.0) for j in n_set]
                for i in n_set] for l in n_set]
    s_2_ijl = [[[Decimal(0.0) for j in n_set]
                for i in n_set] for l in n_set]
    m_st_ijl = [[[0 for j in n_set] for i in n_set] for l in n_set]

    for l in n_set:
        for j in n_set:
            for i in n_set:
                if j != i:
                    l_proc, i_proc, j_proc, sh_l_ij_proc, s_2_ijl_proc = estimate_s_2_ijl(
                        l, i, j, n_set, v, original, m_exp_ijl)
                    sh_l_ij[l_proc][i_proc][j_proc] += sh_l_ij_proc
                    s_2_ijl[l_proc][i_proc][j_proc] = s_2_ijl_proc

    s_2_ij_k_sum = 0
    for o in s_2_ijl:
        for p in o:
            for q in p:
                s_2_ij_k_sum += float(q)

    for l in n_set:
        for i in n_set:
            for j in n_set:
                if j != i:
                    m_ijl = m * (float(s_2_ijl[l][i][j]) ** 2) / s_2_ij_k_sum
                    m_st_ijl[l][i][j] = \
                        ceil(m_ijl - m_exp_ijl) \
                        if m_ijl > m_exp_ijl else 1

    for l in n_set:
        for j in n_set:
            for i in n_set:
                if j != i:
                    l_proc, i_proc, j_proc, sh_l_ij_proc = estimate_sh_l_ij(
                        l, i, j, n_set, v, original, m_exp_ijl,
                        m_st_ijl[l][i][j], sh_l_ij[l][i][j]
                    )
                    sh_l_ij[l_proc][i_proc][j_proc] = sh_l_ij_proc

    sh_st_opt_ij = [[0.0 for j in n_set] for i in n_set]
    for l in n_set:
        for i in n_set:
            for j in n_set:
                if j != i:
                    sh_st_opt_ij[i][j] += sh_l_ij[l][i][j]
    for i in n_set:
        for j in n_set:
            if j != i:
                sh_st_opt_ij[i][j] = float(sh_st_opt_ij[i][j] / Decimal(n))

    return sh_st_opt_ij