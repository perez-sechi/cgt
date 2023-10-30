from decimal import *


def float_round_to_zero(x):
    precision = getcontext().prec
    lowest_num = Decimal(f"1e-{precision}")
    return float(x) if abs(x) > lowest_num else 0
