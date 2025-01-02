from decimal import *


def float_round_to_zero(x):
    precision = getcontext().prec
    lowest_num = Decimal(f"1e-{precision}")
    if abs(x) > lowest_num:
        return float(x)
    else:
        return 0
