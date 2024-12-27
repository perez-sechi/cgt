import numpy as np
import pandas as pd
from scipy import stats


def sample_size_kruskal_wallis(
    X: pd.DataFrame,
    shap_values: np.ndarray,
    variable_name: str,
    acceleration: int = 1
) -> int:
    '''
    Find the sample size of the groups divide by the input variable name
    such that it maximises the significance of the Kruskal-Wallis Test for the
    shape values.
    Parameters:
    X: pandas.DataFrame
        The input data.
    shap_values: numpy.ndarray
        The SHAP values.
    variable_name: str
        The name of the variable to divide the groups by.
    acceleration: int
        The acceleration factor to increase the sample size.
    Returns:
    int
        The sample size that maximises the significance of the Kruskal-Wallis Test.
    '''
    X_sorted = X.sort_values(by=variable_name)
    variable_index = X_sorted.columns.get_loc(variable_name)
    shap_values_sorted = shap_values[:, variable_index][X_sorted.index]
    n = len(X_sorted)
    p_values = []
    third = n // 3
    if third < 5:
        return None
    sample_sizes = list(range(5, third, acceleration))
    for sample_size in sample_sizes:
        groups = [
            shap_values_sorted[i:i+sample_size]
            for i in range(0, n, sample_size)
        ]
        if len(groups[-1]) < 5:
            groups[-2] = np.concatenate([groups[-2], groups[-1]])
            groups.pop()
        if len(groups) < 3:
            continue
        p_value = stats.kruskal(*groups).pvalue
        p_values.append(p_value)

    if len(p_values) == 0:
        return None

    min_p_value = min(p_values)
    min_p_value_index = p_values.index(min_p_value)
    sample_size = sample_sizes[min_p_value_index]

    return sample_size
