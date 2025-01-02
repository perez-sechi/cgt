import numpy as np
import pandas as pd


def code_interval(
    X: pd.DataFrame,
    variable_A_name: str,
    sample_size: int
) -> pd.DataFrame:
    '''
    Code the interval of the input variable A and the SHAP values.
    Parameters:
    X: pandas.DataFrame
        The input data.
    variable_A_name: str
        The name of variable A.
    sample_size: int
        The sample size.
    Returns:
    pandas.DataFrame
        The coded interval.
    '''

    X_sorted = X.sort_values(by=variable_A_name)
    variable_A_index = X.columns.get_loc(variable_A_name)

    labels = []
    current_sample_size = 0
    for i, row in X_sorted.iterrows():
        if current_sample_size == 0:
            current_label = f"[{row[variable_A_name]},"
        current_sample_size += 1
        if current_sample_size == sample_size or len(labels) + current_sample_size >= len(X):
            current_label += f"{row[variable_A_name]}]"
            current_label_array = np.repeat(current_label, current_sample_size)
            labels.extend(current_label_array)
            current_sample_size = 0

    X_sorted[f'{variable_A_name}_interval'] = labels
    X_unsorted = X_sorted.sort_index()

    return X_unsorted