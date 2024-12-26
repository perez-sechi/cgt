from cgt_seccijr.exploration.schema import is_categorical

import shap
import os
import sys
import numpy as np
import matplotlib.pylab as pl
import matplotlib.ticker as mtick


def shap_percentile_distribution_value(
    variable_index,
    X_variable,
    shap_values,
    starting_percentile,
    ending_percentile
):
    starting_percentile_value = np.percentile(
        X_variable, starting_percentile
    )
    ending_percentile_value = np.percentile(
        X_variable, ending_percentile
    )
    idx_survivors = np.where(
        (X_variable >= starting_percentile_value)
        & (X_variable < ending_percentile_value)
    )[0]

    sum_shap_values = np.sum(shap_values[idx_survivors], axis=(0, 1))
    top_variable_shap_values = shap_values[idx_survivors, variable_index]

    return 100 * np.sum(top_variable_shap_values) / sum_shap_values


def shap_interaction_percentile_distribution_value(
    starting_percentile,
    ending_percentile,
    shap_interaction_values,
    variable_A_index,
    variable_B_index
):
    variable_shap_interaction_values = shap_interaction_values[
        :,
        (variable_A_index, variable_B_index),
        (variable_B_index, variable_A_index)
    ]
    starting_percentile_value = np.percentile(
        variable_shap_interaction_values, starting_percentile
    )
    ending_percentile_value = np.percentile(
        variable_shap_interaction_values, ending_percentile
    )
    idx_survivors = np.where(
        (variable_shap_interaction_values >= starting_percentile_value)
        & (variable_shap_interaction_values < ending_percentile_value)
    )[0]

    sum_ineraction_shap_values = np.sum(
        shap_interaction_values[idx_survivors], axis=(0, 1, 2)
    )
    top_variable_interaction_shap_values = variable_shap_interaction_values[idx_survivors]

    return 100 * np.sum(top_variable_interaction_shap_values) / sum_ineraction_shap_values


def shap_category_distribution_value(
    category,
    variable_index,
    X_variable,
    shap_values
):
    idx_survivors = np.where(
        X_variable == category
    )[0]

    sum_shap_values = np.sum(shap_values[idx_survivors], axis=(0, 1))
    top_variable_shap_values = shap_values[idx_survivors, variable_index]

    return 100 * np.sum(top_variable_shap_values) / sum_shap_values

def plot_shap_categorical_distribution(variable_name, X, shap_values):
    X_variable = X[variable_name]

    if not is_categorical(X_variable):
        raise ValueError(f"The variable {variable_name} is not categorical")

    variable_index = X.columns.get_loc(variable_name)
    X_unique = X_variable.unique()
    
    variable_shap_values = shap_values[:, variable_index]

    fig, ax = pl.subplots()
    width = 0.5

    distribution = [
        shap_category_distribution_value(
            category,
            variable_index,
            X_variable,
            shap_values
        ) for category in X_unique
    ]
    cmap = shap.plots.colors._colors.red_blue
    colors = cmap(np.linspace(0, 1, len(X_unique)))

    ticks = [str(x) for x in X_unique]
    p = ax.bar(ticks, distribution, width, color=colors)
    ax.bar_label(p, label_type='center', color='white', fontsize=8, fmt='%.1f%%')
    pl.xlabel(f"{variable_name} categories")
    pl.ylabel(f"Percentage")
    pl.title(f"{variable_name} SHAP percentage distribution")

    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(labelsize=8)
    pl.grid(axis='y', linestyle=':', linewidth=0.5, alpha=0.5)
    pl.show()


def plot_shap_numerical_distribution(variable_name, X, shap_values, n_bins):
    bins = np.arange(n_bins)
    X_variable = X[variable_name]

    X_max = np.max(X_variable)
    shap_distribution = np.zeros(n_bins)
    variable_distribution = np.zeros(n_bins)
    variable_index = X.columns.get_loc(variable_name)

    variable_shap_values = shap_values[:, variable_index]
    for i in bins:
        starting_percentile = i * 100 / n_bins
        ending_percentile = (i + 1) * 100 / n_bins
        shap_distribution[i] = shap_percentile_distribution_value(
            variable_index,
            X_variable,
            shap_values,
            starting_percentile,
            ending_percentile
        )

        starting_percentile_value = np.percentile(
            variable_shap_values, starting_percentile
        )
        ending_percentile_value = np.percentile(
            variable_shap_values, ending_percentile
        )
        idx_survivors = np.where(
            (variable_shap_values >= starting_percentile_value)
            & (variable_shap_values < ending_percentile_value)
        )[0]
        variable_mean = X_variable.iloc[idx_survivors].mean()
        variable_distribution[i] = 100 * variable_mean / X_max

    fig, ax = pl.subplots()

    cmap = shap.plots.colors._colors.red_blue
    colors = cmap(np.linspace(0, 1, n_bins))
    ax.bar(
        bins * 100 / n_bins, shap_distribution, color=colors
    )
    ax.plot(bins * 100 / n_bins, variable_distribution, color="grey")

    pl.legend([
        f"{variable_name} mean value percentage",
        f"{variable_name} SHAP percentage in current percentile"
    ])
    pl.xlabel(f"{variable_name} SHAP percentile")
    pl.ylabel(f"Percentage")
    pl.title(f"{variable_name} SHAP percentage and value distribution")

    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(labelsize=8)
    pl.grid(axis='y', linestyle=':', linewidth=0.5, alpha=0.5)
    
    pl.show()


def plot_shap_distribution(variable_name, X, shap_values, n_bins):
    if is_categorical(X[variable_name]):
        plot_shap_categorical_distribution(
            variable_name, X, shap_values
        )
    else:
        plot_shap_numerical_distribution(
            variable_name, X, shap_values, n_bins
        )


def plot_shap_interaction_numerical_numerical_distribution(
    variable_A_name, variable_B_name, X, shap_interaction_values, n_bins
):
    A_variable = X[variable_A_name]
    B_variable = X[variable_B_name]

    if is_categorical(A_variable) or is_categorical(B_variable):
        raise ValueError(
            "Both variables must be numerical to plot the distribution"
        )

    A_max = np.max(A_variable)
    B_max = np.max(B_variable)
    shap_interaction_distribution = np.zeros(n_bins)
    variable_A_distribution = np.zeros(n_bins)
    variable_A_index = X.columns.get_loc(variable_A_name)
    variable_B_distribution = np.zeros(n_bins)
    variable_B_index = X.columns.get_loc(variable_B_name)

    bins = np.arange(n_bins)

    variable_shap_interaction_values = shap_interaction_values[
        :,
        (variable_A_index, variable_B_index),
        (variable_B_index, variable_A_index)
    ]
    for i in bins:
        starting_percentile = i * 100 / n_bins
        ending_percentile = (i + 1) * 100 / n_bins

        shap_interaction_distribution[i] = shap_interaction_percentile_distribution_value(
            starting_percentile,
            ending_percentile,
            shap_interaction_values,
            variable_A_index,
            variable_B_index
        )
        starting_percentile_value = np.percentile(
            variable_shap_interaction_values, starting_percentile
        )
        ending_percentile_value = np.percentile(
            variable_shap_interaction_values, ending_percentile
        )
        idx_survivors = np.where(
            (variable_shap_interaction_values >= starting_percentile_value)
            & (variable_shap_interaction_values < ending_percentile_value)
        )[0]
        variable_A_mean = A_variable.iloc[idx_survivors].mean()
        variable_A_distribution[i] = 100 * variable_A_mean / A_max
        variable_B_mean = B_variable.iloc[idx_survivors].mean()
        variable_B_distribution[i] = 100 * variable_B_mean / B_max

    fig, ax = pl.subplots()

    ax.bar(
        bins * 100 / n_bins, shap_interaction_distribution
    )
    ax.plot(bins * 100 / n_bins, variable_A_distribution, color="green")
    ax.plot(bins * 100 / n_bins, variable_B_distribution, color="red")

    pl.legend([
        f"{variable_A_name} mean value percentage",
        f"{variable_B_name} mean value percentage",
        "SHAP Interaction percentage in current percentile"
    ])
    pl.xlabel("SHAP Interaction percentile")
    pl.ylabel(f"Percentage")
    pl.title(
        f"{variable_A_name}, {
            variable_B_name} SHAP Interaction percentage and value distribution"
    )

    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    
    pl.show()


def plot_shap_interaction_numerical_categorical_distribution(
    variable_A_name, variable_B_name, X, shap_interaction_values, n_bins
):
    A_variable = X[variable_A_name]
    B_variable = X[variable_B_name]

    if is_categorical(A_variable) and not is_categorical(B_variable):
        variable_N_name = variable_B_name
        variable_C_name = variable_A_name
    elif is_categorical(B_variable) and not is_categorical(A_variable):
        variable_N_name = variable_A_name
        variable_C_name = variable_B_name
    else:
        raise ValueError(
            "One variable must be categorical and the other numerical"
        )

    N_variable = X[variable_N_name]
    variable_N_index = X.columns.get_loc(variable_N_name)
    C_variable = X[variable_C_name]
    variable_C_index = X.columns.get_loc(variable_C_name)

    bins = np.arange(n_bins)
    N_max = np.max(N_variable)
    C_unique = C_variable.unique()

    variable_N_distribution = np.zeros(n_bins)
    shap_interaction_distribution = dict(
        [(str(category), np.zeros(n_bins)) for category in C_unique]
    )

    variable_shap_interaction_values = shap_interaction_values[
        :,
        (variable_N_index, variable_C_index),
        (variable_C_index, variable_N_index)
    ]
    for i in bins:
        starting_percentile = i * 100 / n_bins
        ending_percentile = (i + 1) * 100 / n_bins

        shap_interaction_distribution_value = shap_interaction_percentile_distribution_value(
            starting_percentile,
            ending_percentile,
            shap_interaction_values,
            variable_N_index,
            variable_C_index
        )

        starting_percentile_value = np.percentile(
            variable_shap_interaction_values, starting_percentile
        )
        ending_percentile_value = np.percentile(
            variable_shap_interaction_values, ending_percentile
        )
        idx_survivors = np.where(
            (variable_shap_interaction_values >= starting_percentile_value)
            & (variable_shap_interaction_values < ending_percentile_value)
        )[0]

        variable_N_mean = N_variable.iloc[idx_survivors].mean()
        variable_N_distribution[i] = 100 * variable_N_mean / N_max

        survivors = C_variable.iloc[idx_survivors]
        all_bin_elements = len(survivors)
        for category in C_unique:
            if all_bin_elements == 0:
                category_distribution_value = 0
            else:
                category_bin_elements = np.sum(survivors == category)
                category_distribution_value = shap_interaction_distribution_value * \
                    (category_bin_elements / all_bin_elements)

            shap_interaction_distribution[
                str(category)
            ][i] = category_distribution_value

    fig, ax = pl.subplots()
    bottom = np.zeros(n_bins)
    width = 0.5

    ax.plot(bins * 100 / n_bins, variable_N_distribution, color="green")

    for category, distribution in shap_interaction_distribution.items():
        p = ax.bar(bins, distribution, width, label=category, bottom=bottom)
        bottom += distribution

    pl.legend([f"Variable {variable_N_name} mean value percentage"] + [
        f"{variable_C_name} = {category}" for category in C_unique
    ])

    pl.xlabel("SHAP Interaction percentile")
    pl.ylabel(f"Percentage")
    pl.title(
        f"{variable_A_name}, {
            variable_B_name} SHAP Interaction percentage and value distribution"
    )

    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    
    pl.show()


def plot_shap_interaction_categorical_categorical_distribution(
    variable_A_name, variable_B_name, X, shap_interaction_values, n_bins, partial_sum=True
):
    A_variable = X[variable_A_name]
    B_variable = X[variable_B_name]

    if not is_categorical(A_variable) or not is_categorical(B_variable):
        raise ValueError(
            "Both variables must be categorical to plot the distribution"
        )

    variable_A_index = X.columns.get_loc(variable_A_name)
    variable_B_index = X.columns.get_loc(variable_B_name)

    bins = np.arange(n_bins)
    A_unique = A_variable.unique()
    B_unique = B_variable.unique()
    T_unique = [
        (category_A, category_B)
        for category_B in B_unique for category_A in A_unique
    ]
    variable_distribution = np.zeros(n_bins)
    shap_interaction_distribution = dict([
        (f"{str(category_A)}-{str(category_B)}", np.zeros(n_bins))
        for (category_A, category_B) in T_unique
    ])

    variable_shap_interaction_values = shap_interaction_values[
        :,
        (variable_A_index, variable_B_index),
        (variable_B_index, variable_A_index)
    ]
    for i in bins:
        starting_percentile = i * 100 / n_bins
        ending_percentile = (i + 1) * 100 / n_bins

        shap_interaction_distribution_value = shap_interaction_percentile_distribution_value(
            starting_percentile,
            ending_percentile,
            shap_interaction_values,
            variable_A_index,
            variable_B_index
        )

        starting_percentile_value = np.percentile(
            variable_shap_interaction_values, starting_percentile
        )
        ending_percentile_value = np.percentile(
            variable_shap_interaction_values, ending_percentile
        )
        idx_survivors = np.where(
            (variable_shap_interaction_values >= starting_percentile_value)
            & (variable_shap_interaction_values < ending_percentile_value)
        )[0]

        survivors = X[[variable_A_name, variable_B_name]].iloc[idx_survivors]
        all_bin_elements = len(survivors)
        for (category_A, category_B) in T_unique:
            if all_bin_elements == 0:
                category_distribution_value = 0
            else:
                category_bin_elements = np.sum(
                    (survivors[variable_A_name] == category_A)
                    & (survivors[variable_B_name] == category_B)
                )
                category_distribution_value = shap_interaction_distribution_value * \
                    (category_bin_elements / all_bin_elements)

            shap_interaction_distribution[
                f"{str(category_A)}-{str(category_B)}"
            ][i] = category_distribution_value

    fig, ax = pl.subplots()
    bottom = np.zeros(n_bins)
    width = 0.5

    for category, distribution in shap_interaction_distribution.items():
        p = ax.bar(bins, distribution, width, label=category, bottom=bottom)
        bottom += distribution

    pl.legend([
        f"{variable_A_name} = {category_A}, {variable_B_name} = {category_B}, " for (category_A, category_B) in T_unique
    ])

    pl.xlabel("SHAP Interaction percentile")
    pl.ylabel(f"Percentage")
    pl.title(
        f"{variable_A_name}, {
            variable_B_name} SHAP Interaction percentage and value distribution"
    )

    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    
    pl.show()


def plot_shap_interaction_distribution(variable_A_name, variable_B_name, X, shap_interaction_values, n_bins):
    variable_A_is_categorical = is_categorical(X[variable_A_name])
    variable_B_is_categorical = is_categorical(X[variable_B_name])

    if variable_A_is_categorical and variable_B_is_categorical:
        plot_shap_interaction_categorical_categorical_distribution(
            variable_A_name, variable_B_name, X, shap_interaction_values, n_bins
        )
    elif (variable_A_is_categorical and not variable_B_is_categorical) or (not variable_A_is_categorical and variable_B_is_categorical):
        plot_shap_interaction_numerical_categorical_distribution(
            variable_A_name, variable_B_name, X, shap_interaction_values, n_bins
        )
    else:
        plot_shap_interaction_numerical_numerical_distribution(
            variable_A_name, variable_B_name, X, shap_interaction_values, n_bins
        )

def shap_acumulated_importance_value(
    percentile,
    variable_shap_values,
    variable_index
):
    percentile_value = np.percentile(
        variable_shap_values, percentile
    )

    sum_shap_values = np.sum(variable_shap_values)
    top_variable_shap_values = variable_shap_values[
        variable_shap_values <= percentile_value
    ]

    return 100 * np.sum(top_variable_shap_values) / sum_shap_values


def plot_shap_lorenz_curve(variable_name, X, shap_values, n_bins):
    bins = np.arange(n_bins)
    X_variable = X[variable_name]

    X_max = np.max(X_variable)
    lorenz_values = np.zeros(n_bins)
    variable_index = X.columns.get_loc(variable_name)

    variable_shap_values = shap_values[:, variable_index]
    for i in bins:
        percentile = (i + 1) * 100 / n_bins
        lorenz_values[i] = shap_acumulated_importance_value(
            percentile, variable_shap_values, variable_index
        )

    fig, ax = pl.subplots()

    ax.bar(
        bins * 100 / n_bins, lorenz_values
    )
    ax.plot(bins * 100 / n_bins, bins * 100 / n_bins, color="green")


    pl.legend([
        f"Perfect equality",
        f"Accumulated SHAP"
    ])
    pl.xlabel(f"SHAP percentile")
    pl.ylabel(f"Percentage")
    pl.title(f"{variable_name} SHAP Lorenz curve")

    ax.yaxis.set_major_formatter(mtick.PercentFormatter())

    pl.show()