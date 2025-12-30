# CGT: Cooperative Game Theory Library

A Python library for Cooperative Game Theory computations and visualizations, specializing in Shapley values, Grabisch interactions, Castro sampling methods, and coalition analyses. This project includes utilities for computing game-theoretic values in both single-processor and multiprocess setups, plus specialized routines for SHAP-based feature analysis and visualization.

---

## Table of Contents
- [CGT: Cooperative Game Theory Library](#cgt-cooperative-game-theory-library)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Installation](#installation)
  - [Dependencies](#dependencies)
  - [Quick Start](#quick-start)
  - [Project Structure](#project-structure)
  - [Features](#features)
  - [Modules Overview](#modules-overview)
    - [Computational Module cgt\_perezsechi.compute](#computational-module-cgt_perezsechicompute)
    - [Exploration Module cgt\_perezsechi.exploration](#exploration-module-cgt_perezsechiexploration)
    - [Manipulation Module cgt\_perezsechi.manipulation](#manipulation-module-cgt_perezsechimanipulation)
    - [Modeling Module cgt\_perezsechi.modeling](#modeling-module-cgt_perezsechimodeling)
    - [Visualization Module cgt\_perezsechi.visualization](#visualization-module-cgt_perezsechivisualization)
    - [Utility Functions cgt\_perezsechi.util](#utility-functions-cgt_perezsechiutil)
  - [Detailed Usage Examples](#detailed-usage-examples)
    - [Computing Shapley Values](#computing-shapley-values)
    - [Castro Sampling for Large-Scale Problems](#castro-sampling-for-large-scale-problems)
    - [Grabisch Interaction Indices](#grabisch-interaction-indices)
    - [Community Detection with Duo-Louvain](#community-detection-with-duo-louvain)
    - [SHAP Feature Importance Ranking](#shap-feature-importance-ranking)
    - [Visualizing Networks](#visualizing-networks)
    - [Visualizing Clusters](#visualizing-clusters)
    - [SHAP Distribution Analysis](#shap-distribution-analysis)
    - [Lorenz Curves for Inequality](#lorenz-curves-for-inequality)
  - [Testing](#testing)
  - [Version](#version)
  - [License](#license)
  - [Contact](#contact)

---

## Overview

The CGT library provides a comprehensive toolkit for:

- **Theoretical Game Theory**: Exact Shapley value computation, Grabisch interaction indices, cost-based coalition analysis
- **Scalable Computation**: Castro stratified sampling methods for large-scale problems with multiprocessing support
- **Machine Learning Explainability**: SHAP integration for feature importance, interaction analysis, and distribution visualization
- **Community Detection**: Advanced Louvain-based clustering algorithms using modularity optimization
- **Rich Visualization**: Network graphs with multiple layouts, cluster visualization, distribution plots, and Lorenz curves

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/perez-sechi/cgt.git
   cd cgt
   ```

2. Install the package:
   ```bash
   python setup.py install
   ```

   Or in editable mode for development:
   ```bash
   pip install -e .
   ```

---

## Dependencies

The library requires the following packages (see `setup.py` for complete list):

**Core Dependencies:**
- `numpy`, `pandas`, `scipy` - Numerical computing
- `multiprocess` - Parallel computation support
- `Decimal` - High-precision arithmetic

**Machine Learning & Statistics:**
- `shap` - SHAP value analysis
- `scikit-learn` - Clustering and preprocessing
- `pingouin`, `statsmodels` - Statistical testing

**Visualization:**
- `matplotlib`, `seaborn` - Plotting
- `networkx` - Graph visualization and layouts
- `pygraphviz` - Advanced graph layouts (optional)

**Development:**
- `pytest` - Testing framework

Install all dependencies via:
```bash
pip install -r requirements.txt
```

---

## Quick Start

```python
from cgt_perezsechi.compute.shapley import exact
from cgt_perezsechi.visualization.graph import draw
import pandas as pd

# Define a simple coalition value function
def v(n_set, coalition, data):
    return sum(coalition) ** 1.5

# Compute exact Shapley values
shapley_values = exact(n=5, v=v, original=None)
print("Shapley values:", shapley_values)

# Visualize as a network (requires interaction matrix)
# See detailed examples below
```

---

## Project Structure

```plaintext
cgt/
├── cgt_perezsechi/              # Core library package
│   ├── compute/                 # Game theory computations
│   │   ├── shapley.py          # Exact Shapley values
│   │   ├── shapley_multiproc.py # Parallel Shapley computation
│   │   ├── castro.py           # Castro sampling methods
│   │   ├── castro_multiproc.py # Parallel Castro sampling
│   │   ├── grabisch.py         # Grabisch interaction indices
│   │   └── sh_i.py             # Core Shapley helpers
│   ├── exploration/             # Feature importance & sampling
│   │   ├── importance.py       # SHAP-based feature ranking
│   │   ├── sampling.py         # Statistical sample size estimation
│   │   └── schema.py           # Data type utilities
│   ├── manipulation/            # Data transformations
│   │   ├── coding.py           # Interval coding/binning
│   │   └── norm.py             # Normalization utilities
│   ├── modeling/                # Clustering & inequality
│   │   ├── cluster.py          # Duo-Louvain community detection
│   │   └── inequality.py       # Gini coefficient
│   ├── visualization/           # Plotting & graph drawing
│   │   ├── graph.py            # Network visualization
│   │   └── plot.py             # SHAP distribution plots
│   └── util/                    # Helper utilities
│       └── float.py            # Precision handling
├── cgt_perezsechi_tests/        # Test suite
│   ├── compute/
│   │   └── test_grabisch.py
│   └── modeling/
│       ├── test_cluster.py
│       └── util/benchmark.py
├── setup.py                     # Package configuration
├── requirements.txt             # Dependencies
└── README.md                    # This file
```

---

## Features

### Computational Capabilities
- **Exact Shapley Values**: Single-threaded and multiprocessing implementations
- **Castro Sampling**: Stratified sampling for scalable Shapley estimation on large problems
- **Grabisch Interaction Indices**: Pairwise feature interaction computation
- **Cost-Based Analysis**: Shapley values with cost functions
- **Difference Analysis**: Comparative Shapley value computation

### Machine Learning Integration
- **SHAP Feature Ranking**: Automatic top-N feature selection
- **SHAP Interaction Analysis**: Top pairwise and higher-order interactions
- **Sample Size Optimization**: Kruskal-Wallis based sample size estimation
- **Categorical Detection**: Automatic variable type inference

### Data Manipulation
- **Interval Coding**: Binning continuous variables
- **Normalization**: PSI (Shapley) and R (interaction) matrix normalization

### Advanced Modeling
- **Duo-Louvain Clustering**: Two-stage community detection using adjacency and modularity matrices
- **Modularity Optimization**: Custom modularity calculations for game-theoretic networks
- **Inequality Metrics**: Gini coefficient for distribution analysis

### Rich Visualization
- **Network Graphs**: 10+ layout algorithms (spring, Kamada-Kawai, circular, spectral, planar, shell, spiral, twopi, etc.)
- **IQR-Based Coloring**: Quartile-based uncertainty visualization (grey for inconsistent signs)
- **Cluster Visualization**: Auto-positioned community graphs with custom color palettes
- **SHAP Distributions**: Categorical, numerical, and interaction distribution plots
- **Lorenz Curves**: Inequality visualization for feature importance
- **Customizable Styling**: Node sizes, colors, edge styles, thresholds, and labels

---

## Modules Overview

### Computational Module cgt_perezsechi.compute

Provides exact and approximate methods for computing game-theoretic values.

**Key Functions:**

**Shapley Values** (`shapley.py`, `shapley_multiproc.py`):
- `exact(n, v, original)` - Exact Shapley values for n players
- `exac_n_set(n_set, v, original)` - Shapley values for custom player sets
- `exact_diff(n, v, original)` - Difference in Shapley values
- `cost(n, c, original)` - Cost-based Shapley values
- `cost_diff(n, c, original)` - Cost-based differences

**Castro Sampling** (`castro.py`, `castro_multiproc.py`):
- `castro(n, m, v, original)` - Stratified sampling for Shapley estimation
- `castro_interaction_index(n, m, v, original)` - Interaction index estimation
- Helper functions for variance estimation and confidence intervals

**Grabisch Interactions** (`grabisch.py`):
- `calculate_interaction_ij(i, j, n_set, v, original)` - Pairwise interaction strength

**Core Helpers** (`sh_i.py`):
- `calculate_sh_i(i, n_set, v, original)` - Individual Shapley value
- `calculate_cost_sh_i(i, n_set, c, original)` - Cost-based variant

---

### Exploration Module cgt_perezsechi.exploration

SHAP-based feature analysis and sampling strategies.

**Key Functions:**

**Importance Analysis** (`importance.py`):
- `shap_ranked_top_features(shap_values, X)` - Rank features by mean absolute SHAP
- `shap_ranked_top_interactions(shap_interaction_values, X, head)` - Top interaction pairs
- `shap_top_important_features(shap_values, X, head)` - Get top N features
- `shap_median_important_features(shap_values, X, elem)` - Median-based importance

**Sampling** (`sampling.py`):
- `sample_size_kruskal_wallis(X, shap_values, variable_name, acceleration)` - Optimal sample size using Kruskal-Wallis test

**Schema Utilities** (`schema.py`):
- `is_categorical(column)` - Automatic categorical variable detection

---

### Manipulation Module cgt_perezsechi.manipulation

Data transformation and normalization utilities.

**Key Functions:**

**Interval Coding** (`coding.py`):
- `code_interval(X, variable_A_name, sample_size)` - Bin continuous variables into intervals

**Normalization** (`norm.py`):
- `normalize_psi(psi)` - Normalize Shapley (PSI) values
- `normalize_r(r)` - Normalize interaction (R) matrix

---

### Modeling Module cgt_perezsechi.modeling

Community detection and inequality measures.

**Key Functions:**

**Clustering** (`cluster.py`):
- `duo_louvain(A, R)` - Two-stage Louvain algorithm using adjacency matrix A and modularity matrix R
- `additional_louvain(A, R)` - Single iteration of Louvain clustering
- `modularity(i, cj, community_index, m, K, Kc, R_array)` - Modularity calculation
- `communities_from_index(community_index)` - Convert cluster indices to community lists

**Inequality Metrics** (`inequality.py`):
- `gini_coefficient(values)` - Calculate Gini coefficient for distribution inequality

---

### Visualization Module cgt_perezsechi.visualization

Network graph rendering and SHAP distribution plotting.

**Key Functions:**

**Graph Visualization** (`graph.py`):
- `draw(psi, r, ...)` - Main network graph drawing function
  - **Parameters**: layout, node colors, edge thresholds (alpha/beta), arched edges, IQR coloring
  - **Layouts**: spring, kamada_kawai, circular, spectral, planar, shell, spiral, twopi
  - **Features**: Quartile-based uncertainty coloring, customizable node sizes, label positioning

- `draw_clusters(psi, r, clusters, ...)` - Cluster-aware visualization
  - Auto-positioned communities
  - Custom color palettes (excluding red/blue reserved for positive/negative values)
  - Community legends

- `get_cmap(n)` - Predefined color palette for cluster visualization

**Distribution Plotting** (`plot.py`):
- `plot_shap_distribution(variable_name, X, shap_values, n_bins)` - Auto-detects variable type
- `plot_shap_categorical_distribution(...)` - SHAP analysis for categorical variables
- `plot_shap_numerical_distribution(...)` - SHAP analysis for numerical variables
- `plot_shap_interaction_distribution(...)` - Interaction plot dispatcher
- `plot_shap_interaction_numerical_numerical_distribution(...)` - Numerical-numerical interactions
- `plot_shap_interaction_numerical_categorical_distribution(...)` - Mixed variable interactions
- `plot_shap_interaction_categorical_categorical_distribution(...)` - Categorical-categorical interactions
- `plot_shap_lorenz_curve(variable_name, X, shap_values, n_bins)` - Lorenz curve for inequality visualization

---

### Utility Functions cgt_perezsechi.util

Helper utilities for precision handling.

**Key Functions:**

**Float Utilities** (`float.py`):
- `float_round_to_zero(x)` - Round tiny decimal values to zero based on precision

---

## Detailed Usage Examples

### Computing Shapley Values

```python
from cgt_perezsechi.compute.shapley import exact, cost

# Define a coalition value function
def v(n_set, coalition, data):
    """Example: Synergy-based value function"""
    return sum(coalition) ** 1.5

# Compute exact Shapley values for 5 players
shapley_values = exact(n=5, v=v, original=None)
print("Shapley values:", shapley_values)

# Cost-based Shapley values
def c(n_set, coalition, data):
    """Cost function for coalition formation"""
    return len(coalition) * 10

cost_values = cost(n=5, c=c, original=None)
print("Cost-based values:", cost_values)
```

### Castro Sampling for Large-Scale Problems

```python
from cgt_perezsechi.compute.castro_multiproc import castro

# For large problems where exact computation is infeasible
def v(n_set, coalition, data):
    # Complex value function
    return sum([data[i] * (i+1) for i in coalition])

# Use Castro sampling with m samples
data = list(range(100))  # 100 players
estimated_shapley = castro(n=100, m=1000, v=v, original=data)
print("Estimated Shapley values:", estimated_shapley)
```

### Grabisch Interaction Indices

```python
from cgt_perezsechi.compute.grabisch import calculate_interaction_ij

# Define value function
def v(n_set, coalition, data):
    return len(coalition) ** 2

# Calculate interaction between players 0 and 1
i, j, interaction_value = calculate_interaction_ij(
    i=0,
    j=1,
    n_set=[0, 1, 2, 3],
    v=v,
    original=[50, 200, 300, 150]
)
print(f"Grabisch interaction between {i} and {j}: {interaction_value}")
```

### Community Detection with Duo-Louvain

```python
from cgt_perezsechi.modeling.cluster import duo_louvain
import numpy as np

# Adjacency matrix (connections between nodes)
A = np.array([
    [0, 1, 1, 0],
    [1, 0, 1, 1],
    [1, 1, 0, 0],
    [0, 1, 0, 0]
])

# Interaction/modularity matrix
R = np.array([
    [0, 0.5, 0.8, -0.2],
    [0.5, 0, 0.6, 0.3],
    [0.8, 0.6, 0, -0.1],
    [-0.2, 0.3, -0.1, 0]
])

# Detect communities
clusters = duo_louvain(A, R)
print("Detected communities:", clusters)
```

### SHAP Feature Importance Ranking

```python
from cgt_perezsechi.exploration.importance import (
    shap_ranked_top_features,
    shap_ranked_top_interactions
)
import shap
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# Example: Train model and get SHAP values
X = pd.DataFrame({
    'feature1': [1, 2, 3, 4, 5],
    'feature2': [5, 4, 3, 2, 1],
    'feature3': [2, 2, 3, 3, 4]
})
y = [10, 20, 30, 40, 50]

model = RandomForestRegressor()
model.fit(X, y)

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

# Rank features by importance
ranked_features = shap_ranked_top_features(shap_values, X)
print("Feature importance ranking:", ranked_features)

# Get top interactions (requires shap_interaction_values)
shap_interaction_values = explainer.shap_interaction_values(X)
top_interactions = shap_ranked_top_interactions(shap_interaction_values, X, head=5)
print("Top 5 interactions:", top_interactions)
```

### Visualizing Networks

```python
import pandas as pd
from cgt_perezsechi.visualization.graph import draw

# Shapley values (PSI)
psi = pd.DataFrame({
    'value': [1.2, -0.8, 0.5, 0.3]
})

# Interaction matrix (R)
r = pd.DataFrame([
    [0,    0.5, -0.3,  0.2],
    [0.5,  0,    0.8, -0.1],
    [-0.3, 0.8,  0,    0.4],
    [0.2, -0.1,  0.4,  0]
])

# Draw network with spring layout
draw(
    psi, r,
    layout='spring',
    alpha=0.3,  # Threshold for positive edges
    beta=-0.3,  # Threshold for negative edges
    arched=True,  # Use arched edges
    node_size=1000
)
```

### Visualizing Clusters

```python
from cgt_perezsechi.visualization.graph import draw_clusters

# Same psi and r as above
clusters = [[0, 1], [2, 3]]  # Two communities

draw_clusters(
    psi, r, clusters,
    layout='kamada_kawai',
    alpha=0.3,
    beta=-0.3,
    node_size=1500
)
```

### SHAP Distribution Analysis

```python
from cgt_perezsechi.visualization.plot import (
    plot_shap_distribution,
    plot_shap_interaction_distribution
)

# Plot SHAP distribution for a single feature
plot_shap_distribution(
    variable_name='feature1',
    X=X,
    shap_values=shap_values,
    n_bins=10
)

# Plot interaction between two features
plot_shap_interaction_distribution(
    variable_A_name='feature1',
    variable_B_name='feature2',
    X=X,
    shap_interaction_values=shap_interaction_values
)
```

### Lorenz Curves for Inequality

```python
from cgt_perezsechi.visualization.plot import plot_shap_lorenz_curve
from cgt_perezsechi.modeling.inequality import gini_coefficient

# Visualize inequality in feature importance
plot_shap_lorenz_curve(
    variable_name='feature1',
    X=X,
    shap_values=shap_values,
    n_bins=10
)

# Calculate Gini coefficient
gini = gini_coefficient(shapley_values)
print(f"Gini coefficient: {gini}")
```

---

## Testing

The library includes a comprehensive test suite using pytest.

Run all tests:
```bash
pytest
```

Run specific test modules:
```bash
pytest cgt_perezsechi_tests/compute/test_grabisch.py
pytest cgt_perezsechi_tests/modeling/test_cluster.py
```

**Test Coverage:**
- `test_grabisch.py`: Validates Grabisch interaction calculations with estate division problems
- `test_cluster.py`: Tests Louvain clustering with various network topologies
- `benchmark.py`: NMI (Normalized Mutual Information) benchmarks for clustering quality

---

## Version

Current version: **0.0.12**

---

## License

This project is released under the MIT License.

---

## Contact

**Author**: Carlos I. Pérez-Sechi

For questions, feedback, or contributions, please open an issue or pull request on GitHub.

**Repository**: https://github.com/perez-sechi/cgt
