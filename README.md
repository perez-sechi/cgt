# CGT: Cooperative Game Theory Library

A Python library designed to support Cooperative Game Theory computations and visualizations, particularly for Shapley values, Grabisch interactions, and coalition analyses. This project includes utilities for computing game-theoretic values in both single-processor and multiprocess setups, plus specialized routines for exploration and visualization in Jupyter notebooks or other Python environments.

---

## Table of Contents
- [CGT: Cooperative Game Theory Library](#cgt-cooperative-game-theory-library)
  - [Table of Contents](#table-of-contents)
  - [Project Structure](#project-structure)
  - [Features](#features)
  - [Installation](#installation)
  - [Dependencies](#dependencies)
  - [Usage](#usage)
  - [Modules Overview](#modules-overview)
    - [Computational Module cgt\_perezsechi.compute](#computational-module-cgt_perezsechicompute)
    - [Exploration Module cgt\_perezsechi.exploration](#exploration-module-cgt_perezsechiexploration)
    - [Manipulation Module cgt\_perezsechi.manipulation](#manipulation-module-cgt_perezsechimanipulation)
    - [Modeling Module cgt\_perezsechi.modeling](#modeling-module-cgt_perezsechimodeling)
    - [Visualization Module cgt\_perezsechi.visualization](#visualization-module-cgt_perezsechivisualization)
    - [Utility Functions cgt\_perezsechi.util](#utility-functions-cgt_perezsechiutil)
  - [Testing](#testing)
  - [License](#license)
  - [Contact](#contact)

---

## Project Structure

```plaintext
cgt_perezsechi/ # Core library
├── compute/ # Shapley, Cost, Grabisch, etc.
├── exploration/ # Sampling, schema, importance
├── manipulation/ # Data transformations, normalizations
├── modeling/ # Clustering, community detection
├── visualization/ # Graph drawing, plotting
├── util/ # Helper utilities (float rounding, etc.)
└── __init__.py # Init for main package

cgt_perezsechi_tests/ # Tests for library modules
├── compute/ # Tests for compute module
├── modeling/ # Tests for modeling module
└── __init__.py

setup.py # Packaging and setup script
requirements.txt # Basic dependencies
settings.json # VS Code configuration
.gitignore # Git ignore rules
README.md # Project documentation (this file)
```
---

## Features

- Computation of various cooperative game theory values:
  - Shapley values (single and multiprocess)
  - Grabisch interaction indices
- Tools for data exploration and feature importance:
  - SHAP-based sampling, ranking, and interaction analysis
- Utilities for manipulating data (interval coding, normalizing)
- Visualization modules for generating:
  - Graph representations
  - Distribution plots
  - Interaction percentile plots
- Fully integrated test suite using pytest.

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/perez-sechi/cgt.git
   cd cgt
   ```


2. Install the package in editable mode:
Alternatively, you can install via the setup.py script:

```bash
python setup.py install
```

## Dependencies

All main dependencies are listed in requirements.txt.
Key libraries include:

```plaintext
numpy, pandas, scipy
multiprocess (for parallel computations)
matplotlib, seaborn, networkx (for plotting)
shap (for SHAP analysis)
pytest (for testing)
```

## Usage
Below are some high-level usage examples:

Computing Shapley Values

```python
from cgt_perezsechi.compute.shapley import exact

# Example cost function
def v(n_set, coalition, data):
    # user-defined value function
    return len(coalition) ** 2

results = exact(n=5, v=v, original=None)
print("Shapley values:", results)
```

Grabisch Interaction

```python
from cgt_perezsechi.compute.grabisch import calculate_interaction_ij

i, j, value = calculate_interaction_ij(
    i=0,
    j=1,
    n_set=[0,1,2],
    v=v,
    original=[50, 200, 300]
)
print("Grabisch interaction for (0, 1) =", value)
```

Visualizing Graph

```python
import pandas as pd
from cgt_perezsechi.visualization.graph import draw

psi_mock = pd.DataFrame({'value': [1.0, -0.5, 0.3]})
r_mock = pd.DataFrame([
    [0,  0.1, -0.8],
    [0.1, 0,  0.2 ],
    [-0.8, 0.2, 0]
])
draw(psi_mock, r_mock)
```

Running Tests
```bash
pytest
```

## Modules Overview

### Computational Module cgt_perezsechi.compute
Contains functionality to compute:

- Shapley values (singlethreaded or via multiprocessing)
- Cost-based Shapley values
- Grabisch interaction indices

### Exploration Module cgt_perezsechi.exploration
Helps in:

- Feature importance ranking (SHAP)
- Data sampling strategies
- Identifying relevant subsets

### Manipulation Module cgt_perezsechi.manipulation
Provides:

- Interval coding (coding.py)
- Normalization utilities (e.g., norm.py)

### Modeling Module cgt_perezsechi.modeling
Focuses on:

- Community detection
- Clustering approaches

### Visualization Module cgt_perezsechi.visualization
Includes tools for:

- Graph drawing with thresholds and arcs
- Plotting SHAP distributions and interactions

### Utility Functions cgt_perezsechi.util

- float_round_to_zero safely rounds small decimal values to zero
- Exposing imports from other subpackages

## Testing

We use pytest for unit testing. The cgt_perezsechi_tests folder houses tests for each main package module. To execute all tests:

```bash
pytest
```

Optionally, you can specify a folder or filename:

```bash
pytest cgt_perezsechi_tests/compute
```

## License
This project is released under the MIT License (if provided).
Check the repository for license details and notices.

## Contact
For questions, feedback, or contributions, please open an issue or pull request on GitHub.

