import numpy as np
from normalize_adjacency import normalize_weighted_adjacency, get_edge_list_from_normalized_matrix


# Original data from your code
nodes = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]

I = np.array([
    [ 0,      0.0142,-0.0048,-0.0024, 0.0047,-0.0024,-0.0024,-0.0072, 0,      0     ],
    [ 0.0142, 0,     -0.0024, 0,     -0.0096, 0,     -0.0024, 0,      0,      0     ],
    [-0.0048,-0.0024, 0,      0.0142, 0,     -0.0024,-0.0024,-0.0072, 0,      0     ],
    [-0.0024, 0,      0.0142, 0,      0,     -0.0096, 0,     -0.0024, 0,      0     ],
    [ 0.0047,-0.0096, 0,      0,      0,      0.0071, 0.0071,-0.0048, 0,      0     ],
    [-0.0024, 0,     -0.0024,-0.0096, 0.0071, 0,      0,     -0.0024, 0,      0     ],
    [-0.0024,-0.0024,-0.0024, 0,      0.0071, 0,      0,     -0.0024, 0,      0     ],
    [-0.0072, 0,     -0.0072,-0.0024,-0.0048,-0.0024,-0.0024, 0,      0.0142, 0.0142],
    [ 0,      0,      0,      0,      0,      0,      0,      0.0142, 0,     -0.0142],
    [ 0,      0,      0,      0,      0,      0,      0,      0.0142,-0.0142, 0     ]
])

# Test the normalization function
print("=" * 80)
print("Testing normalize_weighted_adjacency function")
print("=" * 80)

# Get normalized matrix with coefficients
M, coeffs = normalize_weighted_adjacency(I, alpha1=0.5, alpha2=0.5, return_coefficients=True)

print("\nNormalization Coefficients:")
print("-" * 40)
for key, value in coeffs.items():
    print(f"{key:20s}: {value:.6f}")

print("\n\nNormalized Adjacency Matrix:")
print("-" * 40)
print(M)

# Get edge list
edges = get_edge_list_from_normalized_matrix(M, nodes, include_zero_edges=True)

print("\n\nEdge List (first 10 edges):")
print("-" * 40)
for i, (source, target, weight) in enumerate(edges[:10]):
    print(f"{source} -- {target}: {weight:.8f}")

# Verify some specific edges against your manual calculation
print("\n\nVerification of specific edges:")
print("-" * 40)

# From your code:
# coefmas*0.0142+ctePositivos for A-B (positive edge 0.0142)
# ctePositivos-0.0048*coefmenos for A-C (negative edge -0.0048)
# ctePositivos for A-I (zero edge)

expected_AB = coeffs['coefmas'] * 0.0142 + coeffs['ctePositivos']
expected_AC = coeffs['ctePositivos'] - 0.0048 * coeffs['coefmenos']
expected_AI = coeffs['ctePositivos']

actual_AB = M[0, 1]  # A-B
actual_AC = M[0, 2]  # A-C
actual_AI = M[0, 8]  # A-I

print(f"A-B (positive 0.0142):")
print(f"  Expected: {expected_AB:.8f}")
print(f"  Actual:   {actual_AB:.8f}")
print(f"  Match: {np.isclose(expected_AB, actual_AB)}")

print(f"\nA-C (negative -0.0048):")
print(f"  Expected: {expected_AC:.8f}")
print(f"  Actual:   {actual_AC:.8f}")
print(f"  Match: {np.isclose(expected_AC, actual_AC)}")

print(f"\nA-I (zero edge):")
print(f"  Expected: {expected_AI:.8f}")
print(f"  Actual:   {actual_AI:.8f}")
print(f"  Match: {np.isclose(expected_AI, actual_AI)}")

print("\n" + "=" * 80)
print("Test complete!")
print("=" * 80)
