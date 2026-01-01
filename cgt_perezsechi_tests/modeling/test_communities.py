import numpy as np
from normalize_adjacency import normalize_weighted_adjacency, detect_communities


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

print("=" * 80)
print("Testing Community Detection Function")
print("=" * 80)

# Step 1: Normalize the adjacency matrix
print("\n1. Normalizing adjacency matrix...")
M = normalize_weighted_adjacency(I, alpha1=0.5, alpha2=0.5)
print("   Normalization complete.")

# Step 2: Detect communities with the same parameters as your code
print("\n2. Detecting communities (resolution=0.65, seed=42)...")
community_map, G = detect_communities(
    M,
    nodes=nodes,
    resolution=0.65,
    seed=42,
    return_graph=True
)

# Display results
print("\n" + "-" * 80)
print("Community Assignments:")
print("-" * 80)
for node in nodes:
    print(f"  Node {node}: Community {community_map[node]}")

# Group nodes by community
print("\n" + "-" * 80)
print("Communities (grouped):")
print("-" * 80)
communities_grouped = {}
for node, comm_id in community_map.items():
    if comm_id not in communities_grouped:
        communities_grouped[comm_id] = []
    communities_grouped[comm_id].append(node)

for comm_id in sorted(communities_grouped.keys()):
    print(f"  Community {comm_id}: {communities_grouped[comm_id]}")

# Graph statistics
print("\n" + "-" * 80)
print("Graph Statistics:")
print("-" * 80)
print(f"  Number of nodes: {G.number_of_nodes()}")
print(f"  Number of edges: {G.number_of_edges()}")
print(f"  Number of communities: {len(communities_grouped)}")

# Test with different resolution parameters
print("\n" + "=" * 80)
print("Testing Different Resolution Parameters")
print("=" * 80)

resolutions = [0.3, 0.65, 1.0, 1.5]
for res in resolutions:
    comm_map = detect_communities(M, nodes=nodes, resolution=res, seed=42)
    num_communities = len(set(comm_map.values()))
    print(f"  Resolution {res:.2f}: {num_communities} communities")

print("\n" + "=" * 80)
print("Test Complete!")
print("=" * 80)
