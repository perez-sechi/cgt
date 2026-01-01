import copy
import numpy as np
import networkx as nx
from itertools import permutations
from random import randint, sample
from typing import Tuple, Optional, Dict, List
import networkx.algorithms.community as nx_comm


def communities_from_index(community_index):
    """
    Convert a list of cluster indices to a list of clusters.
    """
    clusters_dict = {}
    for idx, cluster in enumerate(community_index):
        if cluster not in clusters_dict or clusters_dict[cluster] is None:
            clusters_dict[cluster] = []
        clusters_dict[cluster].append(idx)

    return list(clusters_dict.values())


def modularity(i, cj, community_index, m, K, Kc, R_array):
    """
    Calculate the modularity of moving node i to community cj.
    https://en.wikipedia.org/wiki/Louvain_method
    Parameters:
    i (int): The index of the node to move.
    cj (int): The index of the community to move the node to.
    community_index (list[int]): The current community assignment of each node.
    m (int): The sum of all edge weights in the graph.
    K (list[int]): The sum of edge weights for each node.
    Kc (list[int]): The sum of edge weights for each community.
    R_array (np.array): The modularity matrix of the graph.
    Returns:
    float: The modularity of moving node i to community cj.
    """
    members_cj = [
        idx for idx, c in enumerate(community_index)
        if c == cj
    ]
    ki_in = 2 * np.sum(R_array[i, members_cj])
    return (ki_in / m) - (2 * K[i] * Kc[cj]) / (m ** 2)


def additional_louvain(A, R):
    """
    Perform an additional Louvain clustering on the given adjacency matrix.
    Parameters:
    A (list[list[int]]): Adjacency matrix representing the graph.
    R (list[list[float]]): Matrix used to create a directed graph for 
    modularity calculation.
    Returns:
    list[int]: The partition of the nodes after applying the Louvain method.
    """
    n = len(A)
    o = sample(range(n), n)
    community_index = list(range(n))

    R_array = np.array(R)
    R_array = R_array + R_array.transpose()
    
    m = np.sum(R_array)
    K = [sum(row) for row in R_array]
    Kc = [sum(row) for row in R_array]
    
    improved = True
    iteration = 0
    while improved:
        iteration += 1
        improved = False
        for i in o:
            ci = community_index[i]
            neighbors = np.nonzero(A[i])[0].tolist()
            new_community = ci
            community_index[i] = -1

            Kc[ci] = Kc[ci] - K[i]

            previous_modularity = modularity(
                i, ci, community_index, m, K, Kc, R_array
            )
            best_increase = 0
            temporal_new_community = None
            for j in neighbors:
                if i == j:
                    continue

                cj = community_index[j]
                current_modularity = modularity(
                    i, cj, community_index, m, K, Kc, R_array
                )
                increase = current_modularity - previous_modularity

                if increase > best_increase:
                    best_increase = increase
                    temporal_new_community = cj

            if best_increase > 0:
                new_community = temporal_new_community

            community_index[i] = new_community
            Kc[new_community] = Kc[new_community] + K[i]

            if new_community != ci:
                improved = True

    return communities_from_index(community_index)


def duo_louvain(A, R):
    """
    Perform a clustering operation using a modified Louvain method.
    This function iteratively applies the Louvain method to cluster the nodes
    of a graph represented by adjacency matrix A and modularity matrix R.
    Parameters:
    A (list[list[int]]): The adjacency matrix of the graph.
    R (list[list[float]]): The modularity matrix of the graph.
    Returns:
    list[int]: A list of sets, where each set contains the indices of nodes
    in the same cluster.
    """
    n = len(A)
    P1 = []
    P2 = [[i] for i in range(n)]
    A1 = np.copy(A)
    A2 = np.copy(A)
    R1 = np.copy(R)
    R2 = np.copy(R)

    is_shape_consistent = (
        len(A2) > 1
        and A2.shape[0] == A2.shape[1]
        and R2.shape[0] == R2.shape[1]
        and A2.shape == R2.shape
    )

    iteration = 0
    partitions = {
        iteration: copy.deepcopy(P2),
    }

    while not np.array_equal(P1, P2) and is_shape_consistent:
        iteration += 1
        P1 = P2
        P2 = additional_louvain(A2.tolist(), R2.tolist())
        partitions[iteration] = copy.deepcopy(P2)

        A2 = np.zeros((len(P2), len(P2)))
        R2 = np.zeros((len(P2), len(P2)))

        for cdx, c in enumerate(P2):
            for i in c:
                for ddx, d in enumerate(P2):
                    for j in d:
                        A2[cdx, ddx] = 1 \
                            if A2[cdx, ddx] > 0 or A1[i, j] > 0 \
                            else 0
                        R2[cdx, ddx] += R1[i, j]

        A1 = np.copy(A2)
        R1 = np.copy(R2)

        is_shape_consistent = (
            len(A2) > 1
            and A2.shape[0] == A2.shape[1]
            and R2.shape[0] == R2.shape[1]
            and A2.shape == R2.shape
        )

    communities = partitions[iteration]
    while iteration > 0:
        iteration -= 1
        for cdx, c in enumerate(communities):
            new_community = []
            for d in c:
                new_community.extend(partitions[iteration][d])
            communities[cdx] = new_community

    return communities


def normalize_weighted_adjacency(
    I: np.ndarray,
    alpha1: float = 0.5,
    alpha2: float = 0.5,
    return_coefficients: bool = False
) -> np.ndarray | Tuple[np.ndarray, dict]:
    """
    Normalize a weighted adjacency matrix with positive and negative relations.

    This function splits positive and negative relations, finds the maximum absolute
    value of negative relations, computes normalization coefficients and constants,
    and builds the resulting normalized adjacency matrix.

    Parameters
    ----------
    I : np.ndarray
        Input interaction/adjacency matrix (n x n) with positive and negative values.
        Diagonal is assumed to be zero.
    alpha1 : float, optional
        Weight parameter for positive relations (default: 0.5)
    alpha2 : float, optional
        Weight parameter for negative relations (default: 0.5)
    return_coefficients : bool, optional
        If True, return coefficients and constants used in normalization (default: False)

    Returns
    -------
    M : np.ndarray
        Normalized adjacency matrix (n x n)
    coefficients : dict, optional
        Dictionary containing normalization parameters (if return_coefficients=True):
        - 'Amas': sum of positive edge weights
        - 'Amenos': sum of absolute negative edge weights
        - 'Pmas': proportion of positive weights
        - 'Pmenos': proportion of negative weights
        - 'coefmas': coefficient for positive edges
        - 'coefmenos': coefficient for negative edges
        - 'ctePositivos': constant added to all edges
        - 'max_abs_negative': maximum absolute value among negative edges

    Examples
    --------
    >>> I = np.array([[0, 0.5, -0.3], [0.5, 0, 0.2], [-0.3, 0.2, 0]])
    >>> M = normalize_weighted_adjacency(I, alpha1=0.5, alpha2=0.5)
    >>> M, info = normalize_weighted_adjacency(I, return_coefficients=True)
    """
    n = I.shape[0]

    # Validate input
    if I.shape[0] != I.shape[1]:
        raise ValueError("Input matrix must be square")

    # Extract upper triangular part (excluding diagonal) to avoid double counting
    # since the matrix should be symmetric
    upper_tri_mask = np.triu(np.ones_like(I, dtype=bool), k=1)

    # Split positive and negative edges
    positive_mask = (I > 0) & upper_tri_mask
    negative_mask = (I < 0) & upper_tri_mask

    # Calculate sums
    Amas = np.sum(I[positive_mask])
    Amenos = np.sum(np.abs(I[negative_mask]))

    # Handle edge case where there are no positive or negative edges
    if Amas == 0 and Amenos == 0:
        return (np.zeros_like(I), {}) if return_coefficients else np.zeros_like(I)

    # Calculate proportions
    total_weight = Amas + Amenos
    Pmas = Amas / total_weight if total_weight > 0 else 0
    Pmenos = Amenos / total_weight if total_weight > 0 else 0

    # Find maximum absolute value among negative edges
    if np.any(negative_mask):
        max_abs_negative = np.max(np.abs(I[negative_mask]))
    else:
        max_abs_negative = 0

    # Calculate number of possible edges in complete graph
    num_possible_edges = n * (n - 1) // 2

    # Calculate coefficients
    if Amas > 0:
        coefmas = alpha1 * Pmas / Amas
    else:
        coefmas = 0

    if max_abs_negative > 0 and Amenos > 0:
        denominator = num_possible_edges * max_abs_negative - Amenos
        if denominator > 0:
            coefmenos = alpha2 * Pmenos / denominator
        else:
            coefmenos = 0
    else:
        coefmenos = 0

    # Calculate constant
    ctePositivos = coefmenos * max_abs_negative

    # Build normalized matrix
    M = np.zeros_like(I, dtype=float)

    for i in range(n):
        for j in range(n):
            if i == j:
                M[i, j] = 0  # Keep diagonal as zero
            elif I[i, j] > 0:
                # Positive edge
                M[i, j] = coefmas * I[i, j] + ctePositivos
            elif I[i, j] < 0:
                # Negative edge
                M[i, j] = ctePositivos - np.abs(I[i, j]) * coefmenos
            else:
                # Zero edge (no relation in original matrix)
                M[i, j] = ctePositivos

    if return_coefficients:
        coefficients = {
            'Amas': Amas,
            'Amenos': Amenos,
            'Pmas': Pmas,
            'Pmenos': Pmenos,
            'coefmas': coefmas,
            'coefmenos': coefmenos,
            'ctePositivos': ctePositivos,
            'max_abs_negative': max_abs_negative,
            'num_possible_edges': num_possible_edges
        }
        return M, coefficients

    return M


def get_edge_list_from_normalized_matrix(
    M: np.ndarray,
    nodes: list,
    include_zero_edges: bool = True
) -> list:
    """
    Extract edge list from normalized adjacency matrix.

    Parameters
    ----------
    M : np.ndarray
        Normalized adjacency matrix (n x n)
    nodes : list
        List of node labels
    include_zero_edges : bool, optional
        Whether to include edges with zero weight (default: True)

    Returns
    -------
    edges : list
        List of tuples (source, target, weight)
    """
    n = M.shape[0]
    edges = []

    # Extract upper triangular part to avoid duplicates
    for i in range(n):
        for j in range(i + 1, n):
            if include_zero_edges or M[i, j] != 0:
                edges.append((nodes[i], nodes[j], M[i, j]))

    return edges


def detect_communities(
    M: np.ndarray,
    nodes: Optional[List[str]] = None,
    resolution: float = 1.0,
    seed: Optional[int] = None,
    weight_attribute: str = 'weight',
    return_graph: bool = False
) -> Dict[str, int] | Tuple[Dict[str, int], nx.Graph]:
    """
    Detect communities in a network from its normalized adjacency matrix using Louvain algorithm.

    Parameters
    ----------
    M : np.ndarray
        Normalized adjacency matrix (n x n). All values should be non-negative.
    nodes : list of str, optional
        List of node labels. If None, nodes will be labeled as 0, 1, 2, ..., n-1
    resolution : float, optional
        Resolution parameter for Louvain algorithm. Higher values lead to more communities.
        Default is 1.0. The original code uses 0.65.
    seed : int, optional
        Random seed for reproducibility. Default is None.
    weight_attribute : str, optional
        Name of the edge attribute to use as weight (default: 'weight')
    return_graph : bool, optional
        If True, also return the NetworkX graph object (default: False)

    Returns
    -------
    community_map : dict
        Dictionary mapping each node to its community ID (integer)
    G : nx.Graph, optional
        The NetworkX graph object (if return_graph=True)

    Examples
    --------
    >>> M = normalize_weighted_adjacency(I)
    >>> community_map = detect_communities(M, nodes=["A", "B", "C"])
    >>> print(community_map)
    {'A': 0, 'B': 0, 'C': 1}

    >>> # With graph return
    >>> community_map, G = detect_communities(M, nodes=["A", "B", "C"], return_graph=True)
    """
    n = M.shape[0]

    # Validate input
    if M.shape[0] != M.shape[1]:
        raise ValueError("Input matrix must be square")

    # Generate default node labels if not provided
    if nodes is None:
        nodes = [str(i) for i in range(n)]
    elif len(nodes) != n:
        raise ValueError(f"Number of nodes ({len(nodes)}) must match matrix size ({n})")

    # Build NetworkX graph from adjacency matrix
    G = nx.Graph()
    G.add_nodes_from(nodes)

    # Add weighted edges from the adjacency matrix
    edge_list = get_edge_list_from_normalized_matrix(M, nodes, include_zero_edges=True)
    G.add_weighted_edges_from(edge_list, weight=weight_attribute)

    # Run Louvain community detection
    communities = nx_comm.louvain_communities(
        G,
        weight=weight_attribute,
        resolution=resolution,
        seed=seed
    )

    # Create community map: node -> community_id
    community_map = {}
    for cid, comm in enumerate(communities):
        for node in comm:
            community_map[node] = cid

    if return_graph:
        return community_map, G

    return community_map
