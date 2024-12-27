import numpy as np
from typing import List
from random import random
from sklearn.metrics import normalized_mutual_info_score


def benchmark_graph(
    nodes: List[int], clusters: List[List[int]], alpha: float, beta: float
):
    '''
    Under the assumption that the expected degree of each node, < k >, and the
    community size of each cluster, |Ct|, are power laws with exponents α and
    β, the construction of benchmark graphs is based on the Equation (4):
    given the cluster Ct , for each pair of nodes (i, j), there will be a link
    between them in the with the probability P(i, j) defined below.
    P(i, j) = α if i and j are in the same cluster
    P(i, j) = β if i and j are in different clusters
    Parameters:
    nodes (list[int]): The nodes in the graph.
    clusters (list[list[int]]): The clusters in the graph.
    alpha (float): The probability of a link between two nodes in the same
    cluster.
    beta (float): The probability of a link between two nodes in different
    clusters.
    Returns:
    np.array: The adjacency matrix of the benchmark graph
    '''
    n_nodes = len(nodes)
    result = np.zeros((n_nodes, n_nodes))

    for i in nodes:
        for j in nodes:
            if i == j:
                result[i][j] = 0
                continue
            located = False
            for c in clusters:
                if i in c and j in c:
                    result[i][j] = 1 if random() <= alpha else 0
                    located = True
                    continue
            if not located:
                result[i][j] = 1 if random() <= beta else 0

    return result


def partitions_nmi(ground, test):
    '''
    The normalized mutual information (NMI) is a measure of the mutual
    dependence between two partitions of a set. It is normalized to lie in the
    range [0, 1]. The NMI is defined as follows:
    NMI = 2 * I(C, T) / (H(C) + H(T))
    where I(C, T) is the mutual information between the two partitions,
    and H(T) are the entropies of the two partitions.
    Parameters:
    ground (list[list[int]]): The ground truth partition.
    test (list[list[int]]): The test partition.
    Returns:
    float: The normalized mutual information between the two partitions.
    '''
    max_index = max(map(max, ground))

    labels_i = np.ones(max_index + 1) * (-1)
    num_communities_i = len(ground)
    for c in range(num_communities_i):
        for k in ground[c]:
            labels_i[k] = c

    labels_j = np.ones(max_index + 1) * (-1)
    num_communities_j = len(test)
    for c in range(num_communities_j):
        for l in test[c]:
            labels_j[l] = c

    return normalized_mutual_info_score(labels_i, labels_j)
