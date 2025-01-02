import os
import sys
import numpy as np
import pandas as pd

from shap_network.modeling.cluster import duo_louvain
from test.modeling.util.benchmark import benchmark_graph, partitions_nmi


def test_dou_louvain_happy_path():
    # Arrange
    A = [[0, 1, 0], [1, 0, 1], [0, 1, 0]]
    M = [[0, 0.2, 0], [0.7, 0, 0.4], [0, 0.5, 0]]

    # Act
    result = duo_louvain(A, M)

    # Assert
    assert len(result) == 1


def test_dou_louvain_politics():
    # Arrange
    A = [
        [0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 1, 1, 0, 0, 0, 0, 0, 0],
        [1, 1, 0, 1, 1, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 1, 1, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 1, 1, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 1, 1],
        [0, 0, 0, 0, 0, 0, 1, 1, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 1, 1, 0]
    ]
    M = (np.array([
        [0, 1, 1, 0, 0, 1, 1, 0, 0, 0],
        [1, 0, 1, 1, 0, 1, 1, 0, 0, 0],
        [1, 1, 0, 0, 0, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 1, 1, 1],
        [0, 0, 0, 1, 0, 0, 0, 1, 1, 1],
        [1, 1, 1, 0, 0, 0, 1, 0, 0, 0],
        [1, 1, 1, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 0, 0, 0, 1, 1],
        [0, 0, 0, 1, 1, 0, 0, 1, 0, 1],
        [0, 0, 0, 1, 1, 0, 0, 1, 1, 0]
    ]) * (1/40)).tolist()
    expected_result = [
        [0, 1, 2],
        [3, 4],
        [5, 6],
        [7, 8, 9]
    ]

    # Act
    result = duo_louvain(A, M)

    # Assert
    assert len(result) == 4
    for cdx, c in enumerate(expected_result):
        for i in c:
            assert i in result[cdx]


def test_duo_louvain_nmi():
    # Arrange
    n_nodes = 256
    nodes = list(range(n_nodes))

    n_modules = 2
    nodes_per_module = n_nodes // n_modules
    modules = []
    module_i = -1
    for i in nodes:
        if i % nodes_per_module == 0:
            modules.append([])
            module_i += 1
        modules[module_i].append(i)

    n_communities = 4
    nodes_per_community = n_nodes // n_communities
    communities = []
    community_i = -1
    for i in nodes:
        if i % nodes_per_community == 0:
            communities.append([])
            community_i += 1
        communities[community_i].append(i)

    labels = {
        1: [0.45, 0.016],
        2: [0.4, 0.033],
        3: [0.35, 0.05],
        4: [0.325, 0.058],
        5: [0.3, 0.066],
        6: [0.275, 0.075],
        7: [0.25, 0.083],
        8: [0.225, 0.091],
        9: [0.2, 0.1]
    }

    # Act
    nmi_results = []

    print("")
    print("Results:")
    print(f"graph,community,nmi")
    for graph_label, [graph_alpha, graph_beta] in labels.items():
        for community_label, [
            community_alpha,
            community_beta
        ] in labels.items():
            A = benchmark_graph(nodes, modules, graph_alpha, graph_beta)
            R = benchmark_graph(
                nodes, communities, community_alpha, community_beta
            )
            result = duo_louvain(A, R)
            nmi = partitions_nmi(communities, result)
            nmi_results.append({
                "graph_label": graph_label,
                "community_label": community_label,
                "nmi": nmi
            })
            print(f"{graph_label},{community_label},{nmi}")

    try:
        file_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        file_dir = "."

    df_results = pd.DataFrame(nmi_results)
    df_results.to_csv(os.path.join(
        file_dir, "test_duo_louvain_nmi.csv"),
        index=False
    )

    # Assert
    for result in nmi_results:
        assert result["nmi"] > 0.7


def test_duo_louvain_worst_nmi():
    # Arrange
    n_nodes = 256
    nodes = list(range(n_nodes))

    n_modules = 2
    nodes_per_module = n_nodes // n_modules
    modules = []
    module_i = -1
    for i in nodes:
        if i % nodes_per_module == 0:
            modules.append([])
            module_i += 1
        modules[module_i].append(i)

    n_communities = 4
    nodes_per_community = n_nodes // n_communities
    communities = []
    community_i = -1
    for i in nodes:
        if i % nodes_per_community == 0:
            communities.append([])
            community_i += 1
        communities[community_i].append(i)

    labels = {
        # 1: [0.45, 0.016],
        # 2: [0.4, 0.033],
        # 3: [0.35, 0.05],
        # 4: [0.325, 0.058],
        # 5: [0.3, 0.066],
        # 6: [0.275, 0.075],
        # 7: [0.25, 0.083],
        # 8: [0.225, 0.091],
        9: [0.2, 0.1]
    }

    # Act
    nmi_results = []

    for graph_label, [graph_alpha, graph_beta] in labels.items():
        for community_label, [
            community_alpha,
            community_beta
        ] in labels.items():
            A = benchmark_graph(nodes, modules, graph_alpha, graph_beta)
            R = benchmark_graph(
                nodes, communities, community_alpha, community_beta
            )
            result = duo_louvain(A, R)
            nmi = partitions_nmi(communities, result)
            nmi_results.append({
                "graph_label": graph_label,
                "community_label": community_label,
                "nmi": nmi
            })

    try:
        file_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        file_dir = "."

    df_results = pd.DataFrame(nmi_results)
    df_results.to_csv(os.path.join(
        file_dir, "test_duo_louvain_worst_nmi.csv"),
        index=False
    )

    # Assert
    for result in nmi_results:
        assert result["nmi"] > 0.7


def test_duo_louvain_small_nmi():
    # Arrange
    n_nodes = 128
    nodes = list(range(n_nodes))

    n_modules = 2
    nodes_per_module = n_nodes // n_modules
    modules = []
    module_i = -1
    for i in nodes:
        if i % nodes_per_module == 0:
            modules.append([])
            module_i += 1
        modules[module_i].append(i)

    n_communities = 4
    nodes_per_community = n_nodes // n_communities
    communities = []
    community_i = -1
    for i in nodes:
        if i % nodes_per_community == 0:
            communities.append([])
            community_i += 1
        communities[community_i].append(i)

    labels = {
        1: [0.45, 0.016],
        2: [0.4, 0.033],
        3: [0.35, 0.05],
        4: [0.325, 0.058],
        5: [0.3, 0.066]
    }

    # Act
    nmi_results = []

    for graph_label, [graph_alpha, graph_beta] in labels.items():
        for community_label, [
            community_alpha,
            community_beta
        ] in labels.items():
            A = benchmark_graph(nodes, modules, graph_alpha, graph_beta)
            R = benchmark_graph(
                nodes, communities, community_alpha, community_beta
            )
            result = duo_louvain(A, R)
            nmi = partitions_nmi(communities, result)
            nmi_results.append({
                "graph_label": graph_label,
                "community_label": community_label,
                "nmi": nmi
            })

    try:
        file_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        file_dir = "."

    df_results = pd.DataFrame(nmi_results)
    df_results.to_csv(os.path.join(
        file_dir, "test_duo_louvain_small_nmi.csv"),
        index=False
    )

    # Assert
    for result in nmi_results:
        assert result["nmi"] > 0.9
