import copy
import numpy as np
import networkx as nx
import matplotlib 
import matplotlib.pyplot as plt

from matplotlib.lines import Line2D
from matplotlib.patches import Patch

def draw(
    psi, r, positive_alpha=0.0, negative_alpha=0.0, positive_beta=0.0,
    negative_beta=0.0, symmetric=True, arched=False, arch_radius=0.2,
    node_size_upper_limit=1500, layout='spring', label_margin=None,
    node_pos=None, label_pos=None, label_color=None,
    label_weight='normal', positive_color='black',
    negative_color='red', output_path=None, plot_margin=None,
    node_label_size_limit=500
):
    '''
    Draws the shapley matrices of Psi and R

    Parameters:
    psi (list(float)): Psi values
    r (list(list(float))): R values
    positive_alpha (float): alpha^{+} threshold
    negative_alpha (float): alpha^{-} threshold
    positive_beta (float): beta^{+} threshold
    negative_beta (float): beta^{-} threshold
    symmetric (bool): Defines the symmetry condition
    layout (str): Defines the Networkx drawing layout

    Returns:
    Graph: Networkx graph
    '''
    def filter_edge(x):
        if x > 0 and abs(x) > positive_beta:
            return x
        elif x < 0 and abs(x) > negative_beta:
            return x
        else:
            return 0

    def filter_node(x):
        if positive_alpha == 0 and negative_alpha == 0:
            return True
        elif x > 0 and abs(x) > positive_alpha:
            return True
        elif x < 0 and abs(x) > negative_alpha:
            return True
        else:
            return False

    r = r.copy().applymap(filter_edge)
    adjacency = r.copy().applymap(lambda x: 1 if x != 0 else 0)
    g = nx.from_pandas_adjacency(adjacency, create_using=nx.DiGraph)

    nodes = g.nodes()
    filtered_nodes = [
        n
        for n in nodes
        if not filter_node(psi['value'][n])
    ]
    g.remove_nodes_from(filtered_nodes)

    edges = g.edges()
    edge_color = [
        negative_color if r[v][u] < 0 else positive_color
        for u, v in edges
    ]

    edge_weight_upper_limit = 3
    edge_weights = [
        edge_weight_upper_limit * abs(r[v][u])
        for u, v in edges
    ]

    node_size_lower_limit = 0
    nodes = g.nodes()
    node_size_dict = {
        n: node_size_lower_limit +
        (node_size_upper_limit * abs(psi['value'][n]))
        for n in nodes
    }
    node_size = list(node_size_dict.values())

    node_color = [
        negative_color if psi['value'][n] < 0 else positive_color
        for n in nodes
    ]

    if node_pos == None:
        if layout == 'kamada_kawai':
            pos = nx.kamada_kawai_layout(g)
        elif layout == 'circular':
            pos = nx.circular_layout(g)
        elif layout == 'spectral':
            pos = nx.spectral_layout(g)
        elif layout == 'spring':
            pos = nx.spring_layout(g)
        elif layout == 'planar':
            pos = nx.planar_layout(g)
        elif layout == 'shell':
            pos = nx.shell_layout(g)
        elif layout == 'spiral':
            pos = nx.spiral_layout(g)
        elif layout == 'twopi':
            pos = nx.nx_agraph.graphviz_layout(g, prog='twopi')
        elif layout == 'fruchterman_reingold_layout':
            pos = nx.fruchterman_reingold_layout(g)
        else:
            pos = nx.random_layout(g)
    else:
        pos = node_pos

    plt.figure()
    if plot_margin != None:
        plt.margins(x=plot_margin, y=plot_margin)
    nx.draw_networkx_nodes(
        g, pos,
        node_size=node_size,
        node_color=node_color
    )
    if label_pos == None:
        nx.draw_networkx_labels(
            g, pos,
            labels={node: node for node in g.nodes(
            ) if node_size_dict[node] > node_label_size_limit},
            font_size=10,
            font_color='white' if label_color is None else label_color,
            font_weight='bold'
        )
        nx.draw_networkx_labels(
            g, pos,
            labels={node: node for node in g.nodes(
            ) if node_size_dict[node] == 0},
            font_size=10,
            font_color='black' if label_color is None else label_color,
            font_weight='bold'
        )

        x_max = max([x for x, y in pos.values()])
        y_max = max([y for x, y in pos.values()])
        x_min = min([x for x, y in pos.values()])
        y_min = min([y for x, y in pos.values()])

        y_diff = y_max - y_min
        x_diff = x_max - x_min
        hipo = (y_diff ** 2 + x_diff ** 2) ** 0.5

        max_node_size = max(node_size)
        max_node_size_canvas_ratio = 8
        font_canvas_ratio = 500
        font_margin = 12 * hipo / 2 / font_canvas_ratio


        for node in [
            node for node in nodes
            if node_size_dict[node] <= node_label_size_limit
            and node_size_dict[node] > 0
        ]:
            x, y = pos[node]
            curr_node_size = node_size_dict[node]

            default_x_margin = curr_node_size * (x_diff / max_node_size / max_node_size_canvas_ratio)
            default_y_margin = curr_node_size * (y_diff / max_node_size  / max_node_size_canvas_ratio)

            x_margin = default_x_margin \
                if label_margin == None \
                else label_margin

            y_margin = default_y_margin \
                if label_margin == None \
                else label_margin

            x_margin_label = font_margin * (4 + len(str(node))) / 2
            y_margin_label = font_margin
            
            if y < 0:
                y_label_pos = y - y_margin - y_margin_label
            elif y > 0:
                y_label_pos = y + y_margin
            else:
                y_label_pos = y

            if x < 0:
                x_label_pos = x - x_margin - x_margin_label
            elif x > 0:
                x_label_pos = x + x_margin + x_margin_label
            else:
                x_label_pos = x

            plt.text(
                x_label_pos, y_label_pos, s=node,
                horizontalalignment='center'
            )
    else:
        for node in g.nodes():
            x, y = label_pos[node]
            plt.text(
                x, y, s=node, color='black' if label_color is None else label_color,
                horizontalalignment='center'
            )

    if symmetric:
        if arched:
            nx.draw_networkx_edges(
                g, pos,
                edge_color=edge_color,
                width=edge_weights,
                node_size=[x + 400 for x in node_size],
                arrows=True,
                arrowsize=0,
                arrowstyle='|-|',
                connectionstyle=f'arc3,rad={arch_radius}',
            )
        else:
            nx.draw_networkx_edges(
                g, pos,
                edge_color=edge_color,
                width=edge_weights,
                node_size=[x + 400 for x in node_size],
                arrows=True,
                arrowsize=0,
                arrowstyle='|-|'
            )
    else:
        nx.draw_networkx_edges(
            g, pos,
            edge_color=edge_color,
            width=edge_weights,
            node_size=[x + 400 for x in node_size],
            connectionstyle=f'arc3,rad={arch_radius}',
        )

    axis = plt.gca()
    axis.set_xlim([x for x in axis.get_xlim()])
    axis.set_ylim([y for y in axis.get_ylim()])
    plt.axis('off')
    plt.show()

    if output_path != None:
        print('Saving the graph to', output_path)
        plt.savefig(output_path, format='jpg', dpi=1200, bbox_inches='tight')

    return g


def get_cmap(n, name='hsv'):
    '''
    Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
    RGB color; the keyword argument name must be a standard mpl colormap name.
    '''
    return plt.cm.get_cmap(name, n + 1)

def draw_clusters(
    psi, r, clusters=[], positive_alpha=0.0, negative_alpha=0.0, positive_beta=0.0,
    negative_beta=0.0, symmetric=True, arched=False, arch_radius=0.2,
    node_size_upper_limit=1500, layout='spring', label_margin=None,
    node_pos=None, label_pos=None, label_color='white',
    label_weight='normal', positive_color='black',
    negative_color='red', output_path=None, plot_margin=None,
    node_label_size_limit=500
):
    '''
    Draws the shapley matrices of Psi and R

    Parameters:
    psi (list(float)): Psi values
    r (list(list(float))): R values
    clusters (list(list(int))): List of clusters
    positive_alpha (float): alpha^{+} threshold
    negative_alpha (float): alpha^{-} threshold
    positive_beta (float): beta^{+} threshold
    negative_beta (float): beta^{-} threshold
    symmetric (bool): Defines the symmetry condition
    layout (str): Defines the Networkx drawing layout

    Returns:
    Graph: Networkx graph
    '''
    def filter_edge(x):
        if x > 0 and abs(x) > positive_beta:
            return x
        elif x < 0 and abs(x) > negative_beta:
            return x
        else:
            return 0

    def filter_node(x):
        if positive_alpha == 0 and negative_alpha == 0:
            return True
        elif x > 0 and abs(x) > positive_alpha:
            return True
        elif x < 0 and abs(x) > negative_alpha:
            return True
        else:
            return False

    r = r.copy().applymap(filter_edge)
    adjacency = r.copy().applymap(lambda x: 1 if x != 0 else 0)
    g = nx.from_pandas_adjacency(adjacency, create_using=nx.DiGraph)

    nodes = g.nodes()
    filtered_nodes = [
        n
        for n in nodes
        if not filter_node(psi['value'][n])
    ]
    g.remove_nodes_from(filtered_nodes)
    nodes = g.nodes()

    clusters = [
        [node for node in c if node in nodes]
        for c in clusters
    ]
    clusters = [
        c for c in clusters if len(c) > 0
    ]

    edges = g.edges()
    edge_color = [
        negative_color if r[v][u] < 0 else positive_color
        for u, v in edges
    ]

    edge_weight_upper_limit = 3
    edge_weights = [
        edge_weight_upper_limit * abs(r[v][u])
        for u, v in edges
    ]

    node_size_lower_limit = 0
    node_size_dict = {
        n: node_size_lower_limit +
        (node_size_upper_limit * abs(psi['value'][n]))
        for n in nodes
    }
    node_size = list(node_size_dict.values())

    node_color = [
        negative_color if psi['value'][n] < 0 else positive_color
        for n in nodes
    ]

    if node_pos == None:
        if layout == 'kamada_kawai':
            pos = nx.kamada_kawai_layout(g)
        elif layout == 'circular':
            pos = nx.circular_layout(g)
        elif layout == 'spectral':
            pos = nx.spectral_layout(g)
        elif layout == 'spring':
            pos = nx.spring_layout(g)
        elif layout == 'planar':
            pos = nx.planar_layout(g)
        elif layout == 'shell':
            pos = nx.shell_layout(g)
        elif layout == 'spiral':
            pos = nx.spiral_layout(g)
        elif layout == 'twopi':
            pos = nx.nx_agraph.graphviz_layout(g, prog='twopi')
        elif layout == 'fruchterman_reingold_layout':
            pos = nx.fruchterman_reingold_layout(g)
        else:
            pos = nx.random_layout(g)
    else:
        pos = node_pos

    plt.figure()

    if plot_margin != None:
        plt.margins(x=plot_margin, y=plot_margin)

    if len(clusters) > 0:
        if node_pos == None:
            # Create well-separated cluster centers
            num_clusters = len(clusters)
            cluster_centers = []
            
            if num_clusters == 1:
                cluster_centers = [(0, 0)]
            elif num_clusters == 2:
                cluster_centers = [(-3.0, 0), (3.0, 0)]
            elif num_clusters == 3:
                # Triangle arrangement for better separation
                import math
                cluster_centers = [
                    (0, 3.0),           # Top
                    (-2.6, -1.5),       # Bottom left
                    (2.6, -1.5)         # Bottom right
                ]
            else:
                # Arrange clusters in a circle for better separation
                import math
                radius = 4.0
                for i in range(num_clusters):
                    angle = 2 * math.pi * i / num_clusters
                    x = radius * math.cos(angle)
                    y = radius * math.sin(angle)
                    cluster_centers.append((x, y))
            
            # Position nodes within each cluster
            pos = {}
            cluster_scale = 0.8  # Scale factor for intra-cluster layout
            
            for center, comm in zip(cluster_centers, clusters):
                if len(comm) == 1:
                    # Single node cluster - place at center
                    pos[comm[0]] = center
                else:
                    # Multiple nodes - use spring layout with smaller scale
                    subgraph = nx.subgraph(g, comm)
                    if layout == 'circular' and len(comm) > 2:
                        # Use circular layout for clusters with circular main layout
                        cluster_pos = nx.circular_layout(subgraph, scale=cluster_scale)
                    else:
                        cluster_pos = nx.spring_layout(subgraph, scale=cluster_scale)
                    
                    # Translate positions to cluster center
                    for node, (x, y) in cluster_pos.items():
                        pos[node] = (center[0] + x, center[1] + y)

        legend_elements = []
        cmap = get_cmap(len(clusters))
        for cdx, cluster_nodes in enumerate(clusters):
            color = matplotlib.colors.rgb2hex(cmap(cdx))
            cluster_node_size = [node_size_dict[node] for node in cluster_nodes]
            nx.draw_networkx_nodes(
                g, pos=pos, nodelist=cluster_nodes,
                node_color=color, node_size=cluster_node_size
            )

            legend_elements.append(
                Line2D(
                    [0], [0], marker='o', color='white', label=f"Community {cdx + 1}",
                    markerfacecolor=color, markersize=10
                )
            )
    else:
        nx.draw_networkx_nodes(
            g, pos,
            node_size=node_size,
            node_color=node_color
        )
    if label_pos == None:
        nx.draw_networkx_labels(
            g, pos,
            labels={node: node for node in g.nodes(
            ) if node_size_dict[node] > node_label_size_limit},
            font_size=10,
            font_color='white',
            font_weight='bold'
        )
        nx.draw_networkx_labels(
            g, pos,
            labels={node: node for node in g.nodes(
            ) if node_size_dict[node] == 0},
            font_size=10,
            font_color='black',
            font_weight='bold'
        )

        x_max = max([x for x, y in pos.values()])
        y_max = max([y for x, y in pos.values()])
        x_min = min([x for x, y in pos.values()])
        y_min = min([y for x, y in pos.values()])

        y_diff = y_max - y_min
        x_diff = x_max - x_min
        hipo = (y_diff ** 2 + x_diff ** 2) ** 0.5

        max_node_size = max(node_size)
        max_node_size_canvas_ratio = 8
        font_canvas_ratio = 500
        font_margin = 12 * hipo / 2 / font_canvas_ratio

        for node in [
            node for node in nodes
            if node_size_dict[node] <= node_label_size_limit
            and node_size_dict[node] > 0
        ]:
            x, y = pos[node]
            curr_node_size = node_size_dict[node]

            default_x_margin = curr_node_size * \
                (x_diff / max_node_size / max_node_size_canvas_ratio)
            default_y_margin = curr_node_size * \
                (y_diff / max_node_size / max_node_size_canvas_ratio)

            x_margin = default_x_margin \
                if label_margin == None \
                else label_margin

            y_margin = default_y_margin \
                if label_margin == None \
                else label_margin

            x_margin_label = font_margin * (4 + len(str(node))) / 2
            y_margin_label = font_margin

            if y < 0:
                y_label_pos = y - y_margin - y_margin_label
            elif y > 0:
                y_label_pos = y + y_margin
            else:
                y_label_pos = y

            if x < 0:
                x_label_pos = x - x_margin - x_margin_label
            elif x > 0:
                x_label_pos = x + x_margin + x_margin_label
            else:
                x_label_pos = x

            plt.text(
                x_label_pos, y_label_pos, s=node,
                horizontalalignment='center'
            )
    else:
        for node in g.nodes():
            x, y = label_pos[node]
            plt.text(
                x, y, s=node,
                horizontalalignment='center'
            )

    if symmetric:
        if arched:
            nx.draw_networkx_edges(
                g, pos,
                edge_color=edge_color,
                width=edge_weights,
                node_size=[x + 400 for x in node_size],
                arrows=True,
                arrowsize=0,
                arrowstyle='|-|',
                connectionstyle=f'arc3,rad={arch_radius}',
            )
        else:
            nx.draw_networkx_edges(
                g, pos,
                edge_color=edge_color,
                width=edge_weights,
                node_size=[x + 400 for x in node_size],
                arrows=True,
                arrowsize=0,
                arrowstyle='|-|'
            )
    else:
        nx.draw_networkx_edges(
            g, pos,
            edge_color=edge_color,
            width=edge_weights,
            node_size=[x + 400 for x in node_size],
            connectionstyle=f'arc3,rad={arch_radius}',
        )

    if len(clusters) > 0:
        ax = plt.gca()
        ax.legend(
            handles=legend_elements,
            loc='center left', bbox_to_anchor=(1, 0.5)
        )

    axis = plt.gca()
    axis.set_xlim([x for x in axis.get_xlim()])
    axis.set_ylim([y for y in axis.get_ylim()])
    plt.tight_layout()
    plt.axis('off')
    plt.show()

    if output_path != None:
        print('Saving the graph to', output_path)
        plt.savefig(output_path, format='jpg', dpi=1200, bbox_inches='tight')

    return g