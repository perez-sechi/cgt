def normalize_psi(psi):
    max_node_size = psi['value'].map(lambda x: abs(x)).max()
    psi['value'] = psi['value'].apply(lambda x: x / max_node_size)

    return psi

def normalize_r(r):
    max_edge_width = r.applymap(lambda x: abs(x)).max().max()
    r = r.copy().applymap(lambda x: x / max_edge_width)

    return r