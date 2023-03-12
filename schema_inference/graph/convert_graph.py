import networkx as nx

import torch


def to_networkx(
    node_weights: torch.Tensor,
    adj_matrix: torch.Tensor,
    edge_threshold: float,
    node_threshold: float,
    node_topk: int = 10,
    edge_topk: int = 5
) -> nx.Graph:
    """
    Args:
        node_weights: [n]
        adj_matrix: [n, n]
    """
    n = node_weights.shape[0]
    node_ids = torch.arange(n)
    # topk nodes
    filter_ids = node_weights.argsort(descending=True)[:node_topk]
    node_weights = node_weights[filter_ids]
    node_ids = node_ids[filter_ids]
    # remove nodes with zero weight
    mask = node_weights > node_threshold
    node_weights = node_weights.masked_select(mask)
    node_ids = node_ids.masked_select(mask)
    # require only upper triangle matrix
    adj_matrix = adj_matrix.triu()
    topk_adj, topk_idx = adj_matrix.topk(edge_topk, dim=-1)

    node_ids = node_ids.tolist()
    node_weights = node_weights.tolist()
    topk_adj = topk_adj.tolist()
    topk_idx = topk_idx.tolist()
    # add nodes
    graph = nx.Graph()
    for node_id, node_w in zip(node_ids, node_weights):
        graph.add_node(node_id, weight=node_w)

    # add edges
    for i in node_ids:
        for j, w in zip(topk_idx[i], topk_adj[i]):
            if j in node_ids and w > edge_threshold:
                graph.add_edge(i, j, weight=w)
    return graph
