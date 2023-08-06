import torch
from torch_geometric.utils import add_random_edge, dropout_edge


def drop_edge(x, edge_index, edge_attr, dropedge, edge_predictor=None):
    if dropedge <= 0:
        return edge_index, edge_attr
    if edge_predictor is None:
        edge_index, edge_mask = dropout_edge(edge_index, dropedge)
        edge_attr = edge_attr[edge_mask]
    else:
        edge_prob = torch.sum(edge_predictor.get_prob(x, edge_index) * edge_attr, -1)
        edge_mask = torch.rand(edge_prob.shape) >= (1 - edge_prob) * dropedge
        edge_index = torch.stack([edge_index[0][edge_mask], edge_index[1][edge_mask]])
        edge_attr = edge_attr[edge_mask]
    return edge_index, edge_attr


def add_edge(x, edge_index, edge_attr, addedge, num_class, size, edge_predictor=None):
    if addedge <= 0:
        return edge_index, edge_attr

    edge_index, added_edges = add_random_edge(edge_index, addedge, num_nodes=size)

    if edge_predictor is None:
        new_label = torch.randint(0, num_class, (added_edges.shape[1],))
    else:
        edge_prob = edge_predictor.get_prob(x, added_edges)
        new_label = torch.multinomial(edge_prob, 1)

    new_attr = torch.eye(num_class)[new_label].squeeze(1)
    edge_attr = torch.cat([edge_attr, new_attr], 0)
    return edge_index, edge_attr


def transform_data(data, dropedge, addedge, edge_predictor=None):
    x = [torch.clone(data["worker"].x), torch.clone(data["task"].x)]
    edge_index = torch.clone(data["worker", "answer", "task"].edge_index)
    edge_attr = torch.clone(data["worker", "answer", "task"].edge_attr)

    size = (data["worker"].num_nodes, data["task"].num_nodes)
    num_class = data["worker", "answer", "task"].num_class

    edge_index, edge_attr = drop_edge(
        x,
        edge_index,
        edge_attr,
        dropedge,
        edge_predictor,
    )
    edge_index, edge_attr = add_edge(
        x,
        edge_index,
        edge_attr,
        addedge,
        num_class,
        size,
        edge_predictor,
    )

    return x, edge_index, edge_attr
