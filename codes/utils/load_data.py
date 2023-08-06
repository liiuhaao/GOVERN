import os

import pandas as pd
import torch
from torch_geometric.data import HeteroData


def build_bipartite(label, truth):
    num_class = truth["truth"].max() + 1
    from_id = []
    to_id = []
    edge_attr = []
    edge_y = []
    item_y = []
    mv = []
    worker_mapping = {}
    item_mapping = {}

    for _, row in label.iterrows():
        if row["worker"] not in worker_mapping:
            worker_mapping[row["worker"]] = len(worker_mapping)
        if row["task"] not in item_mapping:
            mv.append([0] * num_class)
            if len(truth[truth["task"] == row["task"]]) == 0:
                item_y.append(-1)
            else:
                item_y.append(truth[truth["task"] == row["task"]]["truth"].values[0])
            item_mapping[row["task"]] = len(item_mapping)

        from_id.append(worker_mapping[row["worker"]])
        to_id.append(item_mapping[row["task"]])
        edge_feature = [0] * num_class
        edge_feature[row["answer"]] = 1
        edge_attr.append(edge_feature)
        edge_y.append(row["answer"])
        mv[item_mapping[row["task"]]][row["answer"]] += 1

    edge_index = torch.tensor([from_id, to_id])
    edge_attr = torch.tensor(edge_attr, dtype=torch.float32)
    edge_y = torch.tensor(edge_y)
    mv = torch.tensor(mv, dtype=torch.float32)
    mv = mv / mv.sum(1, keepdim=True)

    feature_size = max(len(item_mapping), len(worker_mapping))
    x = torch.eye(feature_size, dtype=torch.float32)
    worker_x = x[: len(worker_mapping)]
    item_x = x[: len(item_mapping)]

    data = HeteroData()
    data["worker"].num_nodes = len(worker_mapping)
    data["worker"].x = worker_x
    data["task"].num_nodes = len(item_mapping)
    data["task"].x = item_x
    data["task"].y = torch.tensor(item_y)
    data["task"].mv = mv
    data["worker", "answer", "task"].edge_index = edge_index
    data["worker", "answer", "task"].edge_attr = edge_attr
    data["worker", "answer", "task"].y = edge_y
    data["worker", "answer", "task"].num_class = num_class
    return data


def load_data(dataset_path):
    graph_path = os.path.join(dataset_path, "bipartite-graph.pkl")

    if not os.path.exists(graph_path):
        print("rebuild graph")
        df_label = pd.read_csv(os.path.join(dataset_path, "label.csv"))
        df_label = df_label.drop_duplicates(keep="first")
        df_truth = pd.read_csv(os.path.join(dataset_path, "truth.csv"))
        data = build_bipartite(df_label, df_truth)
        torch.save(data, graph_path)
    else:
        print("load graph from " + graph_path)
        data = torch.load(graph_path)
    return data
