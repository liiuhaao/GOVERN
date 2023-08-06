import os

import torch
from tqdm import tqdm

from models.EdgePredictor import EdgePredictor
from utils.set_seed import set_seed


def load_edge_predictor(dataset_path, data, lr=0.01, weight_decay=5e-4, epochs=500):
    predictor_path = os.path.join(dataset_path, "edge-predictor.pkl")
    if os.path.exists(predictor_path):
        print("load edge predictor from " + predictor_path)
        edge_predictor = torch.load(predictor_path)
    else:
        print("rebuild edge predictor")
        edge_predictor = EdgePredictor(
            data["worker"].x.shape[1], 64, data["worker", "answer", "task"].num_class
        )
        train_edge_predictor(
            data["worker"].x,
            data["task"].x,
            edge_predictor,
            data["worker", "answer", "task"].edge_index,
            data["worker", "answer", "task"].edge_attr,
            lr,
            weight_decay,
            epochs,
        )
        torch.save(edge_predictor, predictor_path)
    return edge_predictor


def train_edge_predictor(
    x_worker, x_item, model, edge_index, edge_attr, lr, weight_decay, epochs
):
    edge_y = torch.argmax(edge_attr, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    bar = tqdm(total=epochs, dynamic_ncols=True)
    for _ in range(1, epochs + 1):
        output = train(model, optimizer, x_worker, x_item, edge_y, edge_index)
        bar.set_postfix(output)
        bar.update(1)
    bar.close()


def train(model, optimizer, x_worker, x_item, edge_y, edge_index):
    model.train()
    optimizer.zero_grad()
    out = model(x_worker, x_item, edge_index)
    loss = torch.nn.NLLLoss()(out[2], edge_y)
    pred = torch.argmax(out[2], -1)

    loss.backward()
    optimizer.step()
    acc = pred.eq(edge_y).sum().item() / pred.numel()
    return {"loss": loss.item(), "accuracy": acc}
