import torch
from sklearn import metrics
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm


def train_model(data, model, params, edge_predictor=None):
    device = torch.device("cpu")
    data = data.to(device)
    model = model.to(device)
    model.reset_parameters()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"]
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=params["epochs"])

    bar = tqdm(total=params["epochs"], dynamic_ncols=True)
    for epoch in range(1, params["epochs"] + 1):
        pseudo_mode = "mv" if epoch < params["pseudo_epoch"] else "knn"
        output = train(
            data, model, optimizer, scheduler, params, pseudo_mode, edge_predictor
        )

        bar.set_postfix(output)
        bar.update(1)
    bar.close()
    
    return {
        k: output[k]
        for k in output.keys()
        if k in ["accuracy", "macro_fscore", "weighted_fscore"]
    }


def train(data, model, optimizer, scheduler, params, pseudo_mode, edge_predictor=None):
    model.train()
    optimizer.zero_grad()
    output = model(data, pseudo_mode, edge_predictor)
    loss = torch.stack(
        [
            v * params["loss_weight"][k]
            for k, v in output.items()
            if k in params["loss_weight"]
        ]
    ).sum()
    loss.backward()
    optimizer.step()
    scheduler.step()

    output = {k: v.item() for k, v in output.items() if k in params["loss_weight"]}

    model.eval()
    pre = model.predict(data)

    mask = data["task"].y != -1
    accuracy = metrics.accuracy_score(
        data["task"].y[mask].cpu().numpy(), pre[mask].cpu().numpy()
    )
    macro_fscore = metrics.f1_score(
        data["task"].y[mask].cpu().numpy(), pre[mask].cpu().numpy(), average="macro"
    )
    weighted_fscore = metrics.f1_score(
        data["task"].y[mask].cpu().numpy(), pre[mask].cpu().numpy(), average="weighted"
    )

    output["loss"] = loss.item()
    output["accuracy"] = accuracy
    output["macro_fscore"] = macro_fscore
    output["weighted_fscore"] = weighted_fscore

    return output
