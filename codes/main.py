import json

import numpy as np

from config import *
from models.GOVERN import GOVERN
from utils.load_data import load_data
from utils.load_edge_predicter import load_edge_predictor
from utils.set_seed import set_seed
from utils.train_model import train_model


def single_data(dataset, params):
    dataset_path = os.path.join(data_path, datasets_dict[dataset])
    data = load_data(dataset_path)
    if params["edge_predictor"]:
        edge_predictor = load_edge_predictor(dataset_path, data)
    else:
        edge_predictor = None

    set_seed(params["seed"])
    model = GOVERN(
        data["worker", "answer", "task"].num_class,
        data["task"].x.shape[-1],
        **params,
    )
    output = train_model(data, model, params, edge_predictor)
    return output


if __name__ == "__main__":
    params = dict(
        out_channels=5,
        proj_channels=5,
        heads=5,
        hiddens=2,
        lr=0.3,
        epochs=100,
        weight_decay=1e-4,
        negative_slope=0.2,
        dropout=0.5,
        dropmessage=0.001,
        dropedge=0.3,
        addedge=0.3,
        maskfeature=0.3,
        tau=0.5,
        k=5,
        pseudo_epoch=60,
        edge_predictor=True,
        loss_weight={
            "loss_cw": 0.1,
            "loss_ct": 1,
            "loss_cc": 0.1,
            "loss_pl": 0.8,
        },
    )

    datasets = datasets_dict.keys() if args.dataset == "all" else [args.dataset]

    reports = dict()
    for dataset in datasets:
        if dataset not in reports:
            reports[dataset] = dict()

        seeds = args.seeds

        for seed in seeds:
            params["seed"] = seed
            output = single_data(dataset, params)
            for metric in output.keys():
                if metric not in reports[dataset]:
                    reports[dataset][metric] = dict()
                    reports[dataset][metric]["values"] = []
                reports[dataset][metric]["values"].append(output[metric] * 100)

        for metric in reports[dataset]:
            reports[dataset][metric]["mean"] = np.mean(
                reports[dataset][metric]["values"]
            )
            reports[dataset][metric]["std"] = np.std(reports[dataset][metric]["values"])

        print(
            dataset, [reports[dataset][metric]["mean"] for metric in reports[dataset]]
        )

    reports["Avg"] = dict()
    for metric in reports[dataset]:
        reports["Avg"][metric] = dict()
        reports["Avg"][metric]["mean"] = np.mean(
            [reports[dataset][metric]["mean"] for dataset in datasets]
        )
        reports["Avg"][metric]["std"] = np.mean(
            [reports[dataset][metric]["std"] for dataset in datasets]
        )

    with open(os.path.join(output_path, args.name + ".json"), "w") as f:
        json.dump(reports, f, indent=4)
