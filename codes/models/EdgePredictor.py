import torch
import torch.nn.functional as F


class EdgePredictor(torch.nn.Module):
    def __init__(self, in_channels, out_channels, num_class):
        super(EdgePredictor, self).__init__()
        self.out_channels = out_channels
        self.worker_encoder = torch.nn.Sequential(
            torch.nn.Linear(in_channels, out_channels),
            torch.nn.LeakyReLU(),
        )
        self.item_encoder = torch.nn.Sequential(
            torch.nn.Linear(in_channels, out_channels),
            torch.nn.LeakyReLU(),
        )
        self.edge_predictor = torch.nn.Linear(out_channels * 2, num_class)

    def forward(self, x_worker, x_item, edge_index):
        x_worker = self.worker_encoder(x_worker)
        x_item = self.item_encoder(x_item)
        x_cat = torch.cat(
            [
                x_worker[
                    edge_index[0],
                ],
                x_item[edge_index[1]],
            ],
            -1,
        )
        edge_prob = self.edge_predictor(x_cat)
        return x_worker, x_item, F.log_softmax(edge_prob, -1)

    @torch.no_grad()
    def get_prob(self, x, edge_index):
        x_worker = self.worker_encoder(x[0])
        x_item = self.item_encoder(x[1])
        x_cat = torch.cat(
            [
                x_worker[
                    edge_index[0],
                ],
                x_item[edge_index[1]],
            ],
            -1,
        )
        edge_prob = self.edge_predictor(x_cat)
        return F.softmax(edge_prob, -1)
