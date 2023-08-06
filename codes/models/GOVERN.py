import torch
import torch.nn.functional as F

from models.GOVERNConv import GOVERNConv
from models.Projector import Projector
from utils.transforms import transform_data


class GOVERNEncoder(torch.nn.Module):
    def __init__(self, num_class, in_channels, out_channels, heads, hiddens, **kwargs):
        super(GOVERNEncoder, self).__init__()
        self.convs = torch.nn.Sequential()
        channel = in_channels
        for i in range(hiddens):
            self.convs.add_module(
                f"GOVERNConv_{i}",
                GOVERNConv(num_class, channel, out_channels, heads, **kwargs),
            )
            channel = out_channels

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, size, edge_index, edge_attr):
        for conv in self.convs:
            x = conv(x, size, edge_index, edge_attr)
        return x


class GOVERN(torch.nn.Module):
    def __init__(
        self,
        num_class,
        in_channels,
        out_channels,
        proj_channels,
        heads,
        hiddens,
        **kwargs,
    ):
        super(GOVERN, self).__init__()
        self.num_class = num_class
        self.negative_slope = kwargs.get("negative_slope", 0.2)
        self.dropout = kwargs.get("dropout", 0.5)
        self.dropedge = kwargs.get("dropedge", 0.3)
        self.addedge = kwargs.get("addedge", 0.3)
        self.tau = kwargs.get("tau", 0.5)
        self.k = kwargs.get("k", 5)

        self.encoder = GOVERNEncoder(
            num_class, in_channels, out_channels, heads, hiddens, **kwargs
        )
        self.projector = Projector(out_channels, proj_channels, **kwargs)
        self.predictor = torch.nn.Linear(out_channels, self.num_class)

        self.reset_parameters()

    def reset_parameters(self):
        self.encoder.reset_parameters()
        self.projector.reset_parameters()
        self.predictor.reset_parameters()

    def forward(self, data, pseudo_mode="knn", edge_predictor=None):
        size = (data["worker"].num_nodes, data["task"].num_nodes)
        x_q, edge_index_q, edge_attr_q = transform_data(
            data, self.dropedge, self.addedge, edge_predictor
        )
        x_k, edge_index_k, edge_attr_k = transform_data(
            data, self.dropedge, self.addedge, edge_predictor
        )

        x_q = self.encoder(x_q, size, edge_index_q, edge_attr_q)
        x_k = self.encoder(x_k, size, edge_index_k, edge_attr_k)

        y_hat_q = self.predictor(x_q[1])
        y_hat_k = self.predictor(x_k[1])

        x_q = [self.projector(x_q[0]), self.projector(x_q[1])]
        x_k = [self.projector(x_k[0]), self.projector(x_k[1])]

        loss_dict = dict()
        loss_dict["loss_cw"] = 0.5 * (
            self.loss_inst(x_q[0], x_k[0]) + self.loss_inst(x_k[0], x_q[0])
        )
        loss_dict["loss_ct"] = 0.5 * (
            self.loss_inst(x_q[1], x_k[1]) + self.loss_inst(x_k[1], x_q[1])
        )

        x = [data["worker"].x, data["task"].x]
        edge_index = data["worker", "answer", "task"].edge_index
        edge_attr = data["worker", "answer", "task"].edge_attr
        size = (data["worker"].num_nodes, data["task"].num_nodes)

        x = self.encoder(x, size, edge_index, edge_attr)
        y = self.pseudo(data, self.projector(x[1]), pseudo_mode)

        loss_dict["loss_cc"] = 0.5 * (
            self.loss_inst(
                F.normalize(y_hat_q.t(), dim=1),
                F.normalize(y_hat_k.t(), dim=1),
            )
            + self.loss_inst(
                F.normalize(y_hat_k.t(), dim=1),
                F.normalize(y_hat_q.t(), dim=1),
            )
        )

        loss_dict["loss_pl"] = F.cross_entropy(y_hat_q, y) + F.cross_entropy(y_hat_k, y)
        return loss_dict

    @torch.no_grad()
    def predict(self, data):
        x = [data["worker"].x, data["task"].x]
        edge_index = data["worker", "answer", "task"].edge_index
        edge_attr = data["worker", "answer", "task"].edge_attr
        size = (data["worker"].num_nodes, data["task"].num_nodes)

        x = self.encoder(x, size, edge_index, edge_attr)
        y = self.predictor(x[1])
        return torch.argmax(y, dim=1)

    def pseudo(self, data, z, pseudo_mode="knn"):
        label = torch.clone(data["task"].mv)
        label = torch.nn.functional.normalize(label, p=1, dim=1)

        if pseudo_mode == "knn":
            z_item = F.normalize(z.detach(), dim=1)
            k = min(self.k, z_item.shape[0])
            topk_sim, topk_index = torch.topk(
                torch.matmul(z_item, z_item.t()), k=k, dim=1
            )
            topk_labels = label[topk_index.flatten()].view(-1, k, self.num_class)
            label = torch.bmm(topk_sim.unsqueeze(1), topk_labels).squeeze(1)
            label = torch.nn.functional.normalize(label, p=1, dim=1)

        return label

    def loss_inst(self, z_q, z_k):
        sim_qq = torch.exp(torch.mm(z_q, z_q.t())) / self.tau
        sim_qk = torch.exp(torch.mm(z_q, z_k.t())) / self.tau
        loss = -torch.log(
            sim_qk.diag() / (sim_qq.sum(1) + sim_qk.sum(1) - sim_qq.diag())
        )
        return loss.mean()
