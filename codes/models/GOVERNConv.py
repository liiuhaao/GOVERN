import torch
import torch.nn.functional as F
from torch.nn import Parameter
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import glorot
from torch_geometric.utils import mask_feature, softmax


class GOVERNConv(MessagePassing):
    def __init__(self, num_class, in_channels, out_channels, heads, **kwargs):
        super(GOVERNConv, self).__init__(aggr="mean", **kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.negative_slope = kwargs.get("negative_slope", 0.2)
        self.dropout = kwargs.get("dropout", 0.5)
        self.dropmessage = kwargs.get("dropmessage", 0.001)
        self.maskfeature = kwargs.get("maskfeature", 0.3)

        self.weight = Parameter(torch.Tensor(in_channels, heads * out_channels))
        self.weight_edge = Parameter(torch.Tensor(num_class, heads * out_channels))

        self.att_j2i = Parameter(torch.Tensor(1, heads, 3 * out_channels))
        self.att_i2j = Parameter(torch.Tensor(1, heads, 3 * out_channels))

        self.message_j2i = torch.nn.Linear(3 * out_channels, out_channels)
        self.message_i2j = torch.nn.Linear(3 * out_channels, out_channels)

        self.lin_update = torch.nn.Linear(heads * out_channels, heads * out_channels)

    def get_alpha(self, x, size, edge_index, edge_attr):
        x = [
            torch.matmul(x[0], self.weight).view(-1, self.heads, self.out_channels),
            torch.matmul(x[1], self.weight).view(-1, self.heads, self.out_channels),
        ]
        edge_attr = torch.matmul(edge_attr, self.weight_edge).view(
            -1, self.heads, self.out_channels
        )
        x_j = x[0][edge_index[0]]
        x_i = x[1][edge_index[1]]
        alpha = torch.cat([x_j, edge_attr, x_i], dim=-1)
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = (alpha * self.att_j2i).sum(dim=-1)
        alpha = softmax(alpha, edge_index[1], num_nodes=size[1])
        alpha = alpha.view(-1, self.heads).mean(dim=-1)
        return alpha

    def reset_parameters(self):
        glorot(self.weight)
        glorot(self.weight_edge)
        glorot(self.att_j2i)
        glorot(self.att_i2j)
        self.message_j2i.reset_parameters()
        self.message_i2j.reset_parameters()
        self.lin_update.reset_parameters()

    def forward(self, x, size, edge_index, edge_attr):
        x = [torch.matmul(x[0], self.weight), torch.matmul(x[1], self.weight)]
        edge_attr = torch.matmul(edge_attr, self.weight_edge)

        self.flow = "source_to_target"
        x_r = self.propagate(edge_index, x=x, size=size, edge_attr=edge_attr)
        self.flow = "target_to_source"
        x_l = self.propagate(edge_index, x=x, size=size, edge_attr=edge_attr)
        return [x_l, x_r]

    def message(self, edge_index_i, edge_index_j, edge_attr, x_i, x_j, size_i, size_j):
        x_i, _ = mask_feature(
            x_i.view(-1, self.heads * self.out_channels),
            p=self.maskfeature,
            training=self.training,
        )
        x_j, _ = mask_feature(
            x_j.view(-1, self.heads * self.out_channels),
            p=self.maskfeature,
            training=self.training,
        )
        edge_attr, _ = mask_feature(
            edge_attr.view(-1, self.heads * self.out_channels),
            p=self.maskfeature,
            training=self.training,
        )

        x_j = x_j.view(-1, self.heads, self.out_channels)
        x_i = x_i.view(-1, self.heads, self.out_channels)
        edge_attr = edge_attr.view(-1, self.heads, self.out_channels)

        if self.flow == "source_to_target":
            alpha = torch.cat([x_i, edge_attr, x_j], dim=-1)
            alpha = F.leaky_relu(alpha, self.negative_slope)
            alpha = (alpha * self.att_j2i).sum(dim=-1)
            alpha = softmax(alpha, edge_index_i, num_nodes=size_i)
            alpha = F.dropout(alpha, p=self.dropout, training=self.training)
            alpha = alpha.view(-1, self.heads, 1)

            # agg = self.message_j2i(torch.cat([x_j, edge_attr, x_i], -1))
            agg = x_j + edge_attr
            agg = agg * alpha
            agg = (1 / (1 - self.dropmessage)) * mask_feature(
                agg, self.dropmessage, "all", training=self.training
            )[0]
            agg = agg.view(-1, self.heads * self.out_channels)
            return agg

        if self.flow == "target_to_source":
            alpha = torch.cat([x_j, edge_attr, x_i], dim=-1)
            alpha = F.leaky_relu(alpha, self.negative_slope)
            alpha = (alpha * self.att_i2j).sum(dim=-1)
            alpha = softmax(alpha, edge_index_j, num_nodes=size_j)
            alpha = F.dropout(alpha, p=self.dropout, training=self.training)
            alpha = alpha.view(-1, self.heads, 1)

            # agg = self.message_i2j(torch.cat([x_i, edge_attr, x_j], -1))
            agg = x_i + edge_attr
            agg = agg * alpha
            agg = (1 / (1 - self.dropmessage)) * mask_feature(
                agg, self.dropmessage, "all", training=self.training
            )[0]
            agg = agg.view(-1, self.heads * self.out_channels)
            return agg

    def update(self, agg):
        agg = self.lin_update(agg)
        agg = F.leaky_relu(agg, self.negative_slope)
        agg = agg.view(-1, self.heads, self.out_channels).mean(dim=1)
        return agg
