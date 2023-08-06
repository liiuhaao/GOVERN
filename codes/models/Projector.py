import torch
import torch.nn.functional as F


class Projector(torch.nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(Projector, self).__init__()
        self.negative_slope = kwargs.get("negative_slope", 0.2)
        self.lin = torch.nn.Linear(in_channels, out_channels)

    def reset_parameters(self):
        self.lin.reset_parameters()

    def forward(self, x):
        x = self.lin(x)
        x = F.leaky_relu(x, self.negative_slope)
        x = F.normalize(x, dim=1)
        return x
