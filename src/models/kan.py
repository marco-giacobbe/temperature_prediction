import torch
import sys

from src.layers import KANLinear

class KAN(torch.nn.Module):
    def __init__(self, layers_hidden):
        super(KAN, self).__init__()
        self.model_type = 'kan'

        self.layers = torch.nn.ModuleList()
        for in_features, out_features in zip(layers_hidden, layers_hidden[1:]):
            self.layers.append(KANLinear(in_features, out_features))

    def forward(self, x: torch.Tensor, update_grid=False):
        for layer in self.layers:
            if update_grid:
                layer.update_grid(x)
            x = layer(x)
        return x