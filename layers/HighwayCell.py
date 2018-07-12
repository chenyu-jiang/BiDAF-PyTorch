import torch
import torch.nn as nn

class HighwayCell(nn.Module):
    def __init__(self,feature_size, **kwargs):
        super(HighwayCell, self).__init__()
        self.activation = nn.functional.relu
        self.gate_activation = nn.functional.sigmoid
        self.transform = nn.Linear(feature_size,feature_size)
        self.gate = nn.Linear(feature_size,feature_size)

    def forward(self, data):
        transformed = self.activation(self.transform(data))
        gated = self.gate_activation(self.gate(data))
        return torch.add(torch.mul(transformed,gated),torch.mul(data,1-gated))
