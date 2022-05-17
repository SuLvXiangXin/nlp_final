import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, in_channel, classes=4, hidden_layers=[362, 942, 1071, 870, 318, 912, 247]):
        super(MLP, self).__init__()
        self.hidden_layers = [in_channel] + list(hidden_layers) + [classes]

        l = []
        for i in range(len(self.hidden_layers)-1):
            l.append(nn.Linear(self.hidden_layers[i], self.hidden_layers[i+1]))
            if i<len(self.hidden_layers)-2:
                l.append(nn.ReLU())
        self.linear = nn.Sequential(*l)

    def forward(self, x):
        return self.linear(x)
