"""
A non-temporal resnet baseline
"""

import torch
from torch import nn


class ResidualLayer(nn.Module):
    def __init__(self, size_hidden, res_hidden, normalization, activation,
                 hidden_dropout=None, residual_dropout=None):
        super(ResidualLayer, self).__init__()
        self.norm = normalization(size_hidden)
        self.linear0 = nn.Linear(size_hidden, res_hidden)
        self.linear1 = nn.Linear(res_hidden, size_hidden)

        self.activation = activation
        self.hidden_dropout = hidden_dropout
        self.residual_dropout = residual_dropout
        if hidden_dropout:
            self.hidden_dropout = nn.Dropout(p=hidden_dropout)
        if residual_dropout:
            self.residual_dropout = nn.Dropout(p=residual_dropout)

    def forward(self, input):
        x = input
        z = self.norm(input)
        z = self.linear0(z)
        z = self.activation(z)
        if self.hidden_dropout:
            z = self.hidden_dropout(z)
        z = self.linear1(z)
        if self.residual_dropout:
            z = self.residual_dropout(z)
        z = z + x
        return z


class ResNet(nn.Module):

    def __init__(self, num_features=None, size_embedding=128, size_hidden=128, num_layers=4,
                 hidden_factor=2, activation=nn.ReLU(), normalization=nn.BatchNorm1d, hidden_dropout=None,
                 residual_dropout=None, dim_out=1, num_numerical_features=1):
        super(ResNet, self).__init__()

        self.embedding = nn.Linear(num_features - num_numerical_features, size_embedding, bias=False)
        self.first_layer = nn.Linear(size_embedding + num_numerical_features, size_hidden)

        res_hidden = size_hidden * hidden_factor

        self.layers = nn.ModuleList(ResidualLayer(size_hidden, res_hidden, normalization,
                                                  activation, hidden_dropout, residual_dropout)
                                    for _ in range(num_layers))
        self.last_norm = normalization(size_hidden)
        self.head = nn.Linear(size_hidden, dim_out)
        self.last_act = activation

    def forward(self, input):
        cat_input = input[0]
        num_input = input[1]
        cat_input = self.embedding(cat_input)
        x = torch.cat((cat_input, num_input), dim=1)
        x = self.first_layer(x)

        for layer in self.layers:
            x = layer(x)
        x = self.last_norm(x)
        x = self.last_act(x)
        x = self.head(x)
        x = x.squeeze(-1)
        return x
