"""
I rewrote the transformer to use more standard layers and optimize for speed.
"""
import math
import typing as ty

import torch
import torch.nn as nn
from torch.nn import functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, dim_model, max_len=1000):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, dim_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim_model, 2).float() * (-math.log(10000.0) / dim_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = F.embedding(x, self.pe).squeeze()  # looks up and retrieves embedding vectors from pe for each index in x
        return x


class ReGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return x * F.relu(gates)


class Transformer(nn.Module):
    def __init__(self, num_features, embedding_dim, num_heads, num_hidden, num_layers,
                 parallel_pools=10,
                 ffn_dropout=0.05,
                 attention_dropout=0.05,
                 residual_dropout=0.05,
                 max_len=365, max_visits=40, num_dim=4,
                 activation='reglu',
                 prenormalization=True, **kwargs):
        super(Transformer, self).__init__()

        self.embedding = nn.Linear(num_features, embedding_dim, bias=False)  # would an embedding layer be faster?
        self.pos_encoder = PositionalEncoding(embedding_dim, max_len)

        self.ffn_dropout = ffn_dropout
        self.residual_dropout = residual_dropout
        self.layers = nn.ModuleList([])
        for layer_idx in range(num_layers):
            layer = nn.ModuleDict(
                {
                    'attention': nn.MultiheadAttention(
                        embedding_dim + num_dim, num_heads, attention_dropout, batch_first=True
                    ),
                    'linear0': nn.Linear(
                        embedding_dim + num_dim, num_hidden * (2 if activation.endswith('glu') else 1)
                    ),
                    'linear1': nn.Linear(num_hidden, embedding_dim + num_dim),
                    'norm1': nn.LayerNorm(embedding_dim + num_dim),
                }
            )
            if not prenormalization or layer_idx:
                layer['norm0'] = nn.LayerNorm(embedding_dim + num_dim)
            self.layers.append(layer)

        self.activation = ReGLU()
        self.last_activation = nn.ReLU()
        self.embedding_dim = embedding_dim
        self.parallel_pools = parallel_pools
        self.num_dim = 4
        self.pooler = nn.Linear(max_visits, self.parallel_pools)
        self.prediction_head = nn.Linear((embedding_dim + num_dim) * self.parallel_pools, 1)
        self.prenormalization = prenormalization

        self.max_visits = max_visits

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.embedding.weight, -initrange, initrange)

    def forward(self, input):
        source = torch.nn.utils.rnn.pad_sequence(input[0]).transpose(0, 1)
        numerical_data = input[1]  # age + gender data and possibly other non-temporal data
        lengths = input[2]  # number of visits per patient
        times = torch.nn.utils.rnn.pad_sequence(input[3]).transpose(0, 1)  # time Ids of visits

        # rescale embeddings with sqrt(embedding_dim), possibly better to downscale time_embeddings instead?
        source_embedding = self.embedding(source) * math.sqrt(self.embedding_dim)
        time_embedding = self.pos_encoder(times)
        total_embedding = source_embedding + time_embedding

        # concat non-temporal data to the visit embeddings
        x = torch.cat((total_embedding, numerical_data[:, None, :].repeat(1, self.max_visits, 1)), dim=2)

        mask = torch.arange(self.max_visits, device=source.device)[None, :] >= lengths[:, None]

        for layer_idx, layer in enumerate(self.layers):
            is_last_layer = layer_idx + 1 == len(self.layers)
            layer = ty.cast(ty.Dict[str, nn.Module], layer)

            x_residual = self._start_residual(x, layer, 0)
            x_residual, _ = layer['attention'](
                x_residual,
                x_residual,
                x_residual,
                key_padding_mask=mask, need_weights=False)

            if is_last_layer:
                x = x[:, : x_residual.shape[1]]
            x = self._end_residual(x, x_residual, layer, 0)

            x_residual = self._start_residual(x, layer, 1)
            x_residual = layer['linear0'](x_residual)
            x_residual = self.activation(x_residual)
            if self.ffn_dropout:
                x_residual = F.dropout(x_residual, self.ffn_dropout, self.training)
            x_residual = layer['linear1'](x_residual)
            x = self._end_residual(x, x_residual, layer, 1)

        output = self.pooler(x.transpose(1, 2)).view(-1, self.parallel_pools * (self.embedding_dim + self.num_dim))
        output = self.prediction_head(output)
        return output.squeeze()

    def _start_residual(self, x, layer, norm_idx):
        x_residual = x
        if self.prenormalization:
            norm_key = f'norm{norm_idx}'
            if norm_key in layer:
                x_residual = layer[norm_key](x_residual)
        return x_residual

    def _end_residual(self, x, x_residual, layer, norm_idx):
        if self.residual_dropout:
            x_residual = F.dropout(x_residual, self.residual_dropout, self.training)
        x = x + x_residual
        if not self.prenormalization:
            x = layer[f'norm{norm_idx}'](x)
        return x
