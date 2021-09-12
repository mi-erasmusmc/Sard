import math
import typing as ty

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence


class Embedding(nn.Module):
    def __init__(self,
                 dim_numerical: int,
                 dim_embedding: int,
                 dim_categorical: int,
                 max_length: int,
                 bias: bool):
        super().__init__()
        self.dim_bias = dim_numerical + max_length
        self.embedder = nn.Embedding(dim_categorical, dim_embedding)
        nn.init.kaiming_uniform_(self.embedder.weight, a=math.sqrt(5))

        self.max_length = max_length
        self.dim_embedding = dim_embedding
        self.dim_categorical = dim_categorical
        self.weight = nn.Parameter(torch.Tensor(dim_numerical + 1, dim_embedding))
        self.bias = nn.Parameter(torch.Tensor(self.dim_bias, dim_embedding)) if bias else None
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            nn.init.kaiming_uniform_(self.bias, a=math.sqrt(5))

    def forward(self, x_num, x_cat):
        x_cat = pad_sequence(x_cat, batch_first=True)
        x_num = torch.cat(
            [
                torch.ones(len(x_num), 1, device=x_num.device),
                x_num,
            ],
            dim=1,
        )
        x = self.weight[None] * x_num[:, :, None]
        if x_cat is not None:
            x = torch.cat(
                [x, self.embedder(x_cat)],
                dim=1,
            )
        if self.bias is not None:
            bias = torch.cat(
                [
                    torch.zeros(1, self.bias.shape[1], device=x_num.device),
                    self.bias,
                ]
            )
            if x.shape[1] < self.dim_bias:
                x = torch.cat((x, torch.zeros((x.shape[0], self.dim_bias + 1 - x.shape[1], self.dim_embedding),
                                              device=x.device)), dim=1)
            x = x + bias[None]
        return x

    @property
    def n_tokens(self) -> int:
        return len(self.weight) + (
            0 if self.dim_categorical is None else self.dim_categorical
        )


class MultiheadAttention(nn.Module):
    def __init__(
            self, d: int, n_heads: int, dropout: float, initialization: str
    ) -> None:
        if n_heads > 1:
            assert d % n_heads == 0
        assert initialization in ['xavier', 'kaiming']

        super().__init__()
        self.W_q = nn.Linear(d, d)
        self.W_k = nn.Linear(d, d)
        self.W_v = nn.Linear(d, d)
        self.W_out = nn.Linear(d, d) if n_heads > 1 else None
        self.n_heads = n_heads
        self.dropout = nn.Dropout(dropout) if dropout else None

        for m in [self.W_q, self.W_k, self.W_v]:
            if initialization == 'xavier' and (n_heads > 1 or m is not self.W_v):
                # gain is needed since W_qkv is represented with 3 separate layers
                nn.init.xavier_uniform_(m.weight, gain=1 / math.sqrt(2))
            nn.init.zeros_(m.bias)
        if self.W_out is not None:
            nn.init.zeros_(self.W_out.bias)

    def _reshape(self, x):
        batch_size, n_tokens, d = x.shape
        d_head = d // self.n_heads
        return (
            x.reshape(batch_size, n_tokens, self.n_heads, d_head)
                .transpose(1, 2)
                .reshape(batch_size * self.n_heads, n_tokens, d_head)
        )

    def forward(
            self,
            x_q,
            x_kv,
            key_compression,
            value_compression):
        q, k, v = self.W_q(x_q), self.W_k(x_kv), self.W_v(x_kv)
        for tensor in [q, k, v]:
            assert tensor.shape[-1] % self.n_heads == 0
        if key_compression is not None:
            assert value_compression is not None
            k = key_compression(k.transpose(1, 2)).transpose(1, 2)
            v = value_compression(v.transpose(1, 2)).transpose(1, 2)
        else:
            assert value_compression is None

        batch_size = len(q)
        d_head_key = k.shape[-1] // self.n_heads
        d_head_value = v.shape[-1] // self.n_heads
        n_q_tokens = q.shape[1]

        q = self._reshape(q)
        k = self._reshape(k)
        attention = F.softmax(q @ k.transpose(1, 2) / math.sqrt(d_head_key), dim=-1)
        if self.dropout is not None:
            attention = self.dropout(attention)
        x = attention @ self._reshape(v)
        x = (
            x.reshape(batch_size, self.n_heads, n_q_tokens, d_head_value)
                .transpose(1, 2)
                .reshape(batch_size, n_q_tokens, self.n_heads * d_head_value)
        )
        if self.W_out is not None:
            x = self.W_out(x)
        return x


class ReGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return x * F.relu(gates)


class FT_Transformer(nn.Module):
    def __init__(
            self,
            *,
            # tokenizer
            dim_numerical,
            dim_categories,
            token_bias,
            max_length,
            # transformer
            n_layers: int,
            dim_embedding: int,
            n_heads: int,
            d_ffn_factor: float,
            attention_dropout: float,
            ffn_dropout: float,
            residual_dropout: float,
            prenormalization=True,
            initialization: str,
            #
            d_out=1,
    ) -> None:

        super().__init__()
        self.tokenizer = Embedding(dim_numerical, dim_embedding, dim_categories, bias=token_bias, max_length=max_length)
        activation = 'reglu'

        def make_normalization():
            return nn.LayerNorm(dim_embedding)

        d_hidden = int(dim_embedding * d_ffn_factor)
        self.layers = nn.ModuleList([])
        for layer_idx in range(n_layers):
            layer = nn.ModuleDict(
                {
                    'attention': MultiheadAttention(
                        dim_embedding, n_heads, attention_dropout, initialization
                    ),
                    'linear0': nn.Linear(
                        dim_embedding, d_hidden * (2 if activation.endswith('glu') else 1)
                    ),
                    'linear1': nn.Linear(d_hidden, dim_embedding),
                    'norm1': make_normalization(),
                }
            )
            if not prenormalization or layer_idx:
                layer['norm0'] = make_normalization()
            self.layers.append(layer)

        self.activation = ReGLU()
        self.last_activation = nn.ReLU()
        self.prenormalization = prenormalization
        self.last_normalization = make_normalization() if prenormalization else None
        self.ffn_dropout = ffn_dropout
        self.residual_dropout = residual_dropout
        self.head = nn.Linear(dim_embedding, d_out)

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

    def forward(self, input):
        x_cat = input[0]
        x_num = input[1]
        x = self.tokenizer(x_num, x_cat)
        for layer_idx, layer in enumerate(self.layers):
            is_last_layer = layer_idx + 1 == len(self.layers)
            layer = ty.cast(ty.Dict[str, nn.Module], layer)

            x_residual = self._start_residual(x, layer, 0)
            x_residual = layer['attention'](
                # for the last attention, it is enough to process only [CLS]
                (x_residual[:, :1] if is_last_layer else x_residual),
                x_residual,
                None,
                None
            )
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

        assert x.shape[1] == 1
        x = x[:, 0]
        if self.last_normalization is not None:
            x = self.last_normalization(x)
        x = self.last_activation(x)
        x = self.head(x)
        x = x.squeeze(-1)
        return x
