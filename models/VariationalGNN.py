import torch
import torch.nn as nn
from torch_geometric.nn import GATv2Conv

MAX_LOGSTD = 10


class VariationalGNN(nn.Module):
    """
    My adaption of the model from http://arxiv.org/abs/1912.03761
    I rewrote to be used with standard layers from pytorch geometric. It's quite a bit faster.
    """

    def __init__(self, num_features, dim_embedding=256, none_graph_features=0, attention_dropout=0.1,
                 dropout=0.1, num_heads=1, num_layers=1):
        super(VariationalGNN, self).__init__()
        self.num_features = num_features + 1 - none_graph_features  # +1 for target node
        self.embedding = nn.Embedding(num_embeddings=num_features, embedding_dim=dim_embedding, padding_idx=0)
        self.GAT_layers = nn.ModuleList(
            nn.ModuleDict({
                'attention': GATv2Conv(in_channels=dim_embedding, out_channels=dim_embedding, heads=num_heads,
                                       concat=True,
                                       dropout=attention_dropout),
                'norm': nn.LayerNorm(dim_embedding * num_heads),
                'act': nn.ReLU(),
                'linear': nn.Linear(dim_embedding * num_heads, dim_embedding)
            }) for _ in range(num_layers))
        self.decoder = nn.ModuleDict({
            'attention': GATv2Conv(in_channels=dim_embedding, out_channels=dim_embedding, heads=num_heads,
                                   dropout=attention_dropout, concat=False),
            'norm': nn.LayerNorm(dim_embedding),
            'act': nn.ReLU(),
            'linear': nn.Linear(dim_embedding, dim_embedding)})

        self.parameterize = nn.Linear(dim_embedding, 2 * dim_embedding)
        self.dim_embedding = dim_embedding

        self.dropout = nn.Dropout(p=dropout)
        self.numerical_features = nn.Sequential(
            nn.Linear(none_graph_features, dim_embedding // 2, bias=False),
            nn.LayerNorm(dim_embedding // 2),
            nn.ReLU(),
            nn.Dropout(p=dropout))

        self.classification = nn.Sequential(
            nn.Linear(dim_embedding + dim_embedding // 2, dim_embedding, bias=False),
            nn.LayerNorm(dim_embedding),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(dim_embedding, 1))

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        output_edge_index = data.output_edge_index
        x = self.embedding(x).squeeze()

        for layer in self.GAT_layers:
            x = layer['attention'](x, edge_index)
            x = layer['linear'](layer['act'](layer['norm'](x)))

        # variational stuff
        x = self.parameterize(x).view(-1, 2, self.dim_embedding)
        x = self.dropout(x)
        mu = x[:, 0, :]
        logvar = x[:, 1, :].clamp(max=MAX_LOGSTD)
        x = self.reparameterise(mu, logvar)

        x = self.decoder['attention'](x, output_edge_index)
        x = self.decoder['norm'](x)
        x = self.decoder['act'](x)

        last_node_ptr = data.ptr[1:] - 1
        first_node_ptr = data.ptr[:-1]
        x = x[data.ptr[1:] - 1]  # get representation for the last node of each graph in batch

        num = self.numerical_features(data.demographic)
        x = self.classification(torch.cat((num, x), dim=1)).squeeze()

        unnormalized = torch.zeros(data.num_graphs, device=x.device).scatter_add_(0, data.batch, (logvar.exp() -
                                                                                                  logvar - 1 + mu.pow(
                    2)).sum(1))
        kld = 0.5 * unnormalized / (
                last_node_ptr - first_node_ptr + 1)  # divide by number of nodes in each graph in batch
        return [x, kld.sum()]

    def reparameterise(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = std.data.new(std.size()).normal_()
            return eps.mul(std).add_(mu)
        else:
            return mu


class VAELoss(nn.Module):
    """
    Loss function to use with variational models. Blends together reconstruction loss and kl-divergence using alpha.
    """

    def __init__(self, alpha=1, pos_weight=50):
        super(VAELoss, self).__init__()
        self.alpha = alpha
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def forward(self, inputs, target):
        logits, kld = inputs
        return self.bce(logits, target) + self.alpha * kld
