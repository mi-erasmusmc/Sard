import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence


class RETAIN(nn.Module):
    def __init__(self, num_features, dim_emb=128, dropout_emb=0.5, dim_alpha=128, dim_beta=128, dim_num=2,
                 dropout_context=0.5, dim_output=1, num_layers=1, **kwargs):
        """
        RETAIN model from http://arxiv.org/abs/1608.05745

        Parameters
        ----------
        num_features :        dimension of input layer
        dim_emb :           dimension of embedding layer
        dim num :           dimensions of numerical variables, which are concatenated after embedding
        dropout_emb :       dropout for embedding layer
        dim_alpha :         dimension for alpha RNN
        dim_beta :          dimension for beta RNN
        dropout_context :   dropout for context
        dim_output :        output dimensions
        """
        super(RETAIN, self).__init__()
        # TODO use embedding layer instead of linear layer, possibly faster? and could use sparse gradients to save
        #  memory
        self.embedding = nn.Sequential(
            nn.Linear(num_features, dim_emb, bias=False),
            nn.Dropout(p=dropout_emb)
        )
        nn.init.xavier_normal_(self.embedding[0].weight)

        self.rnn_alpha = nn.LSTM(input_size=dim_emb + dim_num, hidden_size=dim_alpha, num_layers=num_layers,
                                 batch_first=True,
                                 bidirectional=True)
        self.alpha_fc = nn.Linear(in_features=2 * dim_alpha, out_features=1)
        nn.init.xavier_normal_(self.alpha_fc.weight)
        self.alpha_fc.bias.data.zero_()

        self.rnn_beta = nn.LSTM(input_size=dim_emb + dim_num, hidden_size=dim_beta, num_layers=num_layers,
                                batch_first=True,
                                bidirectional=True)
        self.beta_fc = nn.Linear(in_features=2 * dim_beta, out_features=dim_emb + dim_num)
        nn.init.xavier_normal_(self.beta_fc.weight, gain=nn.init.calculate_gain('tanh'))
        self.beta_fc.bias.data.zero_()

        self.output = nn.Sequential(
            nn.Dropout(p=dropout_context),
            nn.Linear(in_features=dim_emb + dim_num, out_features=dim_output)
        )
        nn.init.xavier_normal_(self.output[1].weight)
        self.output[1].bias.data.zero_()

    def forward(self, input):
        x_cat = input[0]
        x_num = input[1]
        lengths = input[2].cpu()
        visits = input[3]  # timeId of visit per patient in batch
        # x_cat is a list of tensors. Length of list is batch size. Tensor is of shape seq_length X num_features
        # unit of sequence is visit
        x = pad_sequence(x_cat,
                         batch_first=True)  # pad to max length in batch and return a tensor of batch_size X max_seq_length X num_features
        padded_visits = pad_sequence(visits, batch_first=True)
        batch_size, max_len = x.size()[:2]

        # emb -> batch_size X max_len X dim_emb
        emb = self.embedding(x)

        # concatenate numerical and/or non-temporal variables to each visit embedding
        # TODO check if changing age to be non constant improves performance
        emb = torch.cat([emb, x_num[:, None, :].repeat(1, x.shape[1], 1), padded_visits[..., None]], dim=2)

        packed_input = pack_padded_sequence(emb, lengths, batch_first=True, enforce_sorted=False)

        g, _ = self.rnn_alpha(packed_input)

        # alpha_unpacked -> batch_size X max_len X dim_alpha
        alpha_unpacked, _ = pad_packed_sequence(g, batch_first=True)

        # mask -> batch_size X max_len X 1
        mask = torch.arange(max_len).expand(len(lengths), max_len) < lengths.unsqueeze(1)
        mask = mask.unsqueeze(-1).to(x.device)

        # e => batch_size X max_len X 1
        e = self.alpha_fc(alpha_unpacked)

        def masked_softmax(batch_tensor, mask):
            exp = torch.exp(batch_tensor)
            masked_exp = exp * mask
            sum_masked_exp = torch.sum(masked_exp, dim=1, keepdim=True)
            return masked_exp / sum_masked_exp

        # Alpha = batch_size X max_len X 1
        # alpha value for padded visits (zero) will be zero
        alpha = masked_softmax(e, mask)

        h, _ = self.rnn_beta(packed_input)

        # beta_unpacked -> batch_size X max_len X dim_beta
        beta_unpacked, _ = pad_packed_sequence(h, batch_first=True)

        # Beta -> batch_size X max_len X dim_emb
        # beta for padded visits will be zero-vectors
        beta = torch.tanh(self.beta_fc(beta_unpacked) * mask)

        # context -> batch_size X (1) X dim_emb (squeezed)
        # Context up to i-th visit context_i = sum(alpha_j * beta_j * emb_j)
        # Vectorized sum
        context = torch.bmm(torch.transpose(alpha, 1, 2), beta * emb).squeeze(1)

        # without applying non-linearity
        logit = self.output(context)

        return logit.squeeze()
