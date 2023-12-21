import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class GAT(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        num_layers,
        dropout,
        apply_batchnorm,
        att_heads=1,
        dataset=None,
    ):
        super(GAT, self).__init__()

        self.dropout = dropout
        self.apply_batchnorm = apply_batchnorm
        self.dataset = dataset

        self.convs = torch.nn.ModuleList()
        self.convs.append(GATConv(in_channels, hidden_channels, heads=att_heads))
        if self.apply_batchnorm:
            self.bns = torch.nn.ModuleList()
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels * att_heads))
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_channels * att_heads, hidden_channels, heads=att_heads))
            if self.apply_batchnorm:
                self.bns.append(torch.nn.BatchNorm1d(hidden_channels * att_heads))
        self.convs.append(GATConv(hidden_channels * att_heads, out_channels, concat=False, heads=1))

    def forward(self, x, adj_t):
        if self.dataset == "Cora":
            x = F.dropout(x, p=self.dropout, training=self.training)  # Cora
        for i, conv in enumerate(self.convs[:-1]):
            # x = F.dropout(x, p=self.dropout, training=self.training)
            x = conv(x, adj_t)
            if self.apply_batchnorm:
                x = self.bns[i](x)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x.log_softmax(dim=-1)
