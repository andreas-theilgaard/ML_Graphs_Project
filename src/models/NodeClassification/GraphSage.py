import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv


class SAGE(torch.nn.Module):
    def __init__(
        self, in_channels, hidden_channels, out_channels, num_layers, dropout, apply_batchnorm: bool = False
    ):
        super(SAGE, self).__init__()

        self.apply_batchnorm = apply_batchnorm
        self.dropout = dropout

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        if self.apply_batchnorm:
            self.bns = torch.nn.ModuleList()
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            if self.apply_batchnorm:
                self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, adj_t):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t)
            if self.apply_batchnorm:
                x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x.log_softmax(
            dim=-1
        )  # make such that not return log probabilites but x instead, also make such that no classification head is returned
