import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.datasets import Planetoid
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
import torch.nn as nn

# Load the ogbn-arxiv dataset
dataset = PygNodePropPredDataset(name="ogbn-arxiv", root="data")
data = dataset[0]
data.y = data.y.view(-1)
split_idx = dataset.get_idx_split()
train_idx, valid_idx, test_idx = (
    split_idx["train"],
    split_idx["valid"],
    split_idx["test"],
)


class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout):
        super(SAGE, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        # self.bns = torch.nn.ModuleList()
        # self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            # self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        # self.convs.append(SAGEConv(hidden_channels, out_channels))

        self.dropout = dropout

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x


class Shallow(torch.nn.Module):
    def __init__(self, num_nodes, embedding_dim):
        super(Shallow, self).__init__()
        self.embeddings = nn.Embedding(
            num_nodes, embedding_dim, _weight=torch.rand((num_nodes, embedding_dim))
        )
        self.beta = nn.Parameter(
            torch.tensor(0.0)
        )  # This sets up β as a trainable parameter

    def forward(self, node_i, node_j):
        z_i = self.embeddings(node_i)
        z_j = self.embeddings(node_j)
        return self.beta - torch.norm(z_i - z_j, dim=-1)


class CombinedModel(torch.nn.Module):
    def __init__(
        self, in_channels, hidden_channels, out_channels, num_nodes, embedding_dim
    ):
        super(CombinedModel, self).__init__()
        self.graphsage = SAGE(
            in_channels, hidden_channels, out_channels, num_layers=2, dropout=0.1
        )
        self.shallow = Shallow(num_nodes, embedding_dim)

    def forward(self, x, edge_index):

        return self.shallow(edge_index[0], edge_index[1]) + self.graphsage(
            x, edge_index
        )


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CombinedModel(
    dataset.num_features, 128, dataset.num_classes, data.num_nodes, 128
).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


def train():
    model.train()
    optimizer.zero_grad()
    out = model(data.x.to(device), data.edge_index.to(device))
    loss = F.cross_entropy(out[train_idx], data.y[train_idx].to(device))
    loss.backward()
    optimizer.step()
    return loss.item()


for epoch in range(100):
    loss = train()
    print(f"Epoch: {epoch+1}, Loss: {loss:.4f}")
