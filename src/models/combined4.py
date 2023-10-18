import torch
import torch.nn as nn
from src.data.get_data import DataLoader
from torch_geometric.utils import to_undirected
from torch_geometric.nn import SAGEConv
import torch.nn.functional as F


##### Shallow Embedding Model #####
class ShallowModel(nn.Module):
    def __init__(self, num_nodes, embedding_dim):
        super(ShallowModel, self).__init__()
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


##### Get and prepare data #####
dataset = DataLoader(
    task_type="LinkPrediction",
    dataset="ogbl-collab",
    model_name="Shallow",
).get_data()
data = dataset[0]
if data.is_directed():
    data.edge_index = to_undirected(data.edge_index)

##### Setup models #####
shallow = ShallowModel(dataset.num_nodes, 8)
deep = SAGE(
    in_channels=data.x, hidden_channels=256, out_channels=128, num_layers=3, dropout=0.0
)

optimizer = torch.optim.Adam(
    list(shallow.parameters()) + list(deep.parameters()), lr=0.01
)
criterion = nn.BCEWithLogitsLoss()

for i in range(500):
    optimizer.zero_grad()

    # Shallow traning

    similarity_shallow = shallow(edge_i, edge_j)
    # deep traning
    deep_output = ...

    # extract deep embedding
    similarity_deep = deep_output[edge_index[0]].T @ deep_output[edge_index[0]]

    loss = similarity_shallow + λ * similarity_deep

    criterion(loss)

    loss.backward()
    optimizer.step()

# okay, but how do we use for downstream?
