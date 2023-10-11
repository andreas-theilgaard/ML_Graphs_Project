from src.models.GNN.GraphSage import SAGE
from src.data.get_data import DataLoader
from torch_geometric.utils import to_undirected, negative_sampling
import torch
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F
from ogb.nodeproppred import Evaluator
import random
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.nn import SAGEConv

seed = 42
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout):
        super(SAGE, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
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
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x  # x.log_softmax(dim=-1)


class EmbeddingModel(nn.Module):
    def __init__(self, num_nodes, embedding_dim):
        super(EmbeddingModel, self).__init__()
        self.embeddings = nn.Embedding(num_nodes, embedding_dim)
        self.beta = nn.Parameter(
            torch.tensor(0.0)
        )  # This sets up β as a trainable parameter

    def forward(self, node_i, node_j):
        z_i = self.embeddings(node_i)
        z_j = self.embeddings(node_j)
        return self.beta - torch.norm(z_i - z_j, dim=-1)


# load data
dataset = DataLoader(
    task_type="NodeClassification",
    dataset="ogbn-arxiv",
    model_name="Shallow",
).get_data()

data = dataset[0]

data.edge_index = to_undirected(data.edge_index)
split_idx = dataset.get_idx_split()
train_idx = split_idx["train"]

GNN_model = SAGE(
    in_channels=data.num_features,
    hidden_channels=256,
    out_channels=dataset.num_classes,
    num_layers=3,
    dropout=0.5,
)

evaluator = Evaluator(name="ogbn-arxiv")
prog_bar = tqdm(range(100))

# initilize EmbeddingModel
shallow_model = EmbeddingModel(num_nodes=dataset.num_nodes, embedding_dim=128)


def get_negative_samples(edge_index, num_nodes, num_neg_samples):
    return negative_sampling(
        edge_index, num_neg_samples=num_neg_samples, num_nodes=num_nodes
    )


# get positive edges
pos_train_edges = ((data.edge_index)[:, [list(train_idx.numpy())]]).squeeze(1)

criterion_shallow = nn.BCEWithLogitsLoss()
λ = nn.Parameter(torch.tensor(1.0))

optimizer = torch.optim.Adam(
    (list(GNN_model.parameters()) + list(shallow_model.parameters()) + [λ]), lr=0.01
)


for i, epoch in enumerate(prog_bar):
    GNN_model.train()
    shallow_model.train()
    optimizer.zero_grad()

    # training Embedding Model
    pos_out = shallow_model(pos_train_edges[0], pos_train_edges[1])
    neg_train_edges = get_negative_samples(
        edge_index=pos_train_edges,
        num_neg_samples=pos_train_edges.shape[1],
        num_nodes=pos_train_edges.shape[1],
    )
    neg_out = shallow_model(neg_train_edges[0], neg_train_edges[1])
    embed_out = torch.cat([pos_out, neg_out], dim=0)
    y = torch.cat([torch.ones(pos_out.size(0)), torch.zeros(neg_out.size(0))], dim=0)
    loss_shallow = criterion_shallow(embed_out, y)

    # training GraphSage
    out = GNN_model(data.x, data.edge_index)[train_idx]
    loss_depp = F.nll_loss(out.log_softmax(dim=-1), data.y.squeeze(1)[train_idx])
    # what should the steps be for traning the combined model?

    loss = loss_shallow + λ * loss_shallow
    print(loss.item(), λ.item())

    loss.backward()
    optimizer.step()
