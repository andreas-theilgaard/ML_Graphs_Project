import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.utils import to_undirected, negative_sampling
from src.data.get_data import DataLoader

import random
import numpy as np

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class EmbeddingModel(nn.Module):
    def __init__(self, num_nodes, embedding_dim):
        super(EmbeddingModel, self).__init__()
        self.embeddings = nn.Embedding(
            num_nodes, embedding_dim, _weight=torch.normal(0, 1, size=(num_nodes, embedding_dim))
        )
        self.beta = nn.Parameter(torch.tensor(0.0))  # This sets up β as a trainable parameter

    def forward(self, node_i, node_j):
        z_i = self.embeddings(node_i)
        z_j = self.embeddings(node_j)
        return self.beta - torch.norm(z_i - z_j, dim=-1)


# Initialize Dataset and Model
dataset = DataLoader(
    task_type="LinkPrediction",
    dataset="ogbl-collab",
    model_name="Shallow",
).get_data()
data = dataset[0]
if data.is_directed():
    data.edge_index = to_undirected(data.edge_index)


if 1 == 1:
    split_edge = dataset.get_edge_split()
    data = (split_edge["train"]["edge"]).T
    assert torch.unique(data.flatten()).size(0) == dataset.num_nodes

model = EmbeddingModel(dataset.num_nodes, 128)
optimizer = optim.Adam(list(model.parameters()), lr=0.1)
criterion = nn.BCEWithLogitsLoss()


def get_negative_samples(edge_index, num_nodes, num_neg_samples):
    return negative_sampling(edge_index, num_neg_samples=num_neg_samples, num_nodes=num_nodes)


def metrics(pred, label, type="accuracy"):
    if type == "accuracy":
        yhat = (pred > 0.5).float()
        return torch.mean((yhat == label).float()).item()


def train(model, data, dataset, criterion, optimizer):
    model.train()
    optimizer.zero_grad()

    # Positive edges
    pos_edge_index = data  # data.edge_index
    pos_out = model(pos_edge_index[0], pos_edge_index[1])

    # Negative edges
    neg_edge_index = get_negative_samples(pos_edge_index, dataset.num_nodes, pos_edge_index.size(1))
    neg_out = model(neg_edge_index[0], neg_edge_index[1])

    # Combining positive and negative edges
    out = torch.cat([pos_out, neg_out], dim=0)
    y = torch.cat([torch.ones(pos_out.size(0)), torch.zeros(neg_out.size(0))], dim=0)
    acc = metrics(out, y, type="accuracy")
    loss = criterion(out, y)
    loss.backward()
    optimizer.step()

    return loss.item(), acc


if __name__ == "__main__":
    for epoch in range(100):
        loss, acc = train(model=model, dataset=dataset, data=data, criterion=criterion, optimizer=optimizer)
        print(f"Epoch {epoch + 1}, Loss: {loss:.4f},Acc{acc}")
    torch.save(model.embeddings.weight.data.cpu(), "Shallow_link.pth")
