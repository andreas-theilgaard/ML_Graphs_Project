import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.datasets import OGB_MAG
from torch_geometric.utils import to_undirected, negative_sampling
from src.data.get_data import DataLoader


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


# Initialize Dataset and Model
dataset = DataLoader(
    task_type="NodeClassification",
    dataset="ogbn-arxiv",
    model_name="Shallow",
).get_data()
data = dataset[0]
data.edge_index = to_undirected(data.edge_index)
model = EmbeddingModel(dataset.num_nodes, 128)
optimizer = optim.Adam(list(model.parameters()), lr=0.1)
criterion = nn.BCEWithLogitsLoss()


def get_negative_samples(edge_index, num_nodes, num_neg_samples):
    return negative_sampling(
        edge_index, num_neg_samples=num_neg_samples, num_nodes=num_nodes
    )


def metrics(pred, label, type="accuracy"):
    if type == "accuracy":
        yhat = (pred > 0.5).float()
        return torch.mean((yhat == label).float()).item()


def train():
    model.train()
    optimizer.zero_grad()

    # Positive edges
    pos_edge_index = data.edge_index
    pos_out = model(pos_edge_index[0], pos_edge_index[1])

    # Negative edges
    neg_edge_index = get_negative_samples(
        pos_edge_index, dataset.num_nodes, pos_edge_index.size(1)
    )
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
        loss, acc = train()
        print(f"Epoch {epoch + 1}, Loss: {loss:.4f},Acc{acc}")
    torch.save(model.embeddings.weight.data.cpu(), "shallow_emb_new.pth")
