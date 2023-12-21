from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import torch
from src.data.get_data import DataLoader
from src.models.NodeClassification.GraphSage import SAGE


def embed_visualizer(Z, dataset):
    Z_tsne = TSNE(n_components=2).fit_transform(Z.detach().numpy())

    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(Z_tsne[:, 0], Z_tsne[:, 1], c=dataset.data.y, cmap="Set2", s=20)
    plt.legend(*scatter.legend_elements(), title="Classes")
    plt.title("t-SNE visualization of Node Embeddings")
    plt.show()


dataset = DataLoader("Shallow", task_type="NodeClassification", dataset="Cora").get_data()

### Node2Vec
Node2Vec = torch.load("outputs/NodeClassification/Cora/Node2Vec/2023-11-13/02-24-39/embedding.pth")

### GraphSage supervised
GraphSage = SAGE(
    in_channels=1433, hidden_channels=16, out_channels=7, num_layers=2, dropout=0.5, apply_batchnorm=False
)
Sage_check = torch.load("results/NodeClassification/Cora/GNN/models/GraphSage_False_model_18.pth")
GraphSage.load_state_dict(Sage_check)
GraphSage.eval()
with torch.no_grad():
    embeddings = GraphSage(dataset.data.x, dataset.data.edge_index)

### Shallow
Shallow = torch.load("results/NodeClassification/Cora/Shallow/shallow_embedding_seed_42.pth")
embed_visualizer(Z=Node2Vec, dataset=dataset)

embed_visualizer(Z=embeddings, dataset=dataset)

embed_visualizer(Z=Shallow, dataset=dataset)
