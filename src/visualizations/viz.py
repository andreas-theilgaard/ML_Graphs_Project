from src.data.get_data import DataLoader
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import torch
from src.models.NodeClassification.GraphSage import SAGE

DATASET = "Cora"
DIM = 16
SEED = 42


def embed_visualizer(Z, dataset, DATASET, DIM, model_name):
    Z_tsne = TSNE(n_components=2).fit_transform(Z.detach().numpy())

    plt.figure(figsize=(10, 6))
    plt.scatter(Z_tsne[:, 0], Z_tsne[:, 1], c=dataset.data.y, cmap="Set2", s=20)
    # plt.legend(*scatter.legend_elements(), title="Classes")
    plt.title(f"t-SNE visualization of Node Embeddings, {DATASET}, d={DIM}, {model_name}")
    plt.xticks([])
    plt.yticks([])
    plt.show()


##############
dataset = DataLoader(model_name="Shallow", task_type="NodeClassification", dataset=DATASET).get_data()

PATH = f"results/NodeClassification/{DATASET}/{DIM}"

SHALLOW = torch.load(f"{PATH}/Shallow/shallow_embedding_seed_{SEED}.pth")
Node2Vec = torch.load(f"{PATH}/Node2Vec/Node2Vec_embedding.pth")
GraphSage_checkpoint = f"{PATH}/GNN/models/GraphSage_False_model_{SEED}_False.pth"
GraphSage_DIRECT = torch.load(f"{PATH}/GNN_DIRECT/models/GNN_DIRECT_GraphSage_False_model_{SEED}_False.pth")
GraphSage = SAGE(
    in_channels=dataset.num_features,
    hidden_channels=DIM,
    out_channels=dataset.num_classes,
    num_layers=2,
    dropout=0.5,
    apply_batchnorm=False,
)
GraphSage.load_state_dict(torch.load(GraphSage_checkpoint))
GraphSage.eval()
with torch.no_grad():
    embeddings = GraphSage(dataset.data.x, dataset.data.edge_index)


embed_visualizer(Z=embeddings, dataset=dataset, DATASET=DATASET, DIM=DIM, model_name="GraphSage")
embed_visualizer(Z=SHALLOW, dataset=dataset, DATASET=DATASET, DIM=DIM, model_name="Shallow")
embed_visualizer(Z=Node2Vec, dataset=dataset, DATASET=DATASET, DIM=DIM, model_name="Node2Vec")
embed_visualizer(Z=GraphSage_DIRECT, dataset=dataset, DATASET=DATASET, DIM=DIM, model_name="GraphSage Direct")
