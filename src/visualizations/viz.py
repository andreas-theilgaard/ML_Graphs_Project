from src.data.get_data import DataLoader
from sklearn.manifold import TSNE
import torch
from src.models.NodeClassification.GraphSage import SAGE
from tqdm import tqdm
import matplotlib

# matplotlib.use('pgf')
# import tikzplotlib as tpl
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams.update({"font.size": 14})

SEED = 42
SAVE = True

mapper = {
    "Cora": {"num_layers": 2, "dropout": 0.5, "batchnorm": False},
    "CiteSeer": {"num_layers": 2, "dropout": 0.5, "batchnorm": False},
    "PubMed": {"num_layers": 2, "dropout": 0.5, "batchnorm": False},
    "Flickr": {"num_layers": 2, "dropout": 0.5, "batchnorm": False},
    "Twitch": {"num_layers": 2, "dropout": 0.5, "batchnorm": False},
    "ogbn-arxiv": {"num_layers": 3, "dropout": 0.5, "batchnorm": True},
}

###########################################################################################################
###########################################################################################################
###########################################################################################################

if __name__ == "__main__":
    for DATASET in tqdm(["Cora", "PubMed", "CiteSeer"]):
        for DIM in [2]:
            #####################################
            LATEX_TXT = "$\\textbf{d}$"
            fig, axs = plt.subplots(2, 2, figsize=(10, 8))
            FIRST_PART = f"Visualization of node embeddings, {DATASET} "
            fig.suptitle(FIRST_PART + "$\\mathbf{d=}$" + str(DIM), weight="bold", fontsize=16)

            def embed_visualizer(Z, dataset, i, j, model_name):
                if DIM == 2 and model_name != "GraphSage":
                    Z_tsne = Z
                else:
                    Z_tsne = TSNE(n_components=2).fit_transform(Z.detach().numpy())
                axs[i, j].scatter(Z_tsne[:, 0], Z_tsne[:, 1], c=dataset.data.y, cmap="Set2", s=20)
                axs[i, j].set_xticks([])
                axs[i, j].set_yticks([])
                axs[i, j].set_title(f"{model_name}", fontsize=12)

            ##############
            dataset = DataLoader(
                model_name="Shallow", task_type="NodeClassification", dataset=DATASET
            ).get_data()

            PATH = f"results/NodeClassification/{DATASET}/{DIM}"

            SHALLOW = torch.load(f"{PATH}/Shallow/shallow_embedding_seed_{SEED}.pth")
            Node2Vec = torch.load(f"{PATH}/Node2Vec/Node2Vec_embedding.pth")
            GraphSage_checkpoint = f"{PATH}/GNN/models/GraphSage_False_model_{SEED}_False.pth"
            GraphSage_DIRECT = torch.load(
                f"{PATH}/GNN_DIRECT/models/GNN_DIRECT_GraphSage_False_model_{SEED}_False.pth"
            )
            GraphSage = SAGE(
                in_channels=dataset.num_features,
                hidden_channels=DIM,
                out_channels=dataset.num_classes,
                num_layers=mapper[DATASET]["num_layers"],
                dropout=mapper[DATASET]["dropout"],
                apply_batchnorm=mapper[DATASET]["batchnorm"],
            )
            GraphSage.load_state_dict(torch.load(GraphSage_checkpoint, map_location="cpu"))
            GraphSage.eval()
            with torch.no_grad():
                embeddings = GraphSage(dataset.data.x, dataset.data.edge_index)

            embed_visualizer(Z=Node2Vec, dataset=dataset, model_name="Node2Vec", i=0, j=0)
            embed_visualizer(Z=SHALLOW, dataset=dataset, model_name="Shallow", i=0, j=1)
            embed_visualizer(Z=GraphSage_DIRECT, dataset=dataset, model_name="GraphSage Unsup.", i=1, j=0)
            embed_visualizer(Z=embeddings, dataset=dataset, model_name="GraphSage Sup.", i=1, j=1)
            plt.tight_layout()
            if SAVE:
                plt.savefig(f"figs/{DATASET}_dim={DIM}_new.pdf", dpi=800)
            else:
                plt.show()
