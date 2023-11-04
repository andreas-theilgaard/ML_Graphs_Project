import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.utils import to_undirected, negative_sampling
from src.models.utils import set_seed, get_k_laplacian_eigenvectors
import networkx as nx
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader
from src.models.utils import create_path

from src.data.data_utils import get_link_data_split

# from ogb.linkproppred import Evaluator
from src.models.metrics import METRICS


def decoder(decode_type, beta, z_i, z_j):
    if decode_type == "dot":
        return torch.multiply(z_i, z_j).sum(dim=-1)
    elif decode_type == "dist":
        return beta - torch.norm(z_i - z_j, dim=-1)
    else:
        raise ValueError("Decoder method not yet implemented")


def initialize_embeddings(
    num_nodes, data, dataset=None, method="random", dim: int = 8, for_link=False, edge_split=None
):
    num_nodes = num_nodes

    if method == "random":
        return torch.normal(0, 1, size=(num_nodes, dim))
    elif method == "nodestatistics":
        G = nx.from_edgelist(data.edge_index.numpy().T)
        degrees = torch.tensor([deg for _, deg in G.degree()], dtype=torch.float).view(-1, 1)
        centrality = torch.tensor(list(nx.eigenvector_centrality(G).values()), dtype=torch.float).view(-1, 1)
        X = torch.cat([degrees, centrality], dim=1)
        extras = dim - 2
        if extras > 0:
            extra_features = torch.normal(0, 1, size=(num_nodes, extras))
        return torch.cat([X, extra_features], dim=1)
    elif method == "laplacian":
        return get_k_laplacian_eigenvectors(
            num_nodes=num_nodes, data=data, dataset=dataset, k=dim, for_link=for_link, edge_split=edge_split
        )
    else:
        raise ValueError(f"method: {method} not implemented yet")


class ShallowModel(nn.Module):
    def __init__(
        self,
        num_nodes,
        embedding_dim,
        beta=0.0,
        init_embeddings=None,
        decoder_type=None,
        device="cpu",
    ):
        super(ShallowModel, self).__init__()
        self.embeddings = nn.Embedding(num_nodes, embedding_dim, _weight=init_embeddings).to(device)
        self.decoder_type = decoder_type
        # make β as a trainable parameter
        self.beta = nn.Parameter(torch.tensor(beta)) if decoder_type in ["dist"] else None

    def forward(self, node_i, node_j):
        z_i = self.embeddings(node_i)
        z_j = self.embeddings(node_j)
        # return self.beta - torch.norm(z_i - z_j, dim=-1)
        return decoder(self.decoder_type, self.beta, z_i, z_j)


@torch.no_grad()
def test_link(model, split_edge, evaluator, batch_size, device):
    model.eval()
    # get training edges
    pos_train_edge = split_edge["train"]["edge"].to(device)
    # get validation edges
    pos_valid_edge = split_edge["valid"]["edge"].to(device)
    neg_valid_edge = split_edge["valid"]["edge_neg"].to(device)
    # get test edges
    pos_test_edge = split_edge["test"]["edge"].to(device)
    neg_test_edge = split_edge["test"]["edge_neg"].to(device)

    pos_train_preds = []
    for perm in DataLoader(range(pos_train_edge.size(0)), batch_size):
        edge = pos_train_edge[perm].t()
        pos_train_preds += [torch.sigmoid(model(edge[0], edge[1])).cpu()]
    pos_train_pred = torch.cat(pos_train_preds, dim=0)

    pos_valid_preds = []
    for perm in DataLoader(range(pos_valid_edge.size(0)), batch_size):
        edge = pos_valid_edge[perm].t()
        pos_valid_preds += [torch.sigmoid(model(edge[0], edge[1])).cpu()]
    pos_valid_pred = torch.cat(pos_valid_preds, dim=0)

    neg_valid_preds = []
    for perm in DataLoader(range(neg_valid_edge.size(0)), batch_size):
        edge = neg_valid_edge[perm].t()
        neg_valid_preds += [torch.sigmoid(model(edge[0], edge[1])).cpu()]
    neg_valid_pred = torch.cat(neg_valid_preds, dim=0)

    pos_test_preds = []
    for perm in DataLoader(range(pos_test_edge.size(0)), batch_size):
        edge = pos_test_edge[perm].t()
        pos_test_preds += [torch.sigmoid(model(edge[0], edge[1])).cpu()]
    pos_test_pred = torch.cat(pos_test_preds, dim=0)

    neg_test_preds = []
    for perm in DataLoader(range(neg_test_edge.size(0)), batch_size):
        edge = neg_test_edge[perm].t()
        neg_test_preds += [torch.sigmoid(model(edge[0], edge[1])).cpu()]
    neg_test_pred = torch.cat(neg_test_preds, dim=0)

    predictions = {
        "train": {"y_pred_pos": pos_train_pred, "y_pred_neg": neg_valid_pred},
        "val": {"y_pred_pos": pos_valid_pred, "y_pred_neg": neg_valid_pred},
        "test": {"y_pred_pos": pos_test_pred, "y_pred_neg": neg_test_pred},
    }
    results = evaluator.collect_metrics(predictions)
    return results


class ShallowTrainer:
    def __init__(self, config, training_args, save_path, log, Logger):
        self.config = config
        self.training_args = training_args
        self.save_path = save_path
        self.log = log
        self.Logger = Logger

    def metrics(self, pred, label, type="accuracy"):
        if type == "accuracy":
            yhat = (pred > 0.5).float()
            return torch.mean((yhat == label).float()).item()

    def get_negative_samples(self, edge_index, num_nodes, num_neg_samples):
        return negative_sampling(edge_index, num_neg_samples=num_neg_samples, num_nodes=num_nodes)

    def train(self, model, data, criterion, optimizer, num_nodes):
        model.train()
        optimizer.zero_grad()

        # Positive edges
        pos_edge_index = data.edge_index if self.config.dataset.task == "NodeClassification" else data
        pos_edge_index = pos_edge_index.to(self.config.device)
        pos_out = model(pos_edge_index[0], pos_edge_index[1])

        # Negative edges
        neg_edge_index = self.get_negative_samples(pos_edge_index, num_nodes, pos_edge_index.size(1))
        neg_edge_index.to(self.config.device)
        neg_out = model(neg_edge_index[0], neg_edge_index[1])

        # Combining positive and negative edges
        out = torch.cat([pos_out, neg_out], dim=0)
        y = torch.cat([torch.ones(pos_out.size(0)), torch.zeros(neg_out.size(0))], dim=0).to(
            self.config.device
        )
        acc = self.metrics(out, y, type="accuracy")
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        return loss.item(), acc

    def save_embeddings(self, model, seed):
        self.embedding_save_path = self.save_path + f"/embedding_seed={seed}.pth"
        torch.save(model.embeddings.weight.data.cpu(), self.embedding_save_path)

    def fit(self, dataset, seeds):
        for_link = False
        split_edge = None
        data = dataset[0]

        NUMBER_NODES = data.num_nodes

        if data.is_directed():
            data.edge_index = to_undirected(data.edge_index)

        # If task is link prediction only use training edges
        if self.config.dataset.task == "LinkPrediction":
            for_link = True
            if self.config.dataset.dataset_name in ["ogbl-collab", "ogbl-ppi"]:
                split_edge = dataset.get_edge_split()
            elif self.config.dataset.dataset_name in ["Cora", "Flickr"]:
                train_data, val_data, test_data = get_link_data_split(data)
                split_edge = {
                    "train": {
                        "edge": train_data.pos_edge_label_index.T,
                        "weight": torch.ones(train_data.pos_edge_label_index.shape[1]),
                    },
                    "valid": {
                        "edge": val_data.pos_edge_label_index.T,
                        "edge_neg": val_data.neg_edge_label_index.T,
                    },
                    "test": {
                        "edge": test_data.pos_edge_label_index.T,
                        "edge_neg": test_data.neg_edge_label_index.T,
                    },
                }
            if "edge_weight" not in data:
                split_edge["train"]

            data = (split_edge["train"]["edge"]).T
            # assert torch.unique(data.flatten()).size(0) == NUMBER_NODES

        for seed in seeds:
            set_seed(seed)

            init_embeddings = initialize_embeddings(
                num_nodes=NUMBER_NODES,
                data=data,
                dataset=dataset,
                method=self.training_args.init,
                dim=self.training_args.embedding_dim,
                for_link=for_link,
                edge_split=split_edge,
            )
            init_embeddings = init_embeddings.to(self.config.device)

            model = ShallowModel(
                num_nodes=NUMBER_NODES,
                embedding_dim=self.training_args.embedding_dim,
                beta=self.training_args.init_beta,
                init_embeddings=init_embeddings,
                decoder_type=self.training_args.decode_type,
                device=self.config.device,
            )
            model = model.to(self.config.device)

            optimizer = optim.Adam(list(model.parameters()), lr=self.training_args.lr)
            # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=10)
            prog_bar = tqdm(range(self.training_args.epochs))

            # applies sigmoid by default
            criterion = nn.BCEWithLogitsLoss()

            self.Logger.start_run()
            evaluator = METRICS(metrics_list=self.config.dataset.metrics, task=self.config.dataset.task)

            for epoch in prog_bar:
                loss, acc = self.train(
                    model=model, data=data, criterion=criterion, optimizer=optimizer, num_nodes=NUMBER_NODES
                )
                prog_bar.set_postfix({"loss": loss, "Train Acc.": acc})
                if for_link:
                    results = test_link(
                        model=model,
                        split_edge=split_edge,
                        evaluator=evaluator,
                        batch_size=self.training_args.batch_size,
                        device=self.config.device,
                    )
                    self.Logger.add_to_run(loss=loss, results=results)
                    if epoch % 10 == 0:
                        self.log.info(
                            f"Epoch {epoch+1}, Train {self.config.dataset.track_metric}: {results['train'][self.config.dataset.track_metric]}, Val {self.config.dataset.track_metric}: {results['val'][self.config.dataset.track_metric]}, Test {self.config.dataset.track_metric}: {results['test'][self.config.dataset.track_metric]}"
                        )

                # scheduler.step(loss)
                # log
                self.log.info(f"Epoch {epoch + 1}, Loss: {loss:.4f}, Acc: {acc:.4f}")

            self.Logger.end_run()

            self.save_embeddings(model, seed=seed)
            if "save_to_folder" in self.config:
                create_path(self.config.save_to_folder)
                additional_save_path = f"{self.config.save_to_folder}/{self.config.dataset.task}/{self.config.dataset.dataset_name}/{self.config.model_type}"
                create_path(f"{additional_save_path}")
                torch.save(
                    model.embeddings.weight.data.cpu(), additional_save_path + f"/embedding_seed_{seed}.pth"
                )

        if for_link:
            self.Logger.save_results(self.save_path + "/results.json")
            if "save_to_folder" in self.config:
                self.Logger.save_results(additional_save_path + f"/results.json")

        self.Logger.save_value({"loss": loss, "acc": acc})
