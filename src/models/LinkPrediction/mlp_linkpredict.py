import torch
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
import numpy as np

# from ogb.linkproppred import Evaluator
from src.models.logger import LoggerClass
from src.models.utils import set_seed
from src.models.utils import prepare_metric_cols
from torch_geometric.utils import negative_sampling
from src.models.utils import get_k_laplacian_eigenvectors
from torch_geometric.utils import to_undirected
from src.models.metrics import METRICS


class LinkPredictor(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int,
        dropout: float,
    ):
        super(LinkPredictor, self).__init__()
        self.dropout = dropout

        # Create linear and batchnorm layers
        self.linear = nn.ModuleList()
        self.linear.append(torch.nn.Linear(in_channels, hidden_channels))

        for _ in range(num_layers - 2):
            self.linear.append(torch.nn.Linear(hidden_channels, hidden_channels))
        self.linear.append(torch.nn.Linear(hidden_channels, out_channels))

    def reset_parameters(self):
        for lin in self.linear:
            lin.reset_parameters()

    def forward(self, x_i, x_j):
        x = x_i * x_j
        for i, lin_layer in enumerate(self.linear[:-1]):
            x = lin_layer(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.linear[-1](x)
        return torch.sigmoid(x)


class MLP_LinkPrediction:
    def __init__(
        self,
        device,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int,
        dropout: float,
        log,
        save_path,
        logger,
        config,
    ):
        self.device = device
        self.model = LinkPredictor(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.model.to(device)
        self.log = log
        self.save_path = save_path
        self.logger = logger
        self.config = config

    def train(self, X, split_edge, optimizer, batch_size):
        self.model.train()

        positive_edges = split_edge["train"]["edge"].to(self.device)
        total_loss, total_examples = 0, 0
        loss_fn = nn.BCELoss()

        for perm in DataLoader(range(positive_edges.size(0)), batch_size=batch_size, shuffle=True):
            optimizer.zero_grad()
            pos_edge = positive_edges[perm].t()

            # option 1
            positive_preds = self.model(X[pos_edge[0]], X[pos_edge[1]])
            pos_loss = -torch.log(positive_preds + 1e-15).mean()
            neg_edge = torch.randint(0, X.size(0), pos_edge.size(), dtype=torch.long, device=self.device)
            neg_preds = self.model(X[neg_edge[0]], X[neg_edge[1]])
            neg_loss = -torch.log(1 - neg_preds + 1e-15).mean()
            loss = pos_loss + neg_loss

            loss.backward()
            optimizer.step()

            num_examples = positive_preds.size(0)
            total_loss += loss.item() * num_examples
            total_examples += num_examples
        return total_loss / total_examples

    @torch.no_grad()
    def test(self, x, split_edge, evaluator, batch_size):
        self.model.eval()

        # get training edges
        pos_train_edge = split_edge["train"]["edge"].to(x.device)

        # get validation edges
        pos_valid_edge = split_edge["valid"]["edge"].to(x.device)
        neg_valid_edge = split_edge["valid"]["edge_neg"].to(x.device)

        # get test edges
        pos_test_edge = split_edge["test"]["edge"].to(x.device)
        neg_test_edge = split_edge["test"]["edge_neg"].to(x.device)

        pos_train_preds = []
        for perm in DataLoader(range(pos_train_edge.size(0)), batch_size):
            edge = pos_train_edge[perm].t()
            pos_train_preds += [self.model(x[edge[0]], x[edge[1]]).squeeze().cpu()]
        pos_train_pred = torch.cat(pos_train_preds, dim=0)

        pos_valid_preds = []
        for perm in DataLoader(range(pos_valid_edge.size(0)), batch_size):
            edge = pos_valid_edge[perm].t()
            pos_valid_preds += [self.model(x[edge[0]], x[edge[1]]).squeeze().cpu()]
        pos_valid_pred = torch.cat(pos_valid_preds, dim=0)

        neg_valid_preds = []
        for perm in DataLoader(range(neg_valid_edge.size(0)), batch_size):
            edge = neg_valid_edge[perm].t()
            neg_valid_preds += [self.model(x[edge[0]], x[edge[1]]).squeeze().cpu()]
        neg_valid_pred = torch.cat(neg_valid_preds, dim=0)

        pos_test_preds = []
        for perm in DataLoader(range(pos_test_edge.size(0)), batch_size):
            edge = pos_test_edge[perm].t()
            pos_test_preds += [self.model(x[edge[0]], x[edge[1]]).squeeze().cpu()]
        pos_test_pred = torch.cat(pos_test_preds, dim=0)

        neg_test_preds = []
        for perm in DataLoader(range(neg_test_edge.size(0)), batch_size):
            edge = neg_test_edge[perm].t()
            neg_test_preds += [self.model(x[edge[0]], x[edge[1]]).squeeze().cpu()]
        neg_test_pred = torch.cat(neg_test_preds, dim=0)

        predictions = {
            "train": {"y_pred_pos": pos_train_pred, "y_pred_neg": neg_valid_pred},
            "val": {"y_pred_pos": pos_valid_pred, "y_pred_neg": neg_valid_pred},
            "test": {"y_pred_pos": pos_test_pred, "y_pred_neg": neg_test_pred},
        }
        results = evaluator.collect_metrics(predictions)
        return results

    def fit(self, X, split_edge, lr, batch_size, epochs, evaluator):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        prog_bar = tqdm(range(epochs))

        for i, epoch in enumerate(prog_bar):
            loss = self.train(X, split_edge, optimizer, batch_size)
            results = self.test(X, split_edge=split_edge, evaluator=evaluator, batch_size=batch_size)
            prog_bar.set_postfix(
                {
                    "Train Loss": loss,
                    f"Train {self.config.dataset.track_metric}": results["train"][
                        self.config.dataset.track_metric
                    ],
                    f"Val {self.config.dataset.track_metric}": results["val"][
                        self.config.dataset.track_metric
                    ],
                    f"Test {self.config.dataset.track_metric}": results["test"][
                        self.config.dataset.track_metric
                    ],
                }
            )

            self.logger.add_to_run(loss=loss, results=results)

        self.logger.save_value(
            {
                "loss": loss,
                f"Test {self.config.dataset.track_metric}": results["test"][self.config.dataset.track_metric],
            }
        )


def mlp_LinkPrediction(dataset, config, training_args, log, save_path, seeds, Logger):
    data = dataset[0]
    if config.dataset.dataset_name in ["ogbl-collab"]:
        split_edge = dataset.get_edge_split()

    if (
        config.dataset[config.model_type].saved_embeddings
        and config.dataset[config.model_type].using_features
    ):
        embedding = torch.load(config.dataset[config.model_type].saved_embeddings, map_location=config.device)
        x = torch.cat([data.x, embedding], dim=-1)
    if (
        config.dataset[config.model_type].saved_embeddings
        and not config.dataset[config.model_type].using_features
    ):
        embedding = torch.load(config.dataset[config.model_type].saved_embeddings, map_location=config.device)
        x = embedding
    if (
        not config.dataset[config.model_type].saved_embeddings
        and config.dataset[config.model_type].using_features
    ):
        x = data.x
    if config.dataset[config.model_type].use_spectral:
        if data.is_directed():
            data.edge_index = to_undirected(data.edge_index)
        x = get_k_laplacian_eigenvectors(
            data=data, dataset=dataset, k=config.dataset[config.model_type].K, is_undirected=True
        )

    X = x.to(config.device)

    # evaluator = Evaluator(name=config.dataset.dataset_name)
    evaluator = METRICS(metrics_list=config.dataset.metrics, task=config.task)

    for seed in seeds:
        set_seed(seed=seed)
        Logger.start_run()

        model = MLP_LinkPrediction(
            device=config.device,
            in_channels=X.shape[-1],
            hidden_channels=training_args.hidden_channels,
            out_channels=1,
            num_layers=training_args.num_layers,
            dropout=training_args.dropout,
            log=log,
            save_path=save_path,
            logger=Logger,
            config=config,
        )
        model.fit(
            X=X,
            split_edge=split_edge,
            lr=training_args.lr,
            batch_size=training_args.batch_size,
            epochs=training_args.epochs,
            evaluator=evaluator,
        )
        Logger.end_run()

    Logger.save_results(save_path + "/results.json")
    Logger.get_statistics(metrics=prepare_metric_cols(config.dataset.metrics))
