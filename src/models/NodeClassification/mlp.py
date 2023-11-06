import torch
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm
import numpy as np


class NodeClassifier(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int,
        dropout: float,
        apply_batchnorm: bool,
    ):
        super(NodeClassifier, self).__init__()
        self.dropout = dropout
        self.apply_batchnorm = apply_batchnorm

        # Create linear and batchnorm layers
        self.linear = nn.ModuleList()
        self.linear.append(torch.nn.Linear(in_channels, hidden_channels))
        if self.apply_batchnorm:
            self.batch_norm = torch.nn.ModuleList()
            self.batch_norm.append(torch.nn.BatchNorm1d(hidden_channels))

        for _ in range(num_layers - 2):
            self.linear.append(torch.nn.Linear(hidden_channels, hidden_channels))
            if self.apply_batchnorm:
                self.batch_norm.append(torch.nn.BatchNorm1d(hidden_channels))
        self.linear.append(torch.nn.Linear(hidden_channels, out_channels))

    def reset_parameters(self):
        for lin in self.linear:
            lin.reset_parameters()
        for bn in self.batch_norm:
            bn.reset_parameters()

    def forward(self, x):
        for i, lin_layer in enumerate(self.linear[:-1]):
            x = lin_layer(x)
            if self.apply_batchnorm:
                x = self.batch_norm[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.linear[-1](x)
        return torch.log_softmax(x, dim=-1)


class MLP_model:
    def __init__(
        self,
        device,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int,
        dropout: float,
        apply_batchnorm: bool,
        log,
        logger,
        config,
    ):
        self.device = device
        self.model = NodeClassifier(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            num_layers=num_layers,
            dropout=dropout,
            apply_batchnorm=apply_batchnorm,
        )
        self.model.to(device)
        self.log = log
        self.logger = logger
        self.config = config

    def train(self, X, y, train_idx, optimizer):
        self.model.train()
        optimizer.zero_grad()
        out = self.model(X[train_idx])
        loss = F.nll_loss(out, y.squeeze(1)[train_idx])
        loss.backward()
        optimizer.step()
        return loss.item()

    @torch.no_grad()
    def test(self, X, y, split_idx, evaluator):
        self.model.eval()
        with torch.no_grad():
            out = self.model(X)
            y_hat = out.argmax(dim=-1, keepdim=True)
            if self.config.dataset.dataset_name != "ogbn-mag":
                predictions = {
                    "train": {"y_true": y[split_idx["train"]], "y_hat": y_hat[split_idx["train"]]},
                    "val": {"y_true": y[split_idx["valid"]], "y_hat": y_hat[split_idx["valid"]]},
                    "test": {"y_true": y[split_idx["test"]], "y_hat": y_hat[split_idx["test"]]},
                }
            else:
                y_true_train = y[split_idx["train"]["paper"]]
                y_true_valid = y[split_idx["valid"]["paper"]]
                y_true_test = y[split_idx["test"]["paper"]]
                predictions = {
                    "train": {"y_true": y_true_train, "y_hat": y_hat[split_idx["train"]["paper"]]},
                    "val": {"y_true": y_true_valid, "y_hat": y_hat[split_idx["valid"]["paper"]]},
                    "test": {"y_true": y_true_test, "y_hat": y_hat[split_idx["test"]["paper"]]},
                }
            results = evaluator.collect_metrics(predictions)
            return results

    def fit(self, X, y, epochs: int, split_idx, evaluator, lr, weight_decay):
        # self.model.reset_parameters()
        train_idx = (
            split_idx["train"].to(self.device)
            if self.config.dataset.dataset_name != "ogbn-mag"
            else split_idx["train"]["paper"]
        )
        prog_bar = tqdm(range(epochs))
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        for i, epoch in enumerate(prog_bar):
            loss = self.train(X=X, y=y, train_idx=train_idx, optimizer=optimizer)
            result = self.test(X=X, y=y, split_idx=split_idx, evaluator=evaluator)
            prog_bar.set_postfix(
                {
                    "Train Loss": loss,
                    f"Train {self.config.dataset.track_metric}": result["train"][
                        self.config.dataset.track_metric
                    ],
                    f"Val {self.config.dataset.track_metric}": result["val"][
                        self.config.dataset.track_metric
                    ],
                    f"Test {self.config.dataset.track_metric}": result["test"][
                        self.config.dataset.track_metric
                    ],
                }
            )

            self.logger.add_to_run(loss=loss, results=result)

        self.logger.save_value(
            {
                "loss": loss,
                f"Test {self.config.dataset.track_metric}": result["test"][self.config.dataset.track_metric],
            }
        )
