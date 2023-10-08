import torch
from src.models.GNN.GCN import GCN
from src.models.GNN.GraphSage import SAGE
import torch.nn.functional as F


class GNN:
    def __init__(
        self,
        GNN_type,
        task,
        in_channels,
        hidden_channels,
        out_channels,
        num_layers,
        dropout,
    ):
        self.GNN_type = GNN_type
        self.task = task
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.dropout = dropout

    def get_gnn_model(self):
        if self.GNN_type == "GCN":
            model = GCN(
                in_channels=self.in_channels,
                hidden_channels=self.hidden_channels,
                out_channels=self.out_channels,
                num_layers=self.num_layers,
                dropout=self.dropout,
            )
        elif self.GNN_type == "GraphSage":
            model = SAGE(
                in_channels=self.in_channels,
                hidden_channels=self.hidden_channels,
                out_channels=self.out_channels,
                num_layers=self.num_layers,
                dropout=self.dropout,
            )
        return model

    def train(self, model, data, train_idx, optimizer):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.adj_t)[train_idx]
        loss = F.nll_loss(out, data.y.squeeze(1)[train_idx])

        loss.backward()
        optimizer.step()

        return loss.item()

    @torch.no_grad()
    def test(self, model, data, split_idx, evaluator):
        model.eval()

        out = model(data.x, data.adj_t)  # data.edge_index
        y_pred = out.argmax(dim=-1, keepdim=True)

        y_true_train = data.y[split_idx["train"]]
        y_true_valid = data.y[split_idx["valid"]]
        y_true_test = data.y[split_idx["test"]]

        train_acc = evaluator.eval(
            {"y_true": y_true_train, "y_pred": y_pred[split_idx["train"]]}
        )["acc"]

        valid_acc = evaluator.eval(
            {"y_true": y_true_valid, "y_pred": y_pred[split_idx["valid"]]}
        )["acc"]

        test_acc = evaluator.eval(
            {"y_true": y_true_test, "y_pred": y_pred[split_idx["test"]]}
        )["acc"]

        return train_acc, valid_acc, test_acc
