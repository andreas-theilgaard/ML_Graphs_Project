import torch
from src.models.NodeClassification.GCN import GCN
from src.models.NodeClassification.GraphSage import SAGE
import torch.nn.functional as F
from ogb.nodeproppred import Evaluator
from tqdm import tqdm
import numpy as np
from src.models.logger import LoggerClass
from src.models.utils import set_seed, prepare_metric_cols


class GNN:
    def __init__(
        self,
        GNN_type,
        in_channels,
        hidden_channels,
        out_channels,
        num_layers,
        dropout,
    ):
        self.GNN_type = GNN_type
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

        train_acc = evaluator.eval({"y_true": y_true_train, "y_pred": y_pred[split_idx["train"]]})[
            evaluator.eval_metric
        ]

        valid_acc = evaluator.eval({"y_true": y_true_valid, "y_pred": y_pred[split_idx["valid"]]})[
            evaluator.eval_metric
        ]

        test_acc = evaluator.eval({"y_true": y_true_test, "y_pred": y_pred[split_idx["test"]]})[
            evaluator.eval_metric
        ]

        return train_acc, valid_acc, test_acc


def GNN_trainer(dataset, config, training_args, log, save_path, seeds, Logger):
    data = dataset[0]
    evaluator = Evaluator(name=config.dataset.dataset_name)
    data.adj_t = data.adj_t.to_symmetric()
    split_idx = dataset.get_idx_split()
    # model.reset_parameters()
    train_idx = split_idx["train"].to(config.device)

    for seed in seeds:
        set_seed(seed)
        Logger.start_run()
        GNN_object = GNN(
            GNN_type=config.dataset[config.model_type].model,
            in_channels=data.num_features,
            hidden_channels=training_args.hidden_channels,
            out_channels=dataset.num_classes,
            dropout=training_args.dropout,
            num_layers=training_args.num_layers,
        )
        model = GNN_object.get_gnn_model()
        model.to(config.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=training_args.lr)
        prog_bar = tqdm(range(training_args.epochs))

        for i, epoch in enumerate(prog_bar):
            loss = GNN_object.train(model, data, train_idx, optimizer)
            result = GNN_object.test(model, data, split_idx, evaluator)
            train_acc, valid_acc, test_acc = result
            prog_bar.set_postfix(
                {
                    "Train Loss": loss,
                    "Train Acc.": train_acc,
                    "Val Acc.": valid_acc,
                    "Test Acc.": test_acc,
                }
            )
            if epoch % 10 == 0:
                log.info(f"Train: {100*train_acc}, Valid: {100*valid_acc}, Test: {100*test_acc}")
            Logger.add_to_run(np.array([loss, train_acc, valid_acc, test_acc]))

    Logger.end_run()
    Logger.save_value({"loss": loss, "acc": test_acc})
    Logger.save_results(save_path + "/results.json")
    Logger.get_statistics(
        metrics=prepare_metric_cols(config.dataset.metrics),
        directions=["-", "+", "+", "+"],
    )

    model.train()
    log.info(f"saved model at {save_path+'/model.pth'}")
    torch.save(model.state_dict(), save_path + "/model.pth")
