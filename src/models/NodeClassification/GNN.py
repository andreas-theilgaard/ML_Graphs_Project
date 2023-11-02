import torch
from src.models.NodeClassification.GCN import GCN
from src.models.NodeClassification.GraphSage import SAGE
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from src.models.utils import set_seed, prepare_metric_cols
from src.models.metrics import METRICS


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

        predictions = {
            "train": {"y_true": y_true_train, "y_hat": y_pred[split_idx["train"]]},
            "val": {"y_true": y_true_valid, "y_hat": y_pred[split_idx["valid"]]},
            "test": {"y_true": y_true_test, "y_hat": y_pred[split_idx["test"]]},
        }
        results = evaluator.collect_metrics(predictions)
        return results


def GNN_trainer(dataset, config, training_args, log, save_path, seeds, Logger):
    data = dataset[0]
    evaluator = METRICS(metrics_list=config.dataset.metrics, task=config.task)
    data.adj_t = data.adj_t.to_symmetric()
    data = data.to(config.device)
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
        model = model.to(config.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=training_args.lr)
        prog_bar = tqdm(range(training_args.epochs))

        for i, epoch in enumerate(prog_bar):
            loss = GNN_object.train(model, data, train_idx, optimizer)
            result = GNN_object.test(model, data, split_idx, evaluator)
            prog_bar.set_postfix(
                {
                    "Train Loss": loss,
                    f"Train {config.dataset.track_metric}": result["train"][config.dataset.track_metric],
                    f"Val {config.dataset.track_metric}": result["val"][config.dataset.track_metric],
                    f"Test {config.dataset.track_metric}": result["test"][config.dataset.track_metric],
                }
            )
            Logger.add_to_run(loss=loss, results=result)

        Logger.end_run()
        model.train()

        model_save_path = save_path + f"/model_{seed}.pth"
        log.info(f"saved model at {model_save_path}")
        torch.save(model.state_dict(), model_save_path)

    Logger.save_value(
        {"loss": loss, f"Test {config.dataset.track_metric}": result["test"][config.dataset.track_metric]}
    )
    Logger.save_results(save_path + "/results.json")
    Logger.get_statistics(metrics=prepare_metric_cols(config.dataset.metrics))
