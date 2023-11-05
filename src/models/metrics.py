from ogb.linkproppred import Evaluator as eval_link
from ogb.nodeproppred import Evaluator as eval_class
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    auc,
    roc_curve,
)
import torch


def directions(metric):
    if metric == "loss":
        return "-"
    return "+"


class METRICS:
    def __init__(self, metrics_list: list, task: str):
        self.metrics_list = metrics_list
        self.task = task
        self.evaluator_link = eval_link(name="ogbl-collab")
        self.evaluator_class = eval_class(name="ogbn-arxiv")

    def hits_k(self, y_pred_pos, y_pred_neg, K=50):
        self.evaluator_link.K = K
        return self.evaluator_link.eval({"y_pred_pos": y_pred_pos, "y_pred_neg": y_pred_neg})

    def ogb_acc(self, y, y_hat):
        return self.evaluator_class.eval({"y_true": y, "y_pred": y_hat})

    def accuracy(self, y, y_hat):
        return accuracy_score(y_true=y, y_pred=y_hat)

    def roc_auc(self, y, y_hat):
        return roc_auc_score(y_true=y, y_pred=y_hat, average="micro")

    def f1(self, y, y_hat):
        return f1_score(y_true=y, y_pred=y_hat, average="micro")

    def precision(self, y, y_hat):
        return precision_score(y_true=y, y_pred=y_hat, average="micro")

    def recall(self, y, y_hat):
        return recall_score(y_true=y, y_pred=y_hat, average="micro")

    def auc_metric(self, y, y_hat):
        fpr, tpr, thresholds = roc_curve(y, y_hat, pos_label=1)
        return auc(fpr, tpr)

    def collect_metrics(self, predictions: dict):
        """
        Example:
        predictions = {'train':{
        'y_pred_pos':torch.rand((100,1)),
        'y_pred_neg':torch.rand((100,1))},
        'valid':{
        'y_pred_pos':torch.rand((100,1)),
        'y_pred_neg':torch.rand((100,1))},
        'test':{
        'y_pred_pos':torch.rand((100,1)),
        'y_pred_neg':torch.rand((100,1))}
        }
        """
        results = {}
        for data_type in predictions.keys():
            inner_results = {}

            if self.task == "LinkPrediction":
                y_hat = torch.cat(
                    [predictions[data_type]["y_pred_pos"], predictions[data_type]["y_pred_neg"]], dim=0
                )
                y_hat = (y_hat >= 0.5).float().numpy()
                y_true = torch.cat(
                    [
                        torch.ones(predictions[data_type]["y_pred_pos"].shape[0]),
                        torch.zeros(predictions[data_type]["y_pred_neg"].shape[0]),
                    ]
                ).numpy()
            elif self.task == "NodeClassification":
                y_hat = predictions[data_type]["y_hat"]
                y_true = predictions[data_type]["y_true"]

            for metric in self.metrics_list:
                if metric == "f1":
                    inner_results[metric] = self.f1(y=y_true, y_hat=y_hat)
                elif metric == "acc":
                    inner_results[metric] = self.accuracy(y=y_true, y_hat=y_hat)
                elif metric == "roc_auc":
                    inner_results[metric] = self.roc_auc(y=y_true, y_hat=y_hat)
                elif metric == "precision":
                    inner_results[metric] = self.precision(y=y_true, y_hat=y_hat)
                elif metric == "recall":
                    inner_results[metric] = self.recall(y=y_true, y_hat=y_hat)
                elif metric == "auc":
                    inner_results[metric] = self.auc_metric(y=y_true, y_hat=y_hat)
                elif "hits" in metric:
                    K = int((metric.split("@"))[1])
                    inner_results[metric] = self.hits_k(
                        K=K,
                        y_pred_pos=predictions[data_type]["y_pred_pos"],
                        y_pred_neg=predictions[data_type]["y_pred_neg"],
                    )[metric]

            results[data_type] = inner_results

        return results
