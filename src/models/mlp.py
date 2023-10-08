import torch
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm


class MLP(nn.Module):
    def __init__(
        self,
        task: str,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int,
        dropout: float,
    ):
        super(MLP, self).__init__()
        self.dropout = dropout
        self.task = task

        # Create linear and batchnorm layers
        self.linear = nn.ModuleList()
        self.linear.append(torch.nn.Linear(in_channels, hidden_channels))
        self.batch_norm = torch.nn.ModuleList()
        self.batch_norm.append(torch.nn.BatchNorm1d(hidden_channels))

        for _ in range(num_layers - 2):
            self.linear.append(torch.nn.Linear(hidden_channels, hidden_channels))
            if self.task == "LinkPrediction":
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
            if self.task == "LinkPrediction":
                x = self.batch_norm[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.linear[-1](x)
        return (
            torch.sigmoid(x)
            if self.task == "LinkPrediction"
            else torch.log_softmax(x, dim=-1)
        )


# class MLP_LinkPrediction:
#     def __init__(self,task:str, device,in_channels:int,hidden_channels:int,out_channels:int,num_layers:int,dropout:float,log):
#         self.device = device
#         self.model = MLP(task=task,in_channels=in_channels,hidden_channels=hidden_channels,out_channels=out_channels,num_layers=num_layers,dropout=dropout)
#         self.model.to(device)
#         self.log=log

#     def train(self,X,split_edge,optimizer):
#         self.model.train()


class MLP_model:
    def __init__(
        self,
        task: str,
        device,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int,
        dropout: float,
        log,
    ):
        self.device = device
        self.model = MLP(
            task=task,
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.model.to(device)
        self.log = log

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
            train_acc = evaluator.eval(
                {
                    "y_true": y[split_idx["train"]],
                    "y_pred": y_hat[split_idx["train"]],
                }
            )["acc"]

            valid_acc = evaluator.eval(
                {
                    "y_true": y[split_idx["valid"]],
                    "y_pred": y_hat[split_idx["valid"]],
                }
            )["acc"]

            test_acc = evaluator.eval(
                {
                    "y_true": y[split_idx["test"]],
                    "y_pred": y_hat[split_idx["test"]],
                }
            )["acc"]

        return train_acc, valid_acc, test_acc

    def fit(self, X, y, epochs: int, split_idx, evaluator, lr):
        # self.model.reset_parameters()
        train_idx = split_idx["train"].to(self.device)
        prog_bar = tqdm(range(epochs))
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        for epoch in prog_bar:
            loss = self.train(X=X, y=y, train_idx=train_idx, optimizer=optimizer)
            result = self.test(X=X, y=y, split_idx=split_idx, evaluator=evaluator)
            prog_bar.set_postfix(
                {
                    "Train Loss": loss,
                    "Train Acc.": result[0],
                    "Val Acc.": result[1],
                    "Test Acc.": result[-1],
                }
            )
        self.log.info(
            f"Finished training - Train Loss: {loss}, Train Acc.: {result[0]}, Val Acc.: {result[1]}, Test Acc.{result[-1]}"
        )
